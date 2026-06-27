//! CLI layer — argument parsing and process lifecycle.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::application::Guardrails;
use crate::domain::metrics::default_db_path;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "guardrail",
    about = "Transparent OpenAI chat-completions proxy with tool-call guardrails"
)]
pub struct Config {
    /// Subcommand to run. When omitted, the proxy server starts.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Address the proxy listens on.
    #[arg(long, env = "GUARDRAIL_LISTEN", default_value = "127.0.0.1:8080")]
    pub listen: SocketAddr,

    /// Base URL of the OpenAI-compatible backend.
    #[arg(
        long,
        env = "GUARDRAIL_BACKEND",
        default_value = "http://127.0.0.1:1234"
    )]
    pub backend: String,

    /// Timeout for establishing the TCP/TLS connection to the backend, in seconds.
    #[arg(long, env = "GUARDRAIL_CONNECT_TIMEOUT_SECS", default_value_t = 10)]
    pub connect_timeout_secs: u64,

    /// Maximum idle gap between read chunks of the backend response, in seconds.
    #[arg(long, env = "GUARDRAIL_READ_TIMEOUT_SECS", default_value_t = 300)]
    pub read_timeout_secs: u64,

    /// Maximum corrective retries before falling back to the model's last text.
    /// Set to `0` to disable retries while keeping the other repairs.
    #[arg(long, env = "GUARDRAIL_MAX_RETRIES", default_value_t = 2)]
    pub max_retries: u32,

    /// Path to the SQLite failure-metrics database. Defaults to
    /// `~/.guardrails/guardrails.sql`; one row is recorded per guarded request.
    #[arg(long, env = "GUARDRAIL_METRICS_DB")]
    pub metrics_db: Option<PathBuf>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Print collected failure metrics as text and exit.
    Stats {
        /// Database to read. Defaults to `~/.guardrails/guardrails.sql`.
        #[arg(long, env = "GUARDRAIL_METRICS_DB")]
        metrics_db: Option<PathBuf>,
    },
}

impl Config {
    /// Resolve the metrics database path, applying the default home location.
    pub fn metrics_db_path(&self) -> PathBuf {
        self.metrics_db.clone().unwrap_or_else(default_db_path)
    }
}

impl Config {
    /// Build the runtime [`Guardrails`] configuration.
    pub fn guardrails(&self) -> Guardrails {
        Guardrails {
            max_retries: self.max_retries,
        }
    }
}

/// Resolve when the process receives Ctrl-C (SIGINT) or, on Unix, SIGTERM.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {}
            Err(e) => {
                tracing::error!(error = %e, "failed to listen for Ctrl-C");
                // Setup failed: never resolve, so select! doesn't read this as a
                // shutdown signal and exit the server prematurely.
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::info!("shutdown signal received");
}
