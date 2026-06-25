//! CLI layer — argument parsing and process lifecycle.

use std::net::SocketAddr;

use clap::{ArgAction, Parser};

use crate::domain::guardrails::Guardrails;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "guardrail",
    about = "Transparent OpenAI chat-completions proxy with tool-call guardrails"
)]
pub struct Config {
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

    /// Rescue malformed tool calls from model text. On by default; pass
    /// `--rescue false` (or `GUARDRAIL_RESCUE=false`) to disable.
    #[arg(long, env = "GUARDRAIL_RESCUE", default_value_t = true, action = ArgAction::Set)]
    pub rescue: bool,

    /// Inject the synthetic `respond` tool and unwrap it to text. On by default;
    /// `--respond false` disables.
    #[arg(long, env = "GUARDRAIL_RESPOND", default_value_t = true, action = ArgAction::Set)]
    pub respond: bool,

    /// Retry the backend with a corrective nudge when a tool call fails
    /// validation. On by default; `--retry false` disables.
    #[arg(long, env = "GUARDRAIL_RETRY", default_value_t = true, action = ArgAction::Set)]
    pub retry: bool,

    /// Maximum corrective retries before falling back to the model's last text.
    #[arg(long, env = "GUARDRAIL_MAX_RETRIES", default_value_t = 2)]
    pub max_retries: u32,
}

impl Config {
    /// Collect the per-guardrail toggles into the runtime [`Guardrails`] set.
    pub fn guardrails(&self) -> Guardrails {
        Guardrails {
            rescue: self.rescue,
            respond: self.respond,
            retry: self.retry,
            max_retries: self.max_retries,
        }
    }
}

/// Resolve when the process receives Ctrl-C (SIGINT) or, on Unix, SIGTERM.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %e, "failed to listen for Ctrl-C");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => tracing::error!(error = %e, "failed to install SIGTERM handler"),
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
