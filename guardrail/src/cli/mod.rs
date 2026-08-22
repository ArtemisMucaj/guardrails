//! CLI layer — argument parsing and process lifecycle.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::application::Guardrails;
use crate::domain::metrics::default_db_path;
use crate::domain::provider::{Provider, DEFAULT_PROVIDER};

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

    /// Address for the read-only admin server (stats, health, info), on a
    /// separate port from the proxy. Disabled unless set; bind to a loopback
    /// address (e.g. `127.0.0.1:8081`) so the metrics are not exposed off-host.
    #[arg(long, env = "GUARDRAIL_ADMIN_LISTEN")]
    pub admin_listen: Option<SocketAddr>,

    /// An OpenAI-compatible backend, as `URL` or `NAME=URL`.
    ///
    /// Repeat to proxy several. Requests route by the model they name; the
    /// first backend listed serves models no other one claims. A bare URL is
    /// named `default`, so a single `--backend` behaves exactly as before.
    ///
    /// The environment variable takes a comma-separated list.
    #[arg(
        long = "backend",
        env = "GUARDRAIL_BACKEND",
        value_delimiter = ',',
        default_value = "http://127.0.0.1:1234"
    )]
    pub backends: Vec<String>,

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

    /// Reconstruct conversations from Chat Completions traffic, so token
    /// metrics can count a resent transcript once instead of once per turn.
    ///
    /// That API is stateless and carries no conversation key, so turns are
    /// matched by their message prefix: turn N resends turn N−1's messages and
    /// appends, which identifies the predecessor. Off by default for two
    /// reasons. It is the only thing that makes the metrics path read message
    /// content — to hash it; message text is never stored, only digests — and
    /// the grouping is approximate, so the figures derived from it are reported
    /// as such. The Responses API is unaffected: it supplies real conversation
    /// edges and never needs the inference.
    #[arg(long, env = "GUARDRAIL_MATCH_CONVERSATIONS", default_value_t = false)]
    pub match_conversations: bool,

    /// Proxy GitHub Copilot models, using a Copilot subscription.
    ///
    /// Requires a device-flow login: start the proxy with `--admin-listen` and
    /// `POST /copilot/login`, or place a token at the store path. Because the
    /// proxy then holds a credential, the admin server must be bound to a
    /// loopback address.
    #[arg(long, env = "GUARDRAIL_COPILOT", default_value_t = false)]
    pub copilot: bool,

    /// Base URL of the Copilot API. Override for an enterprise deployment.
    #[arg(
        long,
        env = "GUARDRAIL_COPILOT_BASE_URL",
        default_value = "https://api.githubcopilot.com"
    )]
    pub copilot_base_url: String,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Print collected failure metrics as text and exit.
    Stats {},
}

impl Config {
    /// Path to the SQLite guardrails database, fixed at
    /// `~/.guardrails/guardrails.sql`.
    pub fn database_path(&self) -> PathBuf {
        default_db_path()
    }
}

impl Config {
    /// Build the runtime [`Guardrails`] configuration.
    pub fn guardrails(&self) -> Guardrails {
        Guardrails {
            max_retries: self.max_retries,
            match_conversations: self.match_conversations,
        }
    }

    /// Parse `--backend` into providers, in the order given.
    ///
    /// Accepts `URL` or `NAME=URL`. Splitting on the *first* `=` keeps query
    /// strings and credentials in a URL intact, since only the name is
    /// `=`-free by construction.
    pub fn providers(&self) -> anyhow::Result<Vec<Provider>> {
        let mut providers: Vec<Provider> = Vec::with_capacity(self.backends.len());

        for (index, spec) in self.backends.iter().enumerate() {
            let spec = spec.trim();
            if spec.is_empty() {
                continue;
            }
            let (name, url) = match spec.split_once('=') {
                // A bare URL: `http://host` splits at no `=`, but `scheme://`
                // means an empty name is impossible here.
                Some((name, url)) if !name.is_empty() && !name.contains("//") => {
                    (name.trim(), url.trim())
                }
                _ if index == 0 => (DEFAULT_PROVIDER, spec),
                // Only the first may be anonymous; naming the rest is what
                // makes their outcomes distinguishable in the stats.
                _ => anyhow::bail!(
                    "backend {index} ({spec}) needs a name: pass it as NAME=URL \
                     so its metrics can be told apart"
                ),
            };

            if url.is_empty() {
                anyhow::bail!("backend '{name}' has no URL");
            }
            if providers.iter().any(|p| p.name() == name) {
                anyhow::bail!(
                    "two backends are both named '{name}'; names must be unique \
                     so routing and metrics are unambiguous"
                );
            }
            providers.push(Provider::new(name, url));
        }

        if providers.is_empty() {
            anyhow::bail!("no backend configured");
        }
        Ok(providers)
    }

    /// Reject an admin server exposed off-host while a credential is held.
    ///
    /// The admin server is unauthenticated by design, and with `--copilot` it
    /// can both start a login and front a Copilot subscription. Binding it to a
    /// non-loopback address would let anything that can reach the host use that
    /// credential, so this is enforced rather than merely documented.
    pub fn check_admin_exposure(&self) -> anyhow::Result<()> {
        if !self.copilot {
            return Ok(());
        }
        if let Some(addr) = self.admin_listen {
            if !addr.ip().is_loopback() {
                anyhow::bail!(
                    "--admin-listen {addr} is not a loopback address, and --copilot makes the \
                     admin server able to start a login and use a Copilot credential. Bind it to \
                     127.0.0.1 (or [::1]) instead."
                );
            }
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn config(args: &[&str]) -> Config {
        let mut argv = vec!["guardrail"];
        argv.extend_from_slice(args);
        Config::try_parse_from(argv).expect("args should parse")
    }

    #[test]
    fn a_bare_url_is_the_default_provider() {
        // The single-backend invocation every existing user is on.
        let providers = config(&["--backend", "http://127.0.0.1:1234"])
            .providers()
            .unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].name(), DEFAULT_PROVIDER);
        assert_eq!(providers[0].base_url(), "http://127.0.0.1:1234");
    }

    #[test]
    fn no_backend_flag_still_yields_the_default() {
        let providers = config(&[]).providers().unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].base_url(), "http://127.0.0.1:1234");
    }

    #[test]
    fn named_backends_keep_their_order() {
        let providers = config(&[
            "--backend",
            "lmstudio=http://127.0.0.1:1234",
            "--backend",
            "copilot=https://api.githubcopilot.com",
        ])
        .providers()
        .unwrap();

        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0].name(), "lmstudio");
        assert_eq!(providers[1].name(), "copilot");
        assert_eq!(providers[1].base_url(), "https://api.githubcopilot.com");
    }

    #[test]
    fn a_url_containing_equals_survives_parsing() {
        // Splitting on the last `=`, or on every `=`, would corrupt a query
        // string — and the resulting URL would fail at request time, far from
        // the cause.
        let providers = config(&["--backend", "azure=https://host/openai?api-version=2024-02-01"])
            .providers()
            .unwrap();
        assert_eq!(providers[0].name(), "azure");
        assert_eq!(
            providers[0].base_url(),
            "https://host/openai?api-version=2024-02-01"
        );
    }

    #[test]
    fn only_the_first_backend_may_be_anonymous() {
        let error = config(&[
            "--backend",
            "http://127.0.0.1:1234",
            "--backend",
            "https://api.githubcopilot.com",
        ])
        .providers()
        .unwrap_err();
        assert!(
            error.to_string().contains("needs a name"),
            "got: {error}"
        );
    }

    #[test]
    fn duplicate_names_are_rejected() {
        // Two providers with one name make routing and metrics ambiguous.
        let error = config(&["--backend", "a=http://one", "--backend", "a=http://two"])
            .providers()
            .unwrap_err();
        assert!(error.to_string().contains("both named"), "got: {error}");
    }

    #[test]
    fn a_named_backend_without_a_url_is_rejected() {
        let error = config(&["--backend", "lmstudio="]).providers().unwrap_err();
        assert!(error.to_string().contains("no URL"), "got: {error}");
    }

    #[test]
    fn copilot_refuses_an_admin_server_bound_off_host() {
        // Unauthenticated + holds a credential + reachable off-host is the
        // combination worth failing at startup rather than documenting.
        let error = config(&["--copilot", "--admin-listen", "0.0.0.0:8081"])
            .check_admin_exposure()
            .unwrap_err();
        assert!(error.to_string().contains("loopback"), "got: {error}");
    }

    #[test]
    fn copilot_allows_a_loopback_admin_server() {
        for addr in ["127.0.0.1:8081", "[::1]:8081"] {
            config(&["--copilot", "--admin-listen", addr])
                .check_admin_exposure()
                .expect("loopback must be allowed");
        }
    }

    #[test]
    fn without_copilot_the_admin_bind_is_unconstrained() {
        // No credential is held, so this stays the operator's call.
        config(&["--admin-listen", "0.0.0.0:8081"])
            .check_admin_exposure()
            .expect("unchanged without --copilot");
    }

    #[test]
    fn copilot_without_an_admin_server_is_fine() {
        config(&["--copilot"]).check_admin_exposure().unwrap();
    }

    #[test]
    fn the_environment_variable_accepts_a_comma_separated_list() {
        let providers = config(&["--backend", "a=http://one,b=http://two"])
            .providers()
            .unwrap();
        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0].name(), "a");
        assert_eq!(providers[1].name(), "b");
    }
}
