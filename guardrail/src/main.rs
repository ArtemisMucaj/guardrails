use std::time::Duration;

use clap::Parser;
use guardrail::application::AppState;
use guardrail::cli::{shutdown_signal, Config};
use guardrail::connector::Backend;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "guardrail=info,warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cfg = Config::parse();

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
        .read_timeout(Duration::from_secs(cfg.read_timeout_secs))
        .build()?;

    let guardrails = cfg.guardrails();
    let state = AppState::new(Backend::new(client), &cfg.backend).with_guardrails(guardrails);
    let app = guardrail::build_app(state);

    // The backend URL is operator-controlled and may embed basic-auth
    // credentials or token-bearing query params; log only scheme/host/port.
    let backend_for_log = reqwest::Url::parse(&cfg.backend)
        .ok()
        .and_then(|url| {
            let host = url.host_str()?;
            Some(match url.port() {
                Some(port) => format!("{}://{}:{}", url.scheme(), host, port),
                None => format!("{}://{}", url.scheme(), host),
            })
        })
        .unwrap_or_else(|| "<redacted>".to_string());

    info!(
        listen = %cfg.listen,
        backend = %backend_for_log,
        ?guardrails,
        "guardrail proxy starting"
    );

    let listener = tokio::net::TcpListener::bind(cfg.listen).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}
