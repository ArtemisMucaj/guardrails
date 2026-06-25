use std::time::Duration;

use clap::Parser;
use guardrail::application::{AppState, Backend};
use guardrail::cli::{shutdown_signal, Config};
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

    info!(
        listen = %cfg.listen,
        backend = %cfg.backend,
        ?guardrails,
        "guardrail proxy starting"
    );

    let listener = tokio::net::TcpListener::bind(cfg.listen).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}
