use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use guardrail::application::AppState;
use guardrail::cli::{shutdown_signal, Command, Config};
use guardrail::connector::Backend;
use guardrail::domain::metrics::{SqliteRecorder, Stats};
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

    // `stats` reads the database and prints a report instead of starting the
    // proxy. Honor an override given either after the subcommand
    // (`stats --metrics-db X`) or before it (`--metrics-db X stats`).
    if let Some(Command::Stats { metrics_db }) = &cfg.command {
        let path = metrics_db
            .clone()
            .or_else(|| cfg.metrics_db.clone())
            .unwrap_or_else(guardrail::domain::metrics::default_db_path);
        print!("{}", Stats::read(&path)?.render());
        return Ok(());
    }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
        .read_timeout(Duration::from_secs(cfg.read_timeout_secs))
        .build()?;

    let guardrails = cfg.guardrails();

    // Metrics are always on; they record to `~/.guardrails/stats.sqlite` unless
    // a path is given. A failure to open the database must not stop the proxy.
    let metrics_path = cfg.metrics_db_path();
    let recorder: guardrail::domain::metrics::SharedRecorder = match SqliteRecorder::open(
        &metrics_path,
    ) {
        Ok(recorder) => Arc::new(recorder),
        Err(e) => {
            tracing::warn!(error = %e, path = %metrics_path.display(), "metrics disabled: could not open database");
            Arc::new(guardrail::domain::metrics::NoopRecorder)
        }
    };

    let state = AppState::new(Backend::new(client), &cfg.backend)
        .with_guardrails(guardrails)
        .with_recorder(recorder);

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
