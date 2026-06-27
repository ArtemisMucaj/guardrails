use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use guardrail::admin::{build_admin_app, redact_backend_url, AdminInfo, AdminState};
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
    // proxy.
    if let Some(Command::Stats {}) = &cfg.command {
        let path = guardrail::domain::metrics::default_db_path();
        print!("{}", Stats::read(&path)?.render());
        return Ok(());
    }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
        .read_timeout(Duration::from_secs(cfg.read_timeout_secs))
        .build()?;

    let guardrails = cfg.guardrails();

    // Metrics are always on; they record to `~/.guardrails/guardrails.sql`. A
    // failure to open the database must not stop the proxy.
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
    // credentials or token-bearing query params; expose only scheme/host/port.
    let backend_for_log = redact_backend_url(&cfg.backend);

    info!(
        listen = %cfg.listen,
        admin_listen = ?cfg.admin_listen,
        backend = %backend_for_log,
        ?guardrails,
        "guardrail proxy starting"
    );

    let listener = tokio::net::TcpListener::bind(cfg.listen).await?;
    let proxy = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal());

    // The admin server is opt-in (only when `--admin-listen` is set) and runs on
    // its own port alongside the proxy, sharing the same shutdown signal.
    if let Some(admin_addr) = cfg.admin_listen {
        let admin_listener = tokio::net::TcpListener::bind(admin_addr).await?;
        let info = AdminInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend: backend_for_log,
            proxy_listen: cfg.listen.to_string(),
            admin_listen: admin_addr.to_string(),
            max_retries: guardrails.max_retries,
            metrics_db: metrics_path.display().to_string(),
        };
        let admin_app = build_admin_app(AdminState::new(metrics_path, info));
        let admin = axum::serve(admin_listener, admin_app).with_graceful_shutdown(shutdown_signal());

        info!(admin_listen = %admin_addr, "admin server starting");
        // Run both to completion; either failing surfaces its error.
        let (proxy_res, admin_res) = tokio::join!(proxy, admin);
        proxy_res?;
        admin_res?;
    } else {
        proxy.await?;
    }
    Ok(())
}
