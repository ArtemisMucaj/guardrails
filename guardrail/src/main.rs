use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use guardrail::admin::{build_admin_app, redact_backend_url, AdminInfo, AdminState};
use guardrail::application::AppState;
use guardrail::cli::{shutdown_signal, Command, Config};
use guardrail::connector::Backend;
use guardrail::domain::metrics::{SqliteRecorder, Stats};
use guardrail::domain::provider::Provider;
use guardrail::domain::registry::Registry;
use tracing::{info, warn};

/// Ask every provider for its model list and record the routes.
///
/// Best-effort by design. A provider that cannot be reached — not started yet,
/// no credential — keeps its place in the registry and simply claims no models;
/// requests naming one of its models still reach it through the default
/// provider fallback. Failing startup here would make the multi-provider proxy
/// less robust than the single-backend one it replaces.
async fn discover_models(backend: &Backend, registry: &mut Registry) {
    // Cloned up front so the registry can be mutated while iterating. Going
    // through the `Backend` means each provider is asked with its own client —
    // Copilot's carries the credential its catalogue requires — and through
    // `target`, so a provider serving its routes at the root is asked at
    // `/models` rather than a `/v1/models` that would 404.
    let providers: Vec<Provider> = registry.providers().map(|p| (**p).clone()).collect();

    for provider in &providers {
        let name = provider.name().to_string();
        let url = provider.target("/v1/models");
        match fetch_model_ids(backend, provider, &url).await {
            Ok(ids) if ids.is_empty() => {
                info!(provider = %name, "no models reported");
            }
            Ok(ids) => {
                let mut claimed = 0usize;
                for id in &ids {
                    if registry.route(id.clone(), &name) {
                        claimed += 1;
                    } else {
                        // Another provider listed this id first. The operator's
                        // ordering decides; say so rather than silently
                        // preferring one.
                        warn!(
                            provider = %name, model = %id,
                            "model already served by an earlier provider; not routed here"
                        );
                    }
                }
                info!(provider = %name, discovered = ids.len(), routed = claimed, "models discovered");
            }
            Err(e) => {
                warn!(
                    provider = %name, error = %e,
                    "could not list models; requests naming them will fall back to the default provider"
                );
            }
        }
    }
}

/// Read `data[].id` from an OpenAI-compatible models response.
async fn fetch_model_ids(
    backend: &Backend,
    provider: &Provider,
    url: &str,
) -> anyhow::Result<Vec<String>> {
    use guardrail::application::BackendPort;

    let response = backend
        .forward(
            provider,
            reqwest::Method::GET,
            url,
            &axum::http::HeaderMap::new(),
            bytes::Bytes::new(),
        )
        .await;
    if !response.status().is_success() {
        anyhow::bail!("models request returned {}", response.status());
    }
    let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024).await?;
    let body: serde_json::Value = serde_json::from_slice(&bytes)?;
    Ok(body
        .get("data")
        .and_then(serde_json::Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|m| m.get("id").and_then(serde_json::Value::as_str))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default())
}

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
    let db_path = cfg.database_path();
    let recorder: guardrail::domain::metrics::SharedRecorder = match SqliteRecorder::open(
        &db_path,
    ) {
        Ok(recorder) => Arc::new(recorder),
        Err(e) => {
            tracing::warn!(error = %e, path = %db_path.display(), "metrics disabled: could not open database");
            Arc::new(guardrail::domain::metrics::NoopRecorder)
        }
    };

    cfg.check_admin_exposure()?;

    let mut providers = cfg.providers()?;

    // Copilot is a provider like any other once built, but it needs a
    // credential and its own HTTP client carrying it.
    let mut backend = Backend::new(client.clone());
    let copilot_login = if cfg.copilot {
        let login = guardrail::copilot::CopilotLogin::new(guardrail::copilot::default_token_path())?;
        match login.token().await {
            Some(token) => {
                let built = guardrail::copilot::provider_at(
                    token,
                    &cfg.copilot_base_url,
                    Duration::from_secs(cfg.connect_timeout_secs),
                    Duration::from_secs(cfg.read_timeout_secs),
                )?;
                backend = backend.with_client_for(built.provider.name(), built.client);
                providers.push(built.provider);
                info!("copilot provider enabled");
            }
            None => {
                // Not fatal: the operator can log in through the admin server
                // and restart, and the other providers still work meanwhile.
                warn!(
                    "copilot is enabled but no token is stored; \
                     POST /copilot/login on the admin server to authorize, then restart"
                );
            }
        }
        Some(login)
    } else {
        None
    };

    // Ask each provider which models it serves, so requests route by model
    // without the operator hand-maintaining a list. A provider that is down at
    // startup is kept, not dropped: a local server is often started after the
    // proxy, and refusing to route to it would be worse than discovering its
    // models late.
    let mut registry = Registry::new(providers).expect("providers() rejects an empty list");
    discover_models(&backend, &mut registry).await;

    // Backend URLs are operator-controlled and may embed basic-auth
    // credentials or token-bearing query params; expose only scheme/host/port.
    let providers_for_log: Vec<String> = registry
        .providers()
        .map(|p| format!("{}={}", p.name(), redact_backend_url(p.base_url())))
        .collect();

    let state = AppState::with_registry(backend, registry)
        .with_guardrails(guardrails)
        .with_recorder(recorder);

    let app = guardrail::build_app(state);

    info!(
        listen = %cfg.listen,
        admin_listen = ?cfg.admin_listen,
        providers = %providers_for_log.join(", "),
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
            providers: providers_for_log,
            proxy_listen: cfg.listen.to_string(),
            admin_listen: admin_addr.to_string(),
            max_retries: guardrails.max_retries,
            database: db_path.display().to_string(),
        };
        let mut admin_state = AdminState::new(db_path, info);
        if let Some(login) = copilot_login.clone() {
            admin_state = admin_state.with_login(login);
        }
        let admin_app = build_admin_app(admin_state);
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
