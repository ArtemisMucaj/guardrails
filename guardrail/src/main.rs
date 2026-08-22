use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use guardrail::admin::{build_admin_app, redact_backend_url, AdminInfo, AdminState};
use guardrail::application::AppState;
use guardrail::cli::{shutdown_signal, Command, Config};
use guardrail::connector::Backend;
use guardrail::domain::metrics::{SqliteRecorder, Stats};
use guardrail::admin::manage::Management;
use guardrail::domain::config::{Config as ProxyConfig, ProviderConfig};
use guardrail::domain::provider::Provider;
use openai_rs::{ApiRoutes, Model, ModelCatalog, OpenAiModelCatalog, Transport};
use guardrail::domain::registry::Registry;
use tracing::{info, warn};

/// Ask every provider for its model list and record the routes.
///
/// Best-effort by design. A provider that cannot be reached — not started yet,
/// no credential — keeps its place in the registry and simply claims no models;
/// requests naming one of its models still reach it through the default
/// provider fallback. Failing startup here would make the multi-provider proxy
/// less robust than the single-backend one it replaces.
async fn discover_models(
    backend: &Backend,
    registry: &mut Registry,
    config: &ProxyConfig,
    management: Option<&Arc<Management>>,
) {
    // Cloned up front so the registry can be mutated while iterating. Going
    // through the `Backend` means each provider is asked with its own client —
    // Copilot's carries the credential its catalogue requires — and through
    // `target`, so a provider serving its routes at the root is asked at
    // `/models` rather than a `/v1/models` that would 404.
    let providers: Vec<Provider> = registry.providers().map(|p| (**p).clone()).collect();

    for provider in &providers {
        let name = provider.name().to_string();
        match fetch_models(backend, provider).await {
            Ok(models) if models.is_empty() => {
                info!(provider = %name, "no models reported");
            }
            Ok(models) => {
                if let Some(management) = management {
                    management.set_discovered(&name, models.clone()).await;
                }
                let mut claimed = 0usize;
                let mut hidden = 0usize;
                for id in models.iter().map(|m| m.id.clone()).collect::<Vec<_>>().iter() {
                    // A model the user chose not to expose is recorded as
                    // hidden rather than routed, so it is neither listed nor
                    // served.
                    if !config
                        .provider(&name)
                        .map(|p| p.exposes(id))
                        .unwrap_or(true)
                    {
                        registry.hide(id.clone());
                        hidden += 1;
                        continue;
                    }
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
                info!(provider = %name, discovered = models.len(), routed = claimed, hidden, "models discovered");
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

/// List a provider's models.
///
/// Delegates to `openai-rs`, which already normalises the several shapes an
/// OpenAI-compatible `/models` response comes in into one `Model` type. It is
/// built on the provider's own HTTP client — Copilot's carries the credential
/// its catalogue requires — via the transport escape hatch, so this shares the
/// connection pool and auth rather than constructing a second client that would
/// not be authenticated.
async fn fetch_models(backend: &Backend, provider: &Provider) -> anyhow::Result<Vec<Model>> {
    let routes = if provider.is_unversioned() {
        ApiRoutes::unversioned()
    } else {
        ApiRoutes::default()
    };
    let transport = Transport::with_http_client(
        backend.client_for(provider).clone(),
        provider.base_url(),
        routes,
    );
    Ok(OpenAiModelCatalog::with_transport(transport)
        .list_models()
        .await?)
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

    // The config file is the source of truth once it exists; CLI flags seed it
    // on first run and are otherwise defaults. Without this, a change made
    // through the management API would be overwritten by whatever flags the
    // supervising app happened to pass on the next restart.
    let config_path = ProxyConfig::default_path();
    let mut config = match ProxyConfig::load(&config_path)? {
        Some(config) => {
            info!(path = %config_path.display(), "loaded configuration");
            config
        }
        None => {
            let seeded = ProxyConfig {
                providers: cfg
                    .providers()?
                    .iter()
                    .map(|p| ProviderConfig::new(p.name(), p.base_url()))
                    .collect(),
            };
            seeded.save(&config_path)?;
            info!(path = %config_path.display(), "seeded configuration from the command line");
            seeded
        }
    };

    let mut providers: Vec<Provider> = config
        .enabled_providers()
        .map(|p| {
            let provider = Provider::new(&p.name, &p.base_url);
            if p.unversioned {
                provider.unversioned()
            } else {
                provider
            }
        })
        .collect();

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
                // Copilot serves its routes at the root, not under `/v1`, so
                // its entry must say so. Correcting an existing entry matters
                // as much as seeding a new one: the management API rebuilds
                // providers from this config, and an entry written before this
                // was set — or edited by hand — makes every rebuild target
                // `/v1/...`, which Copilot answers with 404. The proxy then
                // serves no Copilot models until the next restart, while the
                // admin API still reports them as routed.
                let path = built.provider.base_url().to_string();
                match config.provider_mut(built.provider.name()) {
                    Some(entry) if !entry.unversioned => {
                        entry.unversioned = true;
                        config.save(&config_path)?;
                    }
                    Some(_) => {}
                    None => {
                        let mut entry = ProviderConfig::new(built.provider.name(), path);
                        entry.unversioned = true;
                        config.providers.push(entry);
                        config.save(&config_path)?;
                    }
                }
                if config
                    .provider(built.provider.name())
                    .is_some_and(|p| p.enabled)
                {
                    // Replace, never append. The config already contributed a
                    // `copilot` entry above, but that one is built from a name
                    // and a URL alone — it cannot carry the OAuth credential or
                    // GitHub's client-identity headers, which live on this
                    // provider's own HTTP client. Pushing beside it left two
                    // providers of the same name: `Registry` keeps both and
                    // resolves by first match, so which one served depended on
                    // ordering, and `/info` listed Copilot twice.
                    Registry::replacing(&mut providers, built.provider);
                    info!("copilot provider enabled");
                }
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
    let Some(mut registry) = Registry::new(providers) else {
        anyhow::bail!(
            "no provider is enabled — enable one in {} or pass --backend",
            config_path.display()
        );
    };

    // Built before discovery so it can record what each provider reports; the
    // registry it shares is replaced wholesale on every change.
    let shared_registry: guardrail::application::SharedRegistry =
        Arc::new(tokio::sync::RwLock::new(Arc::new(registry.clone())));
    let management = Management::new(
        shared_registry.clone(),
        config.clone(),
        config_path.clone(),
    );

    discover_models(&backend, &mut registry, &config, Some(&management)).await;
    *shared_registry.write().await = Arc::new(registry.clone());

    // Backend URLs are operator-controlled and may embed basic-auth
    // credentials or token-bearing query params; expose only scheme/host/port.
    let providers_for_log: Vec<String> = registry
        .providers()
        .map(|p| format!("{}={}", p.name(), redact_backend_url(p.base_url())))
        .collect();

    let state = AppState::with_shared_registry(backend, shared_registry.clone())
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
        let mut admin_state = AdminState::new(db_path, info).with_management(management.clone());
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
