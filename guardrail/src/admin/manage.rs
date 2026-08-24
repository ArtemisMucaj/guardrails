//! The management API — read and change what the proxy exposes, at runtime.
//!
//! Everything here is scoped to configuration a supervising app (Hoplon) needs
//! to drive: which providers exist, which of their models are discovered, and
//! which of those are exposed to clients.
//!
//! Changes are applied to the live registry *and* written to the config file in
//! the same call, so a restart does not undo them and a user editing the file by
//! hand sees the same shape the API produces.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::application::DiscoveryPort;
use crate::domain::config::{Config, ProviderConfig};
use crate::domain::provider::Provider;
use crate::domain::registry::Registry;

/// Live, mutable proxy configuration shared by the proxy and the admin server.
///
/// The registry behind an `RwLock` is what makes a change take effect without a
/// restart: the proxy reads it per request, the management API replaces it.
pub struct Management {
    registry: Arc<RwLock<Arc<Registry>>>,
    config: RwLock<Config>,
    config_path: std::path::PathBuf,
    /// Models each provider reported at discovery, exposed or not, so the UI can
    /// offer the full list to choose from.
    discovered: RwLock<std::collections::BTreeMap<String, Vec<openai_rs::Model>>>,
    /// How to ask a provider what it serves. `None` leaves [`Self::discover`]
    /// answering that discovery is not configured, and the catalogue stays
    /// whatever was recorded at startup.
    discovery: Option<Arc<dyn DiscoveryPort>>,
}

impl Management {
    pub fn new(
        registry: Arc<RwLock<Arc<Registry>>>,
        config: Config,
        config_path: std::path::PathBuf,
    ) -> Self {
        Self {
            registry,
            config: RwLock::new(config),
            config_path,
            discovered: RwLock::new(std::collections::BTreeMap::new()),
            discovery: None,
        }
    }

    /// Enable re-discovery, so the catalogue routing is decided from can be
    /// rebuilt without restarting the proxy.
    pub fn with_discovery(mut self, discovery: Arc<dyn DiscoveryPort>) -> Self {
        self.discovery = Some(discovery);
        self
    }

    /// Record what a provider reported at discovery.
    pub async fn set_discovered(&self, provider: &str, models: Vec<openai_rs::Model>) {
        self.discovered
            .write()
            .await
            .insert(provider.to_string(), models);
    }

    async fn snapshot(&self) -> ProvidersResponse {
        let config = self.config.read().await;
        let discovered = self.discovered.read().await;
        let registry = self.registry.read().await;

        ProvidersResponse {
            providers: config
                .providers
                .iter()
                .map(|p| {
                    let models = discovered
                        .get(&p.name)
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|model| ModelDto {
                            exposed: p.exposes(&model.id) && p.enabled,
                            routed: registry.has_route(&model.id),
                            display_name: model.display_name,
                            vendor: model.vendor,
                            id: model.id,
                        })
                        .collect();
                    ProviderDto {
                        name: p.name.clone(),
                        base_url: super::redact_backend_url(&p.base_url),
                        enabled: p.enabled,
                        expose_by_default: p.expose_by_default,
                        models,
                    }
                })
                .collect(),
        }
    }

    /// Re-ask every enabled provider what it serves, then rebuild.
    ///
    /// Without this the catalogue is a startup snapshot while `/v1/models` is
    /// answered live, so the two disagree the moment a provider loads a model:
    /// the id is advertised but nothing routes it, and with more than one
    /// provider the proxy refuses it rather than guessing. This is the path
    /// that closes that gap without a restart.
    ///
    /// Best-effort per provider, in the same sense startup discovery is. A
    /// provider that cannot be reached **keeps the catalogue it had** rather
    /// than losing it — a transient failure must not start refusing models
    /// that are still live, which would turn a refresh into an outage.
    ///
    /// An empty reply is treated the same way, as "nothing to report" rather
    /// than "nothing served". A local server answering `200 []` while it loads
    /// its index is the same transient wearing different clothes, and wiping a
    /// populated catalogue on it would refuse every model that server is about
    /// to serve. The cost is that a provider which genuinely stops serving
    /// everything keeps its stale ids until it is removed — a stale route
    /// forwards and gets the provider's own `404`, which is a better failure
    /// than refusing a model that exists.
    pub async fn discover(&self) -> anyhow::Result<Vec<ProviderDiscovery>> {
        let Some(discovery) = self.discovery.clone() else {
            anyhow::bail!("model discovery is not configured");
        };

        // Cloned out from under the lock: the fan-out below is slow (a network
        // round trip per provider) and `rebuild` takes these locks itself, so
        // holding either across the await would stall every other reader and
        // deadlock against the rebuild.
        let providers = {
            let config = self.config.read().await;
            providers_from(&config)
        };

        // Concurrently, like the `/v1/models` aggregate: one unreachable
        // provider should cost its own timeout, not everyone else's in turn.
        // Claim order is unaffected — `rebuild` decides that, in config order.
        let replies = futures_util::future::join_all(providers.iter().map(|provider| async {
            let name = provider.name().to_string();
            (name, discovery.list_models(provider).await)
        }))
        .await;

        let mut report = Vec::with_capacity(replies.len());
        {
            let mut discovered = self.discovered.write().await;
            for (name, reply) in replies {
                let (refreshed, error) = match reply {
                    Ok(models) if models.is_empty() => {
                        info!(provider = %name, "no models reported; keeping the previous catalogue");
                        (false, None)
                    }
                    Ok(models) => {
                        info!(provider = %name, discovered = models.len(), "models discovered");
                        discovered.insert(name.clone(), models);
                        (true, None)
                    }
                    Err(e) => {
                        warn!(
                            provider = %name, error = %e,
                            "could not list models; keeping the previous catalogue"
                        );
                        (false, Some(e.to_string()))
                    }
                };
                report.push(ProviderDiscovery {
                    models: discovered.get(&name).map_or(0, Vec::len),
                    name,
                    refreshed,
                    error,
                });
            }
        }

        self.rebuild().await?;
        Ok(report)
    }

    /// Rebuild the live registry from the current config and discovery.
    ///
    /// Called after every mutation so the proxy's behaviour and the stored
    /// configuration cannot drift apart, and after every refresh so a newly
    /// discovered model becomes routable.
    async fn rebuild(&self) -> anyhow::Result<()> {
        let config = self.config.read().await;
        let discovered = self.discovered.read().await;

        let Some(mut registry) = Registry::new(providers_from(&config)) else {
            anyhow::bail!("at least one provider must be enabled");
        };

        for provider in config.enabled_providers() {
            let Some(models) = discovered.get(&provider.name) else {
                continue;
            };
            let (mut routed, mut hidden) = (0usize, 0usize);
            for model in models {
                // A model the user chose not to expose is recorded as hidden
                // rather than routed, so it is neither listed nor served.
                if !provider.exposes(&model.id) {
                    registry.hide(model.id.clone(), &provider.name);
                    hidden += 1;
                } else if registry.route(model.id.clone(), &provider.name) {
                    routed += 1;
                } else {
                    // An earlier provider listed this id first. The operator's
                    // ordering decides the bare id; say so rather than silently
                    // preferring one. This copy is still reachable —
                    // `/v1/models` lists it qualified, and routing accepts that
                    // form.
                    warn!(
                        provider = %provider.name, model = %model.id,
                        qualified = %format!("{}/{}", provider.name, model.id),
                        "model already served by an earlier provider; reachable under its qualified id"
                    );
                }
            }
            debug!(provider = %provider.name, discovered = models.len(), routed, hidden, "routes rebuilt");
        }

        *self.registry.write().await = Arc::new(registry);
        Ok(())
    }

    async fn persist(&self) -> anyhow::Result<()> {
        self.config.read().await.save(&self.config_path)
    }
}

/// The providers an enabled configuration describes.
///
/// Shared by [`Management::rebuild`] and [`Management::discover`] so the set
/// asked for a catalogue is exactly the set routed to. Building them
/// separately is how a provider ends up discovered but unreachable, or asked
/// at the wrong routes.
fn providers_from(config: &Config) -> Vec<Provider> {
    config
        .enabled_providers()
        .map(|p| {
            let provider = Provider::new(&p.name, &p.base_url);
            let provider = if p.unversioned {
                provider.unversioned()
            } else {
                provider
            };
            // Copilot's credential lives on its HTTP client, which the Backend
            // still holds by name; the reserved headers must be re-declared
            // here or a rebuild would drop them.
            if p.name == crate::copilot::COPILOT_PROVIDER {
                provider.owning_credential().reserving([
                    "copilot-integration-id",
                    "editor-version",
                    "editor-plugin-version",
                    "x-github-api-version",
                    "openai-intent",
                    "user-agent",
                ])
            } else {
                provider
            }
        })
        .collect()
}

// ── DTOs ────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ProvidersResponse {
    providers: Vec<ProviderDto>,
}

/// What a discovery run did to one provider's catalogue.
#[derive(Serialize)]
pub struct ProviderDiscovery {
    pub name: String,
    /// Models in the catalogue now in effect for this provider — the fresh
    /// reply when there was one, otherwise the one it kept.
    pub models: usize,
    /// Whether this run replaced that catalogue with a fresh reply.
    /// `false` with no `error` means the provider answered, but reported
    /// nothing, so what it had was kept.
    pub refreshed: bool,
    /// Why the provider could not be asked, when it could not be. Reported per
    /// provider rather than failing the call, because the other providers were
    /// refreshed and the caller should see that.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// What discovery found, and the state it produced.
///
/// Carries the same `providers` snapshot the other management endpoints answer
/// with, so a caller re-runs discovery and re-reads the result in one round
/// trip.
#[derive(Serialize)]
struct DiscoveryResponse {
    discovery: Vec<ProviderDiscovery>,
    providers: Vec<ProviderDto>,
}

#[derive(Serialize)]
struct ProviderDto {
    name: String,
    /// Reduced to scheme/host/port — a base URL may embed credentials.
    base_url: String,
    enabled: bool,
    expose_by_default: bool,
    models: Vec<ModelDto>,
}

#[derive(Serialize)]
struct ModelDto {
    id: String,
    /// Present when the provider names the model distinctly from its id — a
    /// picker should show this instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
    /// Vendor or owning organisation, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    vendor: Option<String>,
    /// Whether clients can see and use this model.
    exposed: bool,
    /// Whether the live registry currently routes it. Differs from `exposed`
    /// when a change has been made but the model was never discovered.
    routed: bool,
}

#[derive(Deserialize)]
pub struct ExposureUpdate {
    /// Per-model exposure to set. Models not named are left as they are.
    #[serde(default)]
    models: std::collections::BTreeMap<String, bool>,
    /// Drop every stored per-model decision before applying `models`, so the
    /// provider falls back to `expose_by_default` for anything not named.
    ///
    /// Decisions are stored per model and outlive the model — deliberately, so
    /// one that disappears and returns keeps its setting. That also means a
    /// caller working from a list of *currently offered* models cannot undo a
    /// decision for a model the provider has since stopped offering: setting
    /// what it can see leaves the rest stranded. Clearing first is the only way
    /// to express "whatever is remembered, forget it".
    #[serde(default)]
    clear_models: bool,
    /// Whether the provider is routed to at all.
    #[serde(default)]
    enabled: Option<bool>,
    /// What to do with models the user has not decided about.
    #[serde(default)]
    expose_by_default: Option<bool>,
}

#[derive(Deserialize)]
pub struct AddProvider {
    name: String,
    base_url: String,
    #[serde(default)]
    unversioned: bool,
    #[serde(default)]
    expose_by_default: Option<bool>,
}

// ── Handlers ────────────────────────────────────────────────────────────────

pub(super) async fn list_providers(State(state): State<super::AdminState>) -> Response {
    let Some(management) = state.management.as_ref() else {
        return disabled();
    };
    Json(management.snapshot().await).into_response()
}

/// `PATCH /providers/:name` — change exposure for one provider.
pub(super) async fn update_provider(
    State(state): State<super::AdminState>,
    Path(name): Path<String>,
    Json(update): Json<ExposureUpdate>,
) -> Response {
    let Some(management) = state.management.as_ref() else {
        return disabled();
    };

    {
        let mut config = management.config.write().await;
        let Some(provider) = config.provider_mut(&name) else {
            return not_found(&name);
        };
        if update.clear_models {
            provider.models.clear();
        }
        for (model, exposed) in &update.models {
            provider.set_exposed(model.clone(), *exposed);
        }
        if let Some(enabled) = update.enabled {
            provider.enabled = enabled;
        }
        if let Some(default) = update.expose_by_default {
            provider.expose_by_default = default;
        }
    }

    apply(management, &format!("updated provider {name}")).await
}

/// `POST /discovery` — re-ask every enabled provider what it serves.
///
/// The answer to a model that was loaded into a provider after the proxy
/// started: discovery is otherwise a startup snapshot, so the id shows up in
/// `GET /v1/models` (served live) while routing refuses it as unserved.
///
/// Partial failure is reported, not raised: an unreachable provider keeps the
/// catalogue it had and says so in its entry, because the providers that *did*
/// answer have been refreshed and the caller should get that result. Only a
/// rebuild that would leave nothing to route to is an error.
pub(super) async fn run_discovery(State(state): State<super::AdminState>) -> Response {
    let Some(management) = state.management.as_ref() else {
        return disabled();
    };

    let discovery = match management.discover().await {
        Ok(discovery) => discovery,
        Err(e) => {
            warn!(error = %e, "could not re-run provider discovery");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    let ProvidersResponse { providers } = management.snapshot().await;
    info!(
        providers = discovery.len(),
        replaced = discovery.iter().filter(|p| p.refreshed).count(),
        "re-ran provider discovery"
    );
    Json(DiscoveryResponse {
        discovery,
        providers,
    })
    .into_response()
}

/// `POST /providers` — add a provider.
pub(super) async fn add_provider(
    State(state): State<super::AdminState>,
    Json(new): Json<AddProvider>,
) -> Response {
    let Some(management) = state.management.as_ref() else {
        return disabled();
    };

    {
        let mut config = management.config.write().await;
        if config.provider(&new.name).is_some() {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": format!("a provider named `{}` already exists", new.name)
                })),
            )
                .into_response();
        }
        let mut provider = ProviderConfig::new(&new.name, &new.base_url);
        provider.unversioned = new.unversioned;
        if let Some(default) = new.expose_by_default {
            provider.expose_by_default = default;
        }
        config.providers.push(provider);
    }

    apply(management, &format!("added provider {}", new.name)).await
}

/// `DELETE /providers/:name` — remove a provider.
pub(super) async fn remove_provider(
    State(state): State<super::AdminState>,
    Path(name): Path<String>,
) -> Response {
    let Some(management) = state.management.as_ref() else {
        return disabled();
    };

    {
        let mut config = management.config.write().await;
        let before = config.providers.len();
        config.providers.retain(|p| p.name != name);
        if config.providers.len() == before {
            return not_found(&name);
        }
    }

    // Forget what it reported, too. The catalogue is keyed by name and outlives
    // the entry otherwise, so re-adding a provider under the same name would
    // resurrect models it may no longer serve — routed immediately, before it
    // has been asked anything.
    management.discovered.write().await.remove(&name);

    apply(management, &format!("removed provider {name}")).await
}

/// Rebuild the registry, persist, and answer with the new state.
///
/// A rebuild that would leave nothing to route to is refused and rolled back by
/// reloading from disk, so the proxy is never left unable to serve.
async fn apply(management: &Arc<Management>, what: &str) -> Response {
    if let Err(e) = management.rebuild().await {
        warn!(error = %e, "rejected a configuration change");
        // Restore whatever is on disk so the in-memory config does not keep a
        // change the registry refused.
        if let Ok(Some(config)) = Config::load(&management.config_path) {
            *management.config.write().await = config;
        }
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response();
    }

    if let Err(e) = management.persist().await {
        // The change is live but not durable; say so rather than reporting
        // success and losing it at the next restart.
        warn!(error = %e, "applied a change but could not persist it");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("change applied but not saved: {e}"),
            })),
        )
            .into_response();
    }

    info!("{what}");
    Json(management.snapshot().await).into_response()
}

fn disabled() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "error": "the management API is not enabled" })),
    )
        .into_response()
}

fn not_found(name: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "error": format!("no provider named `{name}`") })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reserved-header list in `providers_from` is a hand-copy of what
    /// `copilot::provider` puts on the real thing — it has to be, because a
    /// rebuild has no token to build an endpoint from. This is what keeps the
    /// copy honest.
    ///
    /// It matters more than it used to: startup now goes through the same
    /// rebuild, so a name drifting out of this list would drop a gating header
    /// on every Copilot request rather than only on those made after the first
    /// configuration change.
    #[test]
    fn a_rebuilt_copilot_provider_reserves_what_the_built_one_does() {
        let built = crate::copilot::provider(
            gh_copilot_rs::CopilotToken::new("ghu_test_token"),
            std::time::Duration::from_secs(10),
            std::time::Duration::from_secs(300),
        )
        .expect("the provider builds with a well-formed token");

        let mut entry =
            ProviderConfig::new(crate::copilot::COPILOT_PROVIDER, built.provider.base_url());
        entry.unversioned = true;
        let rebuilt = providers_from(&Config {
            providers: vec![entry],
        });

        assert_eq!(rebuilt.len(), 1);
        let expected: Vec<&str> = built.provider.reserved_headers().collect();
        let actual: Vec<&str> = rebuilt[0].reserved_headers().collect();
        assert_eq!(
            actual, expected,
            "a rebuild must reserve exactly what the credential-carrying provider does"
        );
        assert!(rebuilt[0].is_unversioned(), "Copilot serves at the root");
    }

    /// A disabled provider is not asked and not routed to — the same list feeds
    /// both, so this is one assertion rather than two.
    #[test]
    fn a_disabled_provider_is_not_built() {
        let mut off = ProviderConfig::new("beta", "http://127.0.0.1:2");
        off.enabled = false;
        let providers = providers_from(&Config {
            providers: vec![ProviderConfig::new("alpha", "http://127.0.0.1:1"), off],
        });
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].name(), "alpha");
    }
}
