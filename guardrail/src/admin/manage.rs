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
use tracing::{info, warn};

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
}

impl Management {
    pub fn new(
        registry: Arc<RwLock<Arc<Registry>>>,
        config: Config,
        config_path: std::path::PathBuf,
    ) -> Arc<Self> {
        Arc::new(Self {
            registry,
            config: RwLock::new(config),
            config_path,
            discovered: RwLock::new(std::collections::BTreeMap::new()),
        })
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

    /// Rebuild the live registry from the current config and discovery.
    ///
    /// Called after every mutation so the proxy's behaviour and the stored
    /// configuration cannot drift apart.
    async fn rebuild(&self) -> anyhow::Result<()> {
        let config = self.config.read().await;
        let discovered = self.discovered.read().await;

        let providers: Vec<Provider> = config
            .enabled_providers()
            .map(|p| {
                let provider = Provider::new(&p.name, &p.base_url);
                let provider = if p.unversioned {
                    provider.unversioned()
                } else {
                    provider
                };
                // Copilot's credential lives on its HTTP client, which the
                // Backend still holds by name; the reserved headers must be
                // re-declared here or a rebuild would drop them.
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
            .collect();

        let Some(mut registry) = Registry::new(providers) else {
            anyhow::bail!("at least one provider must be enabled");
        };

        for provider in config.enabled_providers() {
            let Some(models) = discovered.get(&provider.name) else {
                continue;
            };
            for model in models {
                if provider.exposes(&model.id) {
                    registry.route(model.id.clone(), &provider.name);
                } else {
                    registry.hide(model.id.clone());
                }
            }
        }

        *self.registry.write().await = Arc::new(registry);
        Ok(())
    }

    async fn persist(&self) -> anyhow::Result<()> {
        self.config.read().await.save(&self.config_path)
    }
}

// ── DTOs ────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ProvidersResponse {
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
