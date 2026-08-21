//! Persisted proxy configuration — which providers exist, and which of their
//! models are exposed.
//!
//! The file at `~/.guardrails/config.json` is the source of truth. CLI flags
//! seed it the first time the proxy runs and are otherwise defaults: once the
//! file exists, it wins, so a change made through the management API survives a
//! restart rather than being overwritten by whatever flags the supervisor
//! happened to pass.
//!
//! It stays hand-editable on purpose. A user without the desktop app should be
//! able to open it, see plain JSON, and change it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The whole persisted configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Providers in routing order. The first serves models no other claims.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

/// One upstream, and the exposure policy for its models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub base_url: String,

    /// Whether this provider is routed to at all. A disabled provider keeps its
    /// configuration — turning it off should not lose the exposure choices made
    /// for it — but claims no models and serves no requests.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Whether the upstream serves its routes at the root rather than under
    /// `/v1`.
    #[serde(default)]
    pub unversioned: bool,

    /// Per-model exposure. A model absent from this map takes
    /// [`Self::expose_by_default`].
    ///
    /// Storing the decision per model — rather than a list of exposed ids —
    /// means a model that disappears from a backend and comes back keeps
    /// whatever the user chose for it.
    #[serde(default)]
    pub models: BTreeMap<String, bool>,

    /// What to do with a model the user has not decided about yet.
    ///
    /// Defaults to exposing it: a new local model appearing in LM Studio should
    /// be usable without a visit to the settings screen.
    #[serde(default = "default_true")]
    pub expose_by_default: bool,
}

fn default_true() -> bool {
    true
}

impl ProviderConfig {
    pub fn new(name: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            enabled: true,
            unversioned: false,
            models: BTreeMap::new(),
            expose_by_default: true,
        }
    }

    /// Whether `model` should be listed and served.
    pub fn exposes(&self, model: &str) -> bool {
        self.models
            .get(model)
            .copied()
            .unwrap_or(self.expose_by_default)
    }

    /// Record an explicit decision for `model`.
    pub fn set_exposed(&mut self, model: impl Into<String>, exposed: bool) {
        self.models.insert(model.into(), exposed);
    }
}

impl Config {
    /// Read the config at `path`, or `None` when it does not exist.
    ///
    /// A malformed file is an error rather than a silent default: overwriting a
    /// user's hand-edited config because of a typo would lose their work.
    pub fn load(path: &Path) -> anyhow::Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(raw) => Ok(Some(serde_json::from_str(&raw).map_err(|e| {
                anyhow::anyhow!("{} is not valid config JSON: {e}", path.display())
            })?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Write the config to `path`, atomically.
    ///
    /// Written to a temporary file and renamed, so a crash mid-write cannot
    /// leave a truncated config that fails to parse on the next start.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.name == name)
    }

    pub fn provider_mut(&mut self, name: &str) -> Option<&mut ProviderConfig> {
        self.providers.iter_mut().find(|p| p.name == name)
    }

    /// Providers that should be routed to.
    pub fn enabled_providers(&self) -> impl Iterator<Item = &ProviderConfig> {
        self.providers.iter().filter(|p| p.enabled)
    }

    /// Default path, alongside the metrics database.
    pub fn default_path() -> PathBuf {
        crate::domain::metrics::default_db_path().with_file_name("config.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("guardrail-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{label}.json"))
    }

    #[test]
    fn a_model_with_no_decision_takes_the_default() {
        // A new model appearing in a backend should be usable without a trip to
        // the settings screen.
        let provider = ProviderConfig::new("lmstudio", "http://127.0.0.1:1234");
        assert!(provider.exposes("brand-new-model"));
    }

    #[test]
    fn an_explicit_decision_beats_the_default() {
        let mut provider = ProviderConfig::new("lmstudio", "http://127.0.0.1:1234");
        provider.set_exposed("noisy-model", false);
        assert!(!provider.exposes("noisy-model"));
        assert!(provider.exposes("other-model"));
    }

    #[test]
    fn expose_by_default_false_hides_undecided_models() {
        // The curated case: expose nothing except what was picked.
        let mut provider = ProviderConfig::new("remote", "https://example.com");
        provider.expose_by_default = false;
        assert!(!provider.exposes("anything"));
        provider.set_exposed("chosen", true);
        assert!(provider.exposes("chosen"));
    }

    #[test]
    fn a_decision_survives_a_model_disappearing_and_returning() {
        // Storing per-model rather than a list of exposed ids is what makes
        // this hold: a backend restarting must not silently re-expose
        // something the user hid.
        let mut provider = ProviderConfig::new("lmstudio", "http://127.0.0.1:1234");
        provider.set_exposed("gone-for-now", false);

        let path = temp_path("persist-decision");
        let config = Config {
            providers: vec![provider],
        };
        config.save(&path).unwrap();

        let loaded = Config::load(&path).unwrap().unwrap();
        assert!(!loaded.provider("lmstudio").unwrap().exposes("gone-for-now"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_round_trip_preserves_everything() {
        let path = temp_path("round-trip");
        let mut lmstudio = ProviderConfig::new("lmstudio", "http://127.0.0.1:1234");
        lmstudio.set_exposed("a", true);
        lmstudio.set_exposed("b", false);
        let mut copilot = ProviderConfig::new("copilot", "https://api.githubcopilot.com");
        copilot.unversioned = true;
        copilot.enabled = false;

        let config = Config {
            providers: vec![lmstudio, copilot],
        };
        config.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap().unwrap(), config);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_is_not_an_error() {
        // First run: no config yet, so the CLI flags seed one.
        let path = temp_path("absent");
        let _ = std::fs::remove_file(&path);
        assert_eq!(Config::load(&path).unwrap(), None);
    }

    #[test]
    fn a_malformed_file_is_an_error_rather_than_a_silent_default() {
        // Defaulting would overwrite a hand-edited config on the next save,
        // losing whatever the user was trying to express.
        let path = temp_path("malformed");
        std::fs::write(&path, "{ not json").unwrap();
        let error = Config::load(&path).unwrap_err();
        assert!(error.to_string().contains("not valid config JSON"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_disabled_provider_keeps_its_model_choices() {
        // Turning a provider off and on again should not lose curation.
        let mut provider = ProviderConfig::new("remote", "https://example.com");
        provider.set_exposed("kept", false);
        provider.enabled = false;

        let config = Config {
            providers: vec![provider],
        };
        assert_eq!(config.enabled_providers().count(), 0);
        assert!(!config.provider("remote").unwrap().exposes("kept"));
    }

    #[test]
    fn fields_absent_from_the_json_take_sensible_defaults() {
        // The file is meant to be hand-editable, so a minimal entry must work.
        let config: Config = serde_json::from_str(
            r#"{"providers":[{"name":"p","base_url":"http://h"}]}"#,
        )
        .unwrap();
        let provider = config.provider("p").unwrap();
        assert!(provider.enabled);
        assert!(provider.expose_by_default);
        assert!(!provider.unversioned);
        assert!(provider.exposes("anything"));
    }
}
