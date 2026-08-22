//! The provider registry — which upstream serves a given model.
//!
//! Routing is by model id, because that is the only thing an OpenAI-compatible
//! client tells the proxy about where a request should go. The registry maps
//! ids to providers and falls back to a default, so a model the proxy has never
//! heard of still reaches somewhere sensible instead of failing.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::provider::Provider;

/// Providers, and the model ids each one serves.
///
/// Always holds at least one provider — the default — so routing can never fail
/// to produce a target.
#[derive(Debug, Clone)]
pub struct Registry {
    providers: Vec<Arc<Provider>>,
    /// Model id → index into `providers`.
    routes: BTreeMap<String, usize>,
    /// Model ids discovered but deliberately not exposed. Kept apart from
    /// `routes` so a hidden model is distinguishable from one that was never
    /// discovered: the first is refused, the second falls back.
    hidden: BTreeSet<String>,
    /// Index of the provider serving unknown and absent models.
    default: usize,
}

impl Registry {
    /// A registry with a single provider serving every request, matching the
    /// behaviour of a single `--backend`.
    pub fn single(provider: Provider) -> Self {
        Self {
            providers: vec![Arc::new(provider)],
            routes: BTreeMap::new(),
            hidden: BTreeSet::new(),
            default: 0,
        }
    }

    /// Build from an ordered list. The first provider is the default.
    ///
    /// Returns `None` if `providers` is empty: a registry with nothing to route
    /// to has no meaningful behaviour, and callers should fail at startup with
    /// a clear message rather than construct one.
    /// Add `provider`, replacing any existing one of the same name.
    ///
    /// Two providers sharing a name is never meaningful: [`Self::route`]
    /// resolves by first match, so the later one is unreachable, and `/info`
    /// reports the name twice. It happens when a provider is both described in
    /// the configuration and constructed in code — Copilot, whose entry is a
    /// name and a URL but whose working provider carries an OAuth credential on
    /// its own HTTP client. The constructed one must win.
    pub fn replacing(providers: &mut Vec<Provider>, provider: Provider) {
        let name = provider.name().to_string();
        providers.retain(|p| p.name() != name);
        providers.push(provider);
    }

    pub fn new(providers: Vec<Provider>) -> Option<Self> {
        if providers.is_empty() {
            return None;
        }
        Some(Self {
            providers: providers.into_iter().map(Arc::new).collect(),
            routes: BTreeMap::new(),
            hidden: BTreeSet::new(),
            default: 0,
        })
    }

    /// Route `model` to the provider named `provider`.
    ///
    /// First registration wins: discovery queries providers in configuration
    /// order, so an id served by several upstreams goes to the one the operator
    /// listed first. Returns whether the route was newly claimed — `false`
    /// means another provider already serves this id, which the caller may want
    /// to log.
    pub fn route(&mut self, model: impl Into<String>, provider: &str) -> bool {
        let Some(index) = self.providers.iter().position(|p| p.name() == provider) else {
            return false;
        };
        let model = model.into();
        if self.routes.contains_key(&model) {
            return false;
        }
        self.routes.insert(model, index);
        true
    }

    /// Record `model` as discovered but not exposed.
    pub fn hide(&mut self, model: impl Into<String>) {
        let model = model.into();
        self.routes.remove(&model);
        self.hidden.insert(model);
    }

    /// Whether `model` was discovered and deliberately hidden.
    pub fn is_hidden(&self, model: &str) -> bool {
        self.hidden.contains(model)
    }

    /// The provider serving `model`, or the default when the id is unknown.
    ///
    /// Unknown ids fall back rather than erroring: discovery can miss a model
    /// that a backend loaded after startup, and refusing those would make the
    /// proxy less useful than the single-backend version it replaces.
    ///
    /// A *hidden* id is not unknown — the user decided against it — so callers
    /// should check [`Self::is_hidden`] first and refuse rather than falling
    /// back, or hiding a model would silently route it somewhere else.
    pub fn resolve(&self, model: Option<&str>) -> &Arc<Provider> {
        model
            .and_then(|model| self.routes.get(model))
            .and_then(|&index| self.providers.get(index))
            .unwrap_or_else(|| self.default_provider())
    }

    /// The provider handling unrouted traffic.
    pub fn default_provider(&self) -> &Arc<Provider> {
        &self.providers[self.default]
    }

    /// Every provider, in configuration order.
    pub fn providers(&self) -> impl Iterator<Item = &Arc<Provider>> {
        self.providers.iter()
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        false // `new` rejects an empty list; `single` always has one.
    }

    /// Whether an explicit route exists for `model`.
    pub fn has_route(&self, model: &str) -> bool {
        self.routes.contains_key(model)
    }

    /// Model ids with an explicit route, paired with their provider's name.
    pub fn routes(&self) -> impl Iterator<Item = (&str, &str)> {
        self.routes
            .iter()
            .map(|(model, &index)| (model.as_str(), self.providers[index].name()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_provider_replaces_one_of_the_same_name() {
        // Copilot arrives twice: once from the config (a name and a URL) and
        // once built with its credential. Keeping both left `route` resolving
        // to whichever came first and `/info` listing the name twice.
        let mut providers = vec![
            Provider::new("mlx", "http://127.0.0.1:8000"),
            Provider::new("copilot", "https://api.githubcopilot.com"),
        ];
        Registry::replacing(
            &mut providers,
            Provider::new("copilot", "https://api.githubcopilot.com").unversioned(),
        );

        assert_eq!(providers.len(), 2, "no duplicate entry");
        assert_eq!(providers.iter().filter(|p| p.name() == "copilot").count(), 1);
        // The replacement is the one kept, credential and routing rules
        // included -- here observable through `unversioned`, which the config
        // one lacked.
        let copilot = providers.iter().find(|p| p.name() == "copilot").unwrap();
        assert!(copilot.is_unversioned(), "the constructed provider must win");
        // Untouched providers keep their place.
        assert_eq!(providers[0].name(), "mlx");
    }

    fn registry() -> Registry {
        Registry::new(vec![
            Provider::new("lmstudio", "http://127.0.0.1:1234"),
            Provider::new("copilot", "https://api.githubcopilot.com"),
        ])
        .expect("two providers is not empty")
    }

    #[test]
    fn routes_a_model_to_its_provider() {
        let mut registry = registry();
        assert!(registry.route("gpt-4o", "copilot"));
        assert!(registry.route("qwen2.5-7b", "lmstudio"));

        assert_eq!(registry.resolve(Some("gpt-4o")).name(), "copilot");
        assert_eq!(registry.resolve(Some("qwen2.5-7b")).name(), "lmstudio");
    }

    #[test]
    fn an_unknown_model_falls_back_to_the_default() {
        // Discovery can miss a model loaded after startup; refusing those would
        // be a regression against the single-backend proxy.
        let mut registry = registry();
        registry.route("gpt-4o", "copilot");
        assert_eq!(registry.resolve(Some("never-heard-of-it")).name(), "lmstudio");
    }

    #[test]
    fn an_absent_model_falls_back_to_the_default() {
        // Non-chat paths carry no model at all.
        assert_eq!(registry().resolve(None).name(), "lmstudio");
    }

    #[test]
    fn the_first_provider_is_the_default() {
        assert_eq!(registry().default_provider().name(), "lmstudio");
    }

    #[test]
    fn a_duplicate_model_id_keeps_the_first_registration() {
        // gpt-4o is reachable through several vendors; the operator's ordering
        // decides, and the loser is reported so it can be logged.
        let mut registry = registry();
        assert!(registry.route("gpt-4o", "lmstudio"));
        assert!(
            !registry.route("gpt-4o", "copilot"),
            "the second claim must be refused, not silently applied"
        );
        assert_eq!(registry.resolve(Some("gpt-4o")).name(), "lmstudio");
    }

    #[test]
    fn routing_to_an_unknown_provider_is_refused() {
        let mut registry = registry();
        assert!(!registry.route("gpt-4o", "no-such-provider"));
        assert!(!registry.has_route("gpt-4o"));
    }

    #[test]
    fn a_hidden_model_is_distinguishable_from_an_unknown_one() {
        // The distinction that keeps hiding meaningful: an unknown id falls
        // back to the default provider, a hidden one must be refused. Without
        // it, hiding a model would route it to the default instead.
        let mut registry = registry();
        registry.route("gpt-4o", "copilot");
        registry.hide("gpt-4o");

        assert!(registry.is_hidden("gpt-4o"));
        assert!(!registry.has_route("gpt-4o"));
        assert!(!registry.is_hidden("never-discovered"));
    }

    #[test]
    fn hiding_removes_the_route_so_it_is_not_listed() {
        let mut registry = registry();
        registry.route("gpt-4o", "copilot");
        registry.route("qwen2.5-7b", "lmstudio");
        registry.hide("gpt-4o");

        let routes: Vec<(&str, &str)> = registry.routes().collect();
        assert_eq!(routes, vec![("qwen2.5-7b", "lmstudio")]);
    }

    #[test]
    fn a_single_provider_registry_serves_everything() {
        let registry = Registry::single(Provider::new("default", "http://127.0.0.1:1234"));
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.resolve(Some("anything")).name(), "default");
        assert_eq!(registry.resolve(None).name(), "default");
    }

    #[test]
    fn an_empty_provider_list_is_rejected() {
        assert!(Registry::new(vec![]).is_none());
    }

    #[test]
    fn routes_are_enumerable_for_reporting() {
        let mut registry = registry();
        registry.route("gpt-4o", "copilot");
        registry.route("qwen2.5-7b", "lmstudio");
        let routes: Vec<(&str, &str)> = registry.routes().collect();
        assert_eq!(
            routes,
            vec![("gpt-4o", "copilot"), ("qwen2.5-7b", "lmstudio")]
        );
    }
}
