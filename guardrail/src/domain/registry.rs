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
    /// Model id → the providers that discovered it and deliberately do not
    /// expose it. Kept apart from `routes` so a hidden model is
    /// distinguishable from one that was never discovered: the first is
    /// refused, the second falls back.
    ///
    /// Per provider, because that is how the configuration models exposure:
    /// `ProviderConfig::exposes` is answered by one provider about one id.
    /// Flattening those decisions into a single set meant hiding `gpt-4o` on a
    /// local server also hid the Copilot one — and worse, left the id routed to
    /// Copilot while every request for it was refused.
    hidden: BTreeMap<String, BTreeSet<usize>>,
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
            hidden: BTreeMap::new(),
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
            hidden: BTreeMap::new(),
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
        let Some(index) = self.index_of(provider) else {
            return false;
        };
        let model = model.into();
        if self.routes.contains_key(&model) {
            return false;
        }
        self.routes.insert(model, index);
        true
    }

    /// Record that `provider` discovered `model` and does not expose it.
    pub fn hide(&mut self, model: impl Into<String>, provider: &str) {
        let Some(index) = self.index_of(provider) else {
            return;
        };
        let model = model.into();
        // Only this provider's claim is withdrawn. Another provider may serve
        // the same id and expose it, and hiding it here must not take that
        // away — that is the whole point of tracking hiding per provider.
        if self.routes.get(&model) == Some(&index) {
            self.routes.remove(&model);
        }
        self.hidden.entry(model).or_default().insert(index);
    }

    /// Whether `provider` discovered `model` and chose not to expose it.
    ///
    /// This is the per-entry question `/v1/models` asks while merging
    /// catalogues: an id hidden on one provider must vanish from that
    /// provider's listing without touching another's.
    pub fn hides(&self, provider: &str, model: &str) -> bool {
        self.index_of(provider)
            .is_some_and(|index| self.hidden_from(index, model))
    }

    /// Whether a request naming `model` must be refused rather than forwarded.
    ///
    /// Hiding is a per-provider decision, so the question is not "is this id
    /// hidden anywhere" but "would this request reach a provider that hid it".
    /// Two ways that happens:
    ///
    /// 1. It resolves to a provider that hid the id — including through a
    ///    qualifier, so `mlx/gpt-4o` is refused while `copilot/gpt-4o` is
    ///    served if only mlx hid it.
    /// 2. Nothing routes the id at all and someone hid it, which means every
    ///    provider holding it hid it. Falling back would either serve what the
    ///    user hid or reach a provider that never had the model, so refusing is
    ///    both safer and the more truthful answer.
    pub fn is_hidden(&self, model: &str) -> bool {
        let (index, upstream) = self.target(model);
        let id = upstream.unwrap_or(model);
        if !self.hidden.contains_key(id) {
            return false;
        }
        self.hidden_from(index, id) || !self.routes.contains_key(id)
    }

    fn hidden_from(&self, provider: usize, model: &str) -> bool {
        self.hidden
            .get(model)
            .is_some_and(|hiders| hiders.contains(&provider))
    }

    fn index_of(&self, provider: &str) -> Option<usize> {
        self.providers.iter().position(|p| p.name() == provider)
    }

    /// The provider a request naming `model` resolves to, and the id to send
    /// upstream when a qualifier was stripped.
    fn target<'a>(&self, model: &'a str) -> (usize, Option<&'a str>) {
        if let Some(&index) = self.routes.get(model) {
            return (index, None);
        }
        match self.qualified(model) {
            Some((index, bare)) => (index, Some(bare)),
            None => (self.default, None),
        }
    }

    /// Split a provider-qualified id — `copilot/gpt-4o` — into the provider it
    /// names and the id to send upstream.
    ///
    /// This is the proxy's own addressing scheme, and the only way to reach a
    /// model id that two providers both serve: `routes` holds one provider per
    /// id, so the second claimant is otherwise unreachable.
    ///
    /// An id the registry already knows is never reinterpreted, because real
    /// model ids contain slashes — `lmstudio-community/Qwen2.5-Coder-7B-GGUF`
    /// is one id, not a qualifier, and stays one even if a provider is named
    /// `lmstudio-community`.
    ///
    /// The provider is not asked whether it serves the bare id. A qualifier is
    /// an instruction about where to send the request, and honouring it for a
    /// model discovery missed is the same leniency [`Self::resolve`] already
    /// extends to unknown ids — with a better destination than the default.
    fn qualified<'a>(&self, model: &'a str) -> Option<(usize, &'a str)> {
        if self.routes.contains_key(model) || self.hidden.contains_key(model) {
            return None;
        }
        let (name, bare) = model.split_once('/')?;
        if bare.is_empty() {
            return None;
        }
        Some((self.index_of(name)?, bare))
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
        self.resolve_upstream(model).0
    }

    /// The provider serving `model`, and the id to send it.
    ///
    /// The second element is `Some` only when the client qualified the id with
    /// a provider name and that qualifier was stripped. The upstream published
    /// the bare id and would not recognise the qualified one, so a caller that
    /// forwards a body must rewrite `model` with this before the hop.
    pub fn resolve_upstream<'a>(
        &self,
        model: Option<&'a str>,
    ) -> (&Arc<Provider>, Option<&'a str>) {
        let Some(model) = model else {
            return (self.default_provider(), None);
        };
        let (index, upstream) = self.target(model);
        (&self.providers[index], upstream)
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
        registry.hide("gpt-4o", "copilot");

        assert!(registry.is_hidden("gpt-4o"));
        assert!(!registry.has_route("gpt-4o"));
        assert!(!registry.is_hidden("never-discovered"));
    }

    #[test]
    fn hiding_removes_the_route_so_it_is_not_listed() {
        let mut registry = registry();
        registry.route("gpt-4o", "copilot");
        registry.route("qwen2.5-7b", "lmstudio");
        registry.hide("gpt-4o", "copilot");

        let routes: Vec<(&str, &str)> = registry.routes().collect();
        assert_eq!(routes, vec![("qwen2.5-7b", "lmstudio")]);
    }

    #[test]
    fn hiding_on_one_provider_leaves_another_serving_the_same_id() {
        // Exposure is a per-provider decision in the configuration, so it has
        // to be one here too. Flattened into a single set, hiding the local
        // gpt-4o also hid Copilot's — and left the id routed to Copilot while
        // every request for it was refused, which is the worst of both.
        let mut registry = registry();
        registry.hide("gpt-4o", "lmstudio");
        assert!(
            registry.route("gpt-4o", "copilot"),
            "the id lmstudio withdrew is free for copilot to claim"
        );

        assert!(!registry.is_hidden("gpt-4o"), "copilot still exposes it");
        assert_eq!(registry.resolve(Some("gpt-4o")).name(), "copilot");
        assert!(registry.hides("lmstudio", "gpt-4o"), "hidden in the listing");
        assert!(!registry.hides("copilot", "gpt-4o"));
    }

    #[test]
    fn hiding_survives_the_other_provider_claiming_first() {
        // Discovery order must not decide the outcome: copilot may be asked
        // before lmstudio, so hiding has to withdraw only the hider's own
        // claim rather than whatever route happens to exist.
        let mut registry = registry();
        registry.route("gpt-4o", "copilot");
        registry.hide("gpt-4o", "lmstudio");

        assert!(registry.has_route("gpt-4o"), "copilot's claim is untouched");
        assert_eq!(registry.resolve(Some("gpt-4o")).name(), "copilot");
        assert!(!registry.is_hidden("gpt-4o"));
    }

    #[test]
    fn an_id_hidden_everywhere_is_refused_rather_than_falling_back() {
        // Nothing routes it, so the fallback would reach the default provider
        // — either serving what the user hid, or a provider that never had the
        // model at all. Refusing is the truthful answer.
        let mut registry = registry();
        registry.hide("gpt-4o", "lmstudio");
        registry.hide("gpt-4o", "copilot");

        assert!(!registry.has_route("gpt-4o"));
        assert!(registry.is_hidden("gpt-4o"));
    }

    #[test]
    fn hiding_the_only_copy_is_refused_even_from_a_non_hiding_default() {
        // beta alone serves the model and hides it. lmstudio is the default and
        // never had it, so "did the resolved provider hide it" is not enough on
        // its own — an unrouted id nobody exposes must still be refused.
        let mut registry = registry();
        registry.hide("text-embedding-3-small", "copilot");

        assert_eq!(registry.default_provider().name(), "lmstudio");
        assert!(registry.is_hidden("text-embedding-3-small"));
    }

    #[test]
    fn a_provider_qualified_id_reaches_that_provider() {
        // The only way to reach the loser of a duplicate id: `routes` holds one
        // provider per id, so without a qualifier copilot's gpt-4o is
        // unreachable once lmstudio has claimed the name.
        let mut registry = registry();
        registry.route("gpt-4o", "lmstudio");

        let (provider, upstream) = registry.resolve_upstream(Some("copilot/gpt-4o"));
        assert_eq!(provider.name(), "copilot");
        assert_eq!(
            upstream,
            Some("gpt-4o"),
            "the qualifier is the proxy's, not the upstream's"
        );
    }

    #[test]
    fn an_unqualified_id_is_sent_on_unchanged() {
        let mut registry = registry();
        registry.route("gpt-4o", "copilot");
        assert_eq!(registry.resolve_upstream(Some("gpt-4o")).1, None);
        assert_eq!(registry.resolve_upstream(None).1, None);
    }

    #[test]
    fn a_model_id_containing_a_slash_is_not_a_qualifier() {
        // Real ids are full of slashes, and one of them may well match a
        // provider name. An id the registry knows is never re-read as an
        // address.
        let mut registry = Registry::new(vec![
            Provider::new("lmstudio", "http://127.0.0.1:1234"),
            Provider::new("lmstudio-community", "http://127.0.0.1:9999"),
        ])
        .unwrap();
        let id = "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF";
        registry.route(id, "lmstudio");

        let (provider, upstream) = registry.resolve_upstream(Some(id));
        assert_eq!(provider.name(), "lmstudio", "the exact route wins");
        assert_eq!(upstream, None, "the id goes upstream whole");
    }

    #[test]
    fn an_unknown_prefix_is_not_treated_as_a_provider() {
        // Only a known provider name qualifies. Anything else is just an
        // unknown id, which falls back as it always did.
        let registry = registry();
        let (provider, upstream) = registry.resolve_upstream(Some("someorg/some-model"));
        assert_eq!(provider.name(), "lmstudio");
        assert_eq!(
            upstream, None,
            "an unrouted id must reach the default provider unchanged"
        );
    }

    #[test]
    fn a_qualifier_does_not_unhide_a_model() {
        // Otherwise `copilot/gpt-4o` would be a one-character bypass of the
        // user's decision not to expose gpt-4o.
        let mut registry = registry();
        registry.route("gpt-4o", "copilot");
        registry.hide("gpt-4o", "copilot");

        assert!(registry.is_hidden("copilot/gpt-4o"));
        assert!(!registry.is_hidden("copilot/qwen2.5-7b"));
    }

    #[test]
    fn a_qualifier_is_refused_only_on_the_provider_that_hid_it() {
        // The pair that shows hiding is now addressed, not global: the same id,
        // refused on the provider that hid it and served on the one that
        // did not.
        let mut registry = registry();
        registry.hide("gpt-4o", "lmstudio");
        registry.route("gpt-4o", "copilot");

        assert!(registry.is_hidden("lmstudio/gpt-4o"));
        assert!(!registry.is_hidden("copilot/gpt-4o"));
        let (provider, upstream) = registry.resolve_upstream(Some("copilot/gpt-4o"));
        assert_eq!(provider.name(), "copilot");
        assert_eq!(upstream, Some("gpt-4o"));
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
