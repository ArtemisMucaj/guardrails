//! Providers — the upstream servers the proxy forwards to, and the rules for
//! reaching each one.
//!
//! A provider is more than a URL. One that carries its own credential (a
//! Copilot subscription, an API-keyed cloud service) also owns a set of headers
//! that must reach the upstream exactly as the provider configured them.
//!
//! That last part is not cosmetic. `reqwest`'s `RequestBuilder::headers()`
//! *replaces* per name rather than merging with the client's configured
//! defaults, so a client sending its own `Authorization` — `Bearer no-key`, as
//! OpenAI-compatible clients routinely do — would silently displace the
//! provider's credential. Every request would fail as `401`, and the failure
//! would read as an expired token rather than a precedence bug. So a provider
//! declares the header names it owns, and those are stripped from the client's
//! set before the hop.

use std::collections::BTreeSet;

/// Name of the provider used when none is configured, preserving the
/// single-backend behaviour of earlier versions.
pub const DEFAULT_PROVIDER: &str = "default";

/// Value recorded for outcomes written before providers existed.
///
/// Rows predating the multi-provider schema have no honest provider, and
/// stamping them with whichever backend happens to be configured now would be a
/// guess that later reads as fact.
pub const UNKNOWN_PROVIDER: &str = "unknown";

/// An upstream server, and what it takes to reach it.
#[derive(Debug, Clone)]
pub struct Provider {
    /// Stable identifier, used for routing and recorded against every outcome.
    name: String,
    /// Base URL, without a trailing slash.
    base_url: String,
    /// Header names this provider owns. Stripped from the client's headers
    /// before forwarding so they cannot displace the provider's own values.
    reserved_headers: BTreeSet<String>,
    /// Whether the upstream serves the OpenAI routes at the root rather than
    /// under `/v1`.
    strip_v1: bool,
}

impl Provider {
    /// A provider that forwards the client's headers untouched — the shape of
    /// every backend before providers existed.
    pub fn new(name: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            reserved_headers: BTreeSet::new(),
            strip_v1: false,
        }
    }

    /// Reserve header names, so the client cannot override what this provider
    /// sends. Names are matched case-insensitively.
    pub fn reserving<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.reserved_headers.extend(
            names
                .into_iter()
                .map(|name| name.as_ref().to_ascii_lowercase()),
        );
        self
    }

    /// Reserve `authorization`, for a provider presenting its own credential.
    ///
    /// Separate from [`Self::reserving`] because it is the case that silently
    /// breaks: the client's key wins, and the resulting `401` looks like a bad
    /// credential rather than a precedence bug.
    pub fn owning_credential(self) -> Self {
        self.reserving(["authorization"])
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Whether this provider owns `name`, and the client's value for it must be
    /// dropped before forwarding.
    pub fn reserves(&self, name: &str) -> bool {
        self.reserved_headers.contains(&name.to_ascii_lowercase())
    }

    /// The reserved names, lowercased.
    pub fn reserved_headers(&self) -> impl Iterator<Item = &str> {
        self.reserved_headers.iter().map(String::as_str)
    }

    /// Serve the OpenAI routes at the root instead of under `/v1`.
    ///
    /// Copilot does this. Getting it wrong 404s every call, and the proxy's own
    /// surface stays `/v1/...` regardless — clients should not have to know
    /// which upstream shape is behind it.
    pub fn unversioned(mut self) -> Self {
        self.strip_v1 = true;
        self
    }

    /// The URL to send a request for `path_and_query` to.
    pub fn target(&self, path_and_query: &str) -> String {
        format!("{}{}", self.base_url, self.upstream_path(path_and_query))
    }

    /// The client's path as this provider expects to receive it.
    fn upstream_path<'a>(&self, path_and_query: &'a str) -> std::borrow::Cow<'a, str> {
        if !self.strip_v1 {
            return std::borrow::Cow::Borrowed(path_and_query);
        }
        match path_and_query.strip_prefix("/v1/") {
            Some(rest) => std::borrow::Cow::Owned(format!("/{rest}")),
            // `/v1` exactly, or a path that was never versioned: leave it be
            // rather than inventing a route the upstream may not serve.
            None => std::borrow::Cow::Borrowed(path_and_query),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_a_trailing_slash_so_targets_do_not_double_up() {
        let provider = Provider::new("p", "http://host:1234/");
        assert_eq!(provider.base_url(), "http://host:1234");
        assert_eq!(
            provider.target("/v1/chat/completions"),
            "http://host:1234/v1/chat/completions"
        );
    }

    #[test]
    fn a_plain_provider_reserves_nothing() {
        // The pre-provider behaviour: every client header is forwarded.
        let provider = Provider::new("lmstudio", "http://127.0.0.1:1234");
        assert!(!provider.reserves("authorization"));
        assert_eq!(provider.reserved_headers().count(), 0);
    }

    #[test]
    fn a_credentialed_provider_reserves_authorization() {
        let provider = Provider::new("copilot", "https://api.githubcopilot.com").owning_credential();
        assert!(provider.reserves("authorization"));
        // Header names are case-insensitive on the wire.
        assert!(provider.reserves("Authorization"));
        assert!(provider.reserves("AUTHORIZATION"));
    }

    #[test]
    fn reserving_is_case_insensitive_in_both_directions() {
        let provider = Provider::new("p", "http://h").reserving(["User-Agent", "X-Custom"]);
        assert!(provider.reserves("user-agent"));
        assert!(provider.reserves("USER-AGENT"));
        assert!(provider.reserves("x-custom"));
        assert!(!provider.reserves("x-other"));
    }

    #[test]
    fn an_unversioned_provider_serves_the_routes_at_the_root() {
        // Copilot's shape. The proxy's own surface stays /v1/... either way.
        let provider = Provider::new("copilot", "https://api.githubcopilot.com").unversioned();
        assert_eq!(
            provider.target("/v1/chat/completions"),
            "https://api.githubcopilot.com/chat/completions"
        );
        assert_eq!(
            provider.target("/v1/models"),
            "https://api.githubcopilot.com/models"
        );
    }

    #[test]
    fn stripping_v1_keeps_the_query_string() {
        let provider = Provider::new("p", "https://host").unversioned();
        assert_eq!(
            provider.target("/v1/models?limit=10"),
            "https://host/models?limit=10"
        );
    }

    #[test]
    fn a_path_without_a_v1_prefix_is_left_alone() {
        // Inventing a route the upstream may not serve would turn a clear 404
        // into a confusing one.
        let provider = Provider::new("p", "https://host").unversioned();
        assert_eq!(provider.target("/healthz"), "https://host/healthz");
        assert_eq!(provider.target("/v1"), "https://host/v1");
    }

    #[test]
    fn a_versioned_provider_forwards_the_path_untouched() {
        let provider = Provider::new("lmstudio", "http://127.0.0.1:1234");
        assert_eq!(
            provider.target("/v1/chat/completions"),
            "http://127.0.0.1:1234/v1/chat/completions"
        );
    }

    #[test]
    fn reserving_accumulates_rather_than_replacing() {
        let provider = Provider::new("p", "http://h")
            .reserving(["user-agent"])
            .owning_credential();
        assert!(provider.reserves("user-agent"));
        assert!(provider.reserves("authorization"));
    }
}
