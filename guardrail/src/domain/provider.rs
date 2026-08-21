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
}

impl Provider {
    /// A provider that forwards the client's headers untouched — the shape of
    /// every backend before providers existed.
    pub fn new(name: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            reserved_headers: BTreeSet::new(),
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

    /// The URL to send a request for `path_and_query` to.
    pub fn target(&self, path_and_query: &str) -> String {
        format!("{}{}", self.base_url, path_and_query)
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
    fn reserving_accumulates_rather_than_replacing() {
        let provider = Provider::new("p", "http://h")
            .reserving(["user-agent"])
            .owning_credential();
        assert!(provider.reserves("user-agent"));
        assert!(provider.reserves("authorization"));
    }
}
