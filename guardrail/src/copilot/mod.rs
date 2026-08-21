//! GitHub Copilot as a provider.
//!
//! Copilot is the first upstream that cannot be described by a URL alone: it
//! needs an OAuth credential, six client-identity headers GitHub gates access
//! on, and its routes live at the root rather than under `/v1`. All three come
//! from [`gh_copilot_rs`], which owns that knowledge so this crate does not have
//! to reverse-engineer it.
//!
//! # Handling the credential
//!
//! [`CopilotToken`] redacts itself in `Debug` and `Display`, but it derives
//! `Serialize` as `#[serde(transparent)]` — serde emits the secret in full.
//! That is deliberate in the upstream crate (it is what lets a host persist the
//! token), but it means the redaction guarantee stops at the serialization
//! boundary. So the token never enters a type that reaches a response body;
//! [`LoginStatus`] is what goes on the wire, and it carries no credential by
//! construction.

use std::sync::Arc;
use std::time::Duration;

use gh_copilot_rs::{
    CopilotEndpoint, CopilotToken, GitHubDeviceFlow, LoginSession, LoginStatus, COPILOT_API_BASE,
};
use tracing::warn;

use crate::domain::provider::Provider;

/// Provider name Copilot registers under.
pub const COPILOT_PROVIDER: &str = "copilot";

/// The six client-identity headers GitHub gates Copilot access on, plus the
/// credential. Reserved on the provider so a client's own values — an LM Studio
/// key, a `Bearer no-key` placeholder, a client's `user-agent` — cannot
/// displace what the endpoint configured.
fn reserved_header_names(endpoint: &CopilotEndpoint) -> Vec<String> {
    endpoint
        .protocol_headers()
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

/// A Copilot provider and the HTTP client that carries its credential.
pub struct CopilotProvider {
    pub provider: Provider,
    pub client: reqwest::Client,
}

/// Build the Copilot provider for `token`.
///
/// The client is built here rather than taken from
/// `CopilotEndpoint::http_client()` so it inherits the proxy's timeout policy.
/// That method applies a single total-request timeout, which would cut off a
/// long but healthy streaming completion; the proxy instead bounds connection
/// setup and the idle gap between chunks, leaving a slow stream alone as long
/// as it keeps producing.
pub fn provider(
    token: CopilotToken,
    connect_timeout: Duration,
    read_timeout: Duration,
) -> anyhow::Result<CopilotProvider> {
    provider_at(token, COPILOT_API_BASE, connect_timeout, read_timeout)
}

/// As [`provider`], against a specific base URL — an enterprise Copilot
/// deployment, or a mock server in a test.
pub fn provider_at(
    token: CopilotToken,
    base_url: &str,
    connect_timeout: Duration,
    read_timeout: Duration,
) -> anyhow::Result<CopilotProvider> {
    let endpoint = CopilotEndpoint::new(token).with_base_url(base_url);
    let mut headers = reqwest::header::HeaderMap::new();

    for (name, value) in endpoint.protocol_headers() {
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())?;
        headers.insert(name, reqwest::header::HeaderValue::from_str(&value)?);
    }

    if let Some(token) = endpoint.token() {
        let mut value =
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token.expose()))?;
        // Keeps the credential out of reqwest's own diagnostics.
        value.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }

    let client = reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .read_timeout(read_timeout)
        .default_headers(headers)
        .build()?;

    Ok(CopilotProvider {
        provider: Provider::new(COPILOT_PROVIDER, endpoint.base_url())
            .unversioned()
            .owning_credential()
            .reserving(reserved_header_names(&endpoint)),
        client,
    })
}

/// Copilot login state shared between the admin server and the proxy.
///
/// Holds the device-flow session and the token it produced. The token is kept
/// behind [`CopilotToken`] and is never serialized.
pub struct CopilotLogin {
    session: Arc<LoginSession>,
    /// Where the token is persisted, so a restart does not require a re-login.
    store: std::path::PathBuf,
}

impl CopilotLogin {
    pub fn new(store: std::path::PathBuf) -> anyhow::Result<Arc<Self>> {
        Ok(Arc::new(Self {
            session: LoginSession::new(Arc::new(GitHubDeviceFlow::new()?)),
            store,
        }))
    }

    /// Start — or restart — the device flow. Returns the status to show the
    /// user; `Pending` carries the code and verification URL.
    pub async fn start(&self) -> LoginStatus {
        let status = self.session.start().await;
        if matches!(status, LoginStatus::Pending { .. }) {
            // Persist as soon as the flow completes, without making the caller
            // poll for it.
            let session = Arc::clone(&self.session);
            let store = self.store.clone();
            tokio::spawn(async move {
                // `Authorized` is only observable once the token is stored, so
                // reading it here can never find an empty slot.
                for _ in 0..600 {
                    match session.status().await {
                        LoginStatus::Authorized => {
                            if let Some(token) = session.token().await {
                                if let Err(e) = save_token(&store, &token) {
                                    warn!(error = %e, "could not persist the Copilot token");
                                }
                            }
                            return;
                        }
                        LoginStatus::Failed { .. } => return,
                        _ => tokio::time::sleep(Duration::from_secs(1)).await,
                    }
                }
            });
        }
        status
    }

    /// The current status — safe to serialize; carries no credential.
    pub async fn status(&self) -> LoginStatus {
        self.session.status().await
    }

    /// The token from this session, or the persisted one from an earlier run.
    pub async fn token(&self) -> Option<CopilotToken> {
        match self.session.token().await {
            Some(token) => Some(token),
            None => load_token(&self.store),
        }
    }
}

/// Read a persisted token, if one is there and usable.
pub fn load_token(path: &std::path::Path) -> Option<CopilotToken> {
    let raw = std::fs::read_to_string(path).ok()?;
    let token = CopilotToken::new(raw.trim());
    (!token.is_empty()).then_some(token)
}

/// Persist the token, readable only by this user.
///
/// A credential on disk should not be world-readable; the file is created with
/// `0600` on Unix rather than inheriting the default umask.
pub fn save_token(path: &std::path::Path, token: &CopilotToken) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(token.expose().as_bytes())?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, token.expose().as_bytes())?;
    }
    Ok(())
}

/// Default location of the persisted token, alongside the metrics database.
pub fn default_token_path() -> std::path::PathBuf {
    crate::domain::metrics::default_db_path().with_file_name("copilot-token")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("guardrail-copilot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(label)
    }

    #[test]
    fn the_provider_is_unversioned_and_owns_its_headers() {
        let built = provider(
            CopilotToken::new("ghu_test"),
            Duration::from_secs(10),
            Duration::from_secs(300),
        )
        .unwrap();

        // Copilot serves at the root; getting this wrong 404s every call.
        assert_eq!(
            built.provider.target("/v1/chat/completions"),
            "https://api.githubcopilot.com/chat/completions"
        );

        // The credential and every gating header must be beyond a client's
        // reach, or the client's values displace them.
        assert!(built.provider.reserves("authorization"));
        for name in [
            "copilot-integration-id",
            "editor-version",
            "editor-plugin-version",
            "x-github-api-version",
            "openai-intent",
            "user-agent",
        ] {
            assert!(built.provider.reserves(name), "{name} must be reserved");
        }
    }

    #[test]
    fn a_token_round_trips_through_the_store() {
        let path = temp_path("token-roundtrip");
        let _ = std::fs::remove_file(&path);
        assert!(load_token(&path).is_none(), "absent file reads as no token");

        save_token(&path, &CopilotToken::new("ghu_secret")).unwrap();
        assert_eq!(
            load_token(&path).map(|t| t.expose().to_string()),
            Some("ghu_secret".to_string())
        );

        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn the_stored_token_is_not_world_readable() {
        use std::os::unix::fs::PermissionsExt;
        let path = temp_path("token-perms");
        let _ = std::fs::remove_file(&path);
        save_token(&path, &CopilotToken::new("ghu_secret")).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(
            mode & 0o077,
            0,
            "a credential must not be readable by group or other, got {:o}",
            mode
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_empty_stored_token_reads_as_absent() {
        // An empty file would otherwise produce a token that fails at request
        // time with a 401 rather than prompting a login.
        let path = temp_path("token-empty");
        std::fs::write(&path, "   \n").unwrap();
        assert!(load_token(&path).is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_login_status_never_carries_the_credential() {
        // The guarantee that makes the admin endpoints safe: `CopilotToken` is
        // serde-transparent, so anything serialized must not contain one.
        let authorized = serde_json::to_string(&LoginStatus::Authorized).unwrap();
        assert_eq!(authorized, r#"{"status":"authorized"}"#);

        let pending = serde_json::to_string(&LoginStatus::Pending {
            user_code: "ABCD-1234".into(),
            verification_uri: "https://github.com/login/device".into(),
        })
        .unwrap();
        assert!(!pending.contains("ghu_"));
        assert!(pending.contains("ABCD-1234"));
    }
}
