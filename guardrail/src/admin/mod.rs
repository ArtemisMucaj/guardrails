//! Admin layer — a small, read-only HTTP server, separate from the proxy.
//!
//! The proxy port (`--listen`) speaks the OpenAI chat-completions protocol and
//! is what model clients point at. The admin port (`--admin-listen`) is a
//! distinct server meant for operators and embedding UIs (a desktop app, a
//! dashboard): it exposes the failure metrics as JSON, a liveness probe, and a
//! description of how the running proxy is configured.
//!
//! It is deliberately decoupled from the request hot path. Every `/stats` read
//! goes straight to the SQLite guardrails database — the same single source of
//! truth the `stats` CLI subcommand reads — so the admin server holds no
//! in-memory counters that could drift from the proxy, and querying it never
//! contends with the proxy's response path (the database runs in WAL mode, so
//! readers and the background writer do not block each other).

pub mod manage;

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::Serialize;
use serde_json::json;
use tracing::error;

use crate::domain::metrics::{ErrorGroup, ModelStats, Stats};

/// Static description of the running proxy, surfaced at `/info` so an embedding
/// UI can show what it is connected to without parsing logs. Holds nothing
/// sensitive: the backend URL is already reduced to scheme/host/port before it
/// reaches here (see `redact_backend_url`).
#[derive(Clone, Debug, Serialize)]
pub struct AdminInfo {
    /// Crate version of the running binary.
    pub version: String,
    /// Backend base URL, reduced to scheme/host/port (no credentials or query).
    /// Each configured provider as `name=scheme://host[:port]`, in routing
    /// order — the first serves models no other one claims.
    pub providers: Vec<String>,
    /// Address the proxy (model-facing) server listens on.
    pub proxy_listen: String,
    /// Address this admin server listens on.
    pub admin_listen: String,
    /// Maximum corrective retries per guarded request.
    pub max_retries: u32,
    /// Filesystem path of the SQLite guardrails database the stats are read from.
    pub database: String,
}

/// Shared state for the admin router: where to read metrics from, and the
/// static description of the proxy. Cheap to clone (the info is shared).
#[derive(Clone)]
pub struct AdminState {
    db_path: PathBuf,
    info: Arc<AdminInfo>,
    /// Present only when a Copilot provider is configured. `None` leaves the
    /// login routes answering `404`, so the mutable surface does not exist at
    /// all unless it is needed.
    login: Option<Arc<crate::copilot::CopilotLogin>>,
    /// Present only when the management API is enabled. `None` leaves those
    /// routes answering `404`.
    management: Option<Arc<manage::Management>>,
}

impl AdminState {
    pub fn new(db_path: PathBuf, info: AdminInfo) -> Self {
        Self {
            db_path,
            info: Arc::new(info),
            login: None,
            management: None,
        }
    }

    /// Enable the management API.
    pub fn with_management(mut self, management: Arc<manage::Management>) -> Self {
        self.management = Some(management);
        self
    }

    /// Enable the Copilot device-flow login routes.
    pub fn with_login(mut self, login: Arc<crate::copilot::CopilotLogin>) -> Self {
        self.login = Some(login);
        self
    }
}

/// Reduce a backend URL to `scheme://host[:port]`, dropping any userinfo, path,
/// and query. The backend URL is operator-controlled and may embed basic-auth
/// credentials or token-bearing query params, so only the non-secret locator is
/// ever exposed (the same reduction the startup log applies). Unparseable input
/// becomes `<redacted>`.
pub fn redact_backend_url(backend: &str) -> String {
    reqwest::Url::parse(backend)
        .ok()
        .and_then(|url| {
            let host = url.host_str()?;
            Some(match url.port() {
                Some(port) => format!("{}://{}:{}", url.scheme(), host, port),
                None => format!("{}://{}", url.scheme(), host),
            })
        })
        .unwrap_or_else(|| "<redacted>".to_string())
}

/// Build the admin router.
///
/// Every route was a `GET` until Copilot login arrived. Starting a device flow
/// mutates state, so `POST /copilot/login` is a deliberate exception to the
/// read-only rule rather than an oversight — see the README's note on the admin
/// server's security posture. The route exists only when a Copilot provider is
/// configured.
pub fn build_admin_app(state: AdminState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/info", get(info))
        .route("/stats", get(stats))
        .route("/copilot/login", get(copilot_login_status).post(copilot_login_start))
        .route(
            "/providers",
            get(manage::list_providers).post(manage::add_provider),
        )
        .route(
            "/providers/:name",
            axum::routing::patch(manage::update_provider).delete(manage::remove_provider),
        )
        .with_state(state)
}

/// `GET /copilot/login` — where the current attempt stands.
///
/// The body is a `LoginStatus`, which carries no credential by construction.
async fn copilot_login_status(State(state): State<AdminState>) -> Response {
    let Some(login) = state.login.as_ref() else {
        return copilot_disabled();
    };
    Json(login.status().await).into_response()
}

/// `POST /copilot/login` — start, or restart, the device flow.
///
/// Returns as soon as GitHub issues the code, so the response carries the
/// `user_code` and `verification_uri` to show the user; the polling continues in
/// the background.
async fn copilot_login_start(State(state): State<AdminState>) -> Response {
    let Some(login) = state.login.as_ref() else {
        return copilot_disabled();
    };
    Json(login.start().await).into_response()
}

fn copilot_disabled() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": "no copilot provider is configured" })),
    )
        .into_response()
}

/// Discoverability root: list the available endpoints so the port is
/// self-describing when opened in a browser or by a new integration.
async fn index(State(state): State<AdminState>) -> Json<serde_json::Value> {
    let mut endpoints = vec!["/healthz", "/info", "/stats"];
    if state.management.is_some() {
        endpoints.push("/providers");
    }
    // Listed only when it exists, so discovery does not advertise a route that
    // answers 404.
    if state.login.is_some() {
        endpoints.push("/copilot/login");
    }
    Json(json!({
        "service": "guardrail-admin",
        "endpoints": endpoints,
    }))
}

/// Liveness probe. The admin server only runs while the process is up, so a
/// reachable `/healthz` is itself the signal — a desktop app can poll this to
/// show connected/disconnected.
async fn healthz() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

/// Describe the running proxy.
async fn info(State(state): State<AdminState>) -> Json<AdminInfo> {
    Json((*state.info).clone())
}

/// Read and return the full failure-metrics rollup as JSON. Reads the database
/// on each request (the same `Stats::read` the CLI uses), so the response is
/// always current without the admin server holding its own counters.
async fn stats(State(state): State<AdminState>) -> Response {
    match Stats::read(&state.db_path) {
        Ok(stats) => Json(StatsResponse::from(stats)).into_response(),
        Err(e) => {
            error!(error = %e, "admin: failed to read guardrails database");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "failed to read guardrails database" })),
            )
                .into_response()
        }
    }
}

// JSON DTOs. The domain `Stats` is a read-model already, but its shape is tuned
// for the text report (tuples, computed-only `succeeded`/`success_rate`). These
// give the HTTP boundary a stable, self-describing JSON shape — computed fields
// materialized, outcome counts as named objects rather than positional tuples —
// so the desktop app does not re-derive them or depend on tuple ordering.

#[derive(Serialize)]
struct StatsResponse {
    per_model: Vec<ModelStatsDto>,
    errors: Vec<ErrorGroupDto>,
}

#[derive(Serialize)]
struct ModelStatsDto {
    provider: String,
    model: String,
    total: i64,
    tool_calls: i64,
    succeeded: i64,
    errors: i64,
    /// Success rate over tool calls in `[0, 1]`, or `null` when the model made
    /// no tool call (so consumers render "n/a" rather than a misleading 0%).
    success_rate: Option<f64>,
    by_outcome: Vec<OutcomeCount>,
    /// Token usage over the requests that reported any, or `null` when none
    /// did — so a consumer shows "not measured" rather than a confident zero.
    usage: Option<UsageDto>,
}

/// Token usage for one (provider, model), summed over every backend attempt the
/// measured requests made.
#[derive(Serialize)]
struct UsageDto {
    /// Prompt tokens as billed. NOT additive across a conversation: each turn
    /// resends the transcript, so shared prefixes are counted once per turn.
    /// Use `distinct_prompt_tokens` to count them once.
    prompt_tokens: i64,
    /// Generated once and never resent, so this figure sums cleanly.
    completion_tokens: i64,
    /// `prompt_tokens + completion_tokens` — what the provider charged for,
    /// not a count of distinct tokens.
    billed_tokens: i64,
    /// Of `prompt_tokens`, the portion served from the prompt cache.
    cached_tokens: i64,
    /// Of `prompt_tokens`, the portion billed at full rate.
    uncached_prompt_tokens: i64,
    /// Cache hit rate over prompt tokens in `[0, 1]`, `null` without any.
    cache_hit_rate: Option<f64>,
    /// Backend calls these totals span, retries included.
    billed_calls: i64,
    /// Client requests the totals are measured over.
    requests: i64,
    /// `billed_calls / requests` — the multiplier retries add to the bill.
    calls_per_request: Option<f64>,
    /// Prompt tokens with resent transcript prefixes counted once — a
    /// conversation contributes its largest prompt, not the sum of its turns.
    /// `null` when conversations cannot be reconstructed (any Chat Completions
    /// traffic), rather than repeating the inflated sum under a better name.
    distinct_prompt_tokens: Option<i64>,
    /// `distinct_prompt_tokens + completion_tokens`; `null` with the above.
    distinct_tokens: Option<i64>,
    /// Conversations the measured requests span; `null` when unknown.
    conversations: Option<i64>,
}

#[derive(Serialize)]
struct OutcomeCount {
    outcome: String,
    count: i64,
}

#[derive(Serialize)]
struct ErrorGroupDto {
    provider: String,
    model: String,
    error_category: Option<String>,
    tool_name: Option<String>,
    detail: Option<String>,
    count: i64,
}

impl From<Stats> for StatsResponse {
    fn from(s: Stats) -> Self {
        Self {
            per_model: s.per_model.into_iter().map(ModelStatsDto::from).collect(),
            errors: s.errors.into_iter().map(ErrorGroupDto::from).collect(),
        }
    }
}

impl From<ModelStats> for ModelStatsDto {
    fn from(m: ModelStats) -> Self {
        // Compute before moving out the fields that feed `by_outcome`.
        let succeeded = m.succeeded();
        let success_rate = m.success_rate();
        let usage = (m.usage_requests > 0).then(|| UsageDto {
            prompt_tokens: m.usage.prompt_tokens,
            completion_tokens: m.usage.completion_tokens,
            billed_tokens: m.billed_tokens(),
            cached_tokens: m.usage.cached_tokens,
            uncached_prompt_tokens: m.usage.uncached_prompt_tokens(),
            cache_hit_rate: m.cache_hit_rate(),
            billed_calls: m.usage.attempts,
            requests: m.usage_requests,
            calls_per_request: m.calls_per_request(),
            distinct_prompt_tokens: m.distinct_prompt_tokens,
            distinct_tokens: m.distinct_tokens(),
            conversations: m.conversations,
        });
        Self {
            provider: m.provider,
            model: m.model,
            total: m.total,
            tool_calls: m.tool_calls,
            succeeded,
            errors: m.errors,
            success_rate,
            by_outcome: m
                .by_outcome
                .into_iter()
                .map(|(outcome, count)| OutcomeCount { outcome, count })
                .collect(),
            usage,
        }
    }
}

impl From<ErrorGroup> for ErrorGroupDto {
    fn from(e: ErrorGroup) -> Self {
        Self {
            provider: e.provider,
            model: e.model,
            error_category: e.error_category,
            tool_name: e.tool_name,
            detail: e.detail,
            count: e.count,
        }
    }
}
