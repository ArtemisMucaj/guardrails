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
    pub backend: String,
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
}

impl AdminState {
    pub fn new(db_path: PathBuf, info: AdminInfo) -> Self {
        Self {
            db_path,
            info: Arc::new(info),
        }
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

/// Build the admin router. Read-only: every route is a `GET`.
pub fn build_admin_app(state: AdminState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/info", get(info))
        .route("/stats", get(stats))
        .with_state(state)
}

/// Discoverability root: list the available endpoints so the port is
/// self-describing when opened in a browser or by a new integration.
async fn index() -> Json<serde_json::Value> {
    Json(json!({
        "service": "guardrail-admin",
        "endpoints": ["/healthz", "/info", "/stats"],
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
    model: String,
    total: i64,
    tool_calls: i64,
    succeeded: i64,
    errors: i64,
    /// Success rate over tool calls in `[0, 1]`, or `null` when the model made
    /// no tool call (so consumers render "n/a" rather than a misleading 0%).
    success_rate: Option<f64>,
    by_outcome: Vec<OutcomeCount>,
}

#[derive(Serialize)]
struct OutcomeCount {
    outcome: String,
    count: i64,
}

#[derive(Serialize)]
struct ErrorGroupDto {
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
        Self {
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
        }
    }
}

impl From<ErrorGroup> for ErrorGroupDto {
    fn from(e: ErrorGroup) -> Self {
        Self {
            model: e.model,
            error_category: e.error_category,
            tool_name: e.tool_name,
            detail: e.detail,
            count: e.count,
        }
    }
}
