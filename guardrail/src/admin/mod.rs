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
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::error;

use crate::domain::metrics::{
    DayActivity, Distribution, ErrorGroup, ModelStats, Range, RequestRow, Stats, MAX_DAYS, MAX_ROWS,
};

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
        .route("/activity", get(activity))
        .route("/requests", get(requests))
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
    let mut endpoints = vec!["/healthz", "/info", "/stats", "/activity", "/requests"];
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

/// A half-open `[since, until)` window, as query parameters.
///
/// Both are optional and absent means unbounded, so `/stats` with no query is
/// the lifetime rollup it has always been.
#[derive(Deserialize, Default)]
struct RangeQuery {
    since: Option<String>,
    until: Option<String>,
}

impl From<&RangeQuery> for Range {
    fn from(q: &RangeQuery) -> Self {
        // `Range::parse` normalizes a seconds-precision instant to the
        // millisecond form the rows are stored in, so `...T00:00:00Z` matches
        // its own midnight row rather than falling on the wrong side of it.
        //
        // Bounds are otherwise not validated as timestamps: the comparison is
        // lexicographic against a fixed-width format, so a prefix like
        // `2026-08-22` is a legitimate and useful bound. Anything that is not a
        // prefix of a real timestamp simply selects nothing, which is the
        // honest answer to a window that matches nothing.
        Range::parse(q.since.as_deref(), q.until.as_deref())
    }
}

/// Read and return the failure-metrics rollup as JSON. Reads the database on
/// each request (the same `Stats::read` the CLI uses), so the response is always
/// current without the admin server holding its own counters.
///
/// `?since=`/`?until=` bound the window every figure is computed over — the
/// rollup, the outcome breakdown, the errors and the distributions alike — so a
/// UI showing them together never mixes a filtered figure with a lifetime one.
async fn stats(State(state): State<AdminState>, Query(range): Query<RangeQuery>) -> Response {
    match Stats::read_in(&state.db_path, &Range::from(&range)) {
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

/// Default number of days `GET /activity` returns.
const DEFAULT_DAY_LIMIT: i64 = 30;

#[derive(Deserialize)]
struct ActivityQuery {
    days: Option<i64>,
    #[serde(flatten)]
    range: RangeQuery,
}

/// `GET /activity` — per-day totals, newest day first.
///
/// The calendar view the rollup cannot serve: `/stats` collapses a window into
/// one set of figures, and a contribution graph needs it broken back out by day.
/// Grouped in SQL, so a year of history is not bounded by the `/requests` row
/// cap.
///
/// Days are UTC (see `Stats::read_activity`), and a day with no traffic is
/// absent rather than zero — the consumer owns the calendar it is drawing.
async fn activity(
    State(state): State<AdminState>,
    Query(query): Query<ActivityQuery>,
) -> Response {
    // Clamped rather than rejected, as on `/requests`: this is a read-only
    // diagnostic surface, and the nearest sane read beats a 400 for a
    // hand-typed URL.
    let days = query.days.unwrap_or(DEFAULT_DAY_LIMIT).clamp(1, MAX_DAYS);
    match Stats::read_activity(&state.db_path, &Range::from(&query.range), days) {
        Ok(rows) => Json(ActivityResponse {
            count: rows.len(),
            days,
            activity: rows.into_iter().map(DayActivityDto::from).collect(),
        })
        .into_response(),
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

/// Default number of request rows `GET /requests` returns.
const DEFAULT_ROW_LIMIT: i64 = 1000;

#[derive(Deserialize)]
struct RequestsQuery {
    limit: Option<i64>,
}

/// `GET /requests` — the individual recorded requests, newest first.
///
/// The rollup at `/stats` answers the questions the report asks. This serves
/// the rows behind it so a consumer can ask its own: a histogram with its own
/// buckets, a grouping by hour, or — on Chat Completions, where the proxy has
/// no conversation key of its own — a grouping by whatever session key that
/// consumer does have.
async fn requests(
    State(state): State<AdminState>,
    Query(query): Query<RequestsQuery>,
) -> Response {
    // A nonsensical limit is clamped rather than rejected: this is a read-only
    // diagnostic endpoint, and answering with the sane neighbouring value is
    // more useful than a 400 for a hand-typed URL. `read_rows` clamps to the
    // same bound itself — this is here so the applied value can be echoed back
    // in the response, not because the read depends on it.
    let limit = query.limit.unwrap_or(DEFAULT_ROW_LIMIT).clamp(1, MAX_ROWS);
    match Stats::read_rows(&state.db_path, limit) {
        Ok(rows) => Json(RequestsResponse {
            count: rows.len(),
            limit,
            requests: rows.into_iter().map(RequestRowDto::from).collect(),
        })
        .into_response(),
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
    /// `null` when conversations cannot be reconstructed — Chat Completions
    /// without `--match-conversations` — rather than repeating the inflated
    /// sum under a better name.
    distinct_prompt_tokens: Option<i64>,
    /// `distinct_prompt_tokens + completion_tokens`; `null` with the above.
    distinct_tokens: Option<i64>,
    /// Conversations the measured requests span; `null` when unknown.
    conversations: Option<i64>,
    /// Whether the three fields above rest on conversation edges the proxy
    /// *inferred* from message prefixes (Chat Completions with matching
    /// enabled) rather than ones the API supplied. Inferred grouping is a
    /// heuristic, so those figures are real but approximate.
    inferred_conversations: bool,
    /// Spread of prompt tokens across single requests. Unlike the three fields
    /// above this needs no conversation key, so it is populated for Chat
    /// Completions traffic too — a sum says what the traffic cost, this says
    /// what it looked like.
    prompt_distribution: Option<DistributionDto>,
    /// Spread of completion tokens across single requests.
    completion_distribution: Option<DistributionDto>,
}

/// Spread of a per-request figure across the measured requests.
///
/// Percentiles are nearest-rank, so every value here is one some request
/// actually reported rather than an interpolated point between two of them.
#[derive(Serialize)]
struct DistributionDto {
    /// Requests the percentiles are over.
    count: i64,
    min: i64,
    p50: i64,
    p90: i64,
    p99: i64,
    max: i64,
}

impl From<Distribution> for DistributionDto {
    fn from(d: Distribution) -> Self {
        Self { count: d.count, min: d.min, p50: d.p50, p90: d.p90, p99: d.p99, max: d.max }
    }
}

#[derive(Serialize)]
struct ActivityResponse {
    /// Days returned, which is `days` when more were available.
    count: usize,
    /// The day limit actually applied, after clamping.
    days: i64,
    activity: Vec<DayActivityDto>,
}

/// One UTC day's traffic.
#[derive(Serialize)]
struct DayActivityDto {
    /// `YYYY-MM-DD` in **UTC**, the timezone the proxy writes timestamps in —
    /// not the consumer's local day.
    date: String,
    /// Every guarded request that day, measured or not.
    requests: i64,
    /// Of `requests`, those the guardrails could not fix.
    errors: i64,
    /// `prompt_tokens + completion_tokens` — what the provider charged for.
    billed_tokens: i64,
    /// Prompt tokens as billed. NOT additive across a conversation; see the
    /// note on `UsageDto::prompt_tokens`.
    prompt_tokens: i64,
    completion_tokens: i64,
    /// Of `prompt_tokens`, the portion served from the prompt cache.
    cached_tokens: i64,
    /// Backend calls that day, retries included.
    billed_calls: i64,
    /// Of `requests`, those that reported usage — so a zero token figure is
    /// distinguishable from one that was never measured.
    usage_requests: i64,
}

impl From<DayActivity> for DayActivityDto {
    fn from(d: DayActivity) -> Self {
        Self {
            billed_tokens: d.billed_tokens(),
            date: d.date,
            requests: d.requests,
            errors: d.errors,
            prompt_tokens: d.prompt_tokens,
            completion_tokens: d.completion_tokens,
            cached_tokens: d.cached_tokens,
            billed_calls: d.billed_calls,
            usage_requests: d.usage_requests,
        }
    }
}

#[derive(Serialize)]
struct RequestsResponse {
    /// Rows returned, which is `limit` when more were available.
    count: usize,
    /// The limit actually applied, after clamping.
    limit: i64,
    requests: Vec<RequestRowDto>,
}

/// One recorded request. Only requests that reported usage appear, matching the
/// population the `/stats` token figures are computed over.
#[derive(Serialize)]
struct RequestRowDto {
    ts: String,
    provider: String,
    model: String,
    outcome: String,
    prompt_tokens: i64,
    completion_tokens: i64,
    cached_tokens: i64,
    /// Backend calls this request made, retries included.
    billed_calls: i64,
    /// This turn's conversation id; `null` on Chat Completions, which carries
    /// no such key — a consumer grouping these rows supplies its own there.
    response_id: Option<String>,
    /// The turn this one continues; `null` on a first turn or without a key.
    parent_id: Option<String>,
}

impl From<RequestRow> for RequestRowDto {
    fn from(r: RequestRow) -> Self {
        Self {
            ts: r.ts,
            provider: r.provider,
            model: r.model,
            outcome: r.outcome,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            cached_tokens: r.cached_tokens,
            billed_calls: r.billed_calls,
            response_id: r.response_id,
            parent_id: r.parent_id,
        }
    }
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
            inferred_conversations: m.inferred_conversations,
            prompt_distribution: m.prompt_distribution.map(DistributionDto::from),
            completion_distribution: m.completion_distribution.map(DistributionDto::from),
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
