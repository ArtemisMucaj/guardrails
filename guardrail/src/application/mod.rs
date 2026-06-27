//! Application layer — port definition, axum router, shared state, and the
//! guardrail loop.
//!
//! This layer owns the `BackendPort` abstraction but never depends on the
//! concrete HTTP infrastructure: the adapter implementing the port lives in the
//! `connector` layer, which depends inward on this trait.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
    Router,
};
use serde_json::Value;
use tracing::{debug, error, info, info_span, warn, Instrument};

use crate::domain::decode::{
    decode_response, response_with_text, response_with_tool_calls, ModelOutput,
};
pub use crate::domain::guardrails::Guardrails;
use crate::domain::metrics::{
    now_rfc3339, redact_args, NoopRecorder, Outcome, OutcomeRecord, SharedRecorder,
};
use crate::domain::model::ChatRequest;
use crate::domain::rescue;
use crate::domain::respond;
use crate::domain::retry::ErrorTracker;
use crate::domain::validate::{
    coerce_arguments, repair_argument_names, validate, ErrorCategory, Validation,
};

/// Port: everything the application layer needs from the HTTP infrastructure.
#[async_trait::async_trait]
pub trait BackendPort: Send + Sync {
    /// POST a buffered body to `target`, return (status, headers, body).
    async fn post(
        &self,
        target: &str,
        headers: &HeaderMap,
        body: Vec<u8>,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>), Response>;

    /// Forward a request verbatim and stream the response back.
    async fn forward(
        &self,
        method: axum::http::Method,
        target: &str,
        headers: &HeaderMap,
        body: bytes::Bytes,
    ) -> Response;
}

/// Upper bound on a request body the proxy will buffer.
const MAX_REQUEST_BODY: usize = 32 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub backend_url: String,
    pub guardrails: Guardrails,
    pub port: Arc<dyn BackendPort>,
    pub recorder: SharedRecorder,
}

impl AppState {
    pub fn new(port: impl BackendPort + 'static, backend_url: impl Into<String>) -> Self {
        Self {
            backend_url: backend_url.into().trim_end_matches('/').to_string(),
            guardrails: Guardrails::default(),
            port: Arc::new(port),
            recorder: Arc::new(NoopRecorder),
        }
    }

    pub fn with_guardrails(mut self, guardrails: Guardrails) -> Self {
        self.guardrails = guardrails;
        self
    }

    /// Install the sink that terminal outcomes are recorded to. Defaults to a
    /// no-op recorder (metrics off).
    pub fn with_recorder(mut self, recorder: SharedRecorder) -> Self {
        self.recorder = recorder;
        self
    }
}

/// Build the axum router.
pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", any(proxy))
        .route("/v1/models", any(proxy))
        .fallback(any(proxy))
        .with_state(state)
}

async fn proxy(State(state): State<AppState>, req: Request) -> Response {
    let method = req.method().clone();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/")
        .to_string();
    let target = format!("{}{}", state.backend_url, path_and_query);

    let span = info_span!("proxy", %method, path = %path_and_query);
    async move {
        debug!(target = %target, "forwarding to backend");

        let (parts, body) = req.into_parts();

        let body_bytes = match axum::body::to_bytes(body, MAX_REQUEST_BODY).await {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "failed to read request body (or exceeded cap)");
                return (StatusCode::PAYLOAD_TOO_LARGE, "request body too large").into_response();
            }
        };

        // Only chat-completions POSTs are eligible for guardrails; other routes
        // whose bodies happen to deserialize as a ChatRequest must pass through.
        let guarded = if parts.method == axum::http::Method::POST
            && parts.uri.path() == "/v1/chat/completions"
        {
            serde_json::from_slice::<ChatRequest>(&body_bytes)
                .ok()
                .filter(|r| r.has_tools() && !r.stream())
        } else {
            None
        };

        if let Some(request) = guarded {
            return guardrail_loop(&state, &target, &parts.headers, request).await;
        }

        state
            .port
            .forward(parts.method, &target, &parts.headers, body_bytes)
            .await
    }
    .instrument(span)
    .await
}

async fn guardrail_loop(
    state: &AppState,
    target: &str,
    headers: &HeaderMap,
    mut request: ChatRequest,
) -> Response {
    let g = state.guardrails;
    // Only own the `respond` tool when the client hasn't defined one itself;
    // otherwise we'd hijack the client's real tool calls as final text.
    let respond_active = !request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .any(|t| t.function.name == respond::RESPOND);
    if respond_active {
        request.push_tool(respond::respond_tool());
    }
    let tools = request.tools.clone().unwrap_or_default();

    let mut tracker = ErrorTracker::new(g.max_retries);
    let mut last_text: Option<String> = None;

    // One terminal outcome is recorded per guarded request. `record` borrows the
    // request's model and the loop-local retry count; building it at each return
    // keeps the metrics next to the decision that produced them.
    let model = request.model.clone();
    let recorder = state.recorder.clone();
    let emit = |outcome: Outcome,
                error_category: Option<ErrorCategory>,
                parser: Option<String>,
                tool_name: Option<String>,
                retries: u32,
                detail: Option<String>| {
        recorder.record(OutcomeRecord {
            ts: now_rfc3339(),
            model: model.clone(),
            outcome,
            error_category,
            parser,
            tool_name,
            retries,
            detail,
        });
    };

    loop {
        let body = match serde_json::to_vec(&request) {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "failed to serialize guardrail request");
                emit(
                    Outcome::InternalError,
                    None,
                    None,
                    None,
                    tracker.attempts(),
                    None,
                );
                return (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
            }
        };

        let (status, out_headers, bytes) = match state.port.post(target, headers, body).await {
            Ok(parts) => parts,
            Err(resp) => {
                emit(
                    Outcome::BackendError,
                    None,
                    None,
                    None,
                    tracker.attempts(),
                    None,
                );
                return resp;
            }
        };

        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "backend response was not JSON; forwarding unverified");
                emit(Outcome::NonJson, None, None, None, tracker.attempts(), None);
                return bytes_response(status, out_headers, bytes);
            }
        };

        let mut rescued_parser: Option<&'static str> = None;
        let calls = match decode_response(&value) {
            ModelOutput::ToolCalls(calls) => Some((calls, false)),
            ModelOutput::Text(text) => {
                last_text = Some(text.clone());
                match rescue::rescue(&text) {
                    Some((parser, calls)) => {
                        info!(parser, count = calls.len(), "rescued tool calls from text");
                        rescued_parser = Some(parser);
                        Some((calls, true))
                    }
                    None => None,
                }
            }
        };

        let Some((mut calls, rescued)) = calls else {
            emit(
                Outcome::PassthroughNoCalls,
                None,
                None,
                None,
                tracker.attempts(),
                None,
            );
            return bytes_response(status, out_headers, bytes);
        };

        if respond_active {
            // Intercept only when the respond call carries usable text. Missing
            // or malformed arguments fall through to validation, which retries
            // against the injected respond tool's required `message` field.
            if let Some(text) = calls
                .iter()
                .find(|c| respond::is_respond(c))
                .and_then(respond::message_text)
            {
                emit(
                    Outcome::RespondIntercept,
                    None,
                    None,
                    Some(respond::RESPOND.to_string()),
                    tracker.attempts(),
                    None,
                );
                return json_response(status, out_headers, &response_with_text(&value, &text));
            }
        }

        // Repair tool-call arguments before validating, so a fixable mistake
        // costs a local rewrite rather than a corrective retry. Any change forces
        // re-emission from `calls` so the fix reaches the client (a native,
        // otherwise-unmodified response forwards raw bytes). Names are repaired
        // first (fill a missing required field from a wrongly-styled key), then
        // types (coerce the now-correctly-named value).
        let mut repaired = false;
        if repair_argument_names(&mut calls, &tools) {
            info!("repaired tool-call argument names to declared schema keys");
            repaired = true;
        }
        if coerce_arguments(&mut calls, &tools) {
            info!("coerced mistyped tool-call arguments to declared types");
            repaired = true;
        }

        match validate(&calls, &tools) {
            Validation::Valid => {
                // Classify how the call became valid: a retry that finally landed
                // outranks the deterministic repairs, which outrank a plain
                // rescue, which outranks an untouched native call.
                let attempts = tracker.attempts();
                let outcome = if attempts > 0 {
                    Outcome::RecoveredAfterRetry
                } else if repaired {
                    Outcome::Repaired
                } else if rescued {
                    Outcome::Rescued
                } else {
                    Outcome::NativeValid
                };
                let tool_name = calls.first().map(|c| c.name.clone());
                emit(
                    outcome,
                    None,
                    rescued_parser.map(str::to_string),
                    tool_name,
                    attempts,
                    None,
                );
                return if rescued || repaired {
                    json_response(
                        status,
                        out_headers,
                        &response_with_tool_calls(&value, &calls),
                    )
                } else {
                    bytes_response(status, out_headers, bytes)
                };
            }
            Validation::NeedsRetry {
                category,
                nudge,
                offending,
            } => {
                if tracker.can_retry() {
                    tracker.record_retry();
                    warn!(attempt = tracker.attempts(), %nudge, "tool call invalid; retrying");
                    request
                        .messages
                        .extend(crate::domain::retry::tool_error_followup(&calls, &nudge));
                    continue;
                }
                warn!("tool call invalid and retries exhausted; falling back");
                // The guardrails could not fix this call. Capture the category,
                // the offending tool, and a redacted argument snippet so it can be
                // triaged and fixed in the tool later. `validate` stops at the
                // first invalid call, so attribute the row to *that* call (which
                // may not be the first) rather than `calls.first()`.
                let offending = calls.get(offending);
                let detail = offending.map(|c| {
                    let snippet = redact_args(&c.arguments);
                    if snippet.is_empty() {
                        nudge.clone()
                    } else {
                        format!("{nudge} | args: {snippet}")
                    }
                });
                emit(
                    Outcome::FallbackUnfixed,
                    Some(category),
                    None,
                    offending.map(|c| c.name.clone()),
                    tracker.attempts(),
                    detail,
                );
                return match &last_text {
                    Some(text) => {
                        json_response(status, out_headers, &response_with_text(&value, text))
                    }
                    None => json_response(status, out_headers, &response_with_text(&value, &nudge)),
                };
            }
        }
    }
}

fn json_response(status: StatusCode, headers: HeaderMap, value: &Value) -> Response {
    match serde_json::to_vec(value) {
        Ok(bytes) => bytes_response(status, headers, bytes),
        Err(e) => {
            error!(error = %e, "failed to serialize guardrail response");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
        }
    }
}

fn bytes_response(status: StatusCode, headers: HeaderMap, bytes: Vec<u8>) -> Response {
    let mut response = Response::new(axum::body::Body::from(bytes));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}
