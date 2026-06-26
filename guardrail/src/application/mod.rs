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
use crate::domain::model::ChatRequest;
use crate::domain::rescue;
use crate::domain::respond;
use crate::domain::retry::ErrorTracker;
use crate::domain::validate::{validate, Validation};

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
}

impl AppState {
    pub fn new(port: impl BackendPort + 'static, backend_url: impl Into<String>) -> Self {
        Self {
            backend_url: backend_url.into().trim_end_matches('/').to_string(),
            guardrails: Guardrails::default(),
            port: Arc::new(port),
        }
    }

    pub fn with_guardrails(mut self, guardrails: Guardrails) -> Self {
        self.guardrails = guardrails;
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

        let guarded = serde_json::from_slice::<ChatRequest>(&body_bytes)
            .ok()
            .filter(|r| r.has_tools() && !r.stream());

        if let Some(request) = guarded {
            if state.guardrails.any_active() {
                return guardrail_loop(&state, &target, &parts.headers, request).await;
            }
            return match state
                .port
                .post(&target, &parts.headers, body_bytes.to_vec())
                .await
            {
                Ok((status, headers, bytes)) => {
                    if let Ok(body) = serde_json::from_slice::<Value>(&bytes) {
                        inspect_response(&body, &request, state.guardrails);
                    } else {
                        warn!("tool-enabled response was not JSON; forwarding unverified");
                    }
                    bytes_response(status, headers, bytes)
                }
                Err(resp) => resp,
            };
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
    if g.respond {
        request.push_tool(respond::respond_tool());
    }
    let tools = request.tools.clone().unwrap_or_default();

    let mut tracker = ErrorTracker::new(if g.retry { g.max_retries } else { 0 });
    let mut last_text: Option<String> = None;

    loop {
        let body = match serde_json::to_vec(&request) {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "failed to serialize guardrail request");
                return (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
            }
        };

        let (status, out_headers, bytes) = match state.port.post(target, headers, body).await {
            Ok(parts) => parts,
            Err(resp) => return resp,
        };

        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "backend response was not JSON; forwarding unverified");
                return bytes_response(status, out_headers, bytes);
            }
        };

        let calls = match decode_response(&value) {
            ModelOutput::ToolCalls(calls) => Some((calls, false)),
            ModelOutput::Text(text) => {
                last_text = Some(text.clone());
                match g.rescue.then(|| rescue::rescue(&text)).flatten() {
                    Some((parser, calls)) => {
                        info!(parser, count = calls.len(), "rescued tool calls from text");
                        Some((calls, true))
                    }
                    None => None,
                }
            }
        };

        let Some((calls, rescued)) = calls else {
            return bytes_response(status, out_headers, bytes);
        };

        if g.respond {
            if let Some(r) = calls.iter().find(|c| respond::is_respond(c)) {
                let text = respond::message_text(r);
                return json_response(status, out_headers, &response_with_text(&value, &text));
            }
        }

        match validate(&calls, &tools) {
            Validation::Valid => {
                return if rescued {
                    json_response(
                        status,
                        out_headers,
                        &response_with_tool_calls(&value, &calls),
                    )
                } else {
                    bytes_response(status, out_headers, bytes)
                };
            }
            Validation::NeedsRetry(nudge) => {
                if g.retry && tracker.can_retry() {
                    tracker.record_retry();
                    warn!(attempt = tracker.attempts(), %nudge, "tool call invalid; retrying");
                    request
                        .messages
                        .extend(crate::domain::retry::tool_error_followup(&calls, &nudge));
                    continue;
                }
                warn!("tool call invalid and retries exhausted; falling back");
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

fn inspect_response(body: &Value, request: &ChatRequest, guardrails: Guardrails) {
    match decode_response(body) {
        ModelOutput::ToolCalls(calls) => {
            let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
            info!(count = calls.len(), tool_calls = ?names, "decoded native tool_calls");
            match validate(&calls, request.tools.as_deref().unwrap_or(&[])) {
                Validation::Valid => info!("tool calls valid"),
                Validation::NeedsRetry(nudge) => warn!(%nudge, "tool calls invalid (log-only)"),
            }
        }
        ModelOutput::Text(text) if guardrails.rescue => match rescue::rescue(&text) {
            Some((parser, calls)) => {
                let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
                info!(parser, count = calls.len(), tool_calls = ?names, "rescued tool calls from text (log-only)");
                match validate(&calls, request.tools.as_deref().unwrap_or(&[])) {
                    Validation::Valid => info!("rescued tool calls valid"),
                    Validation::NeedsRetry(nudge) => warn!(%nudge, "rescued tool calls invalid"),
                }
            }
            None => debug!(len = text.len(), "model returned text, no tool calls"),
        },
        ModelOutput::Text(_) => debug!("model returned text; rescue disabled"),
    }
}
