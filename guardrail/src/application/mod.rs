//! Application layer — port definition, axum router, shared state, and the
//! guardrail loop.
//!
//! ## Streaming architecture
//!
//! All guarded requests are sent to the backend with `stream: true`. The proxy
//! returns an SSE response body to the client immediately, backed by a channel.
//! Inside a spawned task the guardrail loop:
//!
//!   1. Reads the backend SSE stream line by line.
//!   2. Text / passthrough chunks → sent to the client body channel live.
//!   3. Tool-call chunks → buffered silently.
//!   4. At stream end, if tool calls were found → validate, repair, re-emit the
//!      corrected chunk into the body channel. On failure → retry (new backend
//!      request, same body channel). On exhaustion → emit an explanation text.
//!   5. Close the body channel → client sees `[DONE]`.
//!
//! This gives zero-latency text streaming while still guarding tool calls.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
    Router,
};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, error, info, info_span, warn, Instrument};

use crate::domain::decode::{response_with_text, response_with_tool_calls};
pub use crate::domain::guardrails::Guardrails;
use crate::domain::metrics::{
    now_rfc3339, redact_args, NoopRecorder, Outcome, OutcomeRecord, SharedRecorder,
};
use crate::domain::model::ChatRequest;
use crate::domain::respond;
use crate::domain::retry::ErrorTracker;
use crate::domain::sse::{assemble_stream, AssembledResponse};
use crate::domain::validate::{
    coerce_arguments, repair_argument_names, validate, ErrorCategory, Validation,
};

/// Port: everything the application layer needs from the HTTP infrastructure.
#[async_trait::async_trait]
pub trait BackendPort: Send + Sync {
    async fn post(
        &self,
        target: &str,
        headers: &HeaderMap,
        body: Vec<u8>,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>), Response>;

    /// POST with `stream: true`. Returns status, headers, a channel of raw SSE
    /// lines (`None` = end-of-stream), and a bool indicating whether the backend
    /// responded with a native `text/event-stream` (`true`) or a JSON body that
    /// was synthetically converted to SSE (`false`).
    ///
    /// Only native SSE streams should have text tokens forwarded live to the
    /// client — JSON backends return a complete response that may trigger rescue
    /// parsing, and forwarding text before rescue detection would leak the raw
    /// tool-call tag syntax to the client.
    async fn stream_post(
        &self,
        target: &str,
        headers: &HeaderMap,
        body: Vec<u8>,
    ) -> Result<(StatusCode, HeaderMap, mpsc::Receiver<Option<String>>, bool), Response>;

    async fn forward(
        &self,
        method: axum::http::Method,
        target: &str,
        headers: &HeaderMap,
        body: bytes::Bytes,
    ) -> Response;
}

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

    pub fn with_recorder(mut self, recorder: SharedRecorder) -> Self {
        self.recorder = recorder;
        self
    }
}

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
                error!(error = %e, "failed to read request body");
                return (StatusCode::PAYLOAD_TOO_LARGE, "request body too large").into_response();
            }
        };

        let chat_request = if parts.method == axum::http::Method::POST
            && parts.uri.path() == "/v1/chat/completions"
        {
            serde_json::from_slice::<ChatRequest>(&body_bytes).ok()
        } else {
            None
        };

        if let Some(request) = chat_request {
            if request.has_tools() {
                let client_wants_stream = request.stream();
                return guardrail_loop(&state, &target, &parts.headers, request, client_wants_stream).await;
            }
            let outcome = if request.stream() { Outcome::StreamedPassthrough } else { Outcome::NonToolPassthrough };
            state.recorder.record(OutcomeRecord {
                ts: now_rfc3339(), model: request.model, outcome,
                error_category: None, parser: None, tool_name: None, retries: 0, detail: None,
            });
        }

        state.port.forward(parts.method, &target, &parts.headers, body_bytes).await
    }
    .instrument(span)
    .await
}

/// The guardrail loop.
///
/// Returns an SSE (or JSON) response immediately. A background task drives the
/// actual backend communication and guardrail logic, writing output into a
/// channel that backs the response body. This means text tokens flow to the
/// client the instant the backend emits them.
async fn guardrail_loop(
    state: &AppState,
    target: &str,
    headers: &HeaderMap,
    mut request: ChatRequest,
    client_wants_stream: bool,
) -> Response {
    let g = state.guardrails;
    request.sanitize();
    request.rest.insert("stream".to_string(), Value::Bool(true));

    let respond_active = !request
        .tools.as_deref().unwrap_or_default()
        .iter().any(|t| t.function.name == respond::RESPOND);
    if respond_active { request.push_tool(respond::respond_tool()); }
    let tools = request.tools.clone().unwrap_or_default();

    // Two channels:
    // - body_tx/body_rx: SSE lines written by the guardrail task, read by the response body.
    // - passthrough_tx/passthrough_rx: used when the backend returns a non-SSE/non-JSON
    //   body that must be forwarded verbatim (can't go through the SSE channel).
    let (body_tx, body_rx) = mpsc::channel::<String>(1024);
    let (passthrough_tx, mut passthrough_rx) = tokio::sync::oneshot::channel::<Response>();

    let port = state.port.clone();
    let recorder = state.recorder.clone();
    let target = target.to_string();
    let headers = headers.clone();
    let model = request.model.clone();

    tokio::spawn(async move {
        run_guardrail(
            port, recorder, target, headers, request, tools,
            respond_active, g, model, body_tx, passthrough_tx,
        ).await;
    });

    if client_wants_stream {
        // Return the SSE body immediately — the guardrail task fills it live.
        // Passthrough (non-JSON backend) is a degenerate case for streaming clients;
        // we drop the receiver and the passthrough is silently discarded.
        drop(passthrough_rx);
        sse_channel_response(StatusCode::OK, HeaderMap::new(), body_rx)
    } else {
        // For non-streaming clients, wait for the guardrail task to finish,
        // then check if it sent a verbatim passthrough or SSE chunks.
        let sse_body = drain_rx(body_rx).await;
        match passthrough_rx.try_recv() {
            Ok(resp) => resp,
            Err(_) => {
                let json_body = sse_chunks_to_json(&sse_body);
                json_response(StatusCode::OK, HeaderMap::new(), &json_body)
            }
        }
    }
}

/// The actual guardrail logic, running inside a spawned task.
///
/// Writes validated SSE output to `body_tx`. When this function returns,
/// `body_tx` is dropped, which closes the body stream and signals `[DONE]`
/// to the client (the `sse_channel_response` appends the sentinel).
async fn run_guardrail(
    port: Arc<dyn BackendPort>,
    recorder: SharedRecorder,
    target: String,
    headers: HeaderMap,
    mut request: ChatRequest,
    tools: Vec<crate::domain::model::Tool>,
    respond_active: bool,
    g: Guardrails,
    model: String,
    body_tx: mpsc::Sender<String>,
    passthrough_tx: tokio::sync::oneshot::Sender<Response>,
) {
    let mut tracker = ErrorTracker::new(g.max_retries);

    let emit_metric = |outcome: Outcome,
                       error_category: Option<ErrorCategory>,
                       parser: Option<String>,
                       tool_name: Option<String>,
                       retries: u32,
                       detail: Option<String>| {
        recorder.record(OutcomeRecord {
            ts: now_rfc3339(), model: model.clone(), outcome,
            error_category, parser, tool_name, retries, detail,
        });
    };

    loop {
        let body_bytes = match serde_json::to_vec(&request) {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "failed to serialize request");
                emit_metric(Outcome::InternalError, None, None, None, tracker.attempts(), None);
                return;
            }
        };

        let (mut sse_rx, is_native_sse) = match port.stream_post(&target, &headers, body_bytes).await {
            Ok((_status, _resp_headers, rx, native)) => (rx, native),
            Err(passthrough_resp) => {
                // Non-JSON/non-SSE backend response — forward verbatim.
                let _ = passthrough_tx.send(passthrough_resp);
                return;
            }
        };

        // Forward text live only on the first attempt AND when the backend is
        // a native SSE stream. JSON backends may embed tool calls in text
        // (rescue format) — we can't forward text until we know it's safe.
        let forward_text = tracker.attempts() == 0 && is_native_sse;

        // Run the assembler. Text lines go directly to body_tx if forward_text.
        // Tool-call lines are buffered inside the assembler.
        let tx = body_tx.clone();
        let assembled = assemble_stream(
            &mut sse_rx,
            |line: &str| {
                if forward_text {
                    let _ = tx.try_send(line.to_string());
                }
            },
            None, // kind_tx not needed here — we use result directly
        )
        .await;

        match assembled {
            // ── Pure text ────────────────────────────────────────────────────
            // Lines were already forwarded live. Just record and return.
            AssembledResponse::Text { .. } => {
                emit_metric(Outcome::PassthroughNoCalls, None, None, None, tracker.attempts(), None);
                return; // body_tx drops → stream closes
            }

            // ── Tool calls ───────────────────────────────────────────────────
            AssembledResponse::ToolCalls { .. } | AssembledResponse::Rescued { .. } => {
                let (mut calls, template, rescued_parser, native_content): (_, _, Option<&'static str>, String) =
                    match assembled {
                        AssembledResponse::Rescued { parser, calls, template } => {
                            info!(parser, count = calls.len(), "rescued tool calls from text");
                            (calls, template, Some(parser), String::new())
                        }
                        AssembledResponse::ToolCalls { calls, template, content } => (calls, template, None, content),
                        AssembledResponse::Text { .. } => unreachable!(),
                    };
                let rescued = rescued_parser.is_some();

                // Respond-tool intercept.
                if respond_active {
                    if let Some(text) = calls.iter().find(|c| respond::is_respond(c)).and_then(respond::message_text) {
                        emit_metric(Outcome::RespondIntercept, None, None, Some(respond::RESPOND.to_string()), tracker.attempts(), None);
                        let value = response_with_text(&template, &text);
                        send_value(&body_tx, &value).await;
                        return;
                    }
                }

                // Precondition check.
                if let crate::domain::precondition::Precondition::Failed { nudge } =
                    crate::domain::precondition::check(&calls)
                {
                    warn!(%nudge, "precondition failed");
                    emit_metric(Outcome::WriteRefused, None, None, calls.first().map(|c| c.name.clone()), tracker.attempts(), Some(nudge.clone()));
                    let value = response_with_text(&template, &format!("The tool call could not be completed. {nudge}"));
                    send_value(&body_tx, &value).await;
                    return;
                }

                // Repair.
                let mut repaired = false;
                if repair_argument_names(&mut calls, &tools) { repaired = true; }
                if coerce_arguments(&mut calls, &tools) { repaired = true; }

                match validate(&calls, &tools) {
                    Validation::Valid => {
                        let attempts = tracker.attempts();
                        let outcome = if attempts > 0 { Outcome::RecoveredAfterRetry }
                            else if repaired { Outcome::Repaired }
                            else if rescued { Outcome::Rescued }
                            else { Outcome::NativeValid };
                        emit_metric(outcome, None, rescued_parser.map(str::to_string), calls.first().map(|c| c.name.clone()), attempts, None);
                        let value = response_with_tool_calls(&template, &calls);
                        send_value(&body_tx, &value).await;
                        return;
                    }

                    Validation::NeedsRetry { category, nudge, offending } => {
                        // Before retrying an unknown-tool error, check whether the
                        // model also emitted a valid tool call in its text content
                        // (e.g. Qwen XML alongside a hallucinated native tool call).
                        // Skip if content was already forwarded live (forward_text=true):
                        // sending a full response on top of already-streamed deltas
                        // would corrupt the SSE stream.
                        if matches!(category, ErrorCategory::UnknownTool) && !native_content.is_empty() && !forward_text {
                            if let Some((parser, rescued_calls)) = crate::domain::rescue::rescue(&native_content) {
                                let mut rescued_calls = rescued_calls;
                                if repair_argument_names(&mut rescued_calls, &tools) {}
                                if coerce_arguments(&mut rescued_calls, &tools) {}
                                if matches!(validate(&rescued_calls, &tools), Validation::Valid) {
                                    info!(parser, count = rescued_calls.len(), "rescued tool calls from content alongside invalid native call");
                                    emit_metric(Outcome::Rescued, None, Some(parser.to_string()), rescued_calls.first().map(|c| c.name.clone()), tracker.attempts(), None);
                                    let value = response_with_tool_calls(&template, &rescued_calls);
                                    send_value(&body_tx, &value).await;
                                    return;
                                }
                            }
                        }
                        if tracker.can_retry() {
                            tracker.record_retry();
                            warn!(attempt = tracker.attempts(), %nudge, "tool call invalid; retrying");
                            request.messages.extend(crate::domain::retry::tool_error_followup(&calls, &nudge));
                            continue;
                        }
                        warn!("retries exhausted");
                        let offending_call = calls.get(offending);
                        let detail = offending_call.map(|c| {
                            let s = redact_args(&c.arguments);
                            if s.is_empty() { nudge.clone() } else { format!("{nudge} | args: {s}") }
                        });
                        emit_metric(Outcome::RetriesExhausted, Some(category), None, offending_call.map(|c| c.name.clone()), tracker.attempts(), detail);
                        let value = response_with_text(&template, &format!("The tool call could not be completed after several attempts. {nudge}"));
                        send_value(&body_tx, &value).await;
                        return;
                    }
                }
            }
        }
    }
}

/// Send a chat-completion value as an SSE chunk into the body channel.
async fn send_value(tx: &mpsc::Sender<String>, value: &Value) {
    let effective = if value.is_object() {
        value.clone()
    } else {
        serde_json::json!({
            "id": "guardrail-0",
            "object": "chat.completion",
            "choices": []
        })
    };
    let chunk = to_sse_chunk(effective);
    let _ = tx.send(chunk).await;
}

/// Convert a `chat.completion` JSON value to a single SSE `data:` line.
fn to_sse_chunk(mut chunk: Value) -> String {
    if let Some(obj) = chunk.as_object_mut() {
        obj.insert("object".to_string(), Value::String("chat.completion.chunk".to_string()));
        if let Some(choices) = obj.get_mut("choices").and_then(Value::as_array_mut) {
            for choice in choices {
                if let Some(co) = choice.as_object_mut() {
                    if let Some(mut msg) = co.remove("message") {
                        if let Some(mo) = msg.as_object_mut() {
                            if let Some(tc) = mo.get_mut("tool_calls") {
                                if let Some(arr) = tc.as_array_mut() {
                                    for (i, c) in arr.iter_mut().enumerate() {
                                        if let Some(co2) = c.as_object_mut() {
                                            co2.insert("index".to_string(), Value::Number(i.into()));
                                        }
                                    }
                                }
                            }
                        }
                        co.insert("delta".to_string(), msg);
                    }
                }
            }
        }
    }
    let s = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
    format!("data: {s}\n\n")
}

/// Build a streaming SSE response backed by a channel.
/// Appends `data: [DONE]\n\n` when the sender drops.
fn sse_channel_response(
    status: StatusCode,
    mut headers: HeaderMap,
    rx: mpsc::Receiver<String>,
) -> Response {
    use futures_util::stream::{self, StreamExt};
    use tokio_stream::wrappers::ReceiverStream;

    headers.remove(header::CONTENT_LENGTH);
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

    let body_stream = ReceiverStream::new(rx)
        .map(|s| Ok::<_, std::convert::Infallible>(bytes::Bytes::from(s)))
        .chain(stream::once(async {
            Ok::<_, std::convert::Infallible>(bytes::Bytes::from_static(b"data: [DONE]\n\n"))
        }));

    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

/// Drain a channel to a String (for non-streaming clients).
async fn drain_rx(mut rx: mpsc::Receiver<String>) -> String {
    let mut out = String::new();
    while let Some(line) = rx.recv().await {
        out.push_str(&line);
    }
    out
}

/// Convert accumulated SSE chunks into a single buffered JSON chat-completion.
/// Handles both text responses (accumulates content) and tool-call responses
/// (converts delta.tool_calls → message.tool_calls in the final chunk).
fn sse_chunks_to_json(sse: &str) -> Value {
    use crate::domain::decode::{decode_response, ModelOutput};
    use crate::domain::sse::parse_sse_line;

    let mut last_chunk = Value::Null;
    let mut text_content = String::new();

    for line in sse.lines() {
        let Some(chunk) = parse_sse_line(line) else { continue };
        // Accumulate text content across chunks.
        if let Some(c) = chunk.get("choices").and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(Value::as_str)
        {
            text_content.push_str(c);
        }
        last_chunk = chunk;
    }

    if last_chunk.is_null() {
        return Value::Null;
    }

    // Convert the last SSE chunk (which has `delta`) into a buffered response
    // (which needs `message`). We do this by converting delta→message.
    let mut out = last_chunk;
    if let Some(obj) = out.as_object_mut() {
        obj.insert("object".to_string(), Value::String("chat.completion".to_string()));
        if let Some(choices) = obj.get_mut("choices").and_then(Value::as_array_mut) {
            for choice in choices.iter_mut() {
                if let Some(co) = choice.as_object_mut() {
                    if let Some(delta) = co.remove("delta") {
                        co.insert("message".to_string(), delta);
                    }
                }
            }
        }
    }

    // If the reconstructed response has tool_calls, return it as-is.
    // Otherwise fall back to building a text response.
    match decode_response(&out) {
        ModelOutput::ToolCalls(calls) => {
            use crate::domain::decode::response_with_tool_calls;
            response_with_tool_calls(&out, &calls)
        }
        ModelOutput::Text(_) => {
            response_with_text(&out, &text_content)
        }
    }
}


fn json_response(status: StatusCode, headers: HeaderMap, value: &Value) -> Response {
    match serde_json::to_vec(value) {
        Ok(bytes) => bytes_response(status, headers, bytes),
        Err(e) => {
            error!(error = %e, "failed to serialize response");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
        }
    }
}

fn bytes_response(status: StatusCode, headers: HeaderMap, bytes: Vec<u8>) -> Response {
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}
