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
use crate::domain::conversation::PrefixChain;
use crate::domain::metrics::{
    now_rfc3339, redact_args, Conversation, NoopRecorder, Outcome, OutcomeRecord, SharedRecorder,
    Usage,
};
use crate::domain::model::ChatRequest;
use crate::domain::provider::{Provider, DEFAULT_PROVIDER};
use crate::domain::registry::Registry;
use crate::domain::respond;
use crate::domain::responses::{self, ResponsesRequest};
use crate::domain::responses_sse::{assemble_responses_stream, AssembledResponses};
use crate::domain::retry::ErrorTracker;
use crate::domain::sse::{assemble_stream, AssembledResponse};
use crate::domain::validate::{
    coerce_arguments, repair_argument_names, validate, ErrorCategory, Validation,
};

/// Port: everything the application layer needs from the HTTP infrastructure.
///
/// Every method takes the [`Provider`] the request is bound for. The provider
/// carries the header names it owns, which the adapter strips from the client's
/// set before the hop — without that, a client-supplied `Authorization` would
/// displace a provider's own credential.
#[async_trait::async_trait]
pub trait BackendPort: Send + Sync {
    async fn post(
        &self,
        provider: &Provider,
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
        provider: &Provider,
        target: &str,
        headers: &HeaderMap,
        body: Vec<u8>,
    ) -> Result<(StatusCode, HeaderMap, mpsc::Receiver<Option<String>>, bool), Response>;

    async fn forward(
        &self,
        provider: &Provider,
        method: axum::http::Method,
        target: &str,
        headers: &HeaderMap,
        body: bytes::Bytes,
    ) -> Response;
}

const MAX_REQUEST_BODY: usize = 32 * 1024 * 1024;

/// The live registry, swappable at runtime by the management API.
///
/// Read per request and replaced wholesale on a configuration change, so a
/// model can be exposed or hidden without restarting the proxy.
pub type SharedRegistry = Arc<tokio::sync::RwLock<Arc<Registry>>>;

#[derive(Clone)]
pub struct AppState {
    pub registry: SharedRegistry,
    pub guardrails: Guardrails,
    pub port: Arc<dyn BackendPort>,
    pub recorder: SharedRecorder,
}

impl AppState {
    /// State with a single unnamed provider serving every request — the shape
    /// of a lone `--backend`.
    pub fn new(port: impl BackendPort + 'static, backend_url: impl Into<String>) -> Self {
        Self::with_registry(
            port,
            Registry::single(Provider::new(DEFAULT_PROVIDER, backend_url)),
        )
    }

    /// State routing across several providers.
    pub fn with_registry(port: impl BackendPort + 'static, registry: Registry) -> Self {
        Self::with_shared_registry(
            port,
            Arc::new(tokio::sync::RwLock::new(Arc::new(registry))),
        )
    }

    /// State sharing a registry the management API can replace.
    pub fn with_shared_registry(port: impl BackendPort + 'static, registry: SharedRegistry) -> Self {
        Self {
            registry,
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
        .route("/v1/responses", any(proxy_responses))
        .route("/v1/models", any(models))
        .fallback(any(proxy))
        .with_state(state)
}

/// `POST /v1/responses` — the Responses API, guarded.
///
/// The same policy as the chat path: requests declaring no tools are forwarded
/// untouched, tool-enabled ones run the guardrail loop. Only the decoding and
/// re-encoding differ; the guardrails in between are shared.
async fn proxy_responses(State(state): State<AppState>, req: Request) -> Response {
    let method = req.method().clone();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/v1/responses")
        .to_string();

    let span = info_span!("responses", %method, path = %path_and_query);
    async move {
        let (parts, body) = req.into_parts();
        let body_bytes = match axum::body::to_bytes(body, MAX_REQUEST_BODY).await {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "failed to read request body");
                return (StatusCode::PAYLOAD_TOO_LARGE, "request body too large").into_response();
            }
        };

        let request = if parts.method == axum::http::Method::POST {
            serde_json::from_slice::<ResponsesRequest>(&body_bytes).ok()
        } else {
            None
        };

        let registry = state.registry.read().await.clone();
        if let Some(request) = request.as_ref() {
            if registry.is_hidden(&request.model) {
                warn!(model = %request.model, "refused: model is not exposed");
                return json_response(
                    StatusCode::NOT_FOUND,
                    HeaderMap::new(),
                    &serde_json::json!({
                        "error": {
                            "message": format!(
                                "The model `{}` is not exposed by this proxy.",
                                request.model
                            ),
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    }),
                );
            }
        }

        let provider = registry
            .resolve(request.as_ref().map(|r| r.model.as_str()))
            .clone();
        let target = provider.target(&path_and_query);
        debug!(provider = %provider.name(), target = %target, "forwarding to provider");

        if let Some(request) = request {
            if request.has_tools() {
                let client_wants_stream = request.stream();
                return responses_loop(
                    &state, &provider, &target, &parts.headers, request, client_wants_stream,
                )
                .await;
            }
            let outcome = if request.stream() {
                Outcome::StreamedPassthrough
            } else {
                Outcome::NonToolPassthrough
            };
            state.recorder.record(OutcomeRecord {
                ts: now_rfc3339(), provider: provider.name().to_string(), model: request.model,
                outcome, error_category: None, parser: None, tool_name: None, retries: 0,
                detail: None,
                // Forwarded verbatim without the proxy reading the body, so
                // there is no usage and no conversation to report.
                usage: None,
                prefix_chain: None,
                conversation: None,
            });
        }

        state.port.forward(&provider, parts.method, &target, &parts.headers, body_bytes).await
    }
    .instrument(span)
    .await
}

/// The guardrail loop for the Responses API.
///
/// Structurally the chat loop, with the Responses decode/encode at the edges.
async fn responses_loop(
    state: &AppState,
    provider: &Arc<Provider>,
    target: &str,
    headers: &HeaderMap,
    mut request: ResponsesRequest,
    client_wants_stream: bool,
) -> Response {
    let g = state.guardrails;
    request.rest.insert("stream".to_string(), Value::Bool(true));

    let tools = request.normalized_tools();
    let respond_active = !tools.iter().any(|t| t.function.name == respond::RESPOND);
    if respond_active {
        request.push_tool(respond::respond_tool());
    }
    let tools = request.normalized_tools();

    let (body_tx, body_rx) = mpsc::channel::<String>(1024);
    let (passthrough_tx, mut passthrough_rx) = tokio::sync::oneshot::channel::<Response>();

    let port = state.port.clone();
    let recorder = state.recorder.clone();
    let provider = provider.clone();
    let target = target.to_string();
    let headers = headers.clone();
    let model = request.model.clone();

    tokio::spawn(async move {
        run_responses_guardrail(
            port, recorder, provider, target, headers, request, tools, respond_active, g, model,
            body_tx, passthrough_tx,
        )
        .await;
    });

    if client_wants_stream {
        drop(passthrough_rx);
        sse_channel_response(StatusCode::OK, HeaderMap::new(), body_rx)
    } else {
        let sse_body = drain_rx(body_rx).await;
        match passthrough_rx.try_recv() {
            Ok(resp) => resp,
            Err(_) => json_response(
                StatusCode::OK,
                HeaderMap::new(),
                &responses_sse_to_json(&sse_body),
            ),
        }
    }
}

/// The Responses guardrail logic, running inside a spawned task.
#[allow(clippy::too_many_arguments)]
async fn run_responses_guardrail(
    port: Arc<dyn BackendPort>,
    recorder: SharedRecorder,
    provider: Arc<Provider>,
    target: String,
    headers: HeaderMap,
    mut request: ResponsesRequest,
    tools: Vec<crate::domain::model::Tool>,
    respond_active: bool,
    g: Guardrails,
    model: String,
    body_tx: mpsc::Sender<String>,
    passthrough_tx: tokio::sync::oneshot::Sender<Response>,
) {
    let mut tracker = ErrorTracker::new(g.max_retries);
    // Totalled over every backend attempt — see the chat loop's counterpart.
    let mut billed = Usage::default();
    // The turn this request continues, read once from the client's own request:
    // the Responses API is stateful, so a chained turn names its predecessor
    // instead of resending the transcript. Paired with the response id the
    // backend assigns, this is the edge that lets the report count a resent
    // prefix once instead of once per turn.
    let parent_id = request.previous_response_id().map(str::to_string);
    // This turn's own id, learned from the backend's terminal event.
    let mut response_id: Option<String> = None;

    let emit_metric = |billed: Usage,
                       response_id: Option<String>,
                       outcome: Outcome,
                       error_category: Option<ErrorCategory>,
                       parser: Option<String>,
                       tool_name: Option<String>,
                       retries: u32,
                       detail: Option<String>| {
        recorder.record(OutcomeRecord {
            ts: now_rfc3339(), provider: provider.name().to_string(), model: model.clone(),
            outcome, error_category, parser, tool_name, retries, detail,
            usage: (!billed.is_empty()).then_some(billed),
            // Only a turn the backend named can anchor a chain; without an id
            // there is nothing for a later turn to point back to.
            conversation: response_id.map(|id| Conversation { id, parent: parent_id.clone() }),
            // The Responses API supplies real edges; nothing to infer.
            prefix_chain: None,
        });
    };

    loop {
        let body_bytes = match serde_json::to_vec(&request) {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "failed to serialize request");
                emit_metric(billed, response_id.clone(), Outcome::InternalError, None, None, None, tracker.attempts(), None);
                return;
            }
        };

        let (mut sse_rx, is_native_sse) =
            match port.stream_post(&provider, &target, &headers, body_bytes).await {
                Ok((_status, _resp_headers, rx, native)) => (rx, native),
                Err(passthrough_resp) => {
                    // See the chat loop: a streaming client already holds a
                    // `200`, so a backend failure is reported in-band.
                    let status = passthrough_resp.status();
                    if !status.is_success() {
                        emit_metric(billed, response_id.clone(), Outcome::InternalError, None, None, None, tracker.attempts(), Some(format!("backend returned {status}")));
                        send_stream_error(&body_tx, status).await;
                    }
                    let _ = passthrough_tx.send(passthrough_resp);
                    return;
                }
            };

        let forward_text = tracker.attempts() == 0 && is_native_sse;
        let tx = body_tx.clone();
        let (assembled, attempt_usage) = assemble_responses_stream(
            &mut sse_rx,
            |line: &str| {
                if forward_text {
                    let _ = tx.try_send(line.to_string());
                }
            },
            None,
        )
        .await;
        if let Some(attempt_usage) = attempt_usage {
            billed.add(attempt_usage);
        }

        // The backend names this turn on its terminal event, which the
        // assembler keeps as the template. Read on every attempt so a retry —
        // which gets a fresh response id — records the one actually delivered,
        // the id a following turn will chain to.
        let capture_id = |template: &Value, into: &mut Option<String>| {
            if let Some(id) = template.get("id").and_then(Value::as_str) {
                *into = Some(id.to_string());
            }
        };

        let (mut calls, template, rescued_parser, rescued_text) = match assembled {
            AssembledResponses::Text { ref template, .. } => {
                capture_id(template, &mut response_id);
                emit_metric(billed, response_id, Outcome::PassthroughNoCalls, None, None, None, tracker.attempts(), None);
                return;
            }
            AssembledResponses::Rescued { parser, calls, template, text } => {
                info!(parser, count = calls.len(), "rescued tool calls from text");
                (calls, template, Some(parser), text)
            }
            AssembledResponses::ToolCalls { calls, template, .. } => {
                (calls, template, None, String::new())
            }
        };
        capture_id(&template, &mut response_id);
        let rescued = rescued_parser.is_some();

        if respond_active {
            if let Some(text) = calls
                .iter()
                .find(|c| respond::is_respond(c))
                .and_then(respond::message_text)
            {
                emit_metric(billed, response_id.clone(), Outcome::RespondIntercept, None, None, Some(respond::RESPOND.to_string()), tracker.attempts(), None);
                send_responses_value(&body_tx, &responses::with_text(&template, &text)).await;
                return;
            }
        }

        if let crate::domain::precondition::Precondition::Failed { nudge } =
            crate::domain::precondition::check(&calls)
        {
            warn!(%nudge, "precondition failed");
            emit_metric(billed, response_id.clone(), Outcome::WriteRefused, None, None, calls.first().map(|c| c.name.clone()), tracker.attempts(), Some(nudge.clone()));
            let text = format!("The tool call could not be completed. {nudge}");
            send_responses_value(&body_tx, &responses::with_text(&template, &text)).await;
            return;
        }

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
                emit_metric(billed, response_id.clone(), outcome, None, rescued_parser.map(str::to_string), calls.first().map(|c| c.name.clone()), attempts, None);
                // Text a rescue was recovered from was held back rather than
                // forwarded, so emitting it here shows it to the client once —
                // without the call syntax the rescue exists to hide.
                let carried = crate::domain::rescue::prose_beside_call(&rescued_text)
                    .unwrap_or_default();
                let body = responses::with_tool_calls_and_text(&template, &calls, &carried);
                send_responses_value(&body_tx, &body).await;
                return;
            }
            Validation::NeedsRetry { category, nudge, offending } => {
                if tracker.can_retry() {
                    tracker.record_retry();
                    warn!(attempt = tracker.attempts(), %nudge, "tool call invalid; retrying");
                    request.extend_input(crate::domain::retry::tool_error_followup(&calls, &nudge));
                    continue;
                }
                warn!("retries exhausted");
                let offending_call = calls.get(offending);
                let detail = offending_call.map(|c| {
                    let s = redact_args(&c.arguments);
                    if s.is_empty() { nudge.clone() } else { format!("{nudge} | args: {s}") }
                });
                emit_metric(billed, response_id.clone(), Outcome::RetriesExhausted, Some(category), None, offending_call.map(|c| c.name.clone()), tracker.attempts(), detail);
                let text = format!("The tool call could not be completed after several attempts. {nudge}");
                send_responses_value(&body_tx, &responses::with_text(&template, &text)).await;
                return;
            }
        }
    }
}

/// Emit a final Responses body as a terminal `response.completed` event.
async fn send_responses_value(tx: &mpsc::Sender<String>, body: &Value) {
    let event = serde_json::json!({"type": "response.completed", "response": body});
    let line = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
    let _ = tx.send(format!("data: {line}\n\n")).await;
}

/// Recover the final Responses body from the SSE the loop produced, for a
/// client that did not ask to stream.
fn responses_sse_to_json(sse: &str) -> Value {
    let mut text = String::new();
    let mut completed = None;
    for line in sse.lines() {
        let Some(event) = crate::domain::sse::parse_sse_line(line) else { continue };
        match event.get("type").and_then(Value::as_str) {
            // The *last* one wins: the guardrail loop emits its own terminal
            // event after any repair, and returning an earlier one would hand
            // back the output the repair replaced.
            Some("response.completed") => {
                if let Some(response) = event.get("response") {
                    completed = Some(response.clone());
                }
            }
            Some("response.output_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    text.push_str(delta);
                }
            }
            _ => {}
        }
    }
    if let Some(response) = completed {
        return response;
    }
    // Only text deltas were forwarded (a plain-text answer), so rebuild the
    // body they add up to.
    responses::with_text(&Value::Null, &text)
}

/// `GET /v1/models` — the union of every provider's catalogue.
///
/// With several providers a client must see all of their models, not the
/// default provider's alone, or it cannot name a model that would route
/// elsewhere. Non-GET requests to this path keep the old behaviour and are
/// forwarded to the default provider.
async fn models(State(state): State<AppState>, req: Request) -> Response {
    // With one provider there is nothing to merge and nothing to disambiguate,
    // so the response is forwarded untouched. Tagging it would break the
    // byte-for-byte passthrough single-backend users have today, for no
    // information they do not already have.
    if req.method() != axum::http::Method::GET || state.registry.read().await.len() == 1 {
        return proxy(State(state), req).await;
    }
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/v1/models")
        .to_string();
    let headers = req.headers().clone();

    // Snapshot the registry once so every lookup in this request sees the same
    // configuration, even if a change lands mid-flight.
    let registry = state.registry.read().await.clone();

    // Ask every provider concurrently: the response is as slow as the slowest
    // one, not the sum.
    let lookups = registry.providers().map(|provider| {
        let provider = provider.clone();
        let port = state.port.clone();
        let headers = headers.clone();
        let target = provider.target(&path_and_query);
        async move {
            let response = port
                .forward(&provider, axum::http::Method::GET, &target, &headers, bytes::Bytes::new())
                .await;
            (provider, read_models(response).await)
        }
    });

    let mut merged: Vec<Value> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut reachable = 0usize;

    for (provider, models) in futures_util::future::join_all(lookups).await {
        match models {
            Some(models) => {
                reachable += 1;
                for model in models {
                    // Tag each entry with the provider that serves it, so a
                    // client (or a UI) can tell two same-named models apart.
                    let id = model.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                    // Hidden models are not advertised, so the listing and what
                    // the proxy will actually serve agree.
                    if registry.is_hidden(&id) {
                        continue;
                    }
                    if !id.is_empty() && !seen.insert(id) {
                        // An id already claimed by an earlier provider routes
                        // there, so listing it twice would misdescribe routing.
                        continue;
                    }
                    let mut model = model;
                    if let Some(obj) = model.as_object_mut() {
                        obj.insert("provider".to_string(), Value::String(provider.name().to_string()));
                    }
                    merged.push(model);
                }
            }
            None => {
                // Unreachable or unparseable: the other providers' models are
                // still worth returning, so this degrades rather than fails.
                warn!(provider = %provider.name(), "could not list models for the aggregate");
            }
        }
    }

    // The registry always holds at least one provider, so no provider
    // answering means every one of them failed.
    if reachable == 0 {
        return (
            StatusCode::BAD_GATEWAY,
            "no provider could list its models",
        )
            .into_response();
    }

    json_response(
        StatusCode::OK,
        HeaderMap::new(),
        &serde_json::json!({ "object": "list", "data": merged }),
    )
}

/// Pull `data[]` out of an OpenAI-compatible `/v1/models` response, or `None`
/// when the provider did not answer with one.
async fn read_models(response: Response) -> Option<Vec<Value>> {
    if !response.status().is_success() {
        return None;
    }
    let bytes = axum::body::to_bytes(response.into_body(), MAX_REQUEST_BODY)
        .await
        .ok()?;
    let body: Value = serde_json::from_slice(&bytes).ok()?;
    Some(body.get("data")?.as_array()?.clone())
}

async fn proxy(State(state): State<AppState>, req: Request) -> Response {
    let method = req.method().clone();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/")
        .to_string();

    let span = info_span!("proxy", %method, path = %path_and_query);
    async move {
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

        // A model the user hid is refused rather than routed. Falling back to
        // the default provider would quietly serve something the user chose not
        // to expose, and would disagree with what /v1/models advertises.
        let registry = state.registry.read().await.clone();
        if let Some(request) = chat_request.as_ref() {
            if registry.is_hidden(&request.model) {
                warn!(model = %request.model, "refused: model is not exposed");
                return json_response(
                    StatusCode::NOT_FOUND,
                    HeaderMap::new(),
                    &serde_json::json!({
                        "error": {
                            "message": format!(
                                "The model `{}` is not exposed by this proxy.",
                                request.model
                            ),
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    }),
                );
            }
        }

        // Route on the model, which is the only routing hint an
        // OpenAI-compatible client gives us. Requests with no parseable model
        // — every non-chat path — go to the default provider.
        let provider = registry
            .resolve(chat_request.as_ref().map(|r| r.model.as_str()))
            .clone();
        let target = provider.target(&path_and_query);
        debug!(provider = %provider.name(), target = %target, "forwarding to provider");

        if let Some(request) = chat_request {
            if request.has_tools() {
                let client_wants_stream = request.stream();
                return guardrail_loop(&state, &provider, &target, &parts.headers, request, client_wants_stream).await;
            }
            let outcome = if request.stream() { Outcome::StreamedPassthrough } else { Outcome::NonToolPassthrough };
            state.recorder.record(OutcomeRecord {
                ts: now_rfc3339(), provider: provider.name().to_string(), model: request.model, outcome,
                error_category: None, parser: None, tool_name: None, retries: 0, detail: None,
                // Forwarded verbatim without the proxy reading the body, so
                // there is no usage and no conversation to report.
                usage: None,
                prefix_chain: None,
                conversation: None,
            });
        }

        state.port.forward(&provider, parts.method, &target, &parts.headers, body_bytes).await
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
    provider: &Arc<Provider>,
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
    let provider = provider.clone();
    let target = target.to_string();
    let headers = headers.clone();
    let model = request.model.clone();

    tokio::spawn(async move {
        run_guardrail(
            port, recorder, provider, target, headers, request, tools,
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
    provider: Arc<Provider>,
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
    // Usage totalled over every backend attempt this request makes. A retry is
    // a second billed call, so the recorded row must reflect the sum, not just
    // the attempt that happened to succeed.
    let mut billed = Usage::default();

    // Chat Completions is stateless: a turn carries no id and no reference to
    // its predecessor, so the `conversation` argument below is always `None`
    // and the edge -- if any -- is inferred by the recorder instead.
    //
    // The chain is computed from the client's *original* messages, captured
    // before the loop appends any corrective nudge. A retry rewrites
    // `request.messages`, and hashing the rewritten array would describe a
    // transcript the client never sent: the next real turn extends what the
    // client sent, not what the guardrails asked in between, and would fail to
    // match. This is the one place the metrics path reads message content, and
    // it reads it only to hash: the digests are what is stored, never the text.
    let prefix_chain = Some(PrefixChain::of(&request.messages));

    let emit_metric = |billed: Usage,
                       conversation: Option<Conversation>,
                       outcome: Outcome,
                       error_category: Option<ErrorCategory>,
                       parser: Option<String>,
                       tool_name: Option<String>,
                       retries: u32,
                       detail: Option<String>| {
        recorder.record(OutcomeRecord {
            ts: now_rfc3339(), provider: provider.name().to_string(), model: model.clone(), outcome,
            error_category, parser, tool_name, retries, detail,
            conversation,
            usage: (!billed.is_empty()).then_some(billed),
            prefix_chain: prefix_chain.clone(),
        });
    };

    loop {
        let body_bytes = match serde_json::to_vec(&request) {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "failed to serialize request");
                emit_metric(billed, None, Outcome::InternalError, None, None, None, tracker.attempts(), None);
                return;
            }
        };

        let (mut sse_rx, is_native_sse) = match port.stream_post(&provider, &target, &headers, body_bytes).await {
            Ok((_status, _resp_headers, rx, native)) => (rx, native),
            Err(passthrough_resp) => {
                // A backend error or a non-JSON/non-SSE body — forward verbatim.
                // A non-streaming client receives this response as-is. A
                // streaming one has already been handed a `200` with SSE
                // headers, so the status can no longer be changed; the failure
                // is relayed in-band instead, which is what an OpenAI-compatible
                // client reads on a stream that fails after it has begun.
                let status = passthrough_resp.status();
                if !status.is_success() {
                    emit_metric(billed, None, Outcome::InternalError, None, None, None, tracker.attempts(), Some(format!("backend returned {status}")));
                    send_stream_error(&body_tx, status).await;
                }
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
        let (assembled, attempt_usage) = assemble_stream(
            &mut sse_rx,
            |line: &str| {
                if forward_text {
                    let _ = tx.try_send(line.to_string());
                }
            },
            None, // kind_tx not needed here — we use result directly
        )
        .await;
        // Counted before any early return below, so an attempt is billed
        // whatever the outcome turns out to be.
        if let Some(attempt_usage) = attempt_usage {
            billed.add(attempt_usage);
        }

        match assembled {
            // ── Pure text ────────────────────────────────────────────────────
            AssembledResponse::Text { ref template, ref content } => {
                emit_metric(billed, None, Outcome::PassthroughNoCalls, None, None, None, tracker.attempts(), None);
                // Live forwarding already delivered the text when it was on —
                // but a JSON backend, or a retry, has `forward_text` off and
                // nothing has reached the client yet. Emitting the assembled
                // answer here is what keeps that case from returning an empty
                // body (`null` once rebuilt for a non-streaming client).
                if !forward_text {
                    let value = response_with_text(template, content);
                    send_value(&body_tx, &value).await;
                }
                return; // body_tx drops → stream closes
            }

            // ── Tool calls ───────────────────────────────────────────────────
            AssembledResponse::ToolCalls { .. } | AssembledResponse::Rescued { .. } => {
                let (mut calls, template, rescued_parser, native_content): (_, _, Option<&'static str>, String) =
                    match assembled {
                        AssembledResponse::Rescued { parser, calls, template, content } => {
                            info!(parser, count = calls.len(), "rescued tool calls from text");
                            (calls, template, Some(parser), content)
                        }
                        AssembledResponse::ToolCalls { calls, template, content } => (calls, template, None, content),
                        AssembledResponse::Text { .. } => unreachable!(),
                    };
                let rescued = rescued_parser.is_some();

                // Respond-tool intercept.
                if respond_active {
                    if let Some(text) = calls.iter().find(|c| respond::is_respond(c)).and_then(respond::message_text) {
                        emit_metric(billed, None, Outcome::RespondIntercept, None, None, Some(respond::RESPOND.to_string()), tracker.attempts(), None);
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
                    emit_metric(billed, None, Outcome::WriteRefused, None, None, calls.first().map(|c| c.name.clone()), tracker.attempts(), Some(nudge.clone()));
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
                        emit_metric(billed, None, outcome, None, rescued_parser.map(str::to_string), calls.first().map(|c| c.name.clone()), attempts, None);
                        // A rescued call was recovered *from* the text, so what
                        // the model wrote around it is its own answer and is
                        // kept — minus the call syntax itself, which the rescue
                        // exists to hide. Text that rode alongside a native
                        // call was already forwarded live when `forward_text`
                        // was set, and repeating it would show it twice.
                        let carried = (rescued && !forward_text)
                            .then(|| crate::domain::rescue::prose_beside_call(&native_content))
                            .flatten()
                            .unwrap_or_default();
                        let value = crate::domain::decode::response_with_tool_calls_and_text(&template, &calls, &carried);
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
                                    emit_metric(billed, None, Outcome::Rescued, None, Some(parser.to_string()), rescued_calls.first().map(|c| c.name.clone()), tracker.attempts(), None);
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
                        emit_metric(billed, None, Outcome::RetriesExhausted, Some(category), None, offending_call.map(|c| c.name.clone()), tracker.attempts(), detail);
                        let value = response_with_text(&template, &format!("The tool call could not be completed after several attempts. {nudge}"));
                        send_value(&body_tx, &value).await;
                        return;
                    }
                }
            }
        }
    }
}

/// Relay a backend failure to a client that is already receiving a stream.
///
/// Once the SSE headers are out the status line is fixed, so a failure can only
/// be reported inside the body. This is the shape OpenAI-compatible clients
/// expect from a stream that fails mid-flight: a `data:` frame carrying an
/// `error` object rather than a chunk.
async fn send_stream_error(tx: &mpsc::Sender<String>, status: StatusCode) {
    let event = serde_json::json!({
        "error": {
            "message": format!("The upstream provider returned {status}."),
            "type": "upstream_error",
            "code": status.as_u16(),
        }
    });
    let line = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
    let _ = tx.send(format!("data: {line}\n\n")).await;
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
            // Text the loop chose to keep beside the calls — a rescued call's
            // surrounding prose — rides on the chunk's own `content`. Rebuilding
            // with a hard `null` here would throw away what the guardrails just
            // took care to preserve.
            let carried = out
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            crate::domain::decode::response_with_tool_calls_and_text(&out, &calls, &carried)
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
