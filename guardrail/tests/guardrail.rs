//! Milestone 5 + 6 acceptance: the active guardrail loop end-to-end.
//!
//! Each test stands up a wiremock backend returning a specific (often malformed)
//! body and asserts the client sees the repaired result: respond unwrapped to
//! text, rescued calls re-emitted canonically, a bad call retried into a good
//! one, and exhaustion falling back to text. A toggles-off test confirms the
//! proxy degrades to passthrough.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use guardrail::{build_app, AppState, Guardrails};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WmRequest, Respond, ResponseTemplate};

async fn spawn(backend: &str, guardrails: Guardrails) -> String {
    let state = AppState::new(reqwest::Client::new(), backend).with_guardrails(guardrails);
    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// A request carrying one real tool, non-streamed — the guardrail path.
fn tool_request() -> Value {
    json!({
        "model": "local-model",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}}
        }]
    })
}

/// Backend that returns a fixed assistant `content` string.
fn text_response(content: &str) -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]
    })
}

async fn post(proxy: &str, body: &Value) -> Value {
    reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

#[tokio::test]
async fn respond_tool_call_is_stripped_to_text() {
    let backend = MockServer::start().await;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "respond", "arguments": "{\"message\":\"hello world\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&resp))
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails::default()).await;
    let got = post(&proxy, &tool_request()).await;

    assert_eq!(got["choices"][0]["message"]["content"], "hello world");
    assert_eq!(got["choices"][0]["finish_reason"], "stop");
    assert!(got["choices"][0]["message"]["tool_calls"].is_null());
}

#[tokio::test]
async fn malformed_text_call_is_rescued_and_re_emitted() {
    let backend = MockServer::start().await;
    // Qwen-style tool call buried in content.
    let content =
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&text_response(content)))
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails::default()).await;
    let got = post(&proxy, &tool_request()).await;

    let call = &got["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(call["function"]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(got["choices"][0]["finish_reason"], "tool_calls");
}

/// Responds with a different body on each successive call.
struct Sequence {
    calls: Arc<AtomicUsize>,
    bodies: Vec<Value>,
}
impl Respond for Sequence {
    fn respond(&self, _: &WmRequest) -> ResponseTemplate {
        let i = self.calls.fetch_add(1, Ordering::SeqCst);
        let body = &self.bodies[i.min(self.bodies.len() - 1)];
        ResponseTemplate::new(200).set_body_json(body)
    }
}

#[tokio::test]
async fn invalid_tool_name_is_retried_then_succeeds() {
    let backend = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let bad = text_response("<tool_call>{\"name\": \"get_wether\", \"arguments\": {}}</tool_call>");
    let good = text_response(
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Paris\"}}</tool_call>",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Sequence {
            calls: calls.clone(),
            bodies: vec![bad, good],
        })
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails::default()).await;
    let got = post(&proxy, &tool_request()).await;

    assert_eq!(
        got["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    // First attempt + one retry.
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn retry_exhaustion_falls_back_to_last_text() {
    let backend = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    // Always an invalid tool name; rescue recovers a call, validation keeps
    // failing, so the loop exhausts and falls back to the last text.
    let bad = "<tool_call>{\"name\": \"nope\", \"arguments\": {}}</tool_call>";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Sequence {
            calls: calls.clone(),
            bodies: vec![text_response(bad)],
        })
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails::default()).await;
    let got = post(&proxy, &tool_request()).await;

    // Default budget is 2 retries → 3 backend calls total.
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(got["choices"][0]["finish_reason"], "stop");
    assert_eq!(got["choices"][0]["message"]["content"], bad);
}

#[tokio::test]
async fn all_toggles_off_is_passthrough() {
    let backend = MockServer::start().await;
    let content = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&text_response(content)))
        .mount(&backend)
        .await;

    let off = Guardrails {
        rescue: false,
        respond: false,
        retry: false,
        max_retries: 0,
    };
    let proxy = spawn(&backend.uri(), off).await;
    let got = post(&proxy, &tool_request()).await;

    // No rescue: the malformed text is forwarded unchanged.
    assert_eq!(got, text_response(content));
}
