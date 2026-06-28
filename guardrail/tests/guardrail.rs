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

use guardrail::connector::Backend;
use guardrail::{build_app, AppState, Guardrails};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WmRequest, Respond, ResponseTemplate};

async fn spawn(backend: &str, guardrails: Guardrails) -> String {
    let state =
        AppState::new(Backend::new(reqwest::Client::new()), backend).with_guardrails(guardrails);
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

fn edit_request() -> Value {
    json!({
        "model": "local-model",
        "messages": [{"role": "user", "content": "edit the file"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "oldString": {"type": "string"},
                        "newString": {"type": "string"}
                    },
                    "required": ["filePath", "oldString", "newString"]
                }
            }
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

fn native_tool_response(name: &str, arguments: Value) -> Value {
    json!({
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
                    "function": {"name": name, "arguments": arguments.to_string()}
                }]
            },
            "finish_reason": "tool_calls"
        }]
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

#[tokio::test]
async fn lfm_pythonic_tool_call_is_rescued_and_re_emitted() {
    let backend = MockServer::start().await;
    // LFM2.5 emits Pythonic calls wrapped in special tokens, with trailing prose.
    let content = "<|tool_call_start|>[get_weather(location=\"Paris\")]<|tool_call_end|>Checking the weather in Paris.";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&text_response(content)))
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails::default()).await;
    let got = post(&proxy, &tool_request()).await;

    let call = &got["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(call["function"]["arguments"], "{\"location\":\"Paris\"}");
    assert_eq!(got["choices"][0]["finish_reason"], "tool_calls");
    // No special tokens leak into the re-emitted body.
    let body = serde_json::to_string(&got).unwrap();
    assert!(!body.contains("<|tool_call"));
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
async fn missing_required_tool_argument_is_retried_then_succeeds() {
    let backend = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let missing_file = text_response(
        "<function=Edit><parameter=oldString>old</parameter><parameter=newString>new</parameter></function>",
    );
    let good = text_response(
        "<function=Edit><parameter=filePath>/tmp/example.rs</parameter><parameter=oldString>old</parameter><parameter=newString>new</parameter></function>",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Sequence {
            calls: calls.clone(),
            bodies: vec![missing_file, good],
        })
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails::default()).await;
    let got = post(&proxy, &edit_request()).await;

    let call = &got["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "Edit");
    let args: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["filePath"], "/tmp/example.rs");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn retry_exhaustion_returns_explanation() {
    let backend = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    // Always an invalid tool name; rescue recovers a call, validation keeps
    // failing, so the loop exhausts and returns an explanation to the model.
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
    let content = got["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.contains("could not be completed"));
    assert!(content.contains("nope")); // nudge names the unknown tool
}

#[tokio::test]
async fn native_invalid_tool_call_exhaustion_does_not_forward_tool_call() {
    let backend = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Sequence {
            calls: calls.clone(),
            bodies: vec![native_tool_response(
                "Edit",
                json!({"oldString": "old", "newString": "new"}),
            )],
        })
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails::default()).await;
    let got = post(&proxy, &edit_request()).await;

    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(got["choices"][0]["finish_reason"], "stop");
    assert!(got["choices"][0]["message"]["tool_calls"].is_null());
    assert!(got["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .contains("filePath"));
}

#[tokio::test]
async fn stringified_native_argument_is_coerced_without_a_retry() {
    let backend = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    // A typed tool whose `count` is an integer; the model stringifies it.
    let request = json!({
        "model": "local-model",
        "messages": [{"role": "user", "content": "count things"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "tally",
                "parameters": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"]
                }
            }
        }]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Sequence {
            calls: calls.clone(),
            bodies: vec![native_tool_response("tally", json!({"count": "3"}))],
        })
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails::default()).await;
    let got = post(&proxy, &request).await;

    // Coerced in place and re-emitted from parsed calls — no retry round-trip.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let call = &got["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "tally");
    assert_eq!(call["function"]["arguments"], "{\"count\":3}");
    assert_eq!(got["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn wrongly_styled_native_argument_key_is_repaired_without_a_retry() {
    let backend = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    // The model emits snake_case `file_path` for a schema that declares
    // camelCase `filePath`; the proxy should rebind it in place.
    let request = json!({
        "model": "local-model",
        "messages": [{"role": "user", "content": "edit the file"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                    "required": ["filePath"]
                }
            }
        }]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Sequence {
            calls: calls.clone(),
            bodies: vec![native_tool_response("Edit", json!({"file_path": "/tmp/x.rs"}))],
        })
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails::default()).await;
    let got = post(&proxy, &request).await;

    // Renamed in place and re-emitted from parsed calls — no retry round-trip.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let call = &got["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "Edit");
    assert_eq!(call["function"]["arguments"], "{\"filePath\":\"/tmp/x.rs\"}");
    assert_eq!(got["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn zero_max_retries_disables_the_retry_loop_but_keeps_repairs() {
    let backend = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    // An always-invalid tool name: rescue recovers the call, validation keeps
    // failing, and with no retry budget the loop falls back immediately instead
    // of re-asking the backend.
    let bad = "<tool_call>{\"name\": \"nope\", \"arguments\": {}}</tool_call>";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Sequence {
            calls: calls.clone(),
            bodies: vec![text_response(bad)],
        })
        .mount(&backend)
        .await;

    let proxy = spawn(&backend.uri(), Guardrails { max_retries: 0 }).await;
    let got = post(&proxy, &tool_request()).await;

    // No retries → exactly one backend call, then return an explanation.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(got["choices"][0]["finish_reason"], "stop");
    let content = got["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.contains("could not be completed"));
    assert!(content.contains("nope")); // nudge names the unknown tool
}
