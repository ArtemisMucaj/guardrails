//! Milestone 3 acceptance: log-only inspection must not alter responses.
//!
//! When a request carries tools and is non-streamed, the proxy buffers the
//! response to decode + validate it (logging the outcome). These tests assert
//! that path still forwards the body unchanged — both for a well-formed native
//! `tool_calls` response and for a response the decoder can't make sense of.
//! Correctness of decode/validate themselves is covered by their unit tests.

use std::net::SocketAddr;

use guardrail::{build_app, AppState};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn spawn_proxy(backend: &str) -> String {
    let state = AppState::new(reqwest::Client::new(), backend);
    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// A tool-enabled, non-streamed request body — this is the path that triggers
/// buffered inspection.
fn tool_request() -> serde_json::Value {
    serde_json::json!({
        "model": "local-model",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}}
        }]
    })
}

#[tokio::test]
async fn native_tool_call_response_is_forwarded_unchanged() {
    let backend = MockServer::start().await;
    let canned = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&canned))
        .mount(&backend)
        .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let got: serde_json::Value = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // Inspection logged but did not mutate the body.
    assert_eq!(got, canned);
}

#[tokio::test]
async fn undecodable_tool_response_is_still_forwarded() {
    let backend = MockServer::start().await;
    // Not a chat-completion shape the decoder understands; must pass through
    // unverified rather than error.
    let weird = "this is not json at all";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(weird.as_bytes(), "text/plain"))
        .mount(&backend)
        .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), weird);
}
