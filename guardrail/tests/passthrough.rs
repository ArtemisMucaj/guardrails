//! Milestone 1 acceptance: the proxy is byte-for-byte transparent.
//!
//! These tests stand up a `wiremock` backend, point the proxy at it, and assert
//! that status, headers, and body round-trip unchanged for the endpoints we
//! care about — including a streamed SSE body and a verbatim-forwarded request
//! body. This is the suite that guarantees "behaves identically through the
//! proxy as direct to the backend" before any guardrail logic exists.

use std::net::SocketAddr;

use guardrail::{build_app, AppState};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WmRequest, Respond, ResponseTemplate};

/// Spawn the proxy on an ephemeral port pointing at `backend`; return its base URL.
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

#[tokio::test]
async fn chat_completions_body_and_headers_round_trip() {
    let backend = MockServer::start().await;

    let canned = serde_json::json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-backend-marker", "lmstudio")
                .set_body_json(&canned),
        )
        .mount(&backend)
        .await;

    let proxy = spawn_proxy(&backend.uri()).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "local-model", "messages": []}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    // Backend response headers are preserved across the hop.
    assert_eq!(resp.headers().get("x-backend-marker").unwrap(), "lmstudio");
    let got: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(got, canned);
}

/// Capture the request the backend received so we can assert the proxy forwarded
/// it verbatim (model field never rewritten, body preserved).
struct Echo;
impl Respond for Echo {
    fn respond(&self, req: &WmRequest) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_bytes(req.body.clone())
    }
}

#[tokio::test]
async fn request_body_is_forwarded_verbatim() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Echo)
        .mount(&backend)
        .await;

    let proxy = spawn_proxy(&backend.uri()).await;

    // Raw bytes (specific whitespace + key order) so the assertion proves
    // byte-for-byte forwarding, not just semantic JSON equality.
    let body = br#"{"model":"some/exact-model-id","messages":[{"role":"user","content":"hello"}],"temperature":0.7}"#;

    let echoed = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(body.to_vec())
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    // Request bytes (model id, sampling params, exact formatting) round-trip untouched.
    assert_eq!(echoed.as_ref(), body.as_slice());
}

#[tokio::test]
async fn models_endpoint_passthrough() {
    let backend = MockServer::start().await;
    let models = serde_json::json!({
        "object": "list",
        "data": [{"id": "local-model", "object": "model"}]
    });
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&models))
        .mount(&backend)
        .await;

    let proxy = spawn_proxy(&backend.uri()).await;

    let got: serde_json::Value = reqwest::Client::new()
        .get(format!("{proxy}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(got, models);
}

#[tokio::test]
async fn streaming_sse_body_passthrough() {
    let backend = MockServer::start().await;
    // A minimal OpenAI-style SSE stream. We assert the raw bytes survive the hop
    // unchanged, which is what fake-stream / passthrough streaming relies on.
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
               data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse.as_bytes(), "text/event-stream"))
        .mount(&backend)
        .await;

    let proxy = spawn_proxy(&backend.uri()).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "m", "messages": [], "stream": true}))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    assert_eq!(resp.text().await.unwrap(), sse);
}
