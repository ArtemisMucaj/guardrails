//! How the proxy behaves when the transport misbehaves: a backend that splits a
//! multi-byte character across network chunks, and a backend that fails.
//!
//! Both used to be absorbed silently — the first truncated the stream, the
//! second was relayed as a `200 OK` carrying an empty answer — so each case is
//! pinned here rather than left to the assembler's tolerance for odd input.

use std::net::SocketAddr;

use guardrail::connector::Backend;
use guardrail::{build_app, AppState};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn spawn(backend: &str) -> String {
    let state = AppState::new(Backend::new(reqwest::Client::new()), backend);
    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// A streaming request that declares a tool, so it runs the guardrail loop.
fn tool_request() -> Value {
    json!({
        "model": "local-model",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]
    })
}

/// Text deltas whose content is non-ASCII, so the SSE bytes contain multi-byte
/// UTF-8 characters that a chunk boundary can land inside.
fn accented_sse() -> String {
    let mut sse = String::new();
    for piece in ["Voilà ", "une réponse ", "très accentuée — 日本語 ", "🎉"] {
        let chunk = json!({
            "id": "c1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]
        });
        sse.push_str(&format!("data: {chunk}\n\n"));
    }
    let finish = json!({
        "id": "c1", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    sse.push_str(&format!("data: {finish}\n\ndata: [DONE]\n\n"));
    sse
}

/// The client's view of a streamed answer: every `content` delta concatenated.
fn streamed_text(body: &str) -> String {
    let mut text = String::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(piece) = chunk
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            text.push_str(piece);
        }
    }
    text
}

/// A backend that writes a chunked SSE body in two deliberate TCP writes,
/// splitting at `split`. wiremock buffers its body into a single write, which
/// would never reproduce a boundary landing inside a character, so this speaks
/// HTTP/1.1 directly.
async fn split_writing_backend(body: Vec<u8>, split: usize) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let body = body.clone();
            tokio::spawn(async move {
                // Read past the request headers; the body is irrelevant here.
                let mut scratch = [0u8; 8192];
                let _ = socket.read(&mut scratch).await;

                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\n\
                          Content-Type: text/event-stream\r\n\
                          Transfer-Encoding: chunked\r\n\r\n",
                    )
                    .await
                    .unwrap();

                // Two chunks, split mid-character.
                for part in [&body[..split], &body[split..]] {
                    socket
                        .write_all(format!("{:x}\r\n", part.len()).as_bytes())
                        .await
                        .unwrap();
                    socket.write_all(part).await.unwrap();
                    socket.write_all(b"\r\n").await.unwrap();
                    socket.flush().await.unwrap();
                    // Ensure the halves land in separate reads downstream.
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
                socket.write_all(b"0\r\n\r\n").await.unwrap();
                let _ = socket.flush().await;
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn a_multibyte_character_split_across_chunks_does_not_truncate_the_stream() {
    // The backend delivers its SSE with a chunk boundary inside a multi-byte
    // character. Decoding each network chunk on its own fails on both halves,
    // which is what used to end the stream early and silently.
    let bytes = accented_sse().into_bytes();
    // Split inside the first multi-byte character (the `à` of "Voilà").
    let split = bytes
        .iter()
        .position(|&b| b == 0xC3)
        .expect("the fixture must contain a multi-byte character")
        + 1;
    assert!(
        std::str::from_utf8(&bytes[..split]).is_err(),
        "the split must actually fall inside a character"
    );

    let backend = split_writing_backend(bytes, split).await;
    let proxy = spawn(&backend).await;
    let response = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();
    assert_eq!(
        streamed_text(&body),
        "Voilà une réponse très accentuée — 日本語 🎉",
        "every accented delta must survive the hop"
    );
}

#[tokio::test]
async fn a_backend_error_is_not_reported_to_the_client_as_success() {
    // A 429 whose body is JSON carries no `choices`, so assembling it used to
    // yield an empty text answer and a `200 OK` — the failure was invisible to
    // the client and to any backoff it would otherwise apply.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {
                "message": "rate limit exceeded",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded"
            }
        })))
        .mount(&server)
        .await;

    let proxy = spawn(&server.uri()).await;

    // A non-streaming client gets the upstream status back verbatim.
    let mut buffered = tool_request();
    buffered["stream"] = json!(false);
    let response = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&buffered)
        .send()
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        429,
        "the upstream status must reach the client"
    );
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "rate_limit_exceeded");
}

#[tokio::test]
async fn a_streaming_client_is_told_in_band_when_the_backend_fails() {
    // The SSE headers are already out by the time the backend answers, so the
    // status can no longer be changed. The failure is relayed as an `error`
    // frame instead of an empty, successful-looking stream.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({
            "error": {"message": "internal", "type": "server_error"}
        })))
        .mount(&server)
        .await;

    let proxy = spawn(&server.uri()).await;
    let response = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap();

    let body = response.text().await.unwrap();
    let error = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(str::trim)
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .find(|v| v.get("error").is_some())
        .expect("the stream must carry an error frame");

    assert_eq!(error["error"]["type"], "upstream_error");
    assert_eq!(error["error"]["code"], 500);
    assert!(
        streamed_text(&body).is_empty(),
        "a failed call must not look like a text answer"
    );
}

/// The `error` frame a streaming client received, if any.
fn error_frame(body: &str) -> Option<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(str::trim)
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .find(|v| v.get("error").is_some())
}

async fn streamed_against(server: &MockServer) -> String {
    let proxy = spawn(&server.uri()).await;
    reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap()
}

#[tokio::test]
async fn a_success_status_with_an_unreadable_body_is_still_reported() {
    // The status says the call worked, but the body is neither JSON nor SSE, so
    // nothing guardable arrived. A streaming client is already committed to an
    // event stream and cannot be handed the verbatim body, so without a frame
    // of its own it would see a bare `[DONE]` and read it as an empty answer.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw("not json, not sse".as_bytes(), "text/plain"),
        )
        .mount(&server)
        .await;

    let body = streamed_against(&server).await;
    let error = error_frame(&body)
        .unwrap_or_else(|| panic!("an unreadable 200 must still be described: {body}"));
    assert_eq!(error["error"]["type"], "upstream_error");
    assert_eq!(
        error["error"]["code"], 502,
        "a 2xx has no failing status of its own, so the proxy names one"
    );
    assert!(
        streamed_text(&body).is_empty(),
        "nothing readable arrived, so there is no answer to show: {body}"
    );
}

#[tokio::test]
async fn a_failing_status_with_an_sse_body_is_reported_rather_than_streamed() {
    // A native `text/event-stream` body on a non-success status used to take the
    // success path, so the failure streamed as though the turn had worked.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_raw(
            "data: {\"error\":\"overloaded\"}\n\n".as_bytes(),
            "text/event-stream",
        ))
        .mount(&server)
        .await;

    let body = streamed_against(&server).await;
    let error = error_frame(&body)
        .unwrap_or_else(|| panic!("a 503 must be described even as SSE: {body}"));
    assert_eq!(error["error"]["code"], 503, "the upstream status is kept");
}

#[tokio::test]
async fn a_failing_status_with_a_json_body_is_reported_rather_than_assembled() {
    // The other half: a JSON error body on a non-success status. Assembling it
    // would find no `choices` and yield an empty text answer.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"error": {"message": "boom"}})),
        )
        .mount(&server)
        .await;

    let body = streamed_against(&server).await;
    let error = error_frame(&body).unwrap_or_else(|| panic!("a 500 must be described: {body}"));
    assert_eq!(error["error"]["code"], 500);
}
