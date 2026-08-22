//! A stream that dies mid-answer must not be served as a finished one.
//!
//! The backend crashing, a proxy timing out, or a network cut all leave the
//! same trace: bytes stop arriving. Treating that as end-of-stream hands the
//! client half a sentence — or half a tool call — with a clean terminator and
//! nothing to distinguish it from a complete answer.

use std::net::SocketAddr;

use guardrail::connector::Backend;
use guardrail::{build_app, AppState};
use serde_json::{json, Value};

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

/// A backend that writes `frames` and then drops the connection without the
/// terminating zero-length chunk — the shape of an upstream that died.
async fn dies_after(frames: Vec<String>) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let frames = frames.clone();
            tokio::spawn(async move {
                let mut scratch = [0u8; 8192];
                let _ = socket.read(&mut scratch).await;
                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                          Transfer-Encoding: chunked\r\n\r\n",
                    )
                    .await
                    .unwrap();
                for frame in frames {
                    socket
                        .write_all(format!("{:x}\r\n", frame.len()).as_bytes())
                        .await
                        .unwrap();
                    socket.write_all(frame.as_bytes()).await.unwrap();
                    socket.write_all(b"\r\n").await.unwrap();
                    socket.flush().await.unwrap();
                }
                // No terminating chunk: the connection just goes away.
                drop(socket);
            });
        }
    });
    format!("http://{addr}")
}

fn tool_request() -> Value {
    json!({
        "model": "m", "stream": true,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]
            }
        }}]
    })
}

/// Every `data:` frame the client received, parsed.
fn frames(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(str::trim)
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str(d).ok())
        .collect()
}

#[tokio::test]
async fn a_half_written_answer_is_reported_rather_than_served_as_whole() {
    let delta = json!({"choices":[{"index":0,"delta":{"content":"The answer is "}}]});
    let backend = dies_after(vec![format!("data: {delta}\n\n")]).await;
    let proxy = spawn(&backend).await;

    let body = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let error = frames(&body)
        .into_iter()
        .find(|f| f.get("error").is_some())
        .unwrap_or_else(|| panic!("a cut stream must say so: {body}"));
    assert_eq!(error["error"]["type"], "upstream_error");
    assert_eq!(error["error"]["code"], 502);
}

#[tokio::test]
async fn a_tool_call_cut_mid_arguments_is_not_validated_as_if_complete() {
    // The dangerous shape: arguments stop partway, so the JSON is incomplete.
    // Guarding this would spend the retry budget on a truncation the model never
    // caused — and a repair that "fixed" the JSON would invent a call the model
    // never finished asking for.
    let partial = json!({"choices":[{"index":0,"delta":{"tool_calls":[{
        "index":0,"id":"call_1","type":"function",
        "function":{"name":"write_file","arguments":"{\"path\":\"/etc/pas"}
    }]}}]});
    let backend = dies_after(vec![format!("data: {partial}\n\n")]).await;
    let proxy = spawn(&backend).await;

    let body = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let received = frames(&body);
    assert!(
        received.iter().any(|f| f.get("error").is_some()),
        "the truncation must be reported: {body}"
    );
    assert!(
        !body.contains("/etc/pas"),
        "a half-written call must not reach the client: {body}"
    );
}

#[tokio::test]
async fn a_complete_stream_is_still_served_normally() {
    // The guard must not fire on a healthy stream: this is the regression that
    // would make every answer look truncated.
    let delta = json!({"choices":[{"index":0,"delta":{"content":"all good"}}]});
    let finish = json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
    let server = wiremock::MockServer::start().await;
    let sse = format!("data: {delta}\n\ndata: {finish}\n\ndata: [DONE]\n\n");
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_raw(sse.into_bytes(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let proxy = spawn(&server.uri()).await;
    let body = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert!(
        !frames(&body).iter().any(|f| f.get("error").is_some()),
        "a healthy stream must not be reported as truncated: {body}"
    );
    assert!(body.contains("all good"), "the answer must arrive: {body}");
}
