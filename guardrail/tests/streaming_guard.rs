//! Streaming requests that declare tools are guarded, not waved through. The
//! proxy forces the upstream call to buffered mode (so the whole response can be
//! inspected), runs the guardrail loop, and re-emits the guarded result to the
//! client as SSE — preserving the client's `stream: true` contract while still
//! repairing the tool call.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use guardrail::connector::Backend;
use guardrail::domain::metrics::{ModelStats, SqliteRecorder, Stats};
use guardrail::{build_app, AppState};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WmRequest, Respond, ResponseTemplate};

fn temp_db(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "guardrail-stream-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("metrics.sqlite")
}

/// A backend that records every request body it sees (so a test can assert the
/// proxy forced `stream: false` on the upstream call) and replies with a fixed
/// body.
struct Capture {
    seen: Arc<Mutex<Vec<Value>>>,
    response: Value,
}

impl Respond for Capture {
    fn respond(&self, req: &WmRequest) -> ResponseTemplate {
        if let Ok(v) = serde_json::from_slice::<Value>(&req.body) {
            self.seen.lock().unwrap().push(v);
        }
        ResponseTemplate::new(200).set_body_json(&self.response)
    }
}

async fn spawn_with_recorder(backend: &str, db: &Path) -> String {
    let recorder = Arc::new(SqliteRecorder::open(db).unwrap());
    let state =
        AppState::new(Backend::new(reqwest::Client::new()), backend).with_recorder(recorder);
    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// A streaming request carrying one real tool.
fn streaming_tool_request() -> Value {
    json!({
        "model": "local-model",
        "stream": true,
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}}
        }]
    })
}

/// Parse a minimal SSE body into (first chunk JSON, saw `[DONE]`).
fn parse_single_chunk_sse(body: &str) -> (Value, bool) {
    let mut chunk = None;
    let mut done = false;
    for event in body.split("\n\n") {
        let data = match event.trim().strip_prefix("data: ") {
            Some(d) => d,
            None => continue,
        };
        if data == "[DONE]" {
            done = true;
        } else if chunk.is_none() {
            chunk = Some(serde_json::from_str::<Value>(data).expect("chunk is JSON"));
        }
    }
    (chunk.expect("an SSE data chunk"), done)
}

fn wait_for_model_stats(db: &Path) -> ModelStats {
    for _ in 0..200 {
        if let Some(m) = Stats::read(db).unwrap().per_model.into_iter().next() {
            return m;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("no metrics were recorded");
}

#[tokio::test]
async fn streaming_native_tool_call_is_guarded_and_re_emitted_as_sse() {
    let backend = MockServer::start().await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let native = json!({
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
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Capture {
            seen: seen.clone(),
            response: native,
        })
        .mount(&backend)
        .await;

    let db = temp_db("native");
    let proxy = spawn_with_recorder(&backend.uri(), &db).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&streaming_tool_request())
        .send()
        .await
        .unwrap();

    // The client asked to stream, so it still gets an event stream back.
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let body = resp.text().await.unwrap();
    let (chunk, done) = parse_single_chunk_sse(&body);

    // Re-emitted as a streaming chunk with the message moved to `delta`.
    assert_eq!(chunk["object"], "chat.completion.chunk");
    assert_eq!(
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(chunk["choices"][0]["finish_reason"], "tool_calls");
    assert!(done, "stream terminates with [DONE]");

    // The upstream call is always sent with stream: true so text can be
    // forwarded live; the proxy assembles tool calls from the SSE chunks.
    let upstream = seen.lock().unwrap();
    assert_eq!(upstream.len(), 1);
    assert_eq!(upstream[0]["stream"], json!(true));

    // It is recorded as a guarded tool call, not a streamed passthrough.
    let m = wait_for_model_stats(&db);
    assert_eq!(m.tool_calls, 1);
    assert_eq!(m.by_outcome, vec![("native_valid".to_string(), 1)]);

    let _ = std::fs::remove_file(&db);
}

#[tokio::test]
async fn streaming_text_tool_call_is_rescued_and_re_emitted_as_sse() {
    let backend = MockServer::start().await;
    // A Qwen-style tool call buried in plain content — the kind of thing a raw
    // stream would have leaked through unrepaired.
    let content =
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
    let text = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&text))
        .mount(&backend)
        .await;

    let db = temp_db("rescue");
    let proxy = spawn_with_recorder(&backend.uri(), &db).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&streaming_tool_request())
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let (chunk, done) = parse_single_chunk_sse(&resp.text().await.unwrap());

    // The buried call is recovered and surfaced as a canonical tool call in the
    // streamed delta.
    let call = &chunk["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(call["function"]["arguments"], "{\"city\":\"Paris\"}");
    assert!(done);

    let m = wait_for_model_stats(&db);
    assert_eq!(m.by_outcome, vec![("rescued".to_string(), 1)]);

    let _ = std::fs::remove_file(&db);
}
