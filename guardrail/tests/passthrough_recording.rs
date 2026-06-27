//! Streaming and non-tool chat requests are forwarded unchanged but still
//! recorded, so `stats` reflects all chat traffic and is not mysteriously empty
//! for clients that only ever stream. The body must still pass through
//! untouched — recording happens alongside forwarding, not instead of it.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use guardrail::connector::Backend;
use guardrail::domain::metrics::{ModelStats, SqliteRecorder, Stats};
use guardrail::{build_app, AppState};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn temp_db(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "guardrail-pass-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("metrics.sqlite")
}

/// Spawn the proxy with a SQLite recorder installed; return its base URL.
async fn spawn_proxy_with_recorder(backend: &str, db: &Path) -> String {
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

/// Read the single model's stats, retrying briefly while the background writer
/// drains the row (the recorder writes off the request path).
fn wait_for_model_stats(db: &Path) -> ModelStats {
    for _ in 0..200 {
        let stats = Stats::read(db).unwrap();
        if let Some(m) = stats.per_model.into_iter().next() {
            return m;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("no metrics were recorded");
}

#[tokio::test]
async fn streaming_tool_request_is_recorded_as_streamed_passthrough() {
    let backend = MockServer::start().await;
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
               data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse.as_bytes(), "text/event-stream"))
        .mount(&backend)
        .await;

    let db = temp_db("stream");
    let proxy = spawn_proxy_with_recorder(&backend.uri(), &db).await;

    // A streaming request that *does* carry tools: the proxy can't guard a
    // stream, so it forwards unchanged and records the passthrough.
    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [],
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "get_weather"}}]
        }))
        .send()
        .await
        .unwrap();
    // Forwarding is untouched: the SSE body survives the hop verbatim.
    assert_eq!(resp.text().await.unwrap(), sse);

    let m = wait_for_model_stats(&db);
    assert_eq!(m.model, "m");
    assert_eq!(m.total, 1);
    assert_eq!(
        m.tool_calls, 0,
        "a streamed call is not a guarded tool call"
    );
    assert_eq!(m.errors, 0);
    assert_eq!(m.success_rate(), None);
    assert_eq!(m.by_outcome, vec![("streamed_passthrough".to_string(), 1)]);

    let _ = std::fs::remove_file(&db);
}

#[tokio::test]
async fn non_tool_request_is_recorded_as_non_tool_passthrough() {
    let backend = MockServer::start().await;
    let canned = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&canned))
        .mount(&backend)
        .await;

    let db = temp_db("notool");
    let proxy = spawn_proxy_with_recorder(&backend.uri(), &db).await;

    // No tools and not streaming: nothing to guard, but it is recorded.
    let got: serde_json::Value = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "m", "messages": []}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(got, canned, "non-tool response forwarded unchanged");

    let m = wait_for_model_stats(&db);
    assert_eq!(m.total, 1);
    assert_eq!(m.tool_calls, 0);
    assert_eq!(m.by_outcome, vec![("non_tool_passthrough".to_string(), 1)]);

    let _ = std::fs::remove_file(&db);
}
