//! The repetition guard cuts off degenerate model loops end-to-end.
//!
//! When a guarded (tool-enabled) request comes back with text that has fallen
//! into a repeating loop, the proxy stops the runaway: it delivers the good
//! prefix plus one clean copy of the repeated unit and records the outcome as
//! `repetition_detected`, instead of forwarding thousands of repeated lines.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use guardrail::connector::Backend;
use guardrail::domain::metrics::{ModelStats, SqliteRecorder, Stats};
use guardrail::{build_app, AppState};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn temp_db(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "guardrail-rep-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("metrics.sqlite")
}

async fn spawn(backend: &str, db: &Path) -> String {
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

/// A tool-enabled request — the path that runs the guardrail loop.
fn tool_request(stream: bool) -> Value {
    json!({
        "model": "local-model",
        "stream": stream,
        "messages": [{"role": "user", "content": "explain something"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}}
        }]
    })
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
async fn buffered_loop_is_truncated_and_recorded() {
    let backend = MockServer::start().await;
    // A JSON (buffered) backend whose answer degenerates into a loop.
    let loop_text = format!("Sure, here is the plan. {}", "I will help you now. ".repeat(30));
    let body = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": loop_text}
        }]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .mount(&backend)
        .await;

    let db = temp_db("buffered");
    let proxy = spawn(&backend.uri(), &db).await;

    let got: Value = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request(false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let content = got["choices"][0]["message"]["content"]
        .as_str()
        .expect("a text answer");
    // The good prefix survives, but the runaway is collapsed to a single copy.
    assert!(content.starts_with("Sure, here is the plan."));
    assert_eq!(
        content.matches("I will help you now.").count(),
        1,
        "the repeated line should be collapsed to one copy, got: {content:?}"
    );

    let m = wait_for_model_stats(&db);
    assert_eq!(m.by_outcome, vec![("repetition_detected".to_string(), 1)]);
    // A loop is not a tool call, so it does not touch the tool-call success rate.
    assert_eq!(m.tool_calls, 0);

    let _ = std::fs::remove_file(&db);
}

#[tokio::test]
async fn streaming_loop_is_cut_off_mid_stream() {
    let backend = MockServer::start().await;
    // A native SSE stream that repeats the same line far more times than the
    // detector's threshold — the proxy should stop forwarding well before the end.
    let copies = 60;
    let mut sse = String::new();
    for _ in 0..copies {
        let chunk = json!({
            "id": "c1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "stuck in a loop forever. "}, "finish_reason": null}]
        });
        sse.push_str(&format!("data: {chunk}\n\n"));
    }
    sse.push_str("data: [DONE]\n\n");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse.into_bytes(), "text/event-stream"))
        .mount(&backend)
        .await;

    let db = temp_db("streaming");
    let proxy = spawn(&backend.uri(), &db).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request(true))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.headers().get("content-type").unwrap(), "text/event-stream");
    let body = resp.text().await.unwrap();

    // The client saw the loop start but not all 60 copies — the tail was cut off.
    let seen = body.matches("stuck in a loop forever.").count();
    assert!(seen >= 2, "the loop should stream until detected, saw {seen}");
    assert!(seen < copies, "the runaway tail must be cut off, saw {seen} of {copies}");

    let m = wait_for_model_stats(&db);
    assert_eq!(m.by_outcome, vec![("repetition_detected".to_string(), 1)]);

    let _ = std::fs::remove_file(&db);
}

#[tokio::test]
async fn detector_can_be_disabled() {
    let backend = MockServer::start().await;
    let loop_text = format!("ok. {}", "repeat me. ".repeat(40));
    let body = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": loop_text}}]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .mount(&backend)
        .await;

    let db = temp_db("disabled");
    // Build the app with repetition detection turned off.
    let recorder = Arc::new(SqliteRecorder::open(&db).unwrap());
    let mut guardrails = guardrail::Guardrails::default();
    guardrails.repetition.min_repeats = 0;
    let state = AppState::new(Backend::new(reqwest::Client::new()), backend.uri())
        .with_guardrails(guardrails)
        .with_recorder(recorder);
    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let proxy = format!("http://{addr}");

    let _got: Value = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request(false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // With the detector off, the loop is treated as an ordinary text passthrough.
    let m = wait_for_model_stats(&db);
    assert_eq!(m.by_outcome, vec![("passthrough_no_calls".to_string(), 1)]);

    let _ = std::fs::remove_file(&db);
}
