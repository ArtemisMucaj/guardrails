//! Work stops when the client stops listening.
//!
//! The guardrail loop retries, so an abandoned request is not one wasted
//! inference but up to `max_retries + 1` of them — every one billed, every one
//! producing output that goes nowhere. The loop checks for a listener before
//! each attempt rather than only the first, because a client typically hangs up
//! while an earlier attempt is already in flight.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use guardrail::connector::Backend;
use guardrail::domain::metrics::{SqliteRecorder, Stats};
use guardrail::{build_app, AppState};
use serde_json::{json, Value};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request as WmRequest, Respond, ResponseTemplate};

fn temp_db(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "guardrail-disconnect-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("metrics.sqlite")
}

/// Answers slowly, with a tool call naming a tool that was never declared, so
/// the guardrails always want another attempt. Counts every call it receives.
struct SlowInvalidCall {
    calls: Arc<AtomicUsize>,
    delay: std::time::Duration,
}

impl Respond for SlowInvalidCall {
    fn respond(&self, _: &WmRequest) -> ResponseTemplate {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ResponseTemplate::new(200)
            .set_delay(self.delay)
            .set_body_json(json!({
                "id": "c1", "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant", "content": null,
                        "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "undeclared_tool", "arguments": "{}"}}]
                    },
                    "finish_reason": "tool_calls"
                }]
            }))
    }
}

async fn backend(calls: Arc<AtomicUsize>, delay_ms: u64) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(SlowInvalidCall {
            calls,
            delay: std::time::Duration::from_millis(delay_ms),
        })
        .mount(&server)
        .await;
    server
}

async fn spawn(backend: &str, db: Option<&PathBuf>) -> String {
    let mut state = AppState::new(Backend::new(reqwest::Client::new()), backend);
    if let Some(db) = db {
        state = state.with_recorder(Arc::new(SqliteRecorder::open(db).unwrap()));
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, build_app(state)).await.unwrap();
    });
    format!("http://{addr}")
}

fn tool_request(stream: bool) -> Value {
    json!({
        "model": "m", "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "declared_tool",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }}]
    })
}

/// Start a request, abandon it while the first attempt is still in flight, and
/// give the proxy room to keep working if it is going to.
async fn abandon(proxy: &str, stream: bool) {
    let url = format!("{proxy}/v1/chat/completions");
    let body = tool_request(stream);
    let request = tokio::spawn(async move {
        let _ = reqwest::Client::new().post(&url).json(&body).send().await;
    });
    // Long enough for the first attempt to be in flight, short enough that it
    // has not finished.
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    request.abort();
    // Generous: the unfixed loop spends two further attempts in this window.
    tokio::time::sleep(std::time::Duration::from_millis(1_500)).await;
}

#[tokio::test]
async fn a_streaming_client_that_hangs_up_does_not_pay_for_retries() {
    let calls = Arc::new(AtomicUsize::new(0));
    let server = backend(calls.clone(), 200).await;
    let proxy = spawn(&server.uri(), None).await;

    abandon(&proxy, true).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "only the in-flight attempt should be billed, not the retries after it"
    );
}

#[tokio::test]
async fn a_buffered_client_that_hangs_up_does_not_pay_for_retries() {
    // The buffered path parks in `drain_rx` inside the handler future, which
    // axum drops on disconnect — closing the channel the same way dropping the
    // response body does for a streaming client.
    let calls = Arc::new(AtomicUsize::new(0));
    let server = backend(calls.clone(), 200).await;
    let proxy = spawn(&server.uri(), None).await;

    abandon(&proxy, false).await;

    assert_eq!(calls.load(Ordering::SeqCst), 1, "same for a buffered client");
}

#[tokio::test]
async fn abandoning_is_recorded_so_its_cost_stays_visible() {
    // The attempts already made were billed. Dropping the request silently
    // would hide that spend from the very report that exists to surface it.
    let db = temp_db("recorded");
    let _ = std::fs::remove_file(&db);
    let calls = Arc::new(AtomicUsize::new(0));
    let server = backend(calls.clone(), 200).await;
    let proxy = spawn(&server.uri(), Some(&db)).await;

    abandon(&proxy, true).await;

    let stats = Stats::read(&db).unwrap();
    let tags: Vec<&str> = stats
        .per_model
        .iter()
        .flat_map(|m| m.by_outcome.iter().map(|(tag, _)| tag.as_str()))
        .collect();
    assert!(
        tags.contains(&"client_disconnected"),
        "the abandonment must be recorded: {tags:?}"
    );

    let _ = std::fs::remove_dir_all(db.parent().unwrap());
}

#[tokio::test]
async fn a_client_that_waits_still_gets_the_full_retry_budget() {
    // The regression that matters: the check must not cut short a request
    // nobody abandoned. A client that waits sees every attempt spent.
    let calls = Arc::new(AtomicUsize::new(0));
    let server = backend(calls.clone(), 0).await;
    let proxy = spawn(&server.uri(), None).await;

    let response = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request(false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "the initial attempt plus the default two corrective retries"
    );
}
