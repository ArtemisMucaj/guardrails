//! A backend that did not answer gets asked again.
//!
//! Distinct from the guardrails' own retries, which answer a model that
//! produced a bad tool call. These answer a backend that produced nothing at
//! all — a rate limit, a restart, a refused connection — which the guardrail
//! loop cannot act on because there is no response to guard.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use guardrail::connector::Backend;
use guardrail::{build_app, AppState};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WmRequest, Respond, ResponseTemplate};

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

fn tool_request() -> Value {
    json!({
        "model": "m", "stream": false,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }}]
    })
}

fn good_answer() -> Value {
    json!({
        "id": "c1", "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "sunny"},
            "finish_reason": "stop"
        }]
    })
}

/// Fails with `status` for the first `fail_times` calls, then succeeds.
/// Counts every call so the test can assert how many attempts were made.
struct FlakyBackend {
    calls: Arc<AtomicUsize>,
    fail_times: usize,
    status: u16,
    retry_after: Option<&'static str>,
}

impl Respond for FlakyBackend {
    fn respond(&self, _: &WmRequest) -> ResponseTemplate {
        let seen = self.calls.fetch_add(1, Ordering::SeqCst);
        if seen < self.fail_times {
            let mut template = ResponseTemplate::new(self.status);
            if let Some(value) = self.retry_after {
                template = template.append_header("retry-after", value);
            }
            return template.set_body_json(json!({"error": {"message": "not now"}}));
        }
        ResponseTemplate::new(200).set_body_json(good_answer())
    }
}

async fn flaky(
    fail_times: usize,
    status: u16,
    retry_after: Option<&'static str>,
) -> (MockServer, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(FlakyBackend {
            calls: calls.clone(),
            fail_times,
            status,
            retry_after,
        })
        .mount(&server)
        .await;
    (server, calls)
}

async fn ask(proxy: &str) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn a_rate_limit_is_absorbed_rather_than_shown_to_the_client() {
    let (server, calls) = flaky(1, 429, None).await;
    let proxy = spawn(&server.uri()).await;

    let response = ask(&proxy).await;
    assert_eq!(response.status(), 200, "the retry should have succeeded");
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "sunny");
    assert_eq!(calls.load(Ordering::SeqCst), 2, "one failure, one retry");
}

#[tokio::test]
async fn a_restarting_backend_is_retried() {
    // 503 is the shape of an instance coming back up.
    let (server, calls) = flaky(2, 503, None).await;
    let proxy = spawn(&server.uri()).await;

    assert_eq!(ask(&proxy).await.status(), 200);
    assert_eq!(calls.load(Ordering::SeqCst), 3, "two failures, then success");
}

#[tokio::test]
async fn a_backend_that_stays_down_surfaces_its_own_status() {
    // Retries are bounded: a backend that is properly down must become an
    // error the client can act on, not an unbounded stall.
    let (server, calls) = flaky(usize::MAX, 503, None).await;
    let proxy = spawn(&server.uri()).await;

    assert_eq!(ask(&proxy).await.status(), 503, "the real status must arrive");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "the initial attempt plus MAX_TRANSPORT_RETRIES"
    );
}

#[tokio::test]
async fn a_bad_request_is_not_retried() {
    // Every 4xx that is not 408 or 429 is the request being wrong. Resending it
    // unchanged would fail identically and only delay the error.
    let (server, calls) = flaky(usize::MAX, 400, None).await;
    let proxy = spawn(&server.uri()).await;

    assert_eq!(ask(&proxy).await.status(), 400);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "asked exactly once");
}

#[tokio::test]
async fn a_context_length_error_is_not_retried() {
    // The specific 400 that matters here: resending the same oversized prompt
    // burns the budget to arrive at the same answer.
    let (server, calls) = flaky(usize::MAX, 413, None).await;
    let proxy = spawn(&server.uri()).await;

    assert_eq!(ask(&proxy).await.status(), 413);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_retry_after_header_is_honoured() {
    // The backend names a delay; the proxy waits at least that long rather than
    // hammering it on its own schedule.
    let (server, calls) = flaky(1, 429, Some("1")).await;
    let proxy = spawn(&server.uri()).await;

    let started = std::time::Instant::now();
    assert_eq!(ask(&proxy).await.status(), 200);
    let waited = started.elapsed();

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        waited >= std::time::Duration::from_millis(900),
        "should have waited the requested second, waited {waited:?}"
    );
}

#[tokio::test]
async fn an_unreachable_backend_becomes_a_gateway_error() {
    // Nothing is listening: every attempt fails to connect, and the client gets
    // a 502 rather than a hang.
    let proxy = spawn("http://127.0.0.1:1").await;
    assert_eq!(ask(&proxy).await.status(), 502);
}
