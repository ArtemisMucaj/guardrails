//! Token usage recorded end to end, through the real proxy.
//!
//! The unit tests cover extraction and aggregation in isolation. What these
//! check is the wiring: that usage reported by a backend survives the guardrail
//! loop and lands in the database, and — the case that motivates summing rather
//! than overwriting — that a request the guardrails retried is billed for every
//! attempt it made, not just the one that produced the answer.

use std::sync::Arc;

use guardrail::application::AppState;
use guardrail::connector::Backend;
use guardrail::domain::metrics::{SqliteRecorder, Stats};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn temp_db(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("guardrail-tokens-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{label}.sqlite"));
    let _ = std::fs::remove_file(&path);
    path
}

async fn spawn_proxy(state: AppState) -> String {
    let app = guardrail::build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// A tool the model is asked to call, so the request takes the guarded path.
fn weather_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    })
}

/// A non-streaming chat completion carrying a valid tool call and a usage block.
fn completion_with_usage(prompt: i64, completion: i64, cached: i64) -> String {
    serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    })
    .to_string()
}

async fn call_tool(proxy: &str) {
    reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [weather_tool()],
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
}

/// Read stats once the recorder has flushed. Dropping the proxy's `Arc` is not
/// enough on its own, so the read is retried briefly.
fn stats_for(db: &std::path::Path) -> Stats {
    for _ in 0..50 {
        let stats = Stats::read(db).unwrap();
        if stats.per_model.iter().any(|m| m.usage_requests > 0) {
            return stats;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    Stats::read(db).unwrap()
}

#[tokio::test]
async fn usage_reported_by_the_backend_reaches_the_database() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(completion_with_usage(120, 8, 90)),
        )
        .mount(&backend)
        .await;

    let db = temp_db("recorded");
    let recorder = Arc::new(SqliteRecorder::open(&db).unwrap());
    let proxy = spawn_proxy(
        AppState::new(Backend::new(reqwest::Client::new()), backend.uri())
            .with_recorder(recorder.clone()),
    )
    .await;

    call_tool(&proxy).await;
    drop(recorder);

    let stats = stats_for(&db);
    let m = stats
        .per_model
        .iter()
        .find(|m| m.usage_requests > 0)
        .expect("a row carrying usage");
    assert_eq!(m.usage.prompt_tokens, 120);
    assert_eq!(m.usage.completion_tokens, 8);
    assert_eq!(m.usage.cached_tokens, 90);
    assert_eq!(m.billed_tokens(), 128);
    assert_eq!(m.usage.uncached_prompt_tokens(), 30);
    assert_eq!(m.cache_hit_rate(), Some(0.75));
    // One backend call for one client request: nothing retried.
    assert_eq!(m.usage.attempts, 1);
    assert_eq!(m.calls_per_request(), Some(1.0));
}

#[tokio::test]
async fn a_retried_request_is_billed_for_every_attempt() {
    // The backend first returns a call missing the required `city`, which the
    // guardrails reject and retry; the second attempt is valid. Both were
    // billed, and the recorded row must say so — reporting only the successful
    // attempt would hide what the guardrails cost.
    let backend = MockServer::start().await;
    let invalid = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    })
    .to_string();

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(invalid))
        .up_to_n_times(1)
        .mount(&backend)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(completion_with_usage(200, 7, 60)),
        )
        .mount(&backend)
        .await;

    let db = temp_db("retried");
    let recorder = Arc::new(SqliteRecorder::open(&db).unwrap());
    let proxy = spawn_proxy(
        AppState::new(Backend::new(reqwest::Client::new()), backend.uri())
            .with_recorder(recorder.clone()),
    )
    .await;

    call_tool(&proxy).await;
    drop(recorder);

    let stats = stats_for(&db);
    let m = stats
        .per_model
        .iter()
        .find(|m| m.usage_requests > 0)
        .expect("a row carrying usage");

    // Both attempts, summed: 100+200 prompt, 5+7 completion, 40+60 cached.
    assert_eq!(m.usage.prompt_tokens, 300);
    assert_eq!(m.usage.completion_tokens, 12);
    assert_eq!(m.usage.cached_tokens, 100);
    assert_eq!(m.billed_tokens(), 312);
    // Two backend calls for a single client request — the retry multiplier the
    // report exists to make visible.
    assert_eq!(m.usage.attempts, 2);
    assert_eq!(m.usage_requests, 1);
    assert_eq!(m.calls_per_request(), Some(2.0));
}

/// A Responses SSE stream ending in a `response.completed` carrying `id` and a
/// usage block — the two fields conversation grouping is built from.
fn responses_sse(id: &str, prompt: i64, completion: i64) -> String {
    let completed = serde_json::json!({
        "type": "response.completed",
        "response": {
            "id": id,
            "model": "m",
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}",
            }],
            "usage": {
                "input_tokens": prompt,
                "output_tokens": completion,
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    });
    format!("data: {completed}\n\ndata: [DONE]\n\n")
}

#[tokio::test]
async fn a_chained_responses_conversation_counts_its_resent_prefix_once() {
    // The end-to-end case for fix 2. Two turns of one conversation, the second
    // naming the first via `previous_response_id`. The proxy must record the
    // chain so the report can charge the shared prefix once.
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            responses_sse("resp_1", 100, 10),
            "text/event-stream",
        ))
        .up_to_n_times(1)
        .mount(&backend)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            responses_sse("resp_2", 400, 20),
            "text/event-stream",
        ))
        .mount(&backend)
        .await;

    let db = temp_db("chained");
    let recorder = Arc::new(SqliteRecorder::open(&db).unwrap());
    let proxy = spawn_proxy(
        AppState::new(Backend::new(reqwest::Client::new()), backend.uri())
            .with_recorder(recorder.clone()),
    )
    .await;

    let tool = serde_json::json!({
        "type": "function", "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    });
    let client = reqwest::Client::new();

    // Turn 1: no parent, opens the conversation.
    client
        .post(format!("{proxy}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "weather in Paris?", "tools": [tool],
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    // Turn 2: continues turn 1 by naming it, the way a stateful client does.
    client
        .post(format!("{proxy}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "and tomorrow?", "tools": [tool],
            "previous_response_id": "resp_1",
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    drop(recorder);

    let stats = stats_for(&db);
    let m = stats
        .per_model
        .iter()
        .find(|m| m.usage_requests > 0)
        .expect("a row carrying usage");

    // Billed: both prompts, as charged.
    assert_eq!(m.usage.prompt_tokens, 500);
    assert_eq!(m.billed_tokens(), 530);
    // Deduplicated: turn 2's prompt already contains turn 1's.
    assert_eq!(m.distinct_prompt_tokens, Some(400));
    assert_eq!(m.distinct_tokens(), Some(430));
    assert_eq!(m.conversations, Some(1), "two turns, one conversation");
}

#[tokio::test]
async fn a_backend_reporting_no_usage_records_the_request_without_tokens() {
    // Most local backends report no usage. The request must still be counted as
    // an outcome, but must not enter the token totals as a zero.
    let backend = MockServer::start().await;
    let no_usage = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    })
    .to_string();

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(no_usage))
        .mount(&backend)
        .await;

    let db = temp_db("nousage");
    let recorder = Arc::new(SqliteRecorder::open(&db).unwrap());
    let proxy = spawn_proxy(
        AppState::new(Backend::new(reqwest::Client::new()), backend.uri())
            .with_recorder(recorder.clone()),
    )
    .await;

    call_tool(&proxy).await;
    drop(recorder);

    // Wait for the outcome row rather than for usage, which never arrives here.
    let mut stats = Stats::read(&db).unwrap();
    for _ in 0..50 {
        if !stats.per_model.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        stats = Stats::read(&db).unwrap();
    }

    let m = &stats.per_model[0];
    assert_eq!(m.total, 1, "the request is still recorded");
    assert_eq!(m.usage_requests, 0, "but it carries no token measurement");
    assert_eq!(m.billed_tokens(), 0);
    assert_eq!(m.calls_per_request(), None, "no basis to report a multiplier");
}
