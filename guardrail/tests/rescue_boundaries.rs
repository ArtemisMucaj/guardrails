//! Where rescue stops.
//!
//! Rescue turns call syntax the model wrote as text into a real tool call. A
//! fenced JSON block is the one rescue form that is also ordinary Markdown, so
//! a model *explaining* an API emits exactly what a model *calling* one does —
//! and by the time validation sees it, a documented example names a real tool
//! with real arguments and is indistinguishable from a genuine call. These
//! tests pin the boundary: the block must be the answer, not an illustration
//! inside one, and whatever the model wrote around it survives.

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

fn tool_request() -> Value {
    json!({
        "model": "m",
        "messages": [{"role": "user", "content": "how do I get the weather?"}],
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

/// Ask the proxy to guard a response whose assistant text is `answer`.
async fn guarded(answer: &str) -> Value {
    let body = json!({
        "id": "c1", "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }]
    });
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .mount(&server)
        .await;

    let proxy = spawn(&server.uri()).await;
    reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&tool_request())
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

#[tokio::test]
async fn prose_explaining_a_call_is_not_executed_as_one() {
    // The model answers the question, shows an example, and asks whether to
    // proceed. Executing the example would answer a question the user was
    // still being asked.
    let resp = guarded(
        "To get the weather you call the API like this:\n\n\
         ```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n```\n\n\
         Would you like me to run that for you?",
    )
    .await;

    assert!(
        resp.pointer("/choices/0/message/tool_calls").is_none(),
        "an explanation must not become a call: {resp}"
    );
    let content = resp
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        content.contains("Would you like me to run that"),
        "the model's question must reach the user: {resp}"
    );
}

#[tokio::test]
async fn a_walkthrough_showing_several_blocks_is_not_executed() {
    // More than one block is documentation, whichever one happens to parse.
    let resp = guarded(
        "First set the city:\n\n```json\n{\"city\": \"Paris\"}\n```\n\n\
         then issue the call:\n\n\
         ```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n```",
    )
    .await;

    assert!(
        resp.pointer("/choices/0/message/tool_calls").is_none(),
        "a walkthrough must not become a call: {resp}"
    );
}

#[tokio::test]
async fn a_bare_block_is_still_rescued() {
    // The regression guard for the narrowing: a model that emits only the call
    // is still recovered, which is the entire point of the parser.
    let resp = guarded("```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n```").await;

    let calls = resp
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("a bare block must still rescue: {resp}"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
}

#[tokio::test]
async fn a_short_lead_in_still_rescues_and_keeps_what_the_model_said() {
    // Models routinely prefix their only call with a few words. That is still
    // a call — and the words are still the model's, so they survive.
    let resp = guarded(
        "Here you go:\n```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n```",
    )
    .await;

    let calls = resp
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("a short lead-in must still rescue: {resp}"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(
        resp.pointer("/choices/0/message/content")
            .and_then(Value::as_str),
        Some("Here you go:"),
        "the lead-in is the model's own text: {resp}"
    );
}

#[tokio::test]
async fn tagged_call_syntax_is_never_echoed_back_to_the_client() {
    // A control-token rescue carries no prose worth keeping, and echoing the
    // markup would leak the syntax the rescue exists to hide.
    let resp = guarded(
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>",
    )
    .await;

    let calls = resp
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("a tagged call must rescue: {resp}"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(
        resp.pointer("/choices/0/message/content"),
        Some(&Value::Null),
        "call markup must not be echoed as content: {resp}"
    );
}
