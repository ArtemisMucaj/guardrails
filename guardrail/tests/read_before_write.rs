//! The read-before-mutate precondition, end to end through the proxy.
//!
//! The unit tests in `domain::precondition` cover the rule itself. These cover
//! the wiring: that the transcript the client sent actually reaches the check on
//! both APIs, that a refusal comes back as plain assistant text rather than a
//! tool call the harness would execute, and that a read in the transcript lets
//! the same edit through.
//!
//! Each test writes its own fixture under the OS temp directory, because the
//! rule only guards edits to files that exist — a path that is not on disk is
//! left to the harness to complain about.

use std::net::SocketAddr;
use std::path::PathBuf;

use guardrail::connector::Backend;
use guardrail::{build_app, AppState, Guardrails};
use serde_json::{json, Value};
use wiremock::matchers::{method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A file that exists for the duration of a test, removed on drop so a failing
/// assertion cannot leave a fixture behind to poison the next run.
struct Fixture(PathBuf);

impl Fixture {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!("guardrail_rbw_{tag}.txt"));
        std::fs::write(&path, "original contents\n").unwrap();
        Self(path)
    }

    fn path(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

async fn spawn(backend: &str) -> String {
    let state = AppState::new(Backend::new(reqwest::Client::new()), backend)
        .with_guardrails(Guardrails::default());
    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// The `Edit` tool, in the Chat Completions (nested) shape.
fn edit_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "Edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"}
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    })
}

/// A backend that answers every chat request with one native `Edit` call on
/// `path` — the model trying to edit a file it may or may not have read.
async fn backend_editing(path: &str) -> MockServer {
    let server = MockServer::start().await;
    let body = json!({
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
                    "function": {
                        "name": "Edit",
                        "arguments": json!({
                            "file_path": path,
                            "old_string": "original contents",
                            "new_string": "guessed replacement"
                        }).to_string()
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    Mock::given(method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .mount(&server)
        .await;
    server
}

/// A backend that answers with one native `Write` call on `path` — the model
/// rewriting a whole file rather than editing it.
async fn backend_writing(path: &str) -> MockServer {
    let server = MockServer::start().await;
    let body = json!({
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
                    "function": {
                        "name": "Write",
                        "arguments": json!({
                            "file_path": path,
                            "content": "a whole new file"
                        }).to_string()
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    Mock::given(method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .mount(&server)
        .await;
    server
}

/// The `Write` tool, in the Chat Completions (nested) shape.
fn write_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "Write",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["file_path", "content"]
            }
        }
    })
}

async fn post_chat_with(proxy: &str, messages: Value, tool: Value) -> Value {
    reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&json!({
            "model": "local-model",
            "messages": messages,
            "tools": [tool],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

async fn post_chat(proxy: &str, messages: Value) -> Value {
    reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&json!({
            "model": "local-model",
            "messages": messages,
            "tools": [edit_tool()],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// A transcript in which the model read *another* file and then grepped, but
/// never read the file it is about to edit.
///
/// The read of the other file is what makes the transcript legible: the rule
/// only refuses once it has seen reads flowing through, so a transcript with no
/// read at all would stand it down instead of exercising it.
fn transcript_without_the_read() -> Value {
    json!([
        {"role": "user", "content": "fix the typo"},
        {
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": "{\"file_path\":\"/etc/passwd\"}"
                }
            }]
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "root:*:0:0:"},
        {
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "Grep", "arguments": "{\"pattern\":\"typo\"}"}
            }]
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "3 matches"}
    ])
}

/// The same transcript, with the read the rule is looking for.
fn transcript_with_the_read(path: &str) -> Value {
    json!([
        {"role": "user", "content": "fix the typo"},
        {
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": json!({"file_path": path}).to_string()
                }
            }]
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "original contents"}
    ])
}

#[tokio::test]
async fn an_edit_without_a_read_is_refused_as_text() {
    let file = Fixture::new("refused");
    let backend = backend_editing(&file.path()).await;
    let proxy = spawn(&backend.uri()).await;

    let got = post_chat(&proxy, transcript_without_the_read()).await;

    // The refusal reaches the client as assistant text, never as the tool call
    // — a forwarded call would be executed by the harness, which is the blind
    // edit the guard exists to stop.
    let message = &got["choices"][0]["message"];
    assert!(
        message["tool_calls"].is_null(),
        "the edit was forwarded: {got}"
    );
    let content = message["content"].as_str().unwrap_or_default();
    assert!(
        content.contains(&file.path()),
        "nudge omits the path: {content}"
    );
    assert!(
        content.contains("not been read"),
        "unexpected nudge: {content}"
    );

    // The file on disk is untouched.
    assert_eq!(
        std::fs::read_to_string(&file.0).unwrap(),
        "original contents\n"
    );
}

#[tokio::test]
async fn the_same_edit_passes_once_the_transcript_shows_the_read() {
    let file = Fixture::new("allowed");
    let backend = backend_editing(&file.path()).await;
    let proxy = spawn(&backend.uri()).await;

    let got = post_chat(&proxy, transcript_with_the_read(&file.path())).await;

    let call = &got["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(
        call["function"]["name"], "Edit",
        "the edit was refused: {got}"
    );
    assert!(call["function"]["arguments"]
        .as_str()
        .unwrap()
        .contains(&file.path()));
}

/// A transcript with no tool traffic is illegible, not empty: the read may have
/// happened under a vocabulary this proxy does not know. The rule stands down.
#[tokio::test]
async fn a_transcript_with_no_tool_traffic_does_not_refuse() {
    let file = Fixture::new("illegible");
    let backend = backend_editing(&file.path()).await;
    let proxy = spawn(&backend.uri()).await;

    let got = post_chat(&proxy, json!([{"role": "user", "content": "fix the typo"}])).await;

    let call = &got["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(
        call["function"]["name"], "Edit",
        "refused on a fail-open case: {got}"
    );
}

/// A read in one conversation must not license an edit in another. Because the
/// set is rebuilt from each request's own `messages[]`, a client that read the
/// file in a different chat sends a transcript without that read — and is
/// refused, exactly as if the read had never happened.
#[tokio::test]
async fn a_read_in_another_conversation_does_not_carry_over() {
    let file = Fixture::new("cross_session");
    let backend = backend_editing(&file.path()).await;
    let proxy = spawn(&backend.uri()).await;

    // Conversation A reads the file and is allowed to edit it.
    let allowed = post_chat(&proxy, transcript_with_the_read(&file.path())).await;
    assert_eq!(
        allowed["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "Edit"
    );

    // Conversation B, over the same proxy, never read it — and is still refused.
    let refused = post_chat(&proxy, transcript_without_the_read()).await;
    assert!(
        refused["choices"][0]["message"]["tool_calls"].is_null(),
        "conversation A's read leaked into conversation B: {refused}"
    );
}

/// The Responses API carries its history as flat `function_call` items rather
/// than nested `tool_calls`, so the scan has to read both shapes.
#[tokio::test]
async fn responses_input_items_are_scanned_for_the_read() {
    let file = Fixture::new("responses");
    let path = file.path();
    let sse = format!(
        concat!(
            r#"data: {{"type":"response.output_item.added","output_index":0,"item":{{"type":"function_call","call_id":"call_1","name":"Edit"}}}}"#,
            "\n\n",
            r#"data: {{"type":"response.function_call_arguments.delta","output_index":0,"delta":{delta}}}"#,
            "\n\n",
            r#"data: {{"type":"response.completed","response":{{"id":"resp_1","model":"test-model","output":[]}}}}"#,
            "\n\n",
        ),
        delta = Value::String(
            json!({"file_path": path, "old_string": "original contents", "new_string": "x"})
                .to_string()
        )
    );
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(wm_path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
        .mount(&backend)
        .await;
    let proxy = spawn(&backend.uri()).await;

    let tools = json!([{
        "type": "function",
        "name": "Edit",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"}
            },
            "required": ["file_path", "old_string", "new_string"]
        }
    }]);

    let ask = |input: Value| {
        let tools = tools.clone();
        let proxy = proxy.clone();
        async move {
            reqwest::Client::new()
                .post(format!("{proxy}/v1/responses"))
                .json(&json!({"model": "test-model", "input": input, "tools": tools}))
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        }
    };

    let is_call = |body: &Value| {
        body["output"]
            .as_array()
            .is_some_and(|items| items.iter().any(|i| i["type"] == "function_call"))
    };

    // Read another file, grepped this one, never read it → refused. The read of
    // the other file is what makes the transcript legible.
    let refused = ask(json!([
        {
            "type": "function_call", "call_id": "c0", "name": "Read",
            "arguments": "{\"file_path\":\"/etc/passwd\"}"
        },
        {"type": "function_call_output", "call_id": "c0", "output": "root:*:0:0:"},
        {"type": "function_call", "call_id": "c1", "name": "Grep", "arguments": "{\"pattern\":\"x\"}"}
    ]))
    .await;
    assert!(!is_call(&refused), "expected a refusal, got: {refused}");

    // Read → allowed.
    let allowed = ask(json!([
        {
            "type": "function_call", "call_id": "c0", "name": "Read",
            "arguments": json!({"file_path": path}).to_string()
        },
        {"type": "function_call_output", "call_id": "c0", "output": "original contents"}
    ]))
    .await;
    assert!(
        is_call(&allowed),
        "expected the edit through, got: {allowed}"
    );
}

/// Trimmed history: a compaction drops the read and keeps the model's own
/// failed edit plus its error. The surviving traffic is recognisable, but it is
/// not evidence that reads are visible — so the rule must stand down rather than
/// tell the model to re-read a file it already read.
#[tokio::test]
async fn a_trimmed_history_keeping_only_a_failed_edit_does_not_refuse() {
    let file = Fixture::new("trimmed");
    let backend = backend_editing(&file.path()).await;
    let proxy = spawn(&backend.uri()).await;

    let got = post_chat(
        &proxy,
        json!([
            {"role": "user", "content": "fix the typo"},
            {
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "arguments": json!({
                            "file_path": file.path(),
                            "old_string": "stale",
                            "new_string": "fixed"
                        }).to_string()
                    }
                }]
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "Error: string not found"}
        ]),
    )
    .await;

    let call = &got["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(
        call["function"]["name"], "Edit",
        "a trimmed history was refused: {got}"
    );
}

/// A chained Responses turn keeps its history on the backend. Clients commonly
/// replay the previous call and its output, which parses as a read and so is
/// legible — legible about a fragment, while the read that matters is upstream.
/// The rule must stand down on `previous_response_id`, not on the resent input
/// happening to be empty.
#[tokio::test]
async fn a_chained_responses_turn_does_not_refuse() {
    let file = Fixture::new("chained");
    let path = file.path();
    let sse = format!(
        concat!(
            r#"data: {{"type":"response.output_item.added","output_index":0,"item":{{"type":"function_call","call_id":"call_1","name":"Edit"}}}}"#,
            "\n\n",
            r#"data: {{"type":"response.function_call_arguments.delta","output_index":0,"delta":{delta}}}"#,
            "\n\n",
            r#"data: {{"type":"response.completed","response":{{"id":"resp_2","model":"test-model","output":[]}}}}"#,
            "\n\n",
        ),
        delta = Value::String(
            json!({"file_path": path, "old_string": "original contents", "new_string": "x"})
                .to_string()
        )
    );
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(wm_path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
        .mount(&backend)
        .await;
    let proxy = spawn(&backend.uri()).await;

    // Chained, and replaying a read of some *other* file — legible, but a
    // fragment. The read of the edited file happened in an earlier turn the
    // backend holds.
    let got: Value = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .json(&json!({
            "model": "test-model",
            "previous_response_id": "resp_1",
            "input": [
                {
                    "type": "function_call", "call_id": "c9", "name": "Read",
                    "arguments": "{\"file_path\":\"/etc/passwd\"}"
                },
                {"type": "function_call_output", "call_id": "c9", "output": "root:*:0:0:"}
            ],
            "tools": [{
                "type": "function",
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"}
                    },
                    "required": ["file_path", "old_string", "new_string"]
                }
            }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let has_call = got["output"]
        .as_array()
        .is_some_and(|items| items.iter().any(|i| i["type"] == "function_call"));
    assert!(has_call, "a chained turn was refused: {got}");
}

// ── Whole-file writes over an existing file ─────────────────────────────────

/// The rule issue #2 produced, now covered end to end: a whole-file `Write`
/// onto a path that exists is refused, because the rewrite would replace
/// contents the model may never have seen.
#[tokio::test]
async fn a_whole_file_write_over_an_existing_file_is_refused() {
    let file = Fixture::new("write_existing");
    let backend = backend_writing(&file.path()).await;
    let proxy = spawn(&backend.uri()).await;

    let got = post_chat_with(&proxy, transcript_without_the_read(), write_tool()).await;

    let message = &got["choices"][0]["message"];
    assert!(
        message["tool_calls"].is_null(),
        "the write was forwarded: {got}"
    );
    let content = message["content"].as_str().unwrap_or_default();
    assert!(
        content.contains(&file.path()),
        "nudge omits the path: {content}"
    );
    assert!(
        content.contains("already exists"),
        "unexpected nudge: {content}"
    );
    assert!(
        content.contains("edit tool"),
        "nudge does not point at edit: {content}"
    );

    assert_eq!(
        std::fs::read_to_string(&file.0).unwrap(),
        "original contents\n"
    );
}

/// The write rule is about the file existing, not about the transcript: having
/// read the file first does not license replacing it wholesale.
#[tokio::test]
async fn reading_the_file_first_does_not_license_a_whole_file_write() {
    let file = Fixture::new("write_after_read");
    let backend = backend_writing(&file.path()).await;
    let proxy = spawn(&backend.uri()).await;

    let got = post_chat_with(&proxy, transcript_with_the_read(&file.path()), write_tool()).await;

    assert!(
        got["choices"][0]["message"]["tool_calls"].is_null(),
        "a read let a whole-file write through: {got}"
    );
}

/// Creating a genuinely new file is the normal use of `Write` and must pass.
#[tokio::test]
async fn a_write_creating_a_new_file_passes() {
    let path = std::env::temp_dir()
        .join("guardrail_rbw_brand_new.txt")
        .to_string_lossy()
        .into_owned();
    let _ = std::fs::remove_file(&path);
    let backend = backend_writing(&path).await;
    let proxy = spawn(&backend.uri()).await;

    let got = post_chat_with(&proxy, transcript_without_the_read(), write_tool()).await;

    let call = &got["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(
        call["function"]["name"], "Write",
        "creating a new file was refused: {got}"
    );
}
