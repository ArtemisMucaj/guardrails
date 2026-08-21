//! `/v1/responses` end to end: the same guardrails as the chat path, over the
//! Responses API's shapes.

use guardrail::application::AppState;
use guardrail::connector::Backend;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A tool the model is expected to call, in the Responses (flat) shape.
fn edit_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "name": "Edit",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "line": {"type": "integer"}
            },
            "required": ["filePath"]
        }
    })
}

async fn spawn_proxy(backend: &str) -> String {
    let state = AppState::new(Backend::new(reqwest::Client::new()), backend);
    let app = guardrail::build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// Mount a backend answering `/v1/responses` with a raw SSE body.
async fn backend_streaming(sse: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse.to_string(), "text/event-stream"))
        .mount(&server)
        .await;
    server
}

/// Ask the proxy for a response, non-streaming.
async fn ask(proxy: &str, tools: Option<serde_json::Value>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": "test-model",
        "input": "do the thing",
    });
    if let Some(tools) = tools {
        body["tools"] = serde_json::json!([tools]);
    }
    reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// The function calls in a Responses body.
fn calls(body: &serde_json::Value) -> Vec<(String, String)> {
    body["output"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter(|i| i["type"] == "function_call")
                .map(|i| {
                    (
                        i["name"].as_str().unwrap_or_default().to_string(),
                        i["arguments"].as_str().unwrap_or_default().to_string(),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The assistant text in a Responses body.
fn text(body: &serde_json::Value) -> String {
    body["output"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter(|i| i["type"] == "message")
                .filter_map(|i| i["content"].as_array())
                .flatten()
                .filter_map(|p| p["text"].as_str())
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn a_valid_function_call_passes_through() {
    let backend = backend_streaming(concat!(
        r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"Edit"}}"#, "\n\n",
        r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"filePath\":\"/tmp/a.txt\"}"}"#, "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_1","model":"test-model","output":[]}}"#, "\n\n",
    ))
    .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let body = ask(&proxy, Some(edit_tool())).await;

    assert_eq!(
        calls(&body),
        vec![("Edit".to_string(), r#"{"filePath":"/tmp/a.txt"}"#.to_string())]
    );
    assert_eq!(body["id"], "resp_1", "the envelope survives");
}

#[tokio::test]
async fn a_tool_call_buried_in_text_is_rescued() {
    // The failure this proxy exists for, now on the Responses path.
    let backend = backend_streaming(concat!(
        r#"data: {"type":"response.output_text.delta","delta":"<tool_call>{\"name\": \"Edit\", \"arguments\": {\"filePath\": \"/tmp/b.txt\"}}</tool_call>"}"#, "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_2","model":"test-model","output":[]}}"#, "\n\n",
    ))
    .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let body = ask(&proxy, Some(edit_tool())).await;

    let calls = calls(&body);
    assert_eq!(calls.len(), 1, "the call must be recovered from the text");
    assert_eq!(calls[0].0, "Edit");
    assert!(calls[0].1.contains("/tmp/b.txt"));
}

#[tokio::test]
async fn a_mistyped_argument_is_coerced() {
    // `line` is declared integer; the model sent a string.
    let backend = backend_streaming(concat!(
        r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"Edit"}}"#, "\n\n",
        r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"filePath\":\"/tmp/c.txt\",\"line\":\"3\"}"}"#, "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_3","model":"test-model","output":[]}}"#, "\n\n",
    ))
    .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let body = ask(&proxy, Some(edit_tool())).await;

    let args: serde_json::Value = serde_json::from_str(&calls(&body)[0].1).unwrap();
    assert_eq!(args["line"], 3, "a stringified integer must be repaired in place");
}

#[tokio::test]
async fn a_snake_case_argument_name_is_repaired() {
    let backend = backend_streaming(concat!(
        r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"Edit"}}"#, "\n\n",
        r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"file_path\":\"/tmp/d.txt\"}"}"#, "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_4","model":"test-model","output":[]}}"#, "\n\n",
    ))
    .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let body = ask(&proxy, Some(edit_tool())).await;

    let args: serde_json::Value = serde_json::from_str(&calls(&body)[0].1).unwrap();
    assert_eq!(args["filePath"], "/tmp/d.txt");
}

#[tokio::test]
async fn an_unfixable_call_falls_back_to_text_rather_than_forwarding_it() {
    // An unknown tool cannot be repaired; the client must not receive a call it
    // would try to execute.
    let backend = backend_streaming(concat!(
        r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"NoSuchTool"}}"#, "\n\n",
        r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{}"}"#, "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_5","model":"test-model","output":[]}}"#, "\n\n",
    ))
    .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let body = ask(&proxy, Some(edit_tool())).await;

    assert!(
        calls(&body).is_empty(),
        "an invalid call must not reach the client: {body}"
    );
    assert!(
        text(&body).contains("could not be completed"),
        "the client should be told why: {body}"
    );
}

#[tokio::test]
async fn a_request_with_no_tools_is_forwarded_untouched() {
    let server = MockServer::start().await;
    let original = serde_json::json!({
        "id": "resp_6",
        "object": "response",
        "model": "test-model",
        "output": [{
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "plain answer", "annotations": []}]
        }],
        "usage": {"input_tokens": 3, "output_tokens": 2}
    });
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&original))
        .mount(&server)
        .await;

    let proxy = spawn_proxy(&server.uri()).await;
    let body = ask(&proxy, None).await;

    // Byte-for-byte: with no tool there is nothing to guard.
    assert_eq!(body, original);
}

#[tokio::test]
async fn text_streams_live_to_a_streaming_client() {
    let backend = backend_streaming(concat!(
        r#"data: {"type":"response.output_text.delta","delta":"Hel"}"#, "\n\n",
        r#"data: {"type":"response.output_text.delta","delta":"lo"}"#, "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_7","model":"test-model","output":[]}}"#, "\n\n",
    ))
    .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let raw = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": "hi",
            "stream": true,
            "tools": [edit_tool()],
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    // The individual deltas reached the client rather than one buffered blob.
    assert!(raw.contains("response.output_text.delta"), "got: {raw}");
    assert!(raw.contains("Hel"), "got: {raw}");
}

#[tokio::test]
async fn a_streamed_tool_call_is_guarded_not_forwarded_raw() {
    // Malformed native arguments are not repairable in place (that fallback
    // applies to calls rescued from text, not to native ones), so the loop
    // retries and then falls back to an explanation. What matters here is that
    // the broken call itself never reaches the client mid-stream.
    let backend = backend_streaming(concat!(
        r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"Edit"}}"#, "\n\n",
        r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{'filePath': '/tmp/e.txt'}"}"#, "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_8","model":"test-model","output":[]}}"#, "\n\n",
    ))
    .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let raw = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": "hi",
            "stream": true,
            "tools": [edit_tool()],
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert!(
        !raw.contains("'filePath'"),
        "the malformed original must not reach the client: {raw}"
    );
    assert!(
        !raw.contains("function_call"),
        "no unchecked tool call may be emitted: {raw}"
    );
    assert!(
        raw.contains("could not be completed"),
        "the client should be told why: {raw}"
    );
}

#[tokio::test]
async fn rescued_call_syntax_never_reaches_a_streaming_client() {
    // Regression: text deltas were forwarded live, so the raw `<tool_call>`
    // markup reached the client and was then contradicted by the rescued call.
    // The backend's own `response.completed` was forwarded too, so a client
    // reading the last completed event got the unrepaired output.
    let backend = backend_streaming(concat!(
        r#"data: {"type":"response.output_text.delta","delta":"<tool_call>{\"name\": \"Edit\", \"arguments\": {\"filePath\": \"/tmp/b.txt\"}}</tool_call>"}"#, "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_2","model":"test-model","output":[]}}"#, "\n\n",
    ))
    .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let raw = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model", "input": "hi", "stream": true, "tools": [edit_tool()],
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert!(
        !raw.contains("<tool_call>"),
        "raw call markup must never be forwarded: {raw}"
    );
    // Exactly one terminal event, and it carries the repaired call.
    assert_eq!(
        raw.matches("response.completed").count(),
        1,
        "the backend's own completed event must not be forwarded: {raw}"
    );
    assert!(raw.contains("function_call"), "the rescued call should: {raw}");
}

#[tokio::test]
async fn ordinary_text_that_merely_starts_with_a_brace_still_streams() {
    // The held-back text must be released when nothing is rescued, or a plain
    // answer that happens to look tool-shaped would be swallowed.
    let backend = backend_streaming(concat!(
        r#"data: {"type":"response.output_text.delta","delta":"{\"name\": not actually a call"}"#, "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_x","model":"test-model","output":[]}}"#, "\n\n",
    ))
    .await;

    let proxy = spawn_proxy(&backend.uri()).await;
    let raw = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model", "input": "hi", "stream": true, "tools": [edit_tool()],
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert!(
        raw.contains("not actually a call"),
        "withheld text must be released when no call is recovered: {raw}"
    );
}
