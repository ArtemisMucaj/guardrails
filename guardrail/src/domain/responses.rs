//! The OpenAI Responses API (`/v1/responses`), translated at the edges.
//!
//! The guardrails themselves — rescue, validate, repair, retry, the synthetic
//! `respond` tool — work on the neutral [`ToolCall`] and know nothing about
//! either wire format. Only the edges differ, so supporting a second API means
//! translating on the way in and out rather than duplicating the pipeline:
//!
//! | | Chat Completions | Responses |
//! |---|---|---|
//! | turns | `messages[]` | `input[]` (+ `instructions`) |
//! | tool calls | `choices[0].message.tool_calls[]` | `output[]` items typed `function_call` |
//! | arguments | `function.arguments` | `arguments`, flat |
//! | tool defs | `tools[].function.{name,parameters}` | `tools[].{name,parameters}`, flat |
//!
//! Everything the proxy does not touch rides along untouched, so a field this
//! crate has never heard of still reaches the backend and the client.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::decode::ToolCall;
use super::model::{Tool, ToolFunction};

/// Output item type carrying a tool call.
pub const FUNCTION_CALL: &str = "function_call";
/// Output item type carrying assistant text.
pub const MESSAGE: &str = "message";
/// Content part type holding assistant text.
pub const OUTPUT_TEXT: &str = "output_text";

/// A `POST /v1/responses` request body, parsed typed-where-touched.
///
/// Mirrors [`ChatRequest`](super::model::ChatRequest) in shape and intent: the
/// fields the guardrails read are typed, everything else round-trips through
/// [`rest`](Self::rest).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesRequest {
    pub model: String,

    /// Tool definitions in the Responses shape — flat, not nested under
    /// `function`. Absent when the client declared none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,

    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

impl ResponsesRequest {
    /// Whether the client asked for a streamed response.
    pub fn stream(&self) -> bool {
        self.rest
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Whether the request carries any tool definitions.
    pub fn has_tools(&self) -> bool {
        self.tools.as_ref().is_some_and(|t| !t.is_empty())
    }

    /// The response this turn continues, when the client chained it.
    ///
    /// The Responses API is stateful: a client continues a conversation by
    /// naming the previous response instead of resending the transcript. That
    /// makes this the parent edge of a conversation — following it back reaches
    /// the first turn, whose id names the whole exchange.
    ///
    /// Absent on the first turn of a conversation, and on any client that
    /// resends full input rather than chaining (`store: false`).
    pub fn previous_response_id(&self) -> Option<&str> {
        self.rest.get("previous_response_id").and_then(Value::as_str)
    }

    /// The declared tools as the neutral [`Tool`] the validator expects.
    ///
    /// A Responses tool is flat (`{"type":"function","name":...,"parameters":...}`)
    /// where Chat Completions nests under `function`. Lifting it lets one
    /// validator serve both APIs.
    pub fn normalized_tools(&self) -> Vec<Tool> {
        self.tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter_map(tool_from_responses)
            .collect()
    }

    /// Append a tool in the Responses (flat) shape.
    pub fn push_tool(&mut self, tool: Tool) {
        let mut flat = Map::new();
        flat.insert("type".to_string(), Value::String("function".to_string()));
        flat.insert("name".to_string(), Value::String(tool.function.name));
        for (key, value) in tool.function.rest {
            flat.insert(key, value);
        }
        self.tools.get_or_insert_with(Vec::new).push(Value::Object(flat));
    }

    /// The `input` items as an array, when the client sent them that way.
    ///
    /// Empty for the bare-string form and for a chained turn, which names its
    /// predecessor with `previous_response_id` instead of resending anything.
    /// A caller that reads this to *look for* something must treat empty as
    /// "nothing to see here" rather than as "it did not happen" — a chained
    /// turn's history lives on the backend, out of the proxy's reach.
    pub fn input_items(&self) -> &[Value] {
        self.rest
            .get("input")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// Append turns to `input`, so a corrective retry can carry the nudge.
    ///
    /// `input` may legally be a bare string; it is promoted to the turn form
    /// first, since appending to a string would lose the original prompt.
    pub fn extend_input(&mut self, turns: Vec<Value>) {
        let existing = self.rest.remove("input").unwrap_or(Value::Null);
        let mut items = match existing {
            Value::Array(items) => items,
            Value::String(text) => vec![json!({"role": "user", "content": text})],
            Value::Null => Vec::new(),
            other => vec![other],
        };
        items.extend(turns);
        self.rest.insert("input".to_string(), Value::Array(items));
    }
}

/// Lift one Responses tool definition into the neutral [`Tool`].
fn tool_from_responses(value: &Value) -> Option<Tool> {
    let object = value.as_object()?;
    // A client may also send the Chat Completions shape; accept both rather
    // than rejecting a request the backend would have understood.
    if let Some(function) = object.get("function").and_then(Value::as_object) {
        let name = function.get("name")?.as_str()?.to_string();
        let mut rest = function.clone();
        rest.remove("name");
        return Some(Tool {
            kind: "function".to_string(),
            function: ToolFunction { name, rest },
            rest: Map::new(),
        });
    }

    let name = object.get("name")?.as_str()?.to_string();
    let mut rest = object.clone();
    rest.remove("name");
    rest.remove("type");
    Some(Tool {
        kind: "function".to_string(),
        function: ToolFunction { name, rest },
        rest: Map::new(),
    })
}

/// Read tool calls out of a Responses body's `output[]`.
///
/// Returns an empty vector when the model produced no tool call, which the
/// caller distinguishes from "produced one we could not parse".
pub fn tool_calls(body: &Value) -> Vec<ToolCall> {
    body.get("output")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some(FUNCTION_CALL))
                .filter_map(|item| {
                    Some(ToolCall {
                        // `call_id` is what a subsequent turn references; `id`
                        // identifies the output item itself. The former is the
                        // one that must survive.
                        id: item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        name: item.get("name")?.as_str()?.to_string(),
                        arguments: item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}")
                            .to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Concatenate the assistant text of a Responses body, skipping reasoning and
/// tool items.
pub fn text(body: &Value) -> String {
    let Some(items) = body.get("output").and_then(Value::as_array) else {
        return String::new();
    };
    items
        .iter()
        .filter(|item| {
            // An item with no declared type is treated as a message: the
            // alternative is discarding the only answer we got.
            matches!(
                item.get("type").and_then(Value::as_str),
                Some(MESSAGE) | None
            )
        })
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some(OUTPUT_TEXT))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect()
}

/// Rebuild a Responses body whose `output[]` is exactly `calls`.
///
/// `template` supplies id, model, and any other envelope fields so the client
/// sees a response shaped like the one the backend would have sent.
pub fn with_tool_calls(template: &Value, calls: &[ToolCall]) -> Value {
    with_tool_calls_and_text(template, calls, "")
}

/// [`with_tool_calls`], preceded by an assistant message carrying `text`.
///
/// `output[]` is ordered, so the message comes first: the model wrote it before
/// the call it was recovered from. Empty text adds no message, which is the
/// shape a pure tool-call turn already had.
pub fn with_tool_calls_and_text(template: &Value, calls: &[ToolCall], text: &str) -> Value {
    let mut body = envelope(template);
    let mut output: Vec<Value> = Vec::with_capacity(calls.len() + 1);
    if !text.is_empty() {
        output.push(json!({
            "type": MESSAGE,
            "id": "msg_0",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": OUTPUT_TEXT, "text": text, "annotations": []}],
        }));
    }
    output.extend(calls.iter().enumerate().map(|(index, call)| {
        let call_id = call
            .id
            .clone()
            .unwrap_or_else(|| format!("call_{index}"));
        json!({
            "type": FUNCTION_CALL,
            "id": format!("fc_{index}"),
            "call_id": call_id,
            "name": call.name,
            "arguments": call.arguments,
            "status": "completed",
        })
    }));
    set(&mut body, "output", Value::Array(output));
    set(&mut body, "status", Value::String("completed".to_string()));
    body
}

/// Rebuild a Responses body whose output is a single assistant message.
pub fn with_text(template: &Value, text: &str) -> Value {
    let mut body = envelope(template);
    set(
        &mut body,
        "output",
        json!([{
            "type": MESSAGE,
            "id": "msg_0",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": OUTPUT_TEXT, "text": text, "annotations": []}],
        }]),
    );
    set(&mut body, "status", Value::String("completed".to_string()));
    body
}

/// Start from the template's envelope, dropping the fields we are replacing.
fn envelope(template: &Value) -> Value {
    let mut body = match template {
        Value::Object(_) => template.clone(),
        _ => json!({"id": "guardrail-0", "object": "response"}),
    };
    if let Some(object) = body.as_object_mut() {
        object.remove("output");
        object.entry("object".to_string())
            .or_insert_with(|| Value::String("response".to_string()));
    }
    body
}

fn set(body: &mut Value, key: &str, value: Value) {
    if let Some(object) = body.as_object_mut() {
        object.insert(key.to_string(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: Value) -> ResponsesRequest {
        serde_json::from_value(v).expect("parse")
    }

    #[test]
    fn reads_tool_calls_out_of_output_items() {
        let body = json!({
            "id": "resp_1",
            "output": [
                {"type": "reasoning", "id": "rs_1", "content": []},
                {"type": "function_call", "id": "fc_1", "call_id": "call_abc",
                 "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            ]
        });
        let calls = tool_calls(&body);
        assert_eq!(calls.len(), 1, "reasoning items must not be read as calls");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"city\":\"Paris\"}");
        // `call_id` is what the next turn references, so it is the id kept.
        assert_eq!(calls[0].id.as_deref(), Some("call_abc"));
    }

    #[test]
    fn a_body_with_no_tool_call_yields_none() {
        let body = json!({"output": [{"type": "message", "content": [
            {"type": "output_text", "text": "hello"}
        ]}]});
        assert!(tool_calls(&body).is_empty());
        assert_eq!(text(&body), "hello");
    }

    #[test]
    fn text_concatenates_parts_and_skips_reasoning() {
        let body = json!({"output": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "hmm"}]},
            {"type": "message", "content": [
                {"type": "output_text", "text": "Hello"},
                {"type": "output_text", "text": ", world"}
            ]}
        ]});
        assert_eq!(text(&body), "Hello, world");
    }

    #[test]
    fn a_flat_responses_tool_is_lifted_for_the_validator() {
        let request = parse(json!({
            "model": "m",
            "input": "hi",
            "tools": [{
                "type": "function",
                "name": "Edit",
                "parameters": {"type": "object", "required": ["filePath"]}
            }]
        }));
        let tools = request.normalized_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "Edit");
        // `parameters` must survive: it is what required-field checking reads.
        assert!(tools[0].function.rest.contains_key("parameters"));
    }

    #[test]
    fn a_chat_shaped_tool_is_also_accepted() {
        // Clients migrating between APIs send the nested shape; rejecting it
        // would fail a request the backend would have understood.
        let request = parse(json!({
            "model": "m",
            "tools": [{"type": "function", "function": {
                "name": "Edit", "parameters": {"type": "object"}
            }}]
        }));
        let tools = request.normalized_tools();
        assert_eq!(tools[0].function.name, "Edit");
        assert!(tools[0].function.rest.contains_key("parameters"));
    }

    #[test]
    fn pushing_a_tool_produces_the_flat_shape() {
        let mut request = parse(json!({"model": "m", "input": "hi"}));
        request.push_tool(Tool {
            kind: "function".into(),
            function: ToolFunction {
                name: "respond".into(),
                rest: [("parameters".to_string(), json!({"type": "object"}))]
                    .into_iter()
                    .collect(),
            },
            rest: Map::new(),
        });
        let tools = request.tools.unwrap();
        assert_eq!(tools[0]["name"], "respond");
        assert_eq!(tools[0]["type"], "function");
        assert!(tools[0].get("function").is_none(), "must not nest");
        assert!(tools[0].get("parameters").is_some());
    }

    #[test]
    fn a_string_input_is_promoted_before_turns_are_appended() {
        // `input` may be a bare string; appending to it would lose the prompt.
        let mut request = parse(json!({"model": "m", "input": "original"}));
        request.extend_input(vec![json!({"role": "user", "content": "nudge"})]);

        let input = request.rest.get("input").unwrap().as_array().unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["content"], "original");
        assert_eq!(input[1]["content"], "nudge");
    }

    #[test]
    fn re_emitted_tool_calls_carry_their_call_id() {
        let template = json!({"id": "resp_1", "model": "m", "object": "response"});
        let calls = vec![ToolCall {
            id: Some("call_abc".into()),
            name: "get_weather".into(),
            arguments: "{}".into(),
        }];
        let body = with_tool_calls(&template, &calls);

        assert_eq!(body["id"], "resp_1", "the envelope is preserved");
        assert_eq!(body["output"][0]["type"], "function_call");
        assert_eq!(body["output"][0]["call_id"], "call_abc");
        assert_eq!(body["output"][0]["name"], "get_weather");
    }

    #[test]
    fn a_call_without_an_id_gets_a_deterministic_one() {
        let body = with_tool_calls(
            &json!({"id": "r"}),
            &[ToolCall { id: None, name: "f".into(), arguments: "{}".into() }],
        );
        assert_eq!(body["output"][0]["call_id"], "call_0");
    }

    #[test]
    fn a_text_response_round_trips_through_the_output_shape() {
        let body = with_text(&json!({"id": "resp_1"}), "the answer");
        assert_eq!(body["output"][0]["type"], "message");
        assert_eq!(body["output"][0]["content"][0]["text"], "the answer");
        // And reading it back gives the same text.
        assert_eq!(text(&body), "the answer");
    }

    #[test]
    fn rebuilding_replaces_the_previous_output() {
        // The template is the backend's own response, which already has an
        // `output`; leaving it would emit both the bad and the repaired call.
        let template = json!({
            "id": "resp_1",
            "output": [{"type": "function_call", "name": "wrong", "arguments": "{}"}]
        });
        let body = with_tool_calls(
            &template,
            &[ToolCall { id: None, name: "right".into(), arguments: "{}".into() }],
        );
        assert_eq!(body["output"].as_array().unwrap().len(), 1);
        assert_eq!(body["output"][0]["name"], "right");
    }

    #[test]
    fn unknown_request_fields_round_trip() {
        let original = json!({
            "model": "m",
            "input": "hi",
            "temperature": 0.5,
            "reasoning": {"effort": "high"},
            "something_new": true
        });
        let request = parse(original.clone());
        let back = serde_json::to_value(&request).unwrap();
        assert_eq!(back, original);
    }
}
