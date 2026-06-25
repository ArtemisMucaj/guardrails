//! Milestone 3: decode an OpenAI chat-completions response into the internal
//! [`ModelOutput`] the guardrails reason about, and re-emit tool calls in
//! canonical OpenAI form.
//!
//! At this milestone decoding is **log-only**: the proxy decodes and validates
//! the backend's response to confirm it correctly detects native `tool_calls`,
//! but still forwards the original body unchanged. The canonical encoder built
//! here is the foundation later milestones use — rescue (M4) produces
//! [`ToolCall`]s from raw text, and the retry loop (M6) re-emits them as a
//! canonical response.

use serde_json::{json, Value};

/// What the model actually produced, normalised across "native tool_calls" and
/// "tool call buried in text". Native decoding yields [`ModelOutput::ToolCalls`];
/// anything else is [`ModelOutput::Text`], which is the input rescue (M4) tries
/// to repair.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelOutput {
    ToolCalls(Vec<ToolCall>),
    Text(String),
}

/// A single tool call. `arguments` is kept as the raw JSON-encoded string OpenAI
/// uses (not a parsed value) so canonical re-emit is lossless; validation parses
/// it on demand. `id` is preserved when the backend supplied one so re-emit can
/// echo it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    /// Build a canonical OpenAI `tool_calls` entry for this call, minting a
    /// deterministic id from `index` when the backend supplied none.
    fn to_canonical(&self, index: usize) -> Value {
        let id = self.id.clone().unwrap_or_else(|| format!("call_{index}"));
        json!({
            "id": id,
            "type": "function",
            "function": { "name": self.name, "arguments": self.arguments },
        })
    }
}

/// Decode a full OpenAI chat-completion response body into [`ModelOutput`].
///
/// The first choice's message is examined: a non-empty `tool_calls` array yields
/// [`ModelOutput::ToolCalls`]; otherwise the (possibly empty/null) `content`
/// yields [`ModelOutput::Text`]. Malformed/missing fields degrade to empty text
/// rather than panicking — decoding is best-effort and never fails the request.
pub fn decode_response(body: &Value) -> ModelOutput {
    let message = body
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"));

    if let Some(calls) = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
    {
        let parsed: Vec<ToolCall> = calls.iter().map(decode_tool_call).collect();
        return ModelOutput::ToolCalls(parsed);
    }

    let text = message
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    ModelOutput::Text(text)
}

fn decode_tool_call(call: &Value) -> ToolCall {
    let id = call.get("id").and_then(Value::as_str).map(str::to_string);
    let function = call.get("function");
    let name = function
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    // `arguments` is canonically a JSON-encoded string. Some backends emit it as
    // a bare object instead; normalise both to a string so re-emit is uniform.
    let arguments = match function.and_then(|f| f.get("arguments")) {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    ToolCall {
        id,
        name,
        arguments,
    }
}

/// Render tool calls as a canonical OpenAI `tool_calls` array. Used by the retry
/// loop (M6) to re-emit rescued calls; exercised now via tests to confirm a
/// native response round-trips through decode → canonical unchanged.
pub fn canonical_tool_calls(calls: &[ToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| c.to_canonical(i))
            .collect(),
    )
}

/// Rebuild a response with canonical `tool_calls`, reusing `template` so id,
/// model, usage, etc. are preserved. Used by the guardrail loop to re-emit
/// rescued or repaired calls.
pub fn response_with_tool_calls(template: &Value, calls: &[ToolCall]) -> Value {
    let mut out = template.clone();
    set_first_choice_message(
        &mut out,
        json!({
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": canonical_tool_calls(calls),
        }),
        "tool_calls",
    );
    out
}

/// Rebuild a response carrying a plain assistant text message (used to unwrap a
/// `respond` call and for fallback-to-last-text on retry exhaustion).
pub fn response_with_text(template: &Value, text: &str) -> Value {
    let mut out = template.clone();
    set_first_choice_message(
        &mut out,
        json!({ "role": "assistant", "content": text }),
        "stop",
    );
    out
}

/// Replace the first choice's `message` and `finish_reason`, synthesising the
/// `choices` array if the template lacks one.
fn set_first_choice_message(resp: &mut Value, message: Value, finish_reason: &str) {
    let choice = resp
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .and_then(|a| a.get_mut(0));
    match choice {
        Some(c) => {
            c["message"] = message;
            c["finish_reason"] = json!(finish_reason);
        }
        None => {
            resp["choices"] =
                json!([{ "index": 0, "message": message, "finish_reason": finish_reason }]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_response(tool_calls: Value) -> Value {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null, "tool_calls": tool_calls},
                "finish_reason": "tool_calls"
            }]
        })
    }

    #[test]
    fn decodes_native_tool_calls() {
        let resp = native_response(json!([{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
        }]));

        let out = decode_response(&resp);
        assert_eq!(
            out,
            ModelOutput::ToolCalls(vec![ToolCall {
                id: Some("call_abc".into()),
                name: "get_weather".into(),
                arguments: "{\"city\":\"Paris\"}".into(),
            }])
        );
    }

    #[test]
    fn decodes_plain_text_when_no_tool_calls() {
        let resp = json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}]
        });
        assert_eq!(decode_response(&resp), ModelOutput::Text("hello".into()));
    }

    #[test]
    fn null_content_decodes_to_empty_text() {
        let resp = json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": null}}]
        });
        assert_eq!(decode_response(&resp), ModelOutput::Text(String::new()));
    }

    #[test]
    fn empty_tool_calls_array_falls_through_to_text() {
        let resp = json!({
            "choices": [{"index": 0, "message": {"content": "fallback", "tool_calls": []}}]
        });
        assert_eq!(decode_response(&resp), ModelOutput::Text("fallback".into()));
    }

    #[test]
    fn arguments_object_is_normalised_to_string() {
        let resp = native_response(json!([{
            "id": "call_1",
            "function": {"name": "f", "arguments": {"a": 1}}
        }]));
        let ModelOutput::ToolCalls(calls) = decode_response(&resp) else {
            panic!("expected tool calls");
        };
        assert_eq!(calls[0].arguments, "{\"a\":1}");
    }

    #[test]
    fn native_tool_calls_round_trip_through_canonical() {
        let original = json!([{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
        }]);
        let resp = native_response(original.clone());

        let ModelOutput::ToolCalls(calls) = decode_response(&resp) else {
            panic!("expected tool calls");
        };
        assert_eq!(canonical_tool_calls(&calls), original);
    }

    #[test]
    fn canonical_mints_id_when_missing() {
        let calls = vec![ToolCall {
            id: None,
            name: "f".into(),
            arguments: "{}".into(),
        }];
        let canonical = canonical_tool_calls(&calls);
        assert_eq!(canonical[0]["id"], json!("call_0"));
        assert_eq!(canonical[0]["type"], json!("function"));
    }

    #[test]
    fn response_with_tool_calls_preserves_template_fields() {
        let template = json!({
            "id": "chatcmpl-7",
            "model": "local-model",
            "usage": {"total_tokens": 9},
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "junk"}}]
        });
        let calls = vec![ToolCall {
            id: Some("call_1".into()),
            name: "f".into(),
            arguments: "{\"a\":1}".into(),
        }];
        let out = response_with_tool_calls(&template, &calls);

        // Top-level metadata preserved.
        assert_eq!(out["id"], "chatcmpl-7");
        assert_eq!(out["model"], "local-model");
        assert_eq!(out["usage"]["total_tokens"], 9);
        // Message replaced with canonical tool calls.
        assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            out["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "f"
        );
        assert_eq!(out["choices"][0]["message"]["content"], Value::Null);
    }

    #[test]
    fn response_with_text_sets_assistant_message() {
        let template = json!({"id": "x", "choices": [{"index": 0, "message": {}}]});
        let out = response_with_text(&template, "hello there");
        assert_eq!(out["id"], "x");
        assert_eq!(out["choices"][0]["message"]["role"], "assistant");
        assert_eq!(out["choices"][0]["message"]["content"], "hello there");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
    }
}
