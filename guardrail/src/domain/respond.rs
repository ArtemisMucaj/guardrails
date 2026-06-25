//! Milestone 5: the synthetic `respond` tool.
//!
//! Small models are more reliable when *every* turn is a tool call than when
//! they must choose between "call a tool" and "write prose". So when the respond
//! guardrail is on we inject a `respond` tool into the request: the model uses it
//! to deliver a final natural-language answer through the same tool-call channel.
//! On the way back we detect that call and **strip it to text** — the client sees
//! an ordinary assistant message, never the synthetic tool.

use serde_json::{json, Value};

use super::decode::ToolCall;
use super::model::Tool;

/// Name of the injected tool. Chosen to be unlikely to collide with a real tool.
pub const RESPOND: &str = "respond";

/// The tool definition to inject into a request's `tools` array.
pub fn respond_tool() -> Tool {
    serde_json::from_value(json!({
        "type": "function",
        "function": {
            "name": RESPOND,
            "description": "Reply to the user with a final natural-language message \
                            when no other tool is needed. Prefer this over answering \
                            in plain text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to show the user."
                    }
                },
                "required": ["message"]
            }
        }
    }))
    .expect("respond tool definition is valid")
}

/// Whether a tool call targets the synthetic `respond` tool.
pub fn is_respond(call: &ToolCall) -> bool {
    call.name == RESPOND
}

/// Extract the user-facing text from a `respond` call's arguments. Accepts
/// `message` (canonical) and the `content`/`text` aliases small models drift to;
/// missing/unparseable arguments yield an empty string rather than failing.
pub fn message_text(call: &ToolCall) -> String {
    serde_json::from_str::<Value>(&call.arguments)
        .ok()
        .and_then(|v| {
            ["message", "content", "text"]
                .iter()
                .find_map(|k| v.get(*k).and_then(Value::as_str).map(str::to_string))
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: None,
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    #[test]
    fn respond_tool_is_a_valid_named_function() {
        let tool = respond_tool();
        assert_eq!(tool.kind, "function");
        assert_eq!(tool.function.name, RESPOND);
    }

    #[test]
    fn detects_respond_call() {
        assert!(is_respond(&call("respond", "{}")));
        assert!(!is_respond(&call("get_weather", "{}")));
    }

    #[test]
    fn extracts_message_and_aliases() {
        assert_eq!(message_text(&call("respond", "{\"message\":\"hi\"}")), "hi");
        assert_eq!(message_text(&call("respond", "{\"content\":\"yo\"}")), "yo");
        assert_eq!(message_text(&call("respond", "{\"text\":\"sup\"}")), "sup");
    }

    #[test]
    fn missing_or_bad_arguments_yield_empty() {
        assert_eq!(message_text(&call("respond", "{}")), "");
        assert_eq!(message_text(&call("respond", "not json")), "");
    }
}
