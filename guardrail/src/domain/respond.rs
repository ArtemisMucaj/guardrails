//! Synthetic `respond` tool.

use serde_json::{json, Value};

use super::decode::ToolCall;
use super::model::Tool;

/// Name of the injected tool.
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

/// Whether a tool call targets the `respond` tool.
pub fn is_respond(call: &ToolCall) -> bool {
    call.name == RESPOND
}

/// Extract the user-facing text from a `respond` call's arguments. Accepts
/// `message` (canonical) and the `content`/`text` aliases. Returns `None` when
/// the arguments are unparseable or carry no recognized text field, so the
/// caller can retry or fall back rather than emit a blank response.
pub fn message_text(call: &ToolCall) -> Option<String> {
    serde_json::from_str::<Value>(&call.arguments)
        .ok()
        .and_then(|v| {
            ["message", "content", "text"]
                .iter()
                .find_map(|k| v.get(*k).and_then(Value::as_str).map(str::to_string))
        })
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
        assert_eq!(
            message_text(&call("respond", "{\"message\":\"hi\"}")).as_deref(),
            Some("hi")
        );
        assert_eq!(
            message_text(&call("respond", "{\"content\":\"yo\"}")).as_deref(),
            Some("yo")
        );
        assert_eq!(
            message_text(&call("respond", "{\"text\":\"sup\"}")).as_deref(),
            Some("sup")
        );
    }

    #[test]
    fn missing_or_bad_arguments_yield_none() {
        assert_eq!(message_text(&call("respond", "{}")), None);
        assert_eq!(message_text(&call("respond", "not json")), None);
    }
}
