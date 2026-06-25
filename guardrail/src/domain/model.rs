//! Milestone 2: typed-where-touched request model with lossless passthrough.
//!
//! The guardrails only ever read a few fields of an OpenAI chat-completions
//! request — the tool names (to validate calls against), the messages (to append
//! nudges), and the stream flag. Everything else (sampling params, the verbatim
//! `model` id, future OpenAI fields) must round-trip untouched so the backend
//! sees exactly what the client sent.
//!
//! The design is "type what you touch, flatten the rest": named fields capture
//! the touched parts; a flattened [`Map`] captures everything else. Because
//! serde consumes named fields before populating a `#[serde(flatten)]` map, the
//! typed fields never also appear in `rest` — no duplication on re-emit.
//!
//! Round-trip fidelity is the acceptance criterion for this milestone, and it is
//! *semantic*, not byte-identical: JSON object key order is insignificant, so the
//! tests compare parsed [`Value`]s. No guardrail behaviour is wired into the
//! proxy path yet — this is purely the data model future milestones build on.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// An OpenAI `POST /v1/chat/completions` request body, parsed
/// typed-where-touched. Unknown / untouched fields live in [`rest`] and
/// round-trip losslessly.
///
/// [`rest`]: ChatRequest::rest
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatRequest {
    /// The conversation so far. Kept as raw [`Value`]s — the retry guardrail
    /// appends a nudge message but never needs to inspect message internals, so
    /// over-typing them would only risk fidelity loss.
    pub messages: Vec<Value>,

    /// Tool definitions, when the request is tool-enabled. Absent (rather than
    /// empty) when the client sent no `tools`, so it is skipped on re-emit and
    /// the passthrough path stays clean for plain chat requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// The model id, forwarded verbatim — never rewritten. Required by the
    /// OpenAI schema, so always present.
    pub model: String,

    /// Every other field (temperature, top_p, stream, tool_choice, …) preserved
    /// losslessly. `stream` deliberately lives here rather than as a typed bool:
    /// pulling it out would re-emit an explicit `"stream":false` for requests
    /// that omitted it, changing the body. Read it via [`stream`] instead.
    ///
    /// [`stream`]: ChatRequest::stream
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

impl ChatRequest {
    /// Whether the client asked for a streamed response. Defaults to `false`
    /// when absent, matching the OpenAI default.
    pub fn stream(&self) -> bool {
        self.rest
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Whether the request carries any tool definitions. The guardrail loop only
    /// engages when this is true; otherwise the request is pure passthrough.
    pub fn has_tools(&self) -> bool {
        self.tools.as_ref().is_some_and(|t| !t.is_empty())
    }

    /// Append a tool definition (used to inject the synthetic `respond` tool).
    pub fn push_tool(&mut self, tool: Tool) {
        self.tools.get_or_insert_with(Vec::new).push(tool);
    }

    /// The set of tool names the model is allowed to call, in declaration order.
    /// This is the one piece of the tool definitions the validate guardrail
    /// actually reads.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|t| t.function.name.as_str())
            .collect()
    }
}

/// A single tool definition. Only `type` and `function.name` are typed; anything
/// else a tool carries is preserved in [`rest`].
///
/// [`rest`]: Tool::rest
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    /// Tool kind — `"function"` for every tool OpenAI currently defines.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunction,
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// The `function` object of a tool. `name` is the only field the guardrails
/// touch; `description`, `parameters`, `strict`, etc. ride along in [`rest`].
///
/// [`rest`]: ToolFunction::rest
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Parse `v` into a `ChatRequest`, serialize it back, and assert the result
    /// is semantically identical to the input.
    fn assert_round_trips(v: Value) {
        let req: ChatRequest = serde_json::from_value(v.clone()).expect("parse");
        let back = serde_json::to_value(&req).expect("serialize");
        assert_eq!(back, v, "round-trip changed the request body");
    }

    #[test]
    fn plain_chat_request_round_trips() {
        assert_round_trips(json!({
            "model": "local-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": false
        }));
    }

    #[test]
    fn tool_request_round_trips_with_unknown_fields() {
        // Includes fields we do not type at every level (a tool-level `strict`,
        // a function-level `parameters`, a request-level `tool_choice`, and a
        // hypothetical future field) to prove lossless passthrough.
        assert_round_trips(json!({
            "model": "org/exact-model-id",
            "messages": [{"role": "user", "content": "weather?"}],
            "tool_choice": "auto",
            "future_openai_field": {"nested": [1, 2, 3]},
            "tools": [
                {
                    "type": "function",
                    "strict": true,
                    "function": {
                        "name": "get_weather",
                        "description": "Look up weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }
            ]
        }));
    }

    #[test]
    fn request_without_stream_does_not_gain_one() {
        // Regression guard for the reason `stream` is not a typed field: a
        // request that omits it must not grow a `"stream": false` on re-emit.
        let v = json!({
            "model": "m",
            "messages": [],
        });
        let req: ChatRequest = serde_json::from_value(v.clone()).unwrap();
        assert!(!req.stream());
        let back = serde_json::to_value(&req).unwrap();
        assert_eq!(back.get("stream"), None);
        assert_eq!(back, v);
    }

    #[test]
    fn accessors_read_touched_fields() {
        let req: ChatRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [],
            "stream": true,
            "tools": [
                {"type": "function", "function": {"name": "a"}},
                {"type": "function", "function": {"name": "b"}}
            ]
        }))
        .unwrap();

        assert!(req.stream());
        assert!(req.has_tools());
        assert_eq!(req.tool_names(), vec!["a", "b"]);
    }

    #[test]
    fn no_tools_means_passthrough() {
        let req: ChatRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();

        assert!(!req.has_tools());
        assert!(req.tool_names().is_empty());
    }

    #[test]
    fn model_id_is_preserved_verbatim() {
        let req: ChatRequest = serde_json::from_value(json!({
            "model": "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF",
            "messages": []
        }))
        .unwrap();
        assert_eq!(
            req.model,
            "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF"
        );
    }
}
