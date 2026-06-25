//! Milestone 4: rescue parsing.
//!
//! When a small model emits a tool call in the *wrong* place — as text in the
//! `content` field rather than as native `tool_calls` — [`decode`] yields
//! [`ModelOutput::Text`]. Rescue tries, in order, a set of format-specific
//! parsers to recover structured [`ToolCall`]s from that text. Each parser
//! targets one model family's malformed output and is independently unit-tested.
//!
//! The families covered are forge's rescue formats plus the native delimiters of
//! the families a transparent proxy actually sees (Anthropic's wire format is
//! out of scope here):
//!
//! | Parser | Format |
//! |---|---|
//! | [`Mistral`] | `[TOOL_CALLS][{...}]` or `[TOOL_CALLS]name{args}` |
//! | [`Rehearsal`] | `name[ARGS]{...}` (reasoning models) |
//! | [`QwenCoder`] | `<function=name><parameter=k>v</parameter></function>` |
//! | [`Qwen`] | `<tool_call>{...}</tool_call>` |
//! | [`Hermes`] | `<function_call>{...}</function_call>` |
//! | [`Llama`] | `<|python_tag|>{...}` |
//! | [`FencedJson`] | a ```json fenced code block |
//! | [`BareJson`] | the whole text is a tool-call JSON value |
//!
//! All JSON-shaped parsers funnel through [`tool_calls_from_value`], which
//! interprets the common shapes: forge's `{tool, args}`, `{name, arguments}`,
//! `{name, parameters}`, OpenAI's `{type, function:{...}}`, arrays, and
//! `{tool_calls:[...]}` wrappers.
//!
//! [`decode`]: crate::decode::decode_response
//! [`ModelOutput::Text`]: crate::decode::ModelOutput::Text

use serde_json::Value;

use super::decode::ToolCall;

/// A format-specific recogniser for tool calls embedded in model text. Parsers
/// are cheap to try and return `None` when the text isn't in their format.
pub trait RescueParser: Send + Sync {
    /// Stable identifier, used in logs to record which parser fired.
    fn name(&self) -> &'static str;
    /// Attempt to extract tool calls from `text`.
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>>;
}

/// The parsers tried, in order. Distinctive-delimiter formats come first;
/// [`BareJson`] is last because it is the most permissive and would otherwise
/// shadow the others.
pub fn default_parsers() -> Vec<Box<dyn RescueParser>> {
    vec![
        Box::new(Mistral),
        Box::new(Rehearsal),
        Box::new(QwenCoder),
        Box::new(Qwen),
        Box::new(Hermes),
        Box::new(Llama),
        Box::new(FencedJson),
        Box::new(BareJson),
    ]
}

/// Try every parser in [`default_parsers`] and return the first match, along
/// with the parser's name (for logging).
pub fn rescue(text: &str) -> Option<(&'static str, Vec<ToolCall>)> {
    for parser in default_parsers() {
        if let Some(calls) = parser.try_parse(text) {
            return Some((parser.name(), calls));
        }
    }
    None
}

// ── Shared JSON interpretation ──────────────────────────────────────────────

/// Interpret a JSON value as one or more tool calls across the shapes small
/// models emit. Returns `None` if nothing call-shaped is present.
fn tool_calls_from_value(v: &Value) -> Option<Vec<ToolCall>> {
    match v {
        Value::Array(items) => {
            let calls: Vec<ToolCall> = items.iter().filter_map(tool_call_from_value).collect();
            (!calls.is_empty()).then_some(calls)
        }
        Value::Object(map) => {
            if let Some(inner) = map.get("tool_calls") {
                return tool_calls_from_value(inner);
            }
            tool_call_from_value(v).map(|c| vec![c])
        }
        _ => None,
    }
}

/// Interpret a single JSON object as one tool call. Accepts the OpenAI
/// `{type, function:{name, arguments}}` shape, the flatter `{name,
/// arguments|parameters}` shape, and forge's `{tool, args}` shape (the format
/// forge's prompt injection elicits). `arguments` is normalised to a
/// JSON-encoded string for lossless re-emit.
fn tool_call_from_value(v: &Value) -> Option<ToolCall> {
    let obj = v.as_object()?;

    let (name, args) = match obj.get("function").and_then(Value::as_object) {
        Some(func) => (func.get("name"), func.get("arguments")),
        None => (
            // forge accepts `tool` or `name`, and `args` or `arguments`.
            obj.get("name").or_else(|| obj.get("tool")),
            obj.get("arguments")
                .or_else(|| obj.get("args"))
                .or_else(|| obj.get("parameters")),
        ),
    };

    let name = name?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }

    let arguments = match args {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };

    Some(ToolCall {
        id: None,
        name,
        arguments,
    })
}

/// Parse the first JSON value out of `s`, ignoring any trailing text. Useful
/// when a delimiter is followed by JSON plus trailing tokens/prose.
fn first_json_value(s: &str) -> Option<Value> {
    serde_json::Deserializer::from_str(s)
        .into_iter::<Value>()
        .next()?
        .ok()
}

/// Collect the inner text of every `<tag>...</tag>` pair in `text`.
fn extract_tagged(text: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(&open) {
        let after = &rest[start + open.len()..];
        match after.find(&close) {
            Some(end) => {
                out.push(after[..end].to_string());
                rest = &after[end + close.len()..];
            }
            None => break,
        }
    }
    out
}

/// Parse every `<tag>JSON</tag>` block as tool calls. Shared by the XML-ish
/// parsers (Qwen, Hermes).
fn parse_tagged(text: &str, tag: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    for inner in extract_tagged(text, tag) {
        if let Some(v) = first_json_value(inner.trim()) {
            if let Some(mut found) = tool_calls_from_value(&v) {
                calls.append(&mut found);
            }
        }
    }
    (!calls.is_empty()).then_some(calls)
}

// ── Parsers ─────────────────────────────────────────────────────────────────

/// Mistral: `[TOOL_CALLS]` followed by a JSON list/object, or the flatter
/// `[TOOL_CALLS]name{args}` form some quantizations emit.
pub struct Mistral;
const MISTRAL_TOKEN: &str = "[TOOL_CALLS]";
impl RescueParser for Mistral {
    fn name(&self) -> &'static str {
        "mistral"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        let idx = text.find(MISTRAL_TOKEN)?;
        let rest = text[idx + MISTRAL_TOKEN.len()..].trim_start();

        // Preferred: a JSON array/object directly after the token.
        if let Some(v) = first_json_value(rest) {
            if let Some(calls) = tool_calls_from_value(&v) {
                return Some(calls);
            }
        }

        // Fallback: `name{args}`.
        let brace = rest.find('{')?;
        let name = rest[..brace].trim();
        if name.is_empty() || name.contains(char::is_whitespace) {
            return None;
        }
        let args = first_json_value(&rest[brace..])?;
        Some(vec![ToolCall {
            id: None,
            name: name.to_string(),
            arguments: args.to_string(),
        }])
    }
}

/// Rehearsal syntax used by some reasoning models: `name[ARGS]{...}`.
/// Mirrors forge's `(\w+)\[ARGS\](\{.*\})`.
pub struct Rehearsal;
const ARGS_MARKER: &str = "[ARGS]";
impl RescueParser for Rehearsal {
    fn name(&self) -> &'static str {
        "rehearsal"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        let marker = text.find(ARGS_MARKER)?;
        // Name is the identifier immediately preceding `[ARGS]`.
        let name: String = text[..marker]
            .chars()
            .rev()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if name.is_empty() {
            return None;
        }
        let args = first_json_value(text[marker + ARGS_MARKER.len()..].trim_start())?;
        if !args.is_object() {
            return None;
        }
        Some(vec![ToolCall {
            id: None,
            name,
            arguments: args.to_string(),
        }])
    }
}

/// Qwen-Coder XML: `<function=name><parameter=key>value</parameter>...</function>`.
/// Parameter values are coerced to JSON scalars/objects when they parse, else
/// kept as strings. Mirrors forge's `_QWEN_FUNCTION_RE` / `_QWEN_PARAMETER_RE`.
pub struct QwenCoder;
impl RescueParser for QwenCoder {
    fn name(&self) -> &'static str {
        "qwen_coder"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        let mut calls = Vec::new();
        for (name, inner) in extract_function_blocks(text) {
            let mut args = serde_json::Map::new();
            // Split on the opening of each `<parameter=`; the first chunk is the
            // text before any parameter and is skipped.
            for chunk in inner.split("<parameter=").skip(1) {
                if let Some((key, rest)) = chunk.split_once('>') {
                    let value = rest.split("</parameter>").next().unwrap_or(rest).trim();
                    args.insert(key.trim().to_string(), coerce_param(value));
                }
            }
            calls.push(ToolCall {
                id: None,
                name,
                arguments: Value::Object(args).to_string(),
            });
        }
        (!calls.is_empty()).then_some(calls)
    }
}

/// Collect `(name, inner)` for each `<function=name>...</function>` block.
fn extract_function_blocks(text: &str) -> Vec<(String, String)> {
    const OPEN: &str = "<function=";
    const CLOSE: &str = "</function>";
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let after = &rest[start + OPEN.len()..];
        let Some(gt) = after.find('>') else { break };
        let name = after[..gt].trim().to_string();
        let body = &after[gt + 1..];
        let Some(end) = body.find(CLOSE) else { break };
        if !name.is_empty() {
            out.push((name, body[..end].to_string()));
        }
        rest = &body[end + CLOSE.len()..];
    }
    out
}

/// Coerce a Qwen-Coder parameter value to JSON: a parseable scalar/object stays
/// typed, anything else becomes a string.
fn coerce_param(value: &str) -> Value {
    serde_json::from_str::<Value>(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

/// Qwen: one or more `<tool_call>{...}</tool_call>` blocks.
pub struct Qwen;
impl RescueParser for Qwen {
    fn name(&self) -> &'static str {
        "qwen"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        parse_tagged(text, "tool_call")
    }
}

/// Hermes (legacy): `<function_call>{...}</function_call>` blocks.
pub struct Hermes;
impl RescueParser for Hermes {
    fn name(&self) -> &'static str {
        "hermes"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        parse_tagged(text, "function_call")
    }
}

/// Llama 3.x: `<|python_tag|>` followed by a JSON call, optionally terminated by
/// a special token (`<|eom_id|>` / `<|eot_id|>`).
pub struct Llama;
const PYTHON_TAG: &str = "<|python_tag|>";
impl RescueParser for Llama {
    fn name(&self) -> &'static str {
        "llama"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        let idx = text.find(PYTHON_TAG)?;
        let rest = &text[idx + PYTHON_TAG.len()..];
        // Cut at the next special token if present (e.g. <|eom_id|>).
        let json_part = rest.split("<|").next().unwrap_or(rest).trim();
        let v = first_json_value(json_part)?;
        tool_calls_from_value(&v)
    }
}

/// A fenced code block (```json … ``` or bare ``` … ```) containing tool-call
/// JSON.
pub struct FencedJson;
impl RescueParser for FencedJson {
    fn name(&self) -> &'static str {
        "fenced_json"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        for block in fenced_blocks(text) {
            if let Some(v) = first_json_value(block.trim()) {
                if let Some(calls) = tool_calls_from_value(&v) {
                    return Some(calls);
                }
            }
        }
        None
    }
}

/// Return the body of each ``` fenced block, stripping an optional language tag
/// line (e.g. `json`).
fn fenced_blocks(text: &str) -> Vec<String> {
    let parts: Vec<&str> = text.split("```").collect();
    let mut blocks = Vec::new();
    let mut i = 1;
    while i < parts.len() {
        let seg = parts[i];
        let body = match seg.split_once('\n') {
            Some((first, rest))
                if !first.trim().is_empty()
                    && first.trim().chars().all(|c| c.is_ascii_alphanumeric()) =>
            {
                rest
            }
            _ => seg,
        };
        blocks.push(body.to_string());
        i += 2;
    }
    blocks
}

/// The entire text is a tool-call JSON value. Most permissive; tried last.
pub struct BareJson;
impl RescueParser for BareJson {
    fn name(&self) -> &'static str {
        "bare_json"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        let v = first_json_value(text.trim())?;
        tool_calls_from_value(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(calls: &[ToolCall]) -> (&str, &str) {
        (calls[0].name.as_str(), calls[0].arguments.as_str())
    }

    #[test]
    fn mistral_json_array() {
        let text =
            "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]";
        let calls = Mistral.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("get_weather", "{\"city\":\"Paris\"}"));
    }

    #[test]
    fn mistral_name_brace_args() {
        let text = "[TOOL_CALLS]get_weather{\"city\": \"Paris\"}";
        let calls = Mistral.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("get_weather", "{\"city\":\"Paris\"}"));
    }

    #[test]
    fn qwen_single_and_multiple() {
        let text = "<tool_call>{\"name\": \"a\", \"arguments\": {\"x\": 1}}</tool_call>\n\
                    <tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>";
        let calls = Qwen.try_parse(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn hermes_function_call() {
        let text = "sure!<function_call>{\"name\": \"search\", \"arguments\": {\"q\": \"rust\"}}</function_call>";
        let calls = Hermes.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("search", "{\"q\":\"rust\"}"));
    }

    #[test]
    fn llama_python_tag_with_parameters_and_eom() {
        let text = "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}<|eom_id|>";
        let calls = Llama.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("get_weather", "{\"city\":\"Paris\"}"));
    }

    #[test]
    fn fenced_json_with_lang_tag() {
        let text = "Here you go:\n```json\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n```";
        let calls = FencedJson.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("f", "{\"a\":1}"));
    }

    #[test]
    fn fenced_json_without_lang_tag() {
        let text = "```\n{\"name\": \"f\", \"arguments\": {}}\n```";
        let calls = FencedJson.try_parse(text).unwrap();
        assert_eq!(calls[0].name, "f");
    }

    #[test]
    fn bare_json_openai_function_shape() {
        let text = "{\"type\": \"function\", \"function\": {\"name\": \"f\", \"arguments\": \"{\\\"a\\\":1}\"}}";
        let calls = BareJson.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("f", "{\"a\":1}"));
    }

    #[test]
    fn bare_json_forge_tool_args_shape() {
        // forge's prompt-injected shape: {"tool": ..., "args": ...}.
        let calls = BareJson
            .try_parse("{\"tool\": \"get_weather\", \"args\": {\"city\": \"Paris\"}}")
            .unwrap();
        assert_eq!(one(&calls), ("get_weather", "{\"city\":\"Paris\"}"));
    }

    #[test]
    fn rehearsal_name_args_marker() {
        let calls = Rehearsal
            .try_parse("thinking... get_weather[ARGS]{\"city\": \"Paris\"}")
            .unwrap();
        assert_eq!(one(&calls), ("get_weather", "{\"city\":\"Paris\"}"));
    }

    #[test]
    fn qwen_coder_function_parameter_xml() {
        let text = "<function=get_weather><parameter=city>Paris</parameter><parameter=days>3</parameter></function>";
        let calls = QwenCoder.try_parse(text).unwrap();
        assert_eq!(calls[0].name, "get_weather");
        // String value stays a string; numeric value is coerced.
        assert_eq!(calls[0].arguments, "{\"city\":\"Paris\",\"days\":3}");
    }

    #[test]
    fn qwen_coder_parameter_without_closing_tag() {
        // Last parameter need not close before </function>.
        let text = "<function=f><parameter=x>hello</function>";
        let calls = QwenCoder.try_parse(text).unwrap();
        assert_eq!(calls[0].arguments, "{\"x\":\"hello\"}");
    }

    #[test]
    fn rescue_prefers_qwen_coder_over_bare_json() {
        let (parser, _) =
            rescue("<function=get_weather><parameter=city>Paris</parameter></function>").unwrap();
        assert_eq!(parser, "qwen_coder");
    }

    #[test]
    fn rescue_dispatches_and_reports_parser() {
        let (parser, calls) =
            rescue("<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>").unwrap();
        assert_eq!(parser, "qwen");
        assert_eq!(calls[0].name, "a");
    }

    #[test]
    fn plain_prose_is_not_rescued() {
        assert!(rescue("I'm not sure which tool to use, can you clarify?").is_none());
        assert!(rescue("").is_none());
    }

    #[test]
    fn json_without_a_name_is_not_a_tool_call() {
        // A bare data object must not be mistaken for a call.
        assert!(BareJson.try_parse("{\"city\": \"Paris\"}").is_none());
    }

    #[test]
    fn arguments_as_object_round_trip_through_canonical() {
        // A rescued call should re-emit canonically (used by M6).
        let calls = Qwen
            .try_parse("<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}</tool_call>")
            .unwrap();
        let canonical = crate::domain::decode::canonical_tool_calls(&calls);
        assert_eq!(canonical[0]["function"]["name"], "f");
        assert_eq!(canonical[0]["function"]["arguments"], "{\"a\":1}");
    }
}
