//! Rescue parsing: recover structured tool calls from model text output.

use serde_json::Value;

use super::decode::ToolCall;

/// A format-specific recogniser for tool calls embedded in model text.
pub trait RescueParser: Send + Sync {
    /// Stable identifier used in logs.
    fn name(&self) -> &'static str;
    /// Attempt to extract tool calls from `text`.
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>>;
}

/// The parsers tried, in order.
pub fn default_parsers() -> Vec<Box<dyn RescueParser>> {
    vec![
        Box::new(Lfm),
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
/// with the parser's name.
pub fn rescue(text: &str) -> Option<(&'static str, Vec<ToolCall>)> {
    for parser in default_parsers() {
        if let Some(calls) = parser.try_parse(text) {
            return Some((parser.name(), calls));
        }
    }
    None
}

// ── Shared JSON interpretation ──────────────────────────────────────────────

/// Interpret a JSON value as one or more tool calls.
fn tool_calls_from_value(v: &Value) -> Option<Vec<ToolCall>> {
    match v {
        Value::Array(items) => {
            // All-or-nothing: if any entry is malformed, reject the whole batch
            // rather than silently dropping it before validation sees it.
            let calls: Vec<ToolCall> = items
                .iter()
                .map(tool_call_from_value)
                .collect::<Option<_>>()?;
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
/// arguments|parameters}` shape, and forge's `{tool, args}` shape.
fn tool_call_from_value(v: &Value) -> Option<ToolCall> {
    let obj = v.as_object()?;

    let (name, args) = match obj.get("function").and_then(Value::as_object) {
        Some(func) => (func.get("name"), func.get("arguments")),
        None => (
            // Accepts `tool` or `name`, and `args` or `arguments`.
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

/// Parse the first JSON value out of `s`, ignoring any trailing text.
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

/// Parse every `<tag>JSON</tag>` block as tool calls.
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

/// LiquidAI LFM2 / LFM2.5: tool calls wrapped in `<|tool_call_start|>` …
/// `<|tool_call_end|>`. The inner payload is either a Python list of call
/// expressions — `[get_weather(location="Paris"), get_time(zone="UTC")]` — or,
/// when the model is asked to "Output function calls as JSON", a JSON
/// object/array. Natural-language text may follow the closing token; the rescue
/// path drops it when re-emitting canonical `tool_calls`, matching every other
/// parser here.
pub struct Lfm;
const LFM_CALL_START: &str = "<|tool_call_start|>";
const LFM_CALL_END: &str = "<|tool_call_end|>";
impl RescueParser for Lfm {
    fn name(&self) -> &'static str {
        "lfm"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        let start = text.find(LFM_CALL_START)?;
        let after = &text[start + LFM_CALL_START.len()..];
        // Prefer the explicit closing token; tolerate its absence by cutting at
        // the next special token (e.g. `<|im_end|>`), so a truncated or
        // differently-wrapped turn still yields the call span.
        let inner = match after.find(LFM_CALL_END) {
            Some(end) => &after[..end],
            None => after.split("<|").next().unwrap_or(after),
        }
        .trim();

        // JSON mode first (cheaper and unambiguous): a single object or an array
        // of `{name, arguments|parameters}`. Fall back to Pythonic call syntax.
        if let Some(v) = first_json_value(inner) {
            if let Some(calls) = tool_calls_from_value(&v) {
                return Some(calls);
            }
        }
        parse_pythonic_calls(inner)
    }
}

/// Mistral: `[TOOL_CALLS]` followed by a JSON list/object, or the flatter
/// `[TOOL_CALLS]name{args}` form.
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

/// Rehearsal syntax: `name[ARGS]{...}`.
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
/// Parameter values are coerced to JSON scalars/objects when they parse, else kept as strings.
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

/// Hermes: `<function_call>{...}</function_call>` blocks.
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

/// The entire text is a tool-call JSON value.
pub struct BareJson;
impl RescueParser for BareJson {
    fn name(&self) -> &'static str {
        "bare_json"
    }
    fn try_parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        // Require the entire response to be JSON. Accepting a valid prefix would
        // let prose that merely starts with a tool-shaped example be re-emitted
        // as real tool_calls.
        let v: Value = serde_json::from_str(text.trim()).ok()?;
        tool_calls_from_value(&v)
    }
}

// ── Pythonic call parsing (LFM) ─────────────────────────────────────────────

/// Parse a Python call expression — `name(k=v, ...)`, optionally a list of them
/// `[name(...), name2(...)]` — into tool calls. Keyword arguments become the
/// call's JSON-object arguments; Python literals (`True`/`False`/`None`,
/// numbers, strings, lists, dicts) are coerced to their JSON equivalents.
///
/// This is a small dedicated parser rather than `eval`/`literal_eval`: the
/// payload is a *call*, which neither accepts, and we must not execute it.
/// Returns `None` on any malformed input so the caller passes content through
/// untouched instead of fabricating a call.
fn parse_pythonic_calls(s: &str) -> Option<Vec<ToolCall>> {
    let mut p = PyParser::new(s);
    p.skip_ws();
    let calls = if p.peek() == Some('[') {
        p.bump();
        let mut calls = Vec::new();
        p.skip_ws();
        if p.peek() != Some(']') {
            loop {
                calls.push(p.parse_call()?);
                p.skip_ws();
                match p.peek() {
                    Some(',') => {
                        p.bump();
                        p.skip_ws();
                        // Allow a trailing comma before the closing bracket.
                        if p.peek() == Some(']') {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }
        p.expect(']')?;
        calls
    } else {
        vec![p.parse_call()?]
    };
    p.skip_ws();
    // The whole payload must be calls; trailing junk means we misread it.
    if !p.at_end() {
        return None;
    }
    (!calls.is_empty()).then_some(calls)
}

/// A cursor over the Pythonic payload, parsed character by character.
struct PyParser {
    chars: Vec<char>,
    pos: usize,
}

impl PyParser {
    fn new(s: &str) -> Self {
        Self {
            chars: s.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, c: char) -> Option<()> {
        (self.bump() == Some(c)).then_some(())
    }

    /// `ident ( arg, arg, ... )`.
    fn parse_call(&mut self) -> Option<ToolCall> {
        self.skip_ws();
        let name = self.parse_ident()?;
        if name.is_empty() {
            return None;
        }
        self.skip_ws();
        self.expect('(')?;
        let mut args = serde_json::Map::new();
        self.skip_ws();
        if self.peek() != Some(')') {
            loop {
                let (key, value) = self.parse_keyword_arg()?;
                args.insert(key, value);
                self.skip_ws();
                match self.peek() {
                    Some(',') => {
                        self.bump();
                        self.skip_ws();
                        // Allow a trailing comma before the closing paren.
                        if self.peek() == Some(')') {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }
        self.expect(')')?;
        Some(ToolCall {
            id: None,
            name,
            arguments: Value::Object(args).to_string(),
        })
    }

    /// `key=value`. Only keyword arguments are supported: without parameter
    /// names, positional values cannot be mapped into a JSON object.
    fn parse_keyword_arg(&mut self) -> Option<(String, Value)> {
        let key = self.parse_ident()?;
        if key.is_empty() {
            return None;
        }
        self.skip_ws();
        self.expect('=')?;
        self.skip_ws();
        let value = self.parse_value()?;
        Some((key, value))
    }

    fn parse_ident(&mut self) -> Option<String> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') {
            self.pos += 1;
        }
        (self.pos > start).then(|| self.chars[start..self.pos].iter().collect())
    }

    fn parse_value(&mut self) -> Option<Value> {
        match self.peek()? {
            '"' | '\'' => self.parse_string().map(Value::String),
            '[' => self.parse_list(),
            '{' => self.parse_dict(),
            c if c == '-' || c.is_ascii_digit() => self.parse_number(),
            _ => self.parse_keyword(),
        }
    }

    /// A quoted string with the common Python escapes.
    fn parse_string(&mut self) -> Option<String> {
        let quote = self.bump()?;
        let mut out = String::new();
        loop {
            match self.bump()? {
                c if c == quote => return Some(out),
                '\\' => match self.bump()? {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    '\\' => out.push('\\'),
                    '\'' => out.push('\''),
                    '"' => out.push('"'),
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                },
                c => out.push(c),
            }
        }
    }

    fn parse_number(&mut self) -> Option<Value> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
        {
            self.pos += 1;
        }
        let raw: String = self.chars[start..self.pos].iter().collect();
        serde_json::from_str::<Value>(&raw)
            .ok()
            .filter(Value::is_number)
    }

    /// `True` / `False` / `None`, mapped to JSON `true` / `false` / `null`.
    fn parse_keyword(&mut self) -> Option<Value> {
        match self.parse_ident()?.as_str() {
            "True" => Some(Value::Bool(true)),
            "False" => Some(Value::Bool(false)),
            "None" => Some(Value::Null),
            _ => None,
        }
    }

    fn parse_list(&mut self) -> Option<Value> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() != Some(']') {
            loop {
                items.push(self.parse_value()?);
                self.skip_ws();
                match self.peek() {
                    Some(',') => {
                        self.bump();
                        self.skip_ws();
                        // Allow a trailing comma before the closing bracket.
                        if self.peek() == Some(']') {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }
        self.expect(']')?;
        Some(Value::Array(items))
    }

    fn parse_dict(&mut self) -> Option<Value> {
        self.expect('{')?;
        let mut map = serde_json::Map::new();
        self.skip_ws();
        if self.peek() != Some('}') {
            loop {
                // JSON object keys must be strings; a non-string Python key is
                // stringified via its JSON rendering.
                let key = match self.parse_value()? {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                self.skip_ws();
                self.expect(':')?;
                self.skip_ws();
                let value = self.parse_value()?;
                map.insert(key, value);
                self.skip_ws();
                match self.peek() {
                    Some(',') => {
                        self.bump();
                        self.skip_ws();
                        // Allow a trailing comma before the closing brace.
                        if self.peek() == Some('}') {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }
        self.expect('}')?;
        Some(Value::Object(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn one(calls: &[ToolCall]) -> (&str, &str) {
        (calls[0].name.as_str(), calls[0].arguments.as_str())
    }

    #[test]
    fn lfm_pythonic_single_call() {
        let text = "<|tool_call_start|>[get_weather(location=\"Paris\")]<|tool_call_end|>Checking the weather in Paris.<|im_end|>";
        let calls = Lfm.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("get_weather", "{\"location\":\"Paris\"}"));
    }

    #[test]
    fn lfm_pythonic_multiple_calls() {
        let text =
            "<|tool_call_start|>[get_weather(location=\"Paris\"), get_time(zone=\"UTC\")]<|tool_call_end|>";
        let calls = Lfm.try_parse(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"location\":\"Paris\"}");
        assert_eq!(calls[1].name, "get_time");
        assert_eq!(calls[1].arguments, "{\"zone\":\"UTC\"}");
    }

    #[test]
    fn lfm_pythonic_single_call_without_list_brackets() {
        let text = "<|tool_call_start|>get_weather(location=\"Paris\")<|tool_call_end|>";
        let calls = Lfm.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("get_weather", "{\"location\":\"Paris\"}"));
    }

    #[test]
    fn lfm_pythonic_coerces_argument_types() {
        let text = "<|tool_call_start|>[f(n=3, ratio=1.5, ok=True, off=False, nothing=None, tags=[\"a\", \"b\"], opts={\"deep\": 1})]<|tool_call_end|>";
        let calls = Lfm.try_parse(text).unwrap();
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["n"], json!(3));
        assert_eq!(args["ratio"], json!(1.5));
        assert_eq!(args["ok"], json!(true));
        assert_eq!(args["off"], json!(false));
        assert_eq!(args["nothing"], Value::Null);
        assert_eq!(args["tags"], json!(["a", "b"]));
        assert_eq!(args["opts"], json!({"deep": 1}));
    }

    #[test]
    fn lfm_pythonic_trailing_commas() {
        // Python permits trailing commas in calls, lists, and dicts.
        let text =
            "<|tool_call_start|>[f(x=1,), g(items=[\"a\",], opts={\"k\": 1,}),]<|tool_call_end|>";
        let calls = Lfm.try_parse(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, "{\"x\":1}");
        let args: Value = serde_json::from_str(&calls[1].arguments).unwrap();
        assert_eq!(args["items"], json!(["a"]));
        assert_eq!(args["opts"], json!({"k": 1}));
    }

    #[test]
    fn lfm_pythonic_no_args() {
        let text = "<|tool_call_start|>[list_files()]<|tool_call_end|>";
        let calls = Lfm.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("list_files", "{}"));
    }

    #[test]
    fn lfm_pythonic_single_quoted_string_with_escape() {
        let text = "<|tool_call_start|>[say(text='it\\'s fine')]<|tool_call_end|>";
        let calls = Lfm.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("say", "{\"text\":\"it's fine\"}"));
    }

    #[test]
    fn lfm_json_mode() {
        // With "Output function calls as JSON" the payload is JSON, not Pythonic.
        let text = "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]<|tool_call_end|>";
        let calls = Lfm.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("get_weather", "{\"location\":\"Paris\"}"));
    }

    #[test]
    fn lfm_tolerates_missing_end_token() {
        let text = "<|tool_call_start|>[get_weather(location=\"Paris\")]<|im_end|>";
        let calls = Lfm.try_parse(text).unwrap();
        assert_eq!(one(&calls), ("get_weather", "{\"location\":\"Paris\"}"));
    }

    #[test]
    fn lfm_absent_token_is_not_rescued() {
        assert!(Lfm
            .try_parse("Just some prose about get_weather(location).")
            .is_none());
    }

    #[test]
    fn lfm_malformed_payload_is_not_fabricated() {
        // A truncated call must not yield a bogus tool call.
        let text = "<|tool_call_start|>[get_weather(location=<|tool_call_end|>";
        assert!(Lfm.try_parse(text).is_none());
    }

    #[test]
    fn rescue_dispatches_lfm() {
        let (parser, calls) =
            rescue("<|tool_call_start|>[get_weather(location=\"Paris\")]<|tool_call_end|>")
                .unwrap();
        assert_eq!(parser, "lfm");
        assert_eq!(calls[0].name, "get_weather");
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
        // A rescued call re-emits canonically.
        let calls = Qwen
            .try_parse("<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}</tool_call>")
            .unwrap();
        let canonical = crate::domain::decode::canonical_tool_calls(&calls);
        assert_eq!(canonical[0]["function"]["name"], "f");
        assert_eq!(canonical[0]["function"]["arguments"], "{\"a\":1}");
    }
}
