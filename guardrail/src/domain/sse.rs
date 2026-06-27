//! SSE stream assembler for OpenAI chat-completion chunks.
//!
//! The assembler processes `chat.completion.chunk` events one by one:
//!
//! - **Text / passthrough deltas** are forwarded to the client immediately via
//!   the `emit_sse` callback — zero extra latency.
//! - **Tool-call deltas** (`delta.tool_calls`) are accumulated in memory. When
//!   the stream ends the assembled calls are returned for validation and repair.
//! - **Rescue**: if no native tool calls appear but the accumulated text matches
//!   a rescue parser, a `Rescued` result is returned instead of `Text`.

use tokio::sync::mpsc;
use serde_json::Value;

use super::decode::ToolCall;

#[derive(Debug, Clone, Default)]
struct CallSlot {
    id: Option<String>,
    name: String,
    arguments: String,
}

/// The result of processing the complete SSE stream.
#[derive(Debug)]
pub enum AssembledResponse {
    /// Stream contained only text / passthrough content. All chunks were
    /// already forwarded to the client via `emit_sse`.
    Text { template: Value },
    /// Stream ended with native tool-call deltas (buffered, not forwarded).
    ToolCalls { calls: Vec<ToolCall>, template: Value },
    /// No native tool calls; accumulated text was parsed by a rescue parser.
    Rescued { parser: &'static str, calls: Vec<ToolCall>, template: Value },
}

/// Parse a single `data:` SSE line into a JSON value.
/// Returns `None` for blank lines, comments, and the `[DONE]` sentinel.
pub fn parse_sse_line(line: &str) -> Option<Value> {
    let data = line.strip_prefix("data:")?;
    let data = data.trim();
    if data == "[DONE]" {
        return None;
    }
    serde_json::from_str(data).ok()
}

/// Consume an SSE line receiver, assembling the stream.
///
/// `emit_sse` is called immediately for text/passthrough lines. Tool-call lines
/// are buffered and NOT forwarded. `kind_tx` receives `false` the moment the
/// first tool-call delta is seen — allowing the caller to switch to buffered
/// mode before returning a response. For text streams, `kind_tx` receives `true`
/// at EOF (after rescue detection).
pub async fn assemble_stream<F>(
    rx: &mut mpsc::Receiver<Option<String>>,
    mut emit_sse: F,
    kind_tx: Option<mpsc::Sender<bool>>,
) -> AssembledResponse
where
    F: FnMut(&str),
{
    let mut slots: Vec<CallSlot> = Vec::new();
    let mut template = Value::Null;
    let mut has_tool_calls = false;
    let mut accumulated_text = String::new();
    let mut kind_fired = false;

    let mut signal = |is_text: bool, tx: &Option<mpsc::Sender<bool>>| {
        if !kind_fired {
            kind_fired = true;
            if let Some(t) = tx {
                let _ = t.try_send(is_text);
            }
        }
    };

    loop {
        let line = match rx.recv().await {
            Some(Some(line)) => line,
            Some(None) | None => break,
        };

        if line.is_empty() || line.starts_with(':') {
            emit_sse(&format!("{line}\n"));
            continue;
        }

        let Some(chunk) = parse_sse_line(&line) else {
            emit_sse(&format!("{line}\n\n"));
            continue;
        };

        let delta = chunk
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"));

        // ── Tool-call delta ── buffer, signal early, do NOT forward ──────────
        if let Some(tool_calls) = delta
            .and_then(|d| d.get("tool_calls"))
            .and_then(Value::as_array)
        {
            signal(false, &kind_tx); // fire early — caller knows it's tool calls
            has_tool_calls = true;
            for tc in tool_calls {
                let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                if slots.len() <= index { slots.resize_with(index + 1, CallSlot::default); }
                let slot = &mut slots[index];
                if let Some(id) = tc.get("id").and_then(Value::as_str) { slot.id = Some(id.to_string()); }
                if let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) { slot.name = name.to_string(); }
                if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")).and_then(Value::as_str) { slot.arguments.push_str(args); }
            }
            template = chunk;
            continue;
        }

        // ── Text content delta ── accumulate and forward immediately ─────────
        if let Some(content) = delta
            .and_then(|d| d.get("content"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            accumulated_text.push_str(content);
            if !has_tool_calls {
                emit_sse(&format!("{line}\n\n"));
            }
            template = chunk;
            continue;
        }

        // ── Passthrough (thinking, role, finish_reason, usage, etc.) ─────────
        if !chunk.is_null() { template = chunk.clone(); }
        if !has_tool_calls { emit_sse(&format!("{line}\n\n")); }
    }

    if has_tool_calls && !slots.is_empty() {
        // kind was already signalled false (early) on first tool-call delta.
        let calls = slots.into_iter()
            .filter(|s| !s.name.is_empty())
            .map(|s| ToolCall {
                id: s.id,
                name: s.name,
                arguments: if s.arguments.is_empty() { "{}".to_string() } else { s.arguments },
            })
            .collect();
        return AssembledResponse::ToolCalls { calls, template };
    }

    if !accumulated_text.is_empty() {
        if let Some((parser, calls)) = crate::domain::rescue::rescue(&accumulated_text) {
            signal(false, &kind_tx); // rescue = treat like tool calls
            return AssembledResponse::Rescued { parser, calls, template };
        }
    }

    signal(true, &kind_tx); // pure text — signal at EOF
    AssembledResponse::Text { template }
}

/// Synchronous version for tests and non-streaming paths.
pub fn assemble<F>(raw_sse: &str, mut emit_text: F) -> AssembledResponse
where
    F: FnMut(&str),
{
    let mut slots: Vec<CallSlot> = Vec::new();
    let mut template = Value::Null;
    let mut has_tool_calls = false;
    let mut accumulated_text = String::new();

    for line in raw_sse.lines() {
        let Some(chunk) = parse_sse_line(line) else {
            if !line.is_empty() && !line.starts_with(':') { emit_text(&format!("{line}\n\n")); }
            continue;
        };
        let delta = chunk.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("delta"));

        if let Some(tool_calls) = delta.and_then(|d| d.get("tool_calls")).and_then(Value::as_array) {
            has_tool_calls = true;
            for tc in tool_calls {
                let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                if slots.len() <= index { slots.resize_with(index + 1, CallSlot::default); }
                let slot = &mut slots[index];
                if let Some(id) = tc.get("id").and_then(Value::as_str) { slot.id = Some(id.to_string()); }
                if let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) { slot.name = name.to_string(); }
                if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")).and_then(Value::as_str) { slot.arguments.push_str(args); }
            }
            template = chunk;
            continue;
        }

        if let Some(content) = delta.and_then(|d| d.get("content")).and_then(Value::as_str).filter(|s| !s.is_empty()) {
            accumulated_text.push_str(content);
            if !has_tool_calls { emit_text(&format!("{line}\n\n")); }
            template = chunk;
            continue;
        }

        if !chunk.is_null() { template = chunk.clone(); }
        if !has_tool_calls { emit_text(&format!("{line}\n\n")); }
    }

    if has_tool_calls && !slots.is_empty() {
        let calls = slots.into_iter().filter(|s| !s.name.is_empty()).map(|s| ToolCall {
            id: s.id, name: s.name,
            arguments: if s.arguments.is_empty() { "{}".to_string() } else { s.arguments },
        }).collect();
        return AssembledResponse::ToolCalls { calls, template };
    }

    if !accumulated_text.is_empty() {
        if let Some((parser, calls)) = crate::domain::rescue::rescue(&accumulated_text) {
            return AssembledResponse::Rescued { parser, calls, template };
        }
    }

    AssembledResponse::Text { template }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_chunks(content_pieces: &[&str]) -> String {
        let mut out = String::new();
        for (i, piece) in content_pieces.iter().enumerate() {
            let chunk = serde_json::json!({
                "id": "chatcmpl-1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]
            });
            out.push_str(&format!("data: {}\n\n", chunk));
            if i == content_pieces.len() - 1 {
                let done_chunk = serde_json::json!({
                    "id": "chatcmpl-1", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                });
                out.push_str(&format!("data: {}\n\n", done_chunk));
            }
        }
        out.push_str("data: [DONE]\n\n");
        out
    }

    fn tool_call_chunks(id: &str, name: &str, args_pieces: &[&str]) -> String {
        let mut out = String::new();
        let chunk = serde_json::json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": id, "type": "function",
                    "function": {"name": name, "arguments": args_pieces.first().unwrap_or(&"")}}]
            }, "finish_reason": null}]
        });
        out.push_str(&format!("data: {}\n\n", chunk));
        for piece in args_pieces.iter().skip(1) {
            let chunk = serde_json::json!({
                "id": "chatcmpl-1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": piece}}]
                }, "finish_reason": null}]
            });
            out.push_str(&format!("data: {}\n\n", chunk));
        }
        let finish = serde_json::json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        });
        out.push_str(&format!("data: {}\n\n", finish));
        out.push_str("data: [DONE]\n\n");
        out
    }

    #[test]
    fn text_stream_is_forwarded_and_returns_text() {
        let sse = text_chunks(&["Hello", ", world", "!"]);
        let mut forwarded = Vec::new();
        let result = assemble(&sse, |line| forwarded.push(line.to_string()));
        assert!(matches!(result, AssembledResponse::Text { .. }));
        assert_eq!(forwarded.iter().filter(|l| l.contains("Hello")).count(), 1);
        assert_eq!(forwarded.iter().filter(|l| l.contains(", world")).count(), 1);
        assert_eq!(forwarded.iter().filter(|l| l.contains('!')).count(), 1);
    }

    #[test]
    fn tool_call_stream_is_assembled_and_not_forwarded() {
        let sse = tool_call_chunks("call_1", "get_weather", &["{\"city\":", "\"Paris\"}"]);
        let mut forwarded = Vec::new();
        let result = assemble(&sse, |line| forwarded.push(line.to_string()));
        let AssembledResponse::ToolCalls { calls, .. } = result else { panic!("expected ToolCalls") };
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"city\":\"Paris\"}");
    }

    #[test]
    fn multiple_tool_calls_assembled() {
        let mut sse = String::new();
        for chunk in [
            serde_json::json!({"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"id0","type":"function","function":{"name":"foo","arguments":"{\"a\":"}}]},"finish_reason":null}]}),
            serde_json::json!({"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"id1","type":"function","function":{"name":"bar","arguments":"{\"b\":"}}]},"finish_reason":null}]}),
            serde_json::json!({"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":null}]}),
            serde_json::json!({"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]},"finish_reason":"tool_calls"}]}),
        ] {
            sse.push_str(&format!("data: {}\n\n", chunk));
        }
        sse.push_str("data: [DONE]\n\n");
        let AssembledResponse::ToolCalls { calls, .. } = assemble(&sse, |_| {}) else { panic!("expected ToolCalls") };
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "foo");
        assert_eq!(calls[0].arguments, "{\"a\":1}");
        assert_eq!(calls[1].name, "bar");
        assert_eq!(calls[1].arguments, "{\"b\":2}");
    }

    #[test]
    fn empty_arguments_defaults_to_empty_object() {
        let sse = tool_call_chunks("id0", "list_files", &[""]);
        let AssembledResponse::ToolCalls { calls, .. } = assemble(&sse, |_| {}) else { panic!("expected ToolCalls") };
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn done_sentinel_is_not_parsed_as_chunk() {
        let sse = "data: [DONE]\n\n";
        let result = assemble(sse, |_| {});
        assert!(matches!(result, AssembledResponse::Text { .. }));
    }

    #[test]
    fn rescue_format_in_text_stream_is_detected() {
        let content = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
        let chunk = serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}]
        });
        let sse = format!("data: {}\n\ndata: [DONE]\n\n", chunk);
        let AssembledResponse::Rescued { parser, calls, .. } = assemble(&sse, |_| {}) else {
            panic!("expected Rescued");
        };
        assert_eq!(parser, "qwen");
        assert_eq!(calls[0].name, "get_weather");
    }
}
