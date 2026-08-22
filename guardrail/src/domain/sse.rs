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
use tracing::warn;

use super::decode::ToolCall;

/// One item read off a backend stream.
///
/// A stream can end two ways, and they are not the same event: the backend
/// finished, or the connection died partway through. Collapsing both into
/// "no more lines" makes a truncated answer indistinguishable from a complete
/// one — the client is handed half a sentence, or half a tool call, with a
/// clean terminator and no indication anything went wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamItem {
    /// One complete line of the backend's SSE body.
    Line(String),
    /// The backend closed the stream normally.
    Eof,
    /// The stream failed partway through; the text describes why.
    Failed(String),
}

#[derive(Debug, Clone, Default)]
struct CallSlot {
    id: Option<String>,
    name: String,
    arguments: String,
}

/// The result of processing the complete SSE stream.
#[derive(Debug)]
pub enum AssembledResponse {
    /// Stream contained only text / passthrough content. Chunks were forwarded
    /// to the client via `emit_sse` when the caller asked for live forwarding;
    /// `content` is the same text assembled, which a caller that suppressed
    /// forwarding (a JSON backend, or a retry) needs in order to answer at all.
    Text { template: Value, content: String },
    /// Stream ended with native tool-call deltas (buffered, not forwarded).
    /// `content` holds any text the model also emitted alongside the tool calls
    /// (some models emit XML in content while also producing a native tool call).
    ToolCalls { calls: Vec<ToolCall>, template: Value, content: String },
    /// No native tool calls; accumulated text was parsed by a rescue parser.
    /// `content` is the text the calls were recovered from. A model that wrote
    /// something around its call said it to the user, so it is carried here
    /// rather than dropped — re-emitting only the calls would delete the
    /// answer, and with it any question the model was still asking.
    Rescued {
        parser: &'static str,
        calls: Vec<ToolCall>,
        template: Value,
        content: String,
    },
}

/// Why a stream stopped, beside what it turned out to contain.
///
/// Orthogonal to [`AssembledResponse`]: a stream that died can still have
/// produced text or a partial tool call, and the caller needs both facts —
/// what arrived, and whether it was all of it. Kept separate rather than added
/// as a fourth variant so every existing arm keeps its meaning.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Completion {
    /// The backend closed the stream itself; what arrived is the whole answer.
    #[default]
    Complete,
    /// The stream died partway through. Whatever was assembled is a fragment.
    Truncated(String),
}

impl Completion {
    /// Whether the stream ended early.
    pub fn is_truncated(&self) -> bool {
        matches!(self, Completion::Truncated(_))
    }
}

/// Token usage seen on a stream, independent of what the stream turned out to
/// be. Kept beside [`AssembledResponse`] rather than inside it because usage is
/// reported the same way for text, tool calls, and rescues, and every variant
/// would otherwise have to carry a copy.
pub type StreamUsage = Option<super::metrics::Usage>;

/// Pick the usage report out of a chunk, keeping the richer of the two when a
/// backend reports more than once.
///
/// A stream carries at most one usage block in practice, but backends differ on
/// which chunk holds it (the last content chunk, the `finish_reason` chunk, or a
/// trailing usage-only chunk), so every chunk is checked. `max` on the token
/// count avoids a later, emptier report overwriting a real one — some backends
/// send a zeroed block on the terminal chunk after the real numbers.
fn merge_usage(seen: &mut StreamUsage, chunk: &Value) {
    let Some(usage) = super::metrics::extract_usage(chunk) else { return };
    if usage.is_empty() {
        return;
    }
    let better = match *seen {
        Some(prev) => {
            usage.prompt_tokens + usage.completion_tokens
                > prev.prompt_tokens + prev.completion_tokens
        }
        None => true,
    };
    if better {
        *seen = Some(usage);
    }
}

fn fill_slots(slots: &mut Vec<CallSlot>, tool_calls: &[Value]) {
    for tc in tool_calls {
        let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        if slots.len() <= index { slots.resize_with(index + 1, CallSlot::default); }
        let slot = &mut slots[index];
        if let Some(id) = tc.get("id").and_then(Value::as_str) { slot.id = Some(id.to_string()); }
        if let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) { slot.name = name.to_string(); }
        if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")).and_then(Value::as_str) { slot.arguments.push_str(args); }
    }
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
///
/// Returns the assembled result, whatever token usage the backend reported, and
/// whether the stream actually finished.
pub async fn assemble_stream<F>(
    rx: &mut mpsc::Receiver<StreamItem>,
    mut emit_sse: F,
    kind_tx: Option<mpsc::Sender<bool>>,
) -> (AssembledResponse, StreamUsage, Completion)
where
    F: FnMut(&str),
{
    let mut slots: Vec<CallSlot> = Vec::new();
    let mut template = Value::Null;
    let mut has_tool_calls = false;
    let mut accumulated_text = String::new();
    let mut kind_fired = false;
    let mut usage: StreamUsage = None;
    let mut completion = Completion::Complete;

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
            Some(StreamItem::Line(line)) => line,
            Some(StreamItem::Failed(why)) => {
                // The sender dropping without a verdict is also a failure, but
                // an explicit one carries the reason, so it wins.
                completion = Completion::Truncated(why);
                break;
            }
            Some(StreamItem::Eof) => break,
            // The sender went away without saying which it was. Treating that
            // as success would be the very assumption this type exists to stop.
            None => {
                completion =
                    Completion::Truncated("the backend stream ended without a terminator".into());
                break;
            }
        };

        if line.is_empty() || line.starts_with(':') {
            emit_sse(&format!("{line}\n"));
            continue;
        }

        let Some(chunk) = parse_sse_line(&line) else {
            emit_sse(&format!("{line}\n\n"));
            continue;
        };

        // Usage can ride on any chunk, including ones that carry no delta at
        // all, so it is read before the delta dispatch below.
        merge_usage(&mut usage, &chunk);

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
            fill_slots(&mut slots, tool_calls);
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
        let (named, unnamed): (Vec<_>, Vec<_>) = slots.into_iter().partition(|s| !s.name.is_empty());
        if !unnamed.is_empty() {
            // Incomplete delta stream — the backend never sent a function name
            // for these slots. Log so it is visible; validation will reject.
            warn!(dropped = unnamed.len(), "tool call without a function name can't be validated");
        }
        let calls = named.into_iter()
            .map(|s| ToolCall {
                id: s.id,
                name: s.name,
                arguments: if s.arguments.is_empty() { "{}".to_string() } else { s.arguments },
            })
            .collect();
        return (
            AssembledResponse::ToolCalls { calls, template, content: accumulated_text },
            usage,
            completion,
        );
    }

    if !accumulated_text.is_empty() {
        if let Some((parser, calls)) = crate::domain::rescue::rescue(&accumulated_text) {
            signal(false, &kind_tx); // rescue = treat like tool calls
            return (
                AssembledResponse::Rescued { parser, calls, template, content: accumulated_text },
                usage,
                completion,
            );
        }
    }

    signal(true, &kind_tx); // pure text — signal at EOF
    (AssembledResponse::Text { template, content: accumulated_text }, usage, completion)
}

/// Synchronous version for tests and non-streaming paths.
pub fn assemble<F>(raw_sse: &str, emit_text: F) -> AssembledResponse
where
    F: FnMut(&str),
{
    assemble_with_usage(raw_sse, emit_text).0
}

/// [`assemble`], also returning the token usage the stream reported.
pub fn assemble_with_usage<F>(raw_sse: &str, mut emit_text: F) -> (AssembledResponse, StreamUsage)
where
    F: FnMut(&str),
{
    let mut slots: Vec<CallSlot> = Vec::new();
    let mut template = Value::Null;
    let mut has_tool_calls = false;
    let mut accumulated_text = String::new();
    let mut usage: StreamUsage = None;

    for line in raw_sse.lines() {
        let Some(chunk) = parse_sse_line(line) else {
            if !line.is_empty() && !line.starts_with(':') { emit_text(&format!("{line}\n\n")); }
            continue;
        };
        merge_usage(&mut usage, &chunk);
        let delta = chunk.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("delta"));

        if let Some(tool_calls) = delta.and_then(|d| d.get("tool_calls")).and_then(Value::as_array) {
            has_tool_calls = true;
            fill_slots(&mut slots, tool_calls);
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
        let (named, unnamed): (Vec<_>, Vec<_>) = slots.into_iter().partition(|s| !s.name.is_empty());
        if !unnamed.is_empty() {
            warn!(dropped = unnamed.len(), "tool call without a function name can't be validated");
        }
        let calls = named.into_iter().map(|s| ToolCall {
            id: s.id, name: s.name,
            arguments: if s.arguments.is_empty() { "{}".to_string() } else { s.arguments },
        }).collect();
        return (
            AssembledResponse::ToolCalls { calls, template, content: accumulated_text },
            usage,
        );
    }

    if !accumulated_text.is_empty() {
        if let Some((parser, calls)) = crate::domain::rescue::rescue(&accumulated_text) {
            return (
                AssembledResponse::Rescued { parser, calls, template, content: accumulated_text },
                usage,
            );
        }
    }

    (AssembledResponse::Text { template, content: accumulated_text }, usage)
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
    fn tool_calls_carries_content_emitted_alongside_native_calls() {
        // Some models (e.g. Qwen) emit XML in the content delta while also
        // producing a native tool_calls delta. The assembler should carry the
        // content through so the application layer can rescue from it if the
        // native call fails validation.
        let mut sse = String::new();
        let content_chunk = serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {
                "content": "<function=bash><parameter=command>ls</parameter></function>"
            }, "finish_reason": null}]
        });
        sse.push_str(&format!("data: {}\n\n", content_chunk));
        let tc_chunk = serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "root", "arguments": "{}"}}]
            }, "finish_reason": "tool_calls"}]
        });
        sse.push_str(&format!("data: {}\n\n", tc_chunk));
        sse.push_str("data: [DONE]\n\n");

        let result = assemble(&sse, |_| {});
        let AssembledResponse::ToolCalls { calls, content, .. } = result else {
            panic!("expected ToolCalls");
        };
        assert_eq!(calls[0].name, "root");
        assert!(content.contains("<function=bash>"), "content should carry the XML text");
    }

    #[test]
    fn usage_is_read_off_a_trailing_usage_only_chunk() {
        // The shape OpenAI uses with `stream_options.include_usage`: a final
        // chunk with an empty `choices` array carrying only the usage block.
        let mut sse = String::new();
        let content = serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]
        });
        sse.push_str(&format!("data: {content}\n\n"));
        let usage_chunk = serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk", "choices": [],
            "usage": {
                "prompt_tokens": 12, "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 8}
            }
        });
        sse.push_str(&format!("data: {usage_chunk}\n\n"));
        sse.push_str("data: [DONE]\n\n");

        let (_, usage) = assemble_with_usage(&sse, |_| {});
        let usage = usage.expect("usage from the trailing chunk");
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 4);
        assert_eq!(usage.cached_tokens, 8);
        assert_eq!(usage.attempts, 1);
    }

    #[test]
    fn usage_is_captured_on_tool_call_streams_too() {
        // Tool-call chunks are buffered rather than forwarded, so usage on that
        // path is only observable if the assembler reads it — and tool calls
        // are precisely the traffic the guardrails retry and bill twice.
        let mut sse = String::new();
        let tc = serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}}]
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 6}
        });
        sse.push_str(&format!("data: {tc}\n\ndata: [DONE]\n\n"));

        let (assembled, usage) = assemble_with_usage(&sse, |_| {});
        assert!(matches!(assembled, AssembledResponse::ToolCalls { .. }));
        let usage = usage.expect("usage on a tool-call stream");
        assert_eq!(usage.prompt_tokens, 30);
        assert_eq!(usage.completion_tokens, 6);
    }

    #[test]
    fn a_zeroed_terminal_report_does_not_erase_the_real_one() {
        // Some backends send the real numbers mid-stream and then a zeroed
        // usage block on the terminal chunk. Taking the last one seen would
        // throw the measurement away.
        let mut sse = String::new();
        let real = serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "hi"}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 9}
        });
        sse.push_str(&format!("data: {real}\n\n"));
        let zeroed = serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk", "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        });
        sse.push_str(&format!("data: {zeroed}\n\ndata: [DONE]\n\n"));

        let (_, usage) = assemble_with_usage(&sse, |_| {});
        let usage = usage.expect("the real report must survive");
        assert_eq!(usage.prompt_tokens, 40);
        assert_eq!(usage.completion_tokens, 9);
    }

    #[test]
    fn a_stream_without_usage_reports_none() {
        let sse = text_chunks(&["Hello"]);
        let (_, usage) = assemble_with_usage(&sse, |_| {});
        assert_eq!(usage, None, "no usage block means no measurement");
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
