//! Assemble a streamed Responses API reply.
//!
//! The Responses stream is a different protocol from Chat Completions': every
//! event is typed, and a tool call arrives across three of them —
//! `response.output_item.added` announces it, `response.function_call_arguments.delta`
//! carries the arguments in pieces, and `response.output_item.done` closes it.
//! Text arrives as `response.output_text.delta`.
//!
//! The policy is the same as the chat assembler's, because it is the proxy's
//! policy and not the protocol's: text is forwarded the instant it arrives, tool
//! calls are buffered silently so they can be checked and repaired before the
//! client sees anything it would act on.

use serde_json::Value;
use tokio::sync::mpsc;

use super::decode::ToolCall;
use super::rescue;
use super::sse::{parse_sse_line, Completion, StreamItem};

/// Event announcing a new output item (message, reasoning, or function call).
const ITEM_ADDED: &str = "response.output_item.added";
/// Event carrying a slice of a function call's arguments.
const ARGS_DELTA: &str = "response.function_call_arguments.delta";
/// Event carrying the complete arguments of a function call.
const ARGS_DONE: &str = "response.function_call_arguments.done";
/// Event carrying a slice of assistant text.
const TEXT_DELTA: &str = "response.output_text.delta";
/// Terminal event carrying the assembled response.
const COMPLETED: &str = "response.completed";

/// What a streamed Responses reply turned out to be.
#[derive(Debug, Clone, PartialEq)]
pub enum AssembledResponses {
    /// Only text; already forwarded live.
    Text { text: String, template: Value },
    /// Native `function_call` items, buffered rather than forwarded.
    ToolCalls {
        calls: Vec<ToolCall>,
        template: Value,
        text: String,
    },
    /// Tool calls recovered from the model's text by a rescue parser.
    /// `text` is what they were recovered from — kept rather than dropped, so
    /// re-emitting the call does not delete the model's own answer.
    Rescued {
        parser: &'static str,
        calls: Vec<ToolCall>,
        template: Value,
        text: String,
    },
}

/// One function call being accumulated across events.
#[derive(Default, Clone)]
struct CallSlot {
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Consume a Responses SSE stream.
///
/// `emit_sse` is called immediately for text and other passthrough events;
/// function-call events are buffered and never forwarded. `kind_tx` receives
/// `false` the moment the first function call is seen, so the caller can switch
/// to buffered mode before it has committed to a streaming response.
pub async fn assemble_responses_stream<F>(
    rx: &mut mpsc::Receiver<StreamItem>,
    mut emit_sse: F,
    kind_tx: Option<mpsc::Sender<bool>>,
) -> (AssembledResponses, super::sse::StreamUsage, Completion)
where
    F: FnMut(&str),
{
    let mut slots: Vec<CallSlot> = Vec::new();
    let mut template = Value::Null;
    let mut text = String::new();
    let mut has_calls = false;
    let mut kind_fired = false;
    let mut usage: super::sse::StreamUsage = None;
    let mut completion = Completion::Complete;
    // Text deltas withheld because the text so far looks like a tool call.
    // Released if the stream ends without one being recovered.
    let mut buffered_text: Vec<String> = Vec::new();

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
                completion = Completion::Truncated(why);
                break;
            }
            Some(StreamItem::Eof) => break,
            None => {
                completion =
                    Completion::Truncated("the backend stream ended without a terminator".into());
                break;
            }
        };

        // Blank lines and comments frame the stream; a `event:` line names the
        // type but the `data:` payload repeats it, so only the latter is read.
        if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
            if !has_calls {
                emit_sse(&format!("{line}\n"));
            }
            continue;
        }

        let Some(event) = parse_sse_line(&line) else {
            if !has_calls {
                emit_sse(&format!("{line}\n\n"));
            }
            continue;
        };

        let kind = event.get("type").and_then(Value::as_str).unwrap_or_default();

        match kind {
            ITEM_ADDED => {
                let item = event.get("item");
                let is_call = item
                    .and_then(|i| i.get("type"))
                    .and_then(Value::as_str)
                    == Some(super::responses::FUNCTION_CALL);
                if is_call {
                    has_calls = true;
                    signal(false, &kind_tx);
                    let index = event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(slots.len() as u64) as usize;
                    if slots.len() <= index {
                        slots.resize(index + 1, CallSlot::default());
                    }
                    let slot = &mut slots[index];
                    slot.call_id = item
                        .and_then(|i| i.get("call_id"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    slot.name = item
                        .and_then(|i| i.get("name"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    // Some servers include the whole arguments string here.
                    if let Some(args) = item
                        .and_then(|i| i.get("arguments"))
                        .and_then(Value::as_str)
                    {
                        slot.arguments.push_str(args);
                    }
                    continue;
                }
                if !has_calls {
                    emit_sse(&format!("{line}\n\n"));
                }
            }

            ARGS_DELTA => {
                has_calls = true;
                signal(false, &kind_tx);
                let index = event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                if slots.len() <= index {
                    slots.resize(index + 1, CallSlot::default());
                }
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    slots[index].arguments.push_str(delta);
                }
            }

            ARGS_DONE => {
                has_calls = true;
                let index = event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                if slots.len() <= index {
                    slots.resize(index + 1, CallSlot::default());
                }
                // `done` carries the complete arguments; prefer it over the
                // accumulated deltas, which a dropped event could have holed.
                if let Some(args) = event.get("arguments").and_then(Value::as_str) {
                    slots[index].arguments = args.to_string();
                }
            }

            TEXT_DELTA => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    text.push_str(delta);
                }
                // Held back once the accumulated text starts looking like a
                // tool call. Models that emit one as text open with a marker
                // (`<tool_call>`, a fence, `{"name":`), so waiting for the
                // stream to end before deciding would stall every plain answer;
                // watching for the marker keeps ordinary text streaming while
                // making sure raw call syntax is never forwarded and then
                // contradicted by a rescued call.
                if !has_calls && !rescue::looks_like_tool_call(&text) {
                    signal(true, &kind_tx);
                    for held in buffered_text.drain(..) {
                        emit_sse(&held);
                    }
                    emit_sse(&format!("{line}\n\n"));
                } else if !has_calls {
                    buffered_text.push(format!("{line}\n\n"));
                }
            }

            COMPLETED => {
                if let Some(response) = event.get("response") {
                    // The Responses protocol reports usage once, on the
                    // terminal event's response object.
                    if let Some(reported) = super::metrics::extract_usage(response) {
                        if !reported.is_empty() {
                            usage = Some(reported);
                        }
                    }
                    template = response.clone();
                }
                // Never forwarded. The caller emits its own terminal event once
                // the output is final; passing the backend's through would give
                // the client a `response.completed` carrying the *unrepaired*
                // output, which a client reading the last completed event would
                // take as the answer.
            }

            _ => {
                if !has_calls {
                    emit_sse(&format!("{line}\n\n"));
                }
            }
        }
    }

    if has_calls {
        let calls: Vec<ToolCall> = slots
            .into_iter()
            .filter_map(|slot| {
                Some(ToolCall {
                    id: slot.call_id,
                    name: slot.name?,
                    arguments: if slot.arguments.is_empty() {
                        "{}".to_string()
                    } else {
                        slot.arguments
                    },
                })
            })
            .collect();
        if !calls.is_empty() {
            return (
                AssembledResponses::ToolCalls {
                    calls,
                    template,
                    text,
                },
                usage,
                completion,
            );
        }
    }

    // No native call: the model may still have written one into its text, the
    // same failure the chat path rescues.
    if let Some((parser, calls)) = rescue::rescue(&text) {
        signal(false, &kind_tx);
        return (
            AssembledResponses::Rescued {
                parser,
                calls,
                template,
                text,
            },
            usage,
            completion,
        );
    }

    // Nothing was recovered after all, so the held-back text was ordinary
    // prose that merely resembled a call. Release it rather than dropping it.
    signal(true, &kind_tx);
    for held in buffered_text.drain(..) {
        emit_sse(&held);
    }
    (AssembledResponses::Text { text, template }, usage, completion)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `lines` through the assembler, returning the result and whatever
    /// was forwarded to the client.
    async fn assemble(lines: &[&str]) -> (AssembledResponses, String) {
        let (assembled, forwarded, _usage) = assemble_usage(lines).await;
        (assembled, forwarded)
    }

    /// [`assemble`], also returning the usage the stream reported.
    async fn assemble_usage(
        lines: &[&str],
    ) -> (AssembledResponses, String, super::super::sse::StreamUsage) {
        let (tx, mut rx) = mpsc::channel::<StreamItem>(64);
        for line in lines {
            tx.send(StreamItem::Line((*line).to_string())).await.unwrap();
        }
        tx.send(StreamItem::Eof).await.unwrap();
        drop(tx);

        let mut forwarded = String::new();
        let (assembled, usage, _) =
            assemble_responses_stream(&mut rx, |s| forwarded.push_str(s), None).await;
        (assembled, forwarded, usage)
    }

    #[tokio::test]
    async fn text_deltas_are_forwarded_live_and_accumulated() {
        let (assembled, forwarded) = assemble(&[
            r#"data: {"type":"response.output_text.delta","delta":"Hel"}"#,
            r#"data: {"type":"response.output_text.delta","delta":"lo"}"#,
        ])
        .await;

        match assembled {
            AssembledResponses::Text { text, .. } => assert_eq!(text, "Hello"),
            other => panic!("expected text, got {other:?}"),
        }
        assert!(forwarded.contains("Hel"), "text must reach the client live");
    }

    #[tokio::test]
    async fn a_function_call_is_assembled_across_events_and_not_forwarded() {
        let (assembled, forwarded) = assemble(&[
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"get_weather"}}"#,
            r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"city\":"}"#,
            r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"Paris\"}"}"#,
        ])
        .await;

        match assembled {
            AssembledResponses::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
                assert_eq!(calls[0].id.as_deref(), Some("call_1"));
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
        assert!(
            !forwarded.contains("get_weather"),
            "an unchecked tool call must not reach the client: {forwarded}"
        );
    }

    #[tokio::test]
    async fn the_done_event_wins_over_accumulated_deltas() {
        // A dropped delta would leave the accumulated string holed; `done`
        // carries the whole thing, so it is authoritative.
        let (assembled, _) = assemble(&[
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"f"}}"#,
            r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"a\":"}"#,
            r#"data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"a\":1}"}"#,
        ])
        .await;

        match assembled {
            AssembledResponses::ToolCalls { calls, .. } => {
                assert_eq!(calls[0].arguments, r#"{"a":1}"#);
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn several_calls_are_kept_apart_by_output_index() {
        let (assembled, _) = assemble(&[
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c0","name":"first"}}"#,
            r#"data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c1","name":"second"}}"#,
            r#"data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"b\":2}"}"#,
            r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"a\":1}"}"#,
        ])
        .await;

        match assembled {
            AssembledResponses::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].name, "first");
                assert_eq!(calls[0].arguments, r#"{"a":1}"#);
                assert_eq!(calls[1].name, "second");
                assert_eq!(calls[1].arguments, r#"{"b":2}"#);
            }
            other => panic!("expected two calls, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn the_completed_event_supplies_the_template() {
        let (assembled, _) = assemble(&[
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"f"}}"#,
            r#"data: {"type":"response.completed","response":{"id":"resp_9","model":"m","output":[]}}"#,
        ])
        .await;

        match assembled {
            AssembledResponses::ToolCalls { template, .. } => {
                assert_eq!(template["id"], "resp_9");
                assert_eq!(template["model"], "m");
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_tool_call_written_into_text_is_rescued() {
        // The failure this proxy exists for, on the Responses path too.
        let (assembled, _) = assemble(&[
            r#"data: {"type":"response.output_text.delta","delta":"<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>"}"#,
        ])
        .await;

        match assembled {
            AssembledResponses::Rescued { calls, .. } => {
                assert_eq!(calls[0].name, "get_weather");
            }
            other => panic!("expected a rescue, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn the_kind_signal_reports_a_tool_call_before_the_stream_ends() {
        // The caller needs to stop streaming before it commits to a response.
        let (tx, mut rx) = mpsc::channel::<StreamItem>(16);
        let (kind_tx, mut kind_rx) = mpsc::channel::<bool>(4);

        tx.send(StreamItem::Line(r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"f"}}"#.to_string())).await.unwrap();
        tx.send(StreamItem::Eof).await.unwrap();
        drop(tx);

        let _ = assemble_responses_stream(&mut rx, |_| {}, Some(kind_tx)).await;
        assert_eq!(kind_rx.recv().await, Some(false), "false = not plain text");
    }

    #[tokio::test]
    async fn unrecognised_events_pass_through_on_a_text_stream() {
        // The proxy must not swallow events it does not model.
        let (_, forwarded) = assemble(&[
            r#"data: {"type":"response.created","response":{"id":"r"}}"#,
            r#"data: {"type":"response.output_text.delta","delta":"hi"}"#,
        ])
        .await;
        assert!(forwarded.contains("response.created"));
    }
}
