//! Semantic precondition checks on tool calls.
//!
//! These run after the model produces a tool call but *before* the
//! repair/validate loop, so a failed precondition short-circuits the whole
//! retry budget and returns a clear explanation to the model immediately.
//!
//! Two rules are enforced, in order:
//!
//! 1. **Write-on-existing** — a write-only tool must not target a path that
//!    already exists on disk. This is stateless: the filesystem is the
//!    authority, so the rule holds on the first turn of a conversation.
//! 2. **Read-before-mutate** — a tool that edits a file in place must not
//!    target a path the transcript never shows being read. This one needs the
//!    conversation, because "did the model look at this file?" is a fact about
//!    the exchange and not about the disk.
//!
//! The second rule is why models corrupt files: an `Edit` is applied against
//! what the model *remembers* the file containing, and a model that never read
//! it is remembering a guess. The write-on-existing rule cannot catch that —
//! `Edit` on an existing file is exactly the correct call, and the disk cannot
//! say whether the model knows what is in it.
//!
//! # Failing open
//!
//! The transcript rule refuses only on positive evidence: a mutating call whose
//! path is absent from a transcript that *is* present and *does* contain
//! recognisable tool traffic. Anything short of that passes.
//!
//! That is deliberate. A client may trim old history, summarise it, or use tool
//! names this file has never heard of, and in each case the read may well have
//! happened where this scan cannot see it. Refusing there would block correct
//! work with a nudge the model cannot act on — it would read the file, be told
//! again to read the file, and burn the turn. A missed refusal costs one bad
//! edit that the model can still be corrected on; a false refusal costs the
//! conversation. The same reasoning the conversation grouping uses: fail toward
//! no finding rather than a wrong one.

use std::collections::HashSet;

use super::decode::ToolCall;
use serde_json::Value;

/// Write-only tools that must not target an already-existing file.
///
/// Each entry is `(tool_name, harness)`. The name match is case-insensitive.
/// When the model calls one of these tools on a path that already exists, the
/// proxy intercepts the call and instructs the model to read the file first and
/// then use the corresponding edit tool instead of overwriting it blindly.
///
/// Sources:
/// - `Write`       — Claude Code
/// - `write`       — OpenCode, Pi (earendil-works/pi)
/// - `write_file`  — Zed AI
/// - `create`      — GitHub Copilot CLI
const WRITE_TOOLS: &[(&str, &str)] = &[
    ("Write", "Claude Code"),
    ("write", "OpenCode, Pi"),
    ("write_file", "Zed AI"),
    ("create", "GitHub Copilot CLI"),
];

/// Tools that rewrite part of an existing file, leaving the rest in place.
///
/// A call to one of these is only as good as the model's knowledge of the
/// current contents: the old string it matches on, the line it targets, or the
/// hunk context it patches against all have to describe what is actually on
/// disk. That knowledge comes from having read the file.
///
/// Sources:
/// - `Edit`, `MultiEdit` — Claude Code
/// - `edit`              — OpenCode, Pi, GitHub Copilot CLI
/// - `apply_patch`       — OpenCode
/// - `edit_file`         — Zed AI
const EDIT_TOOLS: &[(&str, &str)] = &[
    ("Edit", "Claude Code"),
    ("MultiEdit", "Claude Code"),
    ("edit", "OpenCode, Pi, GitHub Copilot CLI"),
    ("apply_patch", "OpenCode"),
    ("edit_file", "Zed AI"),
];

/// Tools whose call means the model has seen a file's contents.
///
/// Only whole-file readers count. A `Grep` returns matching lines and a `Glob`
/// returns names, so neither tells the model what an `Edit`'s context looks
/// like — treating them as reads would license exactly the blind edit this rule
/// exists to stop.
///
/// Sources:
/// - `Read`      — Claude Code
/// - `read`      — OpenCode, Pi, GitHub Copilot CLI
/// - `read_file` — Zed AI
const READ_TOOLS: &[&str] = &["Read", "read", "read_file"];

/// Outcome of a precondition check.
pub enum Precondition {
    /// All preconditions satisfied; proceed normally.
    Ok,
    /// A precondition failed. `nudge` is the explanation to return to the
    /// model as a plain assistant text message.
    Failed { nudge: String },
}

/// The files a conversation shows having been read.
///
/// Built once per guardrail loop from the request's message array, then
/// consulted per tool call. An empty set is ambiguous — it means either "no
/// read happened" or "this scan understood nothing here" — so it is paired with
/// [`Transcript::legible`] to tell those apart.
#[derive(Debug, Default, Clone)]
pub struct Transcript {
    read_paths: HashSet<String>,
    legible: bool,
}

impl Transcript {
    /// An unavailable transcript. Every read-before-mutate check passes.
    pub fn unavailable() -> Self {
        Self::default()
    }

    /// Scan a message array for reads.
    ///
    /// Accepts both wire shapes without distinguishing them: Chat Completions
    /// `messages[]` carries an assistant message with `tool_calls[]`, while a
    /// Responses `input[]` carries flat `function_call` items. Both are walked
    /// by looking for the name/arguments pair wherever it appears, so one
    /// scanner serves both APIs.
    pub fn of(messages: &[Value]) -> Self {
        let mut read_paths = HashSet::new();
        let mut legible = false;
        for message in messages {
            for (name, arguments) in tool_invocations(message) {
                legible = true;
                if !READ_TOOLS.iter().any(|t| t.eq_ignore_ascii_case(name)) {
                    continue;
                }
                if let Some(path) = file_path_arg(arguments) {
                    read_paths.insert(path);
                }
            }
        }
        Self {
            read_paths,
            legible,
        }
    }

    /// Whether the scan recognised any tool traffic at all.
    ///
    /// False for a transcript that is absent, is pure chat, or names its tools
    /// in a vocabulary this file does not know. In each case the absence of a
    /// read is not evidence that none happened, so the rule stands down.
    pub fn legible(&self) -> bool {
        self.legible
    }

    fn has_read(&self, path: &str) -> bool {
        self.read_paths.contains(path)
    }
}

/// Check semantic preconditions for `calls`.
///
/// `transcript` describes the conversation the calls came from; pass
/// [`Transcript::unavailable`] when there is none to inspect, which leaves only
/// the filesystem rule in effect.
pub fn check(calls: &[ToolCall], transcript: &Transcript) -> Precondition {
    for call in calls {
        let Some(path) = file_path_arg(&call.arguments) else {
            continue;
        };

        if is_write_tool(&call.name) {
            if let Ok(meta) = std::fs::metadata(&path) {
                let nudge = if meta.is_dir() {
                    format!(
                        "\"{path}\" is a directory, not a file. \
                         Provide the full path to the specific file you want to create."
                    )
                } else {
                    format!(
                        "The file \"{path}\" already exists. \
                         Read it first to understand its current contents, \
                         then use the edit tool to make your changes."
                    )
                };
                return Precondition::Failed { nudge };
            }
            continue;
        }

        if is_edit_tool(&call.name) && transcript.legible() && !transcript.has_read(&path) {
            // A path that is not on disk is a different mistake — the edit will
            // fail on its own terms, with a message from the harness that says
            // more than this one would. Only guard edits to files that exist.
            if !std::fs::metadata(&path).is_ok_and(|m| m.is_file()) {
                continue;
            }
            return Precondition::Failed {
                nudge: format!(
                    "The file \"{path}\" has not been read in this conversation. \
                     Read it first so the edit matches its current contents, \
                     then make your change."
                ),
            };
        }
    }
    Precondition::Ok
}

fn is_write_tool(name: &str) -> bool {
    WRITE_TOOLS
        .iter()
        .any(|(tool, _)| tool.eq_ignore_ascii_case(name))
}

fn is_edit_tool(name: &str) -> bool {
    EDIT_TOOLS
        .iter()
        .any(|(tool, _)| tool.eq_ignore_ascii_case(name))
}

/// Every `(name, arguments)` pair a transcript message invokes.
///
/// One message can hold several: a Chat Completions assistant turn may call two
/// tools at once. A Responses `function_call` item names the pair at the top
/// level instead of nesting it under `function`, so both spellings are read.
fn tool_invocations(message: &Value) -> Vec<(&str, &str)> {
    let mut found = Vec::new();
    let Some(obj) = message.as_object() else {
        return found;
    };

    // Responses: a flat `function_call` item.
    if let (Some(name), Some(arguments)) = (
        obj.get("name").and_then(Value::as_str),
        obj.get("arguments").and_then(Value::as_str),
    ) {
        found.push((name, arguments));
    }

    // Chat Completions: an assistant message with `tool_calls[]`.
    if let Some(calls) = obj.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let Some(function) = call.get("function") else {
                continue;
            };
            if let (Some(name), Some(arguments)) = (
                function.get("name").and_then(Value::as_str),
                function.get("arguments").and_then(Value::as_str),
            ) {
                found.push((name, arguments));
            }
        }
    }

    found
}

/// Extract the `file_path` or `path` string argument from a raw JSON arguments
/// string. Returns `None` if the arguments are not a valid object or neither
/// key is present.
fn file_path_arg(arguments: &str) -> Option<String> {
    let obj: Value = serde_json::from_str(arguments).ok()?;
    let map = obj.as_object()?;
    map.get("file_path")
        .or_else(|| map.get("path"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::decode::ToolCall;
    use serde_json::json;

    /// Exists on every Unix system, so an "the file is there" case needs no
    /// fixture to set up or clean away.
    const EXISTING: &str = "/etc/hosts";
    const MISSING: &str = "/tmp/guardrail_test_nonexistent_xyz.txt";

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: None,
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    /// A Chat Completions assistant turn calling one tool.
    fn assistant_call(name: &str, arguments: Value) -> Value {
        json!({
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": arguments.to_string()},
            }],
        })
    }

    /// A Responses `input[]` function-call item calling one tool.
    fn responses_call(name: &str, arguments: Value) -> Value {
        json!({
            "type": "function_call",
            "call_id": "call_1",
            "name": name,
            "arguments": arguments.to_string(),
        })
    }

    fn none() -> Transcript {
        Transcript::unavailable()
    }

    // ── Write-on-existing (the filesystem rule) ──────────────────────────────

    #[test]
    fn non_write_tool_always_passes() {
        let calls = vec![call("Read", r#"{"file_path":"/etc/passwd"}"#)];
        assert!(matches!(check(&calls, &none()), Precondition::Ok));
    }

    #[test]
    fn write_tool_on_nonexistent_file_passes() {
        let calls = vec![call(
            "Write",
            &format!(r#"{{"file_path":"{MISSING}","content":"hi"}}"#),
        )];
        assert!(matches!(check(&calls, &none()), Precondition::Ok));
    }

    #[test]
    fn write_tool_on_existing_file_fails() {
        let calls = vec![call("Write", r#"{"file_path":"/etc/hosts","content":"x"}"#)];
        assert!(matches!(
            check(&calls, &none()),
            Precondition::Failed { .. }
        ));
    }

    #[test]
    fn case_insensitive_tool_name_match() {
        let calls = vec![call("WRITE", r#"{"file_path":"/etc/hosts","content":"x"}"#)];
        assert!(matches!(
            check(&calls, &none()),
            Precondition::Failed { .. }
        ));
    }

    #[test]
    fn path_key_accepted_as_alternative() {
        let calls = vec![call("create", r#"{"path":"/etc/hosts","content":"x"}"#)];
        assert!(matches!(
            check(&calls, &none()),
            Precondition::Failed { .. }
        ));
    }

    #[test]
    fn nudge_mentions_file_and_edit() {
        let calls = vec![call("write", r#"{"file_path":"/etc/hosts","content":"x"}"#)];
        let Precondition::Failed { nudge } = check(&calls, &none()) else {
            panic!("expected Failed");
        };
        assert!(nudge.contains("/etc/hosts"));
        assert!(nudge.contains("edit"));
    }

    #[test]
    fn write_to_directory_gives_directory_nudge() {
        let calls = vec![call("Write", r#"{"file_path":"/tmp","content":"x"}"#)];
        let Precondition::Failed { nudge } = check(&calls, &none()) else {
            panic!("expected Failed");
        };
        assert!(nudge.contains("directory"));
        assert!(nudge.contains("/tmp"));
    }

    /// The transcript rule must not reach a write tool: a `Write` to a path
    /// nothing has read is the normal way to create a file.
    #[test]
    fn write_to_new_file_passes_even_with_a_legible_transcript() {
        let messages = vec![assistant_call("Read", json!({"file_path": "/etc/passwd"}))];
        let calls = vec![call(
            "Write",
            &format!(r#"{{"file_path":"{MISSING}","content":"x"}}"#),
        )];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Ok
        ));
    }

    // ── Read-before-mutate (the transcript rule) ─────────────────────────────

    #[test]
    fn edit_without_a_prior_read_fails() {
        let messages = vec![assistant_call("Grep", json!({"pattern": "fn main"}))];
        let calls = vec![call(
            "Edit",
            r#"{"file_path":"/etc/hosts","old_string":"a","new_string":"b"}"#,
        )];
        let Precondition::Failed { nudge } = check(&calls, &Transcript::of(&messages)) else {
            panic!("expected Failed");
        };
        assert!(nudge.contains("/etc/hosts"));
        assert!(nudge.contains("not been read"));
    }

    #[test]
    fn edit_after_reading_the_same_path_passes() {
        let messages = vec![
            assistant_call("Read", json!({"file_path": EXISTING})),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "127.0.0.1 localhost"}),
        ];
        let calls = vec![call(
            "Edit",
            r#"{"file_path":"/etc/hosts","old_string":"a","new_string":"b"}"#,
        )];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Ok
        ));
    }

    #[test]
    fn reading_a_different_path_does_not_license_the_edit() {
        let messages = vec![assistant_call("Read", json!({"file_path": "/etc/passwd"}))];
        let calls = vec![call(
            "Edit",
            r#"{"file_path":"/etc/hosts","old_string":"a","new_string":"b"}"#,
        )];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Failed { .. }
        ));
    }

    /// A grep or a glob returns lines or names, never the file, so neither tells
    /// the model what an edit's context looks like.
    #[test]
    fn grep_does_not_count_as_a_read() {
        let messages = vec![assistant_call(
            "Grep",
            json!({"path": EXISTING, "pattern": "x"}),
        )];
        let calls = vec![call(
            "Edit",
            r#"{"file_path":"/etc/hosts","old_string":"a","new_string":"b"}"#,
        )];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Failed { .. }
        ));
    }

    #[test]
    fn responses_function_call_items_are_scanned() {
        let messages = vec![responses_call("read", json!({"path": EXISTING}))];
        let calls = vec![call(
            "edit",
            r#"{"path":"/etc/hosts","old_string":"a","new_string":"b"}"#,
        )];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Ok
        ));
    }

    #[test]
    fn every_edit_family_tool_is_guarded() {
        let messages = vec![assistant_call("Grep", json!({"pattern": "x"}))];
        let transcript = Transcript::of(&messages);
        for (tool, _) in EDIT_TOOLS {
            let calls = vec![call(tool, r#"{"file_path":"/etc/hosts","old_string":"a"}"#)];
            assert!(
                matches!(check(&calls, &transcript), Precondition::Failed { .. }),
                "{tool} was not guarded"
            );
        }
    }

    // ── Failing open ─────────────────────────────────────────────────────────

    /// No transcript to inspect: the rule cannot know a read did not happen.
    #[test]
    fn edit_passes_when_the_transcript_is_unavailable() {
        let calls = vec![call(
            "Edit",
            r#"{"file_path":"/etc/hosts","old_string":"a","new_string":"b"}"#,
        )];
        assert!(matches!(check(&calls, &none()), Precondition::Ok));
    }

    /// A transcript with no recognisable tool traffic is illegible, not empty.
    /// The reads may be there under names this scan does not know.
    #[test]
    fn edit_passes_when_the_transcript_shows_no_tool_traffic() {
        let messages = vec![
            json!({"role": "system", "content": "You are a helpful assistant."}),
            json!({"role": "user", "content": "fix the hosts file"}),
        ];
        let transcript = Transcript::of(&messages);
        assert!(!transcript.legible());
        let calls = vec![call(
            "Edit",
            r#"{"file_path":"/etc/hosts","old_string":"a","new_string":"b"}"#,
        )];
        assert!(matches!(check(&calls, &transcript), Precondition::Ok));
    }

    /// One recognised call is enough to trust the vocabulary, even when that
    /// call is not itself a read.
    #[test]
    fn any_recognised_tool_call_makes_the_transcript_legible() {
        let messages = vec![assistant_call("Bash", json!({"command": "ls"}))];
        assert!(Transcript::of(&messages).legible());
    }

    /// An edit to a path that is not on disk fails on its own terms, with a
    /// harness message that says more than this nudge would.
    #[test]
    fn edit_of_a_missing_file_is_left_to_the_harness() {
        let messages = vec![assistant_call("Grep", json!({"pattern": "x"}))];
        let calls = vec![call(
            "Edit",
            &format!(r#"{{"file_path":"{MISSING}","old_string":"a"}}"#),
        )];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Ok
        ));
    }

    #[test]
    fn a_call_with_no_path_argument_is_ignored() {
        let messages = vec![assistant_call("Bash", json!({"command": "ls"}))];
        let calls = vec![call("Edit", r#"{"old_string":"a","new_string":"b"}"#)];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Ok
        ));
    }

    #[test]
    fn unparseable_arguments_are_ignored() {
        let messages = vec![assistant_call("Bash", json!({"command": "ls"}))];
        let calls = vec![call("Edit", "{not json")];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Ok
        ));
    }

    // ── Transcript scanning ──────────────────────────────────────────────────

    #[test]
    fn several_calls_in_one_assistant_turn_are_all_scanned() {
        let messages = vec![json!({
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {
                    "name": "Read", "arguments": json!({"file_path": "/etc/passwd"}).to_string()}},
                {"id": "b", "type": "function", "function": {
                    "name": "Read", "arguments": json!({"file_path": EXISTING}).to_string()}},
            ],
        })];
        let transcript = Transcript::of(&messages);
        assert!(transcript.has_read("/etc/passwd"));
        assert!(transcript.has_read(EXISTING));
    }

    #[test]
    fn malformed_transcript_entries_are_skipped_not_fatal() {
        let messages = vec![
            json!("just a string"),
            json!({"role": "assistant", "tool_calls": "not an array"}),
            json!({"role": "assistant", "tool_calls": [{"id": "x"}]}),
            assistant_call("Read", json!({"file_path": EXISTING})),
        ];
        let transcript = Transcript::of(&messages);
        assert!(transcript.legible());
        assert!(transcript.has_read(EXISTING));
    }
}
