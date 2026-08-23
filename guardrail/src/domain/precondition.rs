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
use super::validate::normalize_key;
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
/// - `edit_file`         — Zed AI
/// - `NotebookEdit`      — Claude Code (targets `notebook_path`)
///
/// `apply_patch` is deliberately absent. It names its target inside the patch
/// body (`*** Update File: …`) rather than in a path argument, so listing it
/// here would guard nothing while the docs claimed otherwise — and a guard
/// believed to be on is worse than one known to be off. Covering it means
/// parsing the patch formats, which is its own change.
const EDIT_TOOLS: &[(&str, &str)] = &[
    ("Edit", "Claude Code"),
    ("MultiEdit", "Claude Code"),
    ("edit", "OpenCode, Pi, GitHub Copilot CLI"),
    ("edit_file", "Zed AI"),
    ("NotebookEdit", "Claude Code"),
];

/// Tools whose call means the model has seen a file's contents, normalized.
///
/// Matched with [`normalize_key`], so one entry covers every casing and
/// separator style a harness might use: `readfile` also admits `read_file`,
/// `readFile`, `ReadFile`, and `read-file`.
///
/// Being generous here is the safe direction, and deliberately so. The two
/// failure modes are not symmetric: a read name this list misses produces a
/// *false refusal*, telling a model to read a file it already read, while an
/// over-broad match merely declines to guard an edit. The first breaks a
/// conversation; the second returns the guard to where it stood before this
/// rule existed.
///
/// Only whole-file readers count. A `Grep` returns matching lines and a `Glob`
/// returns names, so neither tells the model what an `Edit`'s context looks
/// like — treating them as reads would license exactly the blind edit this rule
/// exists to stop. `WebFetch` and `TodoRead` are reads of something that is not
/// a project file, and are likewise out.
///
/// Sources:
/// - `Read`      — Claude Code
/// - `read`      — OpenCode, Pi, GitHub Copilot CLI
/// - `read_file` — Zed AI
const READ_TOOLS: &[&str] = &["read", "readfile"];

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
/// read happened" or "this scan cannot see the reads here" — so it is paired
/// with [`Transcript::legible`], which is true only once a read has been seen
/// and understood, to tell those apart.
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
                if !READ_TOOLS.contains(&normalize_key(name).as_str()) {
                    continue;
                }
                // Only a read the scan fully understood is evidence that reads
                // are visible here. A `Read` whose path will not parse — the
                // arguments came through as an object rather than the wire's
                // JSON string, say — is a read this scan just missed, and
                // counting it as legible would assert the opposite.
                if let Some(path) = file_path_arg(arguments) {
                    read_paths.insert(canonical(&path));
                    legible = true;
                }
            }
        }
        Self {
            read_paths,
            legible,
        }
    }

    /// Whether this scan is in a position to say a read did *not* happen.
    ///
    /// True only when at least one read was seen and understood. That is a
    /// deliberately strong bar, and it is not the same as "some tool call was
    /// recognised": a transcript whose only surviving tool traffic is the
    /// model's own failed edit proves the scan can read *calls*, not that it
    /// can see *reads*. Trimmed history is exactly that shape — the read is the
    /// first thing a compaction drops and the failed edit is the last — so
    /// treating any recognised call as legibility refuses precisely the
    /// conversation that already did the read.
    ///
    /// The cost is that the very first edit of a conversation is unguarded,
    /// since nothing has been read yet. That is the right side to err on: the
    /// rule exists to catch a model editing from memory across a long session,
    /// and one unguarded opening edit is cheaper than telling a model to
    /// re-read a file it just read.
    pub fn legible(&self) -> bool {
        self.legible
    }

    fn has_read(&self, path: &str) -> bool {
        self.read_paths.contains(&canonical(path))
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

    // Responses: a flat `function_call` item. The type is checked rather than
    // inferred from the name/arguments pair, because other item types carry the
    // same pair — an `mcp_call` names a tool on some third-party server, and a
    // server exposing one called `read` would otherwise license an edit to a
    // file nothing on this machine ever opened.
    if obj.get("type").and_then(Value::as_str) == Some(super::responses::FUNCTION_CALL) {
        if let (Some(name), Some(arguments)) = (
            obj.get("name").and_then(Value::as_str),
            obj.get("arguments").and_then(Value::as_str),
        ) {
            found.push((name, arguments));
        }
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

/// Resolve a path to the identity the read set is keyed on.
///
/// Two spellings of one file must compare equal, or the rule refuses an edit
/// whose read it is holding under another name — `/etc/hosts` against a read of
/// `/private/etc/hosts` on macOS, or an absolute edit against a relative read.
/// `canonicalize` collapses symlinks, `.`/`..`, and trailing slashes, and on a
/// case-insensitive filesystem it also settles case.
///
/// It touches the disk and fails on a path that is not there. That is fine for
/// both callers: a read of a since-deleted file cannot license an edit to a
/// file that must exist, and an unresolvable path falls back to its literal
/// form, which is the exact-match behaviour this replaced.
fn canonical(path: &str) -> String {
    std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string())
}

/// The argument keys that name a tool's target file, normalized.
///
/// Ordered: a call carrying both `notebook_path` and `file_path` means the
/// notebook, which is the more specific target.
const PATH_KEYS: &[&str] = &["notebookpath", "filepath", "path"];

/// Extract the target path from a raw JSON arguments string, under any spelling
/// of the keys the supported harnesses use. Returns `None` if the arguments are not a
/// valid object or no such key holds a string — which is why a caller must not
/// read `None` as "this call has no target".
fn file_path_arg(arguments: &str) -> Option<String> {
    let obj: Value = serde_json::from_str(arguments).ok()?;
    let map = obj.as_object()?;
    // Matched on the normalized key rather than a literal list, because the
    // same argument is spelled differently by every harness — `file_path`
    // (Claude Code), `filePath` (OpenCode), `path` (Pi, Zed, Copilot CLI) — and
    // a literal list silently lets the spellings it forgot walk past the guard.
    // `normalize_key` is the repo's existing answer to that, already used to
    // repair argument names against a schema.
    PATH_KEYS.iter().find_map(|wanted| {
        map.iter()
            .find(|(key, _)| normalize_key(key) == *wanted)
            .and_then(|(_, value)| value.as_str())
            .map(str::to_string)
    })
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
        // A read of *another* file: enough to know reads are visible here, and
        // it is not a read of the file being edited.
        let messages = vec![assistant_call("Read", json!({"file_path": "/etc/passwd"}))];
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
        let messages = vec![
            assistant_call("Read", json!({"file_path": "/etc/passwd"})),
            assistant_call("Grep", json!({"path": EXISTING, "pattern": "x"})),
        ];
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
    fn every_edit_family_tool_is_guarded_in_its_own_argument_shape() {
        let messages = vec![assistant_call("Read", json!({"file_path": "/etc/passwd"}))];
        let transcript = Transcript::of(&messages);
        for (tool, _) in EDIT_TOOLS {
            let key = if *tool == "NotebookEdit" {
                "notebook_path"
            } else {
                "file_path"
            };
            let arguments = json!({key: EXISTING, "old_string": "a"}).to_string();
            assert!(
                matches!(
                    check(&[call(tool, &arguments)], &transcript),
                    Precondition::Failed { .. }
                ),
                "{tool} was not guarded"
            );
        }
    }

    /// `apply_patch` names its target inside the patch body, so it is not in
    /// `EDIT_TOOLS`. This pins that it stays out until the body is parsed —
    /// listing it would guard nothing while the docs claimed otherwise.
    #[test]
    fn apply_patch_is_not_claimed_as_guarded() {
        assert!(!EDIT_TOOLS.iter().any(|(t, _)| *t == "apply_patch"));
        let messages = vec![assistant_call("Read", json!({"file_path": "/etc/passwd"}))];
        let calls = vec![call(
            "apply_patch",
            r#"{"patchText":"*** Update File: /etc/hosts"}"#,
        )];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Ok
        ));
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

    /// A transcript with no understood read is illegible, not empty. The reads
    /// may be there under names or shapes this scan does not know.
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

    /// A recognised call that is not a read proves the scan can read *calls*,
    /// not that it can see *reads*. It must not make the transcript legible.
    #[test]
    fn a_non_read_call_alone_does_not_make_the_transcript_legible() {
        let messages = vec![assistant_call("Bash", json!({"command": "ls"}))];
        assert!(!Transcript::of(&messages).legible());
    }

    /// One understood read is the bar, and it licenses the rule for *other*
    /// paths — that is what makes a refusal possible at all.
    #[test]
    fn an_understood_read_makes_the_transcript_legible() {
        let messages = vec![assistant_call("Read", json!({"file_path": "/etc/passwd"}))];
        assert!(Transcript::of(&messages).legible());
    }

    /// A `Read` whose path will not parse is a read this scan *missed*. Calling
    /// that legible would assert the opposite and refuse the next edit.
    #[test]
    fn a_read_with_unparseable_arguments_does_not_make_the_transcript_legible() {
        // `arguments` as a decoded object rather than the wire's JSON string:
        // a real shape for a client that round-trips its own history.
        let messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "Read", "arguments": {"file_path": EXISTING}},
            }],
        })];
        let transcript = Transcript::of(&messages);
        assert!(!transcript.legible());
        let calls = vec![call(
            "Edit",
            r#"{"file_path":"/etc/hosts","old_string":"a"}"#,
        )];
        assert!(matches!(check(&calls, &transcript), Precondition::Ok));
    }

    /// The trimmed-history shape: a compaction drops the read and keeps the
    /// failed edit. Refusing here tells the model to re-read a file it read.
    #[test]
    fn a_surviving_failed_edit_does_not_license_a_refusal() {
        let messages = vec![
            assistant_call("Edit", json!({"file_path": EXISTING, "old_string": "a"})),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "string not found"}),
        ];
        let transcript = Transcript::of(&messages);
        assert!(!transcript.legible());
        let calls = vec![call(
            "Edit",
            r#"{"file_path":"/etc/hosts","old_string":"a"}"#,
        )];
        assert!(matches!(check(&calls, &transcript), Precondition::Ok));
    }

    /// An `mcp_call` carries the same name/arguments pair as a `function_call`.
    /// A third-party server exposing `read` must not license a real edit.
    #[test]
    fn an_mcp_call_named_read_does_not_count_as_a_read() {
        let messages = vec![json!({
            "type": "mcp_call",
            "server_label": "docs",
            "name": "read",
            "arguments": json!({"path": EXISTING}).to_string(),
        })];
        let transcript = Transcript::of(&messages);
        assert!(!transcript.legible());
        assert!(!transcript.has_read(EXISTING));
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

    /// `/etc/hosts` is a symlink to `/private/etc/hosts` on macOS. Two spellings
    /// of one file must not read as two files.
    #[test]
    fn two_spellings_of_one_path_compare_equal() {
        let canonical_form = std::fs::canonicalize(EXISTING).unwrap();
        let other_spelling = canonical_form.to_string_lossy().into_owned();
        let messages = vec![assistant_call("Read", json!({"file_path": other_spelling}))];
        let calls = vec![call(
            "Edit",
            r#"{"file_path":"/etc/hosts","old_string":"a"}"#,
        )];
        assert!(matches!(
            check(&calls, &Transcript::of(&messages)),
            Precondition::Ok
        ));
    }

    /// Every harness spells the path argument differently. A literal key list
    /// silently let the spellings it forgot walk past both rules — including
    /// the write rule, which predates this one.
    #[test]
    fn every_spelling_of_the_path_argument_is_found() {
        for key in ["file_path", "filePath", "path", "File_Path", "file-path"] {
            let arguments = json!({key: EXISTING, "content": "x"}).to_string();
            assert!(
                matches!(
                    check(&[call("Write", &arguments)], &none()),
                    Precondition::Failed { .. }
                ),
                "write guard missed the key {key}"
            );
        }
    }

    /// A read named in a different separator style is the same read. Missing it
    /// costs a false refusal, so matching is normalized rather than literal.
    #[test]
    fn read_tools_are_matched_across_casing_and_separators() {
        for name in [
            "Read",
            "read",
            "READ",
            "read_file",
            "readFile",
            "ReadFile",
            "read-file",
        ] {
            let messages = vec![assistant_call(name, json!({"file_path": EXISTING}))];
            let transcript = Transcript::of(&messages);
            assert!(transcript.legible(), "{name} was not recognised as a read");
            assert!(transcript.has_read(EXISTING), "{name} recorded no path");
        }
    }

    /// A reader of something that is not a project file must not license an
    /// edit to one.
    #[test]
    fn non_file_readers_do_not_count_as_reads() {
        for name in ["WebFetch", "web_fetch", "TodoRead", "Grep", "Glob"] {
            let messages = vec![assistant_call(name, json!({"file_path": EXISTING}))];
            assert!(
                !Transcript::of(&messages).legible(),
                "{name} was treated as a file read"
            );
        }
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
