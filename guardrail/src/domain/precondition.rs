//! Semantic precondition checks on tool calls.
//!
//! These run after the model produces a tool call but *before* the
//! repair/validate loop, so a failed precondition short-circuits the whole
//! retry budget and returns a clear explanation to the model immediately.

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

/// Outcome of a precondition check.
pub enum Precondition {
    /// All preconditions satisfied; proceed normally.
    Ok,
    /// A precondition failed. `nudge` is the explanation to return to the
    /// model as a plain assistant text message.
    Failed { nudge: String },
}

/// Check semantic preconditions for `calls`.
///
/// Currently enforces one rule: a write-only tool must not target a file that
/// already exists on disk. When violated the model receives a nudge to read the
/// file first and then use the appropriate edit tool.
pub fn check(calls: &[ToolCall]) -> Precondition {
    for call in calls {
        if !is_write_tool(&call.name) {
            continue;
        }
        let Some(path) = file_path_arg(&call.arguments) else {
            continue;
        };
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
    }
    Precondition::Ok
}

fn is_write_tool(name: &str) -> bool {
    WRITE_TOOLS
        .iter()
        .any(|(tool, _)| tool.eq_ignore_ascii_case(name))
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

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: None,
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    #[test]
    fn non_write_tool_always_passes() {
        let calls = vec![call("Read", r#"{"file_path":"/etc/passwd"}"#)];
        assert!(matches!(check(&calls), Precondition::Ok));
    }

    #[test]
    fn write_tool_on_nonexistent_file_passes() {
        let calls = vec![call(
            "Write",
            r#"{"file_path":"/tmp/guardrail_test_nonexistent_xyz.txt","content":"hi"}"#,
        )];
        assert!(matches!(check(&calls), Precondition::Ok));
    }

    #[test]
    fn write_tool_on_existing_file_fails() {
        // /etc/hosts is guaranteed to exist on any Unix system.
        let calls = vec![call("Write", r#"{"file_path":"/etc/hosts","content":"x"}"#)];
        assert!(matches!(check(&calls), Precondition::Failed { .. }));
    }

    #[test]
    fn case_insensitive_tool_name_match() {
        let calls = vec![call("WRITE", r#"{"file_path":"/etc/hosts","content":"x"}"#)];
        assert!(matches!(check(&calls), Precondition::Failed { .. }));
    }

    #[test]
    fn path_key_accepted_as_alternative() {
        let calls = vec![call("create", r#"{"path":"/etc/hosts","content":"x"}"#)];
        assert!(matches!(check(&calls), Precondition::Failed { .. }));
    }

    #[test]
    fn nudge_mentions_file_and_edit() {
        let calls = vec![call("write", r#"{"file_path":"/etc/hosts","content":"x"}"#)];
        let Precondition::Failed { nudge } = check(&calls) else {
            panic!("expected Failed");
        };
        assert!(nudge.contains("/etc/hosts"));
        assert!(nudge.contains("edit"));
    }

    #[test]
    fn write_to_directory_gives_directory_nudge() {
        let calls = vec![call("Write", r#"{"file_path":"/tmp","content":"x"}"#)];
        let Precondition::Failed { nudge } = check(&calls) else {
            panic!("expected Failed");
        };
        assert!(nudge.contains("directory"));
        assert!(nudge.contains("/tmp"));
    }
}
