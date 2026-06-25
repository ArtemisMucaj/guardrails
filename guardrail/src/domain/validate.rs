//! Milestone 3: validate decoded tool calls against the request's tool set.
//!
//! A tool call is valid when its name is one the request actually declared and
//! its arguments parse as a JSON object. Anything else produces a corrective
//! nudge the retry loop (M6) will append as a `user` message. At this milestone
//! validation is **log-only** — the proxy logs the outcome but does not yet
//! retry or alter the response.

use super::decode::ToolCall;

/// Outcome of validating a batch of tool calls.
#[derive(Debug, Clone, PartialEq)]
pub enum Validation {
    /// Every call names a declared tool and carries object arguments.
    Valid,
    /// At least one call is malformed but the situation is recoverable by asking
    /// the model to try again. Carries the nudge text to feed back.
    NeedsRetry(String),
}

/// Validate `calls` against the set of declared `tool_names`.
///
/// Returns [`Validation::NeedsRetry`] with a corrective nudge on the first
/// problem found (unknown tool name, or arguments that are not a JSON object),
/// otherwise [`Validation::Valid`]. An empty `calls` slice is vacuously valid —
/// the "model emitted no calls at all" case is handled upstream by [`decode`],
/// which yields `Text` rather than empty `ToolCalls`.
///
/// [`decode`]: crate::decode::decode_response
pub fn validate(calls: &[ToolCall], tool_names: &[&str]) -> Validation {
    for call in calls {
        if !tool_names.contains(&call.name.as_str()) {
            return Validation::NeedsRetry(unknown_tool_nudge(&call.name, tool_names));
        }
        if !arguments_are_object(&call.arguments) {
            return Validation::NeedsRetry(bad_arguments_nudge(&call.name));
        }
    }
    Validation::Valid
}

/// Arguments must be a JSON object (`{...}`). A bare string, array, number, or
/// invalid JSON all fail.
fn arguments_are_object(arguments: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(arguments)
        .map(|v| v.is_object())
        .unwrap_or(false)
}

fn unknown_tool_nudge(name: &str, tool_names: &[&str]) -> String {
    format!(
        "You called a tool named \"{name}\" which does not exist. \
         Call one of the available tools instead: {}.",
        tool_names.join(", ")
    )
}

fn bad_arguments_nudge(name: &str) -> String {
    format!(
        "The arguments for tool \"{name}\" were not a valid JSON object. \
         Reply with a single tool call whose arguments are a JSON object."
    )
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
    fn valid_when_name_known_and_args_object() {
        let calls = [call("get_weather", "{\"city\":\"Paris\"}")];
        assert_eq!(validate(&calls, &["get_weather"]), Validation::Valid);
    }

    #[test]
    fn empty_calls_are_vacuously_valid() {
        assert_eq!(validate(&[], &["get_weather"]), Validation::Valid);
    }

    #[test]
    fn unknown_tool_needs_retry_and_lists_options() {
        let calls = [call("get_wether", "{}")];
        let Validation::NeedsRetry(nudge) = validate(&calls, &["get_weather", "search"]) else {
            panic!("expected NeedsRetry");
        };
        assert!(nudge.contains("get_wether"));
        assert!(nudge.contains("get_weather"));
        assert!(nudge.contains("search"));
    }

    #[test]
    fn non_object_arguments_need_retry() {
        for bad in ["\"just a string\"", "[1,2,3]", "42", "not json"] {
            let calls = [call("get_weather", bad)];
            assert!(
                matches!(
                    validate(&calls, &["get_weather"]),
                    Validation::NeedsRetry(_)
                ),
                "expected NeedsRetry for arguments {bad:?}"
            );
        }
    }

    #[test]
    fn first_problem_wins() {
        let calls = [call("bad_name", "{}"), call("get_weather", "not json")];
        let Validation::NeedsRetry(nudge) = validate(&calls, &["get_weather"]) else {
            panic!("expected NeedsRetry");
        };
        // The unknown-name problem comes first in the slice.
        assert!(nudge.contains("bad_name"));
    }
}
