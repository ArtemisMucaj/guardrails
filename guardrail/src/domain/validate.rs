//! Validate decoded tool calls against the request's tool set.

use super::decode::ToolCall;
use super::model::Tool;
use serde_json::{Map, Value};

/// Outcome of validating a batch of tool calls.
#[derive(Debug, Clone, PartialEq)]
pub enum Validation {
    /// Every call names a declared tool and carries object arguments.
    Valid,
    /// At least one call is malformed but the situation is recoverable by asking
    /// the model to try again. Carries the nudge text to feed back.
    NeedsRetry(String),
}

/// Validate `calls` against the set of declared tools.
///
/// Returns [`Validation::NeedsRetry`] with a corrective nudge on the first
/// problem found (unknown tool name, or arguments that are not a JSON object),
/// otherwise [`Validation::Valid`]. An empty `calls` slice is vacuously valid.
///
/// [`decode`]: crate::domain::decode::decode_response
pub fn validate(calls: &[ToolCall], tools: &[Tool]) -> Validation {
    for call in calls {
        let Some(tool) = tools.iter().find(|tool| tool.function.name == call.name) else {
            return Validation::NeedsRetry(unknown_tool_nudge(&call.name, tools));
        };
        let Some(arguments) = arguments_object(&call.arguments) else {
            return Validation::NeedsRetry(bad_arguments_nudge(&call.name));
        };
        for required in required_parameters(tool) {
            if !arguments.contains_key(required) {
                return Validation::NeedsRetry(missing_argument_nudge(&call.name, required));
            }
        }
        if let Some(properties) = parameter_properties(tool) {
            for (key, value) in &arguments {
                let declared = properties
                    .get(key)
                    .and_then(|schema| schema.get("type"))
                    .and_then(Value::as_str);
                if let Some(declared) = declared {
                    if !type_matches(declared, value) {
                        return Validation::NeedsRetry(wrong_type_nudge(&call.name, key, declared));
                    }
                }
            }
        }
    }
    Validation::Valid
}

/// Arguments must be a JSON object (`{...}`). A bare string, array, number, or
/// invalid JSON all fail.
fn arguments_object(arguments: &str) -> Option<Map<String, Value>> {
    match serde_json::from_str::<Value>(arguments).ok()? {
        Value::Object(map) => Some(map),
        _ => None,
    }
}

fn required_parameters(tool: &Tool) -> Vec<&str> {
    tool.function
        .rest
        .get("parameters")
        .and_then(|parameters| parameters.get("required"))
        .and_then(Value::as_array)
        .map(|required| required.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

/// The JSON-schema `properties` map declared for a tool's parameters, if any.
fn parameter_properties(tool: &Tool) -> Option<&Map<String, Value>> {
    tool.function
        .rest
        .get("parameters")?
        .get("properties")?
        .as_object()
}

/// Whether `value` satisfies a JSON-schema scalar/container `type`. Unknown
/// type names are accepted (we don't reject what we don't understand).
fn type_matches(declared: &str, value: &Value) -> bool {
    match declared {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn unknown_tool_nudge(name: &str, tools: &[Tool]) -> String {
    let tool_names: Vec<&str> = tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect();
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

fn missing_argument_nudge(name: &str, required: &str) -> String {
    format!(
        "The arguments for tool \"{name}\" were missing required field \
         \"{required}\". Reply with a single tool call whose arguments include \
         all required fields."
    )
}

fn wrong_type_nudge(name: &str, field: &str, expected: &str) -> String {
    format!(
        "The argument \"{field}\" for tool \"{name}\" had the wrong type; it \
         must be a {expected}. Reply with a single tool call whose arguments \
         match the declared types."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: None,
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    fn tool(name: &str, required: &[&str]) -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "required": required
                }
            }
        }))
        .unwrap()
    }

    #[test]
    fn valid_when_name_known_and_args_object() {
        let calls = [call("get_weather", "{\"city\":\"Paris\"}")];
        assert_eq!(
            validate(&calls, &[tool("get_weather", &[])]),
            Validation::Valid
        );
    }

    #[test]
    fn empty_calls_are_vacuously_valid() {
        assert_eq!(
            validate(&[], &[tool("get_weather", &[])]),
            Validation::Valid
        );
    }

    #[test]
    fn unknown_tool_needs_retry_and_lists_options() {
        let calls = [call("get_wether", "{}")];
        let tools = [tool("get_weather", &[]), tool("search", &[])];
        let Validation::NeedsRetry(nudge) = validate(&calls, &tools) else {
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
                    validate(&calls, &[tool("get_weather", &[])]),
                    Validation::NeedsRetry(_)
                ),
                "expected NeedsRetry for arguments {bad:?}"
            );
        }
    }

    #[test]
    fn missing_required_argument_needs_retry() {
        let calls = [call(
            "Edit",
            "{\"oldString\":\"old\",\"newString\":\"new\"}",
        )];
        let Validation::NeedsRetry(nudge) = validate(
            &calls,
            &[tool("Edit", &["filePath", "oldString", "newString"])],
        ) else {
            panic!("expected NeedsRetry");
        };
        assert!(nudge.contains("filePath"));
    }

    #[test]
    fn wrong_argument_type_needs_retry() {
        let typed_tool: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                    "required": ["filePath"]
                }
            }
        }))
        .unwrap();

        let tools = [typed_tool];

        // Right key, wrong type (number instead of string).
        let calls = [call("Edit", "{\"filePath\":123}")];
        let Validation::NeedsRetry(nudge) = validate(&calls, &tools) else {
            panic!("expected NeedsRetry");
        };
        assert!(nudge.contains("filePath"));
        assert!(nudge.contains("string"));

        // Correct type passes.
        let calls = [call("Edit", "{\"filePath\":\"/tmp/x.rs\"}")];
        assert_eq!(validate(&calls, &tools), Validation::Valid);
    }

    #[test]
    fn first_problem_wins() {
        let calls = [call("bad_name", "{}"), call("get_weather", "not json")];
        let Validation::NeedsRetry(nudge) = validate(&calls, &[tool("get_weather", &[])]) else {
            panic!("expected NeedsRetry");
        };
        // The unknown-name problem comes first in the slice.
        assert!(nudge.contains("bad_name"));
    }
}
