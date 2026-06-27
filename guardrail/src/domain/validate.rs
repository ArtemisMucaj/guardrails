//! Validate decoded tool calls against the request's tool set.

use super::decode::ToolCall;
use super::model::Tool;
use serde_json::{Map, Value};

/// Why a batch of tool calls failed validation. Stable, low-cardinality tags
/// suitable for grouping failure metrics by category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// The call named a tool that was not declared in the request.
    UnknownTool,
    /// The call's arguments did not parse as a JSON object.
    BadArguments,
    /// A required schema field was absent from the arguments.
    MissingArgument,
    /// An argument's value did not match its declared JSON-schema type.
    WrongType,
}

impl ErrorCategory {
    /// Stable snake_case tag for storage and grouping.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCategory::UnknownTool => "unknown_tool",
            ErrorCategory::BadArguments => "bad_arguments",
            ErrorCategory::MissingArgument => "missing_argument",
            ErrorCategory::WrongType => "wrong_type",
        }
    }
}

/// Outcome of validating a batch of tool calls.
#[derive(Debug, Clone, PartialEq)]
pub enum Validation {
    /// Every call names a declared tool and carries object arguments.
    Valid,
    /// At least one call is malformed but the situation is recoverable by asking
    /// the model to try again. Carries the failure category (for metrics), the
    /// nudge text to feed back, and the index of the offending call (so metrics
    /// attribute the failure to the right tool, not just the first call).
    NeedsRetry {
        category: ErrorCategory,
        nudge: String,
        offending: usize,
    },
}

/// Validate `calls` against the set of declared tools.
///
/// Returns [`Validation::NeedsRetry`] with a corrective nudge on the first
/// problem found (unknown tool name, or arguments that are not a JSON object),
/// otherwise [`Validation::Valid`]. An empty `calls` slice is vacuously valid.
///
/// [`decode`]: crate::domain::decode::decode_response
pub fn validate(calls: &[ToolCall], tools: &[Tool]) -> Validation {
    for (offending, call) in calls.iter().enumerate() {
        let Some(tool) = tools.iter().find(|tool| tool.function.name == call.name) else {
            return Validation::NeedsRetry {
                category: ErrorCategory::UnknownTool,
                nudge: unknown_tool_nudge(&call.name, tools),
                offending,
            };
        };
        let Some(arguments) = arguments_object(&call.arguments) else {
            return Validation::NeedsRetry {
                category: ErrorCategory::BadArguments,
                nudge: bad_arguments_nudge(&call.name),
                offending,
            };
        };
        for required in required_parameters(tool) {
            if !arguments.contains_key(required) {
                return Validation::NeedsRetry {
                    category: ErrorCategory::MissingArgument,
                    nudge: missing_argument_nudge(&call.name, required),
                    offending,
                };
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
                        return Validation::NeedsRetry {
                            category: ErrorCategory::WrongType,
                            nudge: wrong_type_nudge(&call.name, key, declared),
                            offending,
                        };
                    }
                }
            }
        }
    }
    Validation::Valid
}

/// Repair argument keys that name a declared property in a different casing or
/// separator style — `file_path` / `filepath` / `FilePath` for a schema's
/// `filePath`. Small models routinely emit snake_case where the schema is
/// camelCase (or vice-versa); rebinding the value to the declared name is
/// cheaper than spending a corrective retry.
///
/// Renaming a key reassigns which parameter a value binds to, so this is
/// deliberately conservative and only acts where the intent is unambiguous:
///
/// - it only fills a *missing required* property (an otherwise-valid call is
///   never touched);
/// - only unknown keys (not matching any declared property) are rename sources;
/// - matching is normalization-only (case- and separator-insensitive), never
///   fuzzy distance or synonyms;
/// - it abstains whenever the mapping is ambiguous — an unknown key normalizing
///   to more than one missing property, or more than one unknown key competing
///   for the same property.
///
/// Returns whether any call's arguments changed, so the caller re-emits the
/// repaired form to the client.
pub fn repair_argument_names(calls: &mut [ToolCall], tools: &[Tool]) -> bool {
    let mut changed = false;
    for call in calls.iter_mut() {
        let Some(tool) = tools.iter().find(|tool| tool.function.name == call.name) else {
            continue;
        };
        let Some(properties) = parameter_properties(tool) else {
            continue;
        };
        let Some(mut arguments) = arguments_object(&call.arguments) else {
            continue;
        };

        // Rename targets: required properties not already present.
        let missing: Vec<String> = required_parameters(tool)
            .into_iter()
            .filter(|req| !arguments.contains_key(*req))
            .map(str::to_string)
            .collect();
        if missing.is_empty() {
            continue;
        }

        // Rename sources: keys that name no declared property.
        let unknown: Vec<String> = arguments
            .keys()
            .filter(|key| !properties.contains_key(key.as_str()))
            .cloned()
            .collect();
        if unknown.is_empty() {
            continue;
        }

        // Resolve source -> target by normalized equality, abstaining when a
        // source matches more than one missing property.
        let mut renames: Vec<(String, String)> = Vec::new();
        for source in &unknown {
            let norm = normalize_key(source);
            let mut hits = missing
                .iter()
                .filter(|target| normalize_key(target.as_str()) == norm);
            if let Some(target) = hits.next() {
                if hits.next().is_none() {
                    renames.push((source.clone(), target.clone()));
                }
            }
        }

        let mut call_changed = false;
        for (source, target) in &renames {
            // Abstain when more than one source competes for the same property.
            if renames.iter().filter(|(_, t)| t == target).count() != 1 {
                continue;
            }
            // Targets are missing by construction, so a rename never overwrites
            // a present value; the guard keeps that invariant explicit.
            if arguments.contains_key(target.as_str()) {
                continue;
            }
            if let Some(value) = arguments.remove(source.as_str()) {
                arguments.insert(target.clone(), value);
                call_changed = true;
            }
        }

        if call_changed {
            call.arguments = Value::Object(arguments).to_string();
            changed = true;
        }
    }
    changed
}

/// Normalize an argument key for style-insensitive comparison: lowercased with
/// `_`, `-`, and spaces removed, so `file_path`, `File-Path`, and `filePath`
/// all collapse to `filepath`.
fn normalize_key(key: &str) -> String {
    key.chars()
        .filter(|c| !matches!(c, '_' | '-' | ' '))
        .flat_map(char::to_lowercase)
        .collect()
}

/// Coerce obviously-mistyped scalar arguments to the type the tool's JSON
/// schema declares, in place. Small models routinely emit `"123"`/`"true"` for
/// a number/boolean field, or a bare scalar where a string is wanted; fixing
/// these here is cheaper and more reliable than spending a corrective retry on a
/// one-character mismatch.
///
/// Only unambiguous scalar reinterpretations are applied — a stringified number,
/// integer, or boolean parsed back to its type, or a scalar rendered as the
/// declared string. Anything else is left untouched for [`validate`] to flag.
/// Returns whether any call's arguments changed, so the caller can re-emit the
/// repaired form to the client instead of forwarding the original bytes.
pub fn coerce_arguments(calls: &mut [ToolCall], tools: &[Tool]) -> bool {
    let mut changed = false;
    for call in calls.iter_mut() {
        let Some(tool) = tools.iter().find(|tool| tool.function.name == call.name) else {
            continue;
        };
        let Some(properties) = parameter_properties(tool) else {
            continue;
        };
        let Some(mut arguments) = arguments_object(&call.arguments) else {
            continue;
        };
        let mut call_changed = false;
        for (key, value) in arguments.iter_mut() {
            let Some(declared) = properties
                .get(key)
                .and_then(|schema| schema.get("type"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            if type_matches(declared, value) {
                continue;
            }
            if let Some(fixed) = coerce_scalar(declared, value) {
                *value = fixed;
                call_changed = true;
            }
        }
        if call_changed {
            call.arguments = Value::Object(arguments).to_string();
            changed = true;
        }
    }
    changed
}

/// Reinterpret a single scalar `value` as the schema-declared `type`, returning
/// the coerced value only when the reinterpretation is unambiguous. Containers
/// and `null` are never coerced — `None` leaves the value for validation to
/// flag.
fn coerce_scalar(declared: &str, value: &Value) -> Option<Value> {
    match declared {
        "string" => match value {
            Value::Number(n) => Some(Value::String(n.to_string())),
            Value::Bool(b) => Some(Value::String(b.to_string())),
            _ => None,
        },
        "integer" => value
            .as_str()?
            .trim()
            .parse::<i64>()
            .ok()
            .map(|n| Value::Number(n.into())),
        "number" => {
            let text = value.as_str()?.trim();
            if let Ok(i) = text.parse::<i64>() {
                return Some(Value::Number(i.into()));
            }
            text.parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number)
        }
        "boolean" => match value.as_str()?.trim() {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            _ => None,
        },
        _ => None,
    }
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
        let Validation::NeedsRetry { nudge, .. } = validate(&calls, &tools) else {
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
                    Validation::NeedsRetry { .. }
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
        let Validation::NeedsRetry { nudge, .. } = validate(
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
        let Validation::NeedsRetry { nudge, .. } = validate(&calls, &tools) else {
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
        let Validation::NeedsRetry { nudge, .. } = validate(&calls, &[tool("get_weather", &[])])
        else {
            panic!("expected NeedsRetry");
        };
        // The unknown-name problem comes first in the slice.
        assert!(nudge.contains("bad_name"));
    }

    #[test]
    fn offending_index_points_at_the_failing_call_not_the_first() {
        // First call is valid; the second is the bad one. Metrics must attribute
        // the failure to index 1, not 0.
        let calls = [
            call("get_weather", "{\"city\":\"Paris\"}"),
            call("unknown_tool", "{}"),
        ];
        let Validation::NeedsRetry { offending, .. } =
            validate(&calls, &[tool("get_weather", &[])])
        else {
            panic!("expected NeedsRetry");
        };
        assert_eq!(offending, 1);
    }

    // ── Argument coercion ────────────────────────────────────────────────────

    fn typed_tool(properties: Value) -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "f",
                "parameters": { "type": "object", "properties": properties }
            }
        }))
        .unwrap()
    }

    #[test]
    fn coerces_stringified_scalars_and_makes_validation_pass() {
        let tools = [typed_tool(json!({
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "enabled": {"type": "boolean"}
        }))];
        let mut calls = [call(
            "f",
            "{\"count\":\"3\",\"ratio\":\"1.5\",\"enabled\":\"true\"}",
        )];

        // Before coercion the stringified scalars fail validation.
        assert!(matches!(
            validate(&calls, &tools),
            Validation::NeedsRetry { .. }
        ));

        assert!(coerce_arguments(&mut calls, &tools));
        assert_eq!(
            calls[0].arguments,
            "{\"count\":3,\"enabled\":true,\"ratio\":1.5}"
        );
        assert_eq!(validate(&calls, &tools), Validation::Valid);
    }

    #[test]
    fn coerces_scalar_to_declared_string() {
        let tools = [typed_tool(json!({"id": {"type": "string"}}))];
        let mut calls = [call("f", "{\"id\":123}")];
        assert!(coerce_arguments(&mut calls, &tools));
        assert_eq!(calls[0].arguments, "{\"id\":\"123\"}");
        assert_eq!(validate(&calls, &tools), Validation::Valid);
    }

    #[test]
    fn coercion_reports_no_change_when_types_already_match() {
        let tools = [typed_tool(json!({"count": {"type": "integer"}}))];
        let mut calls = [call("f", "{\"count\":3}")];
        assert!(!coerce_arguments(&mut calls, &tools));
    }

    #[test]
    fn coercion_leaves_unparseable_and_container_values_alone() {
        let tools = [typed_tool(json!({
            "count": {"type": "integer"},
            "items": {"type": "string"}
        }))];
        // "abc" is not an integer; an array is not "obviously" the intended string.
        let mut calls = [call("f", "{\"count\":\"abc\",\"items\":[1,2]}")];
        assert!(!coerce_arguments(&mut calls, &tools));
        // Still invalid, so the retry path is preserved for genuine mismatches.
        assert!(matches!(
            validate(&calls, &tools),
            Validation::NeedsRetry { .. }
        ));
    }

    #[test]
    fn coercion_skips_unknown_tools_and_non_object_arguments() {
        let tools = [typed_tool(json!({"count": {"type": "integer"}}))];
        let mut calls = [call("other", "{\"count\":\"3\"}"), call("f", "not json")];
        assert!(!coerce_arguments(&mut calls, &tools));
    }

    // ── Argument name repair ─────────────────────────────────────────────────

    fn typed_tool_req(properties: Value, required: &[&str]) -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }))
        .unwrap()
    }

    #[test]
    fn repairs_snake_case_key_to_declared_camel_case() {
        let tools = [typed_tool_req(
            json!({"filePath": {"type": "string"}}),
            &["filePath"],
        )];
        let mut calls = [call("f", "{\"file_path\":\"/tmp/x.rs\"}")];

        // The wrongly-styled key leaves the required field missing.
        assert!(matches!(
            validate(&calls, &tools),
            Validation::NeedsRetry { .. }
        ));

        assert!(repair_argument_names(&mut calls, &tools));
        assert_eq!(calls[0].arguments, "{\"filePath\":\"/tmp/x.rs\"}");
        assert_eq!(validate(&calls, &tools), Validation::Valid);
    }

    #[test]
    fn repairs_case_and_separator_variants() {
        for variant in [
            "file_path",
            "FilePath",
            "File-Path",
            "filepath",
            "FILE_PATH",
        ] {
            let tools = [typed_tool_req(
                json!({"filePath": {"type": "string"}}),
                &["filePath"],
            )];
            let mut calls = [call("f", &format!("{{\"{variant}\":\"/x\"}}"))];
            assert!(
                repair_argument_names(&mut calls, &tools),
                "expected repair for {variant:?}"
            );
            assert_eq!(calls[0].arguments, "{\"filePath\":\"/x\"}");
        }
    }

    #[test]
    fn does_not_touch_an_already_valid_call() {
        // Required field present and an extra unknown key alongside it: the call
        // already validates, so name repair must leave it alone.
        let tools = [typed_tool_req(
            json!({"filePath": {"type": "string"}}),
            &["filePath"],
        )];
        let mut calls = [call("f", "{\"filePath\":\"/x\",\"note\":\"hi\"}")];
        assert!(!repair_argument_names(&mut calls, &tools));
    }

    #[test]
    fn does_not_overwrite_a_present_required_key() {
        // `filePath` is already correct; a stray `file_path` must not clobber it.
        let tools = [typed_tool_req(
            json!({"filePath": {"type": "string"}}),
            &["filePath"],
        )];
        let mut calls = [call(
            "f",
            "{\"filePath\":\"/right\",\"file_path\":\"/wrong\"}",
        )];
        assert!(!repair_argument_names(&mut calls, &tools));
    }

    #[test]
    fn abstains_when_two_unknown_keys_compete_for_one_property() {
        let tools = [typed_tool_req(
            json!({"filePath": {"type": "string"}}),
            &["filePath"],
        )];
        // Both keys normalize to "filepath"; ambiguous source, so abstain.
        let mut calls = [call("f", "{\"file_path\":\"/a\",\"filepath\":\"/b\"}")];
        assert!(!repair_argument_names(&mut calls, &tools));
        assert!(matches!(
            validate(&calls, &tools),
            Validation::NeedsRetry { .. }
        ));
    }

    #[test]
    fn does_not_repair_a_non_required_property() {
        // `limit` is declared but optional; an unknown `lim_it` is not rebound,
        // because only missing *required* fields are repair targets.
        let tools = [typed_tool_req(
            json!({"query": {"type": "string"}, "limit": {"type": "integer"}}),
            &["query"],
        )];
        let mut calls = [call("f", "{\"query\":\"x\",\"lim_it\":5}")];
        assert!(!repair_argument_names(&mut calls, &tools));
    }

    #[test]
    fn name_repair_then_coercion_compose() {
        // A wrongly-styled key carrying a stringified scalar: rename fills the
        // required field, then coercion fixes its type.
        let tools = [typed_tool_req(
            json!({"maxItems": {"type": "integer"}}),
            &["maxItems"],
        )];
        let mut calls = [call("f", "{\"max_items\":\"5\"}")];
        assert!(repair_argument_names(&mut calls, &tools));
        assert!(coerce_arguments(&mut calls, &tools));
        assert_eq!(calls[0].arguments, "{\"maxItems\":5}");
        assert_eq!(validate(&calls, &tools), Validation::Valid);
    }

    #[test]
    fn name_repair_reports_no_change_when_nothing_matches() {
        let tools = [typed_tool_req(
            json!({"filePath": {"type": "string"}}),
            &["filePath"],
        )];
        // `destination` shares no normalized form with `filePath`.
        let mut calls = [call("f", "{\"destination\":\"/x\"}")];
        assert!(!repair_argument_names(&mut calls, &tools));
    }
}
