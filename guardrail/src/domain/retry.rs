//! Milestone 6: retry bookkeeping.
//!
//! When a tool call fails validation, the guardrail loop appends a corrective
//! nudge and asks the backend again, up to a budget. [`ErrorTracker`] holds that
//! budget. The nudge is delivered on the **canonical tool channel** (matching
//! forge): the model's tool call is echoed as an `assistant` message and the
//! correction comes back as a `tool` message per call — the channel the model
//! was pretrained on, which survives heavy-context attention drop-off better
//! than a trailing `user` message.

use serde_json::{json, Value};

use super::decode::{canonical_tool_calls, ToolCall};

/// Tracks how many corrective retries remain in a single guardrail loop.
#[derive(Debug)]
pub struct ErrorTracker {
    max_retries: u32,
    attempts: u32,
}

impl ErrorTracker {
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            attempts: 0,
        }
    }

    /// Whether another retry is still within budget.
    pub fn can_retry(&self) -> bool {
        self.attempts < self.max_retries
    }

    /// Record that a retry is being issued.
    pub fn record_retry(&mut self) {
        self.attempts += 1;
    }

    /// Number of retries issued so far.
    pub fn attempts(&self) -> u32 {
        self.attempts
    }
}

/// Build the follow-up messages to append after a failed tool call: the
/// `assistant` message echoing the (canonical) tool calls, then one `tool`
/// message per call carrying the corrective nudge. Every tool call is answered
/// so the next request is schema-valid.
pub fn tool_error_followup(calls: &[ToolCall], nudge: &str) -> Vec<Value> {
    let canonical = canonical_tool_calls(calls);
    let mut messages = vec![json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": canonical.clone(),
    })];
    if let Some(array) = canonical.as_array() {
        for call in array {
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call.get("id").cloned().unwrap_or(Value::Null),
                "content": nudge,
            }));
        }
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_respected() {
        let mut t = ErrorTracker::new(2);
        assert!(t.can_retry());
        t.record_retry();
        assert!(t.can_retry());
        t.record_retry();
        assert!(!t.can_retry());
        assert_eq!(t.attempts(), 2);
    }

    #[test]
    fn zero_budget_never_retries() {
        let t = ErrorTracker::new(0);
        assert!(!t.can_retry());
    }

    #[test]
    fn followup_echoes_calls_and_answers_each_on_tool_channel() {
        let calls = vec![
            ToolCall {
                id: Some("call_x".into()),
                name: "a".into(),
                arguments: "{}".into(),
            },
            ToolCall {
                id: None,
                name: "b".into(),
                arguments: "{}".into(),
            },
        ];
        let msgs = tool_error_followup(&calls, "fix it");

        // assistant echo + one tool message per call.
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["tool_calls"].as_array().unwrap().len(), 2);

        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_call_id"], "call_x");
        assert_eq!(msgs[1]["content"], "fix it");
        // Minted id for the call that lacked one.
        assert_eq!(msgs[2]["tool_call_id"], "call_1");
    }
}
