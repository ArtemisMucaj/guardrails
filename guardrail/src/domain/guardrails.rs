/// Runtime configuration for the guardrails.
///
/// Rescue, respond, and retry are always on and no longer individually gated;
/// the only knob is the retry budget. A `max_retries` of `0` disables the retry
/// loop while leaving the deterministic repairs (rescue, argument coercion, and
/// name repair) in effect.
#[derive(Clone, Copy, Debug)]
pub struct Guardrails {
    pub max_retries: u32,
    /// Reconstruct conversations from Chat Completions traffic by matching each
    /// request's message prefix against recent turns.
    ///
    /// Off by default, and deliberately opt-in: it is the one thing that makes
    /// the metrics path read message content (to hash it — no text is stored;
    /// see [`crate::domain::conversation`]), and the grouping it produces is
    /// approximate. The Responses API is unaffected either way, since it
    /// supplies real conversation edges.
    pub match_conversations: bool,
}

impl Default for Guardrails {
    fn default() -> Self {
        Self {
            max_retries: 2,
            match_conversations: false,
        }
    }
}
