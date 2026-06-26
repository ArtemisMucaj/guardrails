/// Runtime configuration for the guardrails.
///
/// Rescue, respond, and retry are always on and no longer individually gated;
/// the only knob is the retry budget. A `max_retries` of `0` disables the retry
/// loop while leaving the deterministic repairs (rescue, argument coercion, and
/// name repair) in effect.
#[derive(Clone, Copy, Debug)]
pub struct Guardrails {
    pub max_retries: u32,
}

impl Default for Guardrails {
    fn default() -> Self {
        Self { max_retries: 2 }
    }
}
