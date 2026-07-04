use crate::domain::repetition::Repetition;

/// Runtime configuration for the guardrails.
///
/// Rescue, respond, retry, and repetition detection are always on and no longer
/// individually gated; the knobs are the retry budget and the repetition
/// threshold. A `max_retries` of `0` disables the retry loop while leaving the
/// deterministic repairs (rescue, argument coercion, and name repair) in effect;
/// a `repetition.min_repeats` below `2` disables the loop detector.
#[derive(Clone, Copy, Debug)]
pub struct Guardrails {
    pub max_retries: u32,
    /// Detection of degenerate repetition loops in the model's text output.
    pub repetition: Repetition,
}

impl Default for Guardrails {
    fn default() -> Self {
        Self {
            max_retries: 2,
            repetition: Repetition::default(),
        }
    }
}
