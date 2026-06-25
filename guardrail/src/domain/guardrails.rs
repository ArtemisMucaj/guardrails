/// Runtime on/off state for each guardrail, plus the retry budget.
/// Every guardrail is independently toggleable so the proxy can degrade to a
/// zero-overhead passthrough; all default on.
#[derive(Clone, Copy, Debug)]
pub struct Guardrails {
    pub rescue: bool,
    pub respond: bool,
    pub retry: bool,
    pub max_retries: u32,
}

impl Default for Guardrails {
    fn default() -> Self {
        Self {
            rescue: true,
            respond: true,
            retry: true,
            max_retries: 2,
        }
    }
}

impl Guardrails {
    /// Whether any guardrail is enabled. When false the tool-enabled path is a
    /// plain passthrough.
    pub fn any_active(&self) -> bool {
        self.rescue || self.respond || self.retry
    }
}
