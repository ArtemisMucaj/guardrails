//! Degenerate-repetition detection for streamed model text.
//!
//! A common failure mode of local / smaller models — the population this proxy
//! targets — is getting stuck in a loop: emitting the same token, phrase, or
//! line over and over until the context or a token limit is hit. The output is
//! useless to the client and, on a streamed connection, floods it with garbage.
//!
//! This module implements a small, allocation-light detector that the SSE
//! assembler runs incrementally over the model's accumulated text. It looks at
//! the *tail* of the text for a unit that repeats back-to-back: the smallest
//! period `p` such that the last `p * repeats` characters are the same `p`-char
//! block repeated `repeats` times. That single check catches both single-token
//! runaways (`"the the the …"`, period 4) and whole-line loops
//! (`"I can't help.\nI can't help.\n…"`, period = the line length), without any
//! model-specific heuristics.
//!
//! When repetition is detected the assembler stops forwarding further tokens and
//! ends the turn, keeping one clean copy of the repeated unit — see
//! [`truncated`]. On a buffered (non-streaming) backend the whole de-looped
//! answer is delivered; on a live stream the runaway tail is simply cut off.
//!
//! The detector is deliberately conservative: the repeating unit must be
//! non-trivial (not whitespace or a run of a single punctuation character, so
//! Markdown rules like `----` and dot leaders never trip it), it must repeat at
//! least [`Repetition::min_repeats`] times, and the whole run must span at least
//! [`Repetition::min_run_len`] characters. That keeps ordinary prose and code —
//! which repeats a little — well clear of the threshold while still catching the
//! dozens-to-thousands of copies a genuine loop produces.

/// Runtime configuration for repetition detection.
///
/// Detection is disabled when [`Repetition::min_repeats`] is below `2` (there is
/// no such thing as a repetition of fewer than two copies), which is how the
/// `--repetition-threshold 0` / `1` CLI setting turns the guard off.
#[derive(Clone, Copy, Debug)]
pub struct Repetition {
    /// Minimum number of back-to-back copies of the repeating unit before the
    /// tail is treated as a degenerate loop.
    pub min_repeats: u32,
    /// Minimum total length, in characters, the repeating run must span. Short
    /// units (a single character, a two-letter token) therefore have to repeat
    /// many more times than a long one before they count — a guard against
    /// flagging incidental short repeats.
    pub min_run_len: usize,
    /// Largest repeating-unit length, in characters, the detector will consider.
    /// Bounds the work per check and reflects that real loops have short-to-
    /// moderate periods (a token, a phrase, a handful of lines).
    pub max_period: usize,
}

impl Default for Repetition {
    fn default() -> Self {
        Self {
            min_repeats: 4,
            min_run_len: 40,
            max_period: 512,
        }
    }
}

impl Repetition {
    /// Whether the detector is active. A `min_repeats` below `2` disables it.
    pub fn enabled(&self) -> bool {
        self.min_repeats >= 2 && self.max_period >= 1
    }

    /// Number of characters at the tail of the text worth scanning to *find* a
    /// candidate period. A run only needs to be visible over a few copies to be
    /// identified; the true extent is then measured over the full text.
    fn scan_window(&self) -> usize {
        self.max_period.saturating_mul(self.min_repeats as usize + 1)
    }
}

/// A detected tail repetition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Detection {
    /// Number of back-to-back copies of the unit at the tail of the text.
    pub repeats: u32,
    /// Length of the repeating unit, in characters.
    pub unit_len: usize,
    /// Character index in the text where the repeating run begins.
    pub run_start: usize,
}

impl Detection {
    /// Character index just past the *first* copy of the repeating unit — the
    /// point to truncate at so one clean copy is kept and the rest of the loop is
    /// dropped.
    pub fn keep_boundary(&self) -> usize {
        self.run_start + self.unit_len
    }
}

/// Detect a degenerate repetition at the tail of `text`.
///
/// Returns the tightest (smallest-period) qualifying run, or `None` when the
/// text does not end in a loop that meets the configured thresholds.
pub fn detect(text: &str, cfg: &Repetition) -> Option<Detection> {
    if !cfg.enabled() {
        return None;
    }

    // Search a bounded tail window for a candidate period — cheap and
    // independent of how long the whole response is.
    let window: Vec<char> = tail_chars(text, cfg.scan_window());
    if window.len() < cfg.min_repeats as usize {
        return None;
    }

    // Measuring the true run length needs the full character sequence, but only
    // once a candidate is found, so the common (no-loop) case never pays for it.
    let mut full: Option<Vec<char>> = None;
    let max_p = cfg.max_period.min(window.len() / cfg.min_repeats as usize);

    for p in 1..=max_p {
        if count_back(&window, p) < cfg.min_repeats {
            continue;
        }
        let unit = &window[window.len() - p..];
        if is_trivial(unit) {
            continue;
        }
        // Candidate period looks periodic in the window; measure how far the run
        // actually extends back over the entire text.
        let full = full.get_or_insert_with(|| text.chars().collect());
        let repeats = count_back(full, p);
        let run_len = repeats as usize * p;
        if repeats >= cfg.min_repeats && run_len >= cfg.min_run_len {
            return Some(Detection {
                repeats,
                unit_len: p,
                run_start: full.len() - run_len,
            });
        }
    }
    None
}

/// Return `text` with the repeated run collapsed to a single copy of its unit:
/// everything up to and including the first copy is kept, the rest is dropped.
pub fn truncated(text: &str, det: &Detection) -> String {
    text.chars().take(det.keep_boundary()).collect()
}

/// Count how many back-to-back copies of the last `p`-character block appear at
/// the tail of `chars` (the tail block itself counts as the first copy).
fn count_back(chars: &[char], p: usize) -> u32 {
    let n = chars.len();
    if p == 0 || p > n {
        return 0;
    }
    let tail = &chars[n - p..];
    let mut reps = 1u32;
    loop {
        let end = n - reps as usize * p;
        if end < p {
            break;
        }
        if &chars[end - p..end] == tail {
            reps += 1;
        } else {
            break;
        }
    }
    reps
}

/// The last `max_chars` characters of `s`, as a `Vec<char>` (fewer when `s` is
/// shorter). Walks back from the end so cost is bounded by `max_chars`, not the
/// full length of `s`.
fn tail_chars(s: &str, max_chars: usize) -> Vec<char> {
    let mut start = s.len();
    let mut seen = 0;
    for (idx, _) in s.char_indices().rev() {
        start = idx;
        seen += 1;
        if seen == max_chars {
            break;
        }
    }
    s[start..].chars().collect()
}

/// Whether a repeating unit is too trivial to count as a degenerate loop:
/// all-whitespace, or made up of a single distinct punctuation character
/// (padding out to any amount of whitespace). This keeps horizontal rules
/// (`----`, `====`, `****`), dot leaders (`....`), and blank runs from tripping
/// the detector, while a repeated word, phrase, line, or single *letter* run
/// still qualifies.
fn is_trivial(unit: &[char]) -> bool {
    let mut marker: Option<char> = None;
    for &c in unit {
        if c.is_whitespace() {
            continue;
        }
        if !c.is_ascii_punctuation() {
            return false; // has a letter/digit/other — not trivial
        }
        match marker {
            Some(m) if m != c => return false, // two different symbols — not trivial
            _ => marker = Some(c),
        }
    }
    // All whitespace, or a single distinct punctuation character.
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Repetition {
        Repetition::default()
    }

    #[test]
    fn clean_prose_is_not_flagged() {
        let text = "The quick brown fox jumps over the lazy dog. \
                    It was a bright cold day in April and the clocks were striking thirteen.";
        assert_eq!(detect(text, &cfg()), None);
    }

    #[test]
    fn repeated_line_is_detected() {
        let text = "Here is my answer.\n".to_string() + &"I cannot help with that.\n".repeat(8);
        let det = detect(&text, &cfg()).expect("loop should be detected");
        assert!(det.repeats >= 4);
        // Truncating keeps the good prefix plus exactly one copy of the line.
        let kept = truncated(&text, &det);
        assert!(kept.starts_with("Here is my answer.\n"));
        assert_eq!(kept.matches("I cannot help with that.").count(), 1);
    }

    #[test]
    fn single_token_runaway_is_detected() {
        let text = "loading the the the the the the the the the the the the ";
        let det = detect(text, &cfg()).expect("token loop should be detected");
        // Smallest period wins: "the " is 4 characters.
        assert_eq!(det.unit_len, 4);
        assert!(det.repeats >= 4);
    }

    #[test]
    fn short_repeat_below_run_length_is_ignored() {
        // "ab" three times is a repeat, but nowhere near the run-length floor.
        assert_eq!(detect("value=abababab", &cfg()), None);
    }

    #[test]
    fn horizontal_rule_is_not_flagged() {
        let text = "Section\n".to_string() + &"-".repeat(80);
        assert_eq!(detect(&text, &cfg()), None);
    }

    #[test]
    fn separator_line_run_is_not_flagged() {
        // A block of dashed rules is formatting, not a degenerate loop.
        let text = "----\n".repeat(20);
        assert_eq!(detect(&text, &cfg()), None);
    }

    #[test]
    fn single_letter_runaway_is_flagged() {
        // A single *letter* repeated is a real loop (unlike a punctuation rule).
        let text = format!("hmm {}", "a".repeat(80));
        let det = detect(&text, &cfg()).expect("letter runaway should be detected");
        assert_eq!(det.unit_len, 1);
    }

    #[test]
    fn disabled_config_detects_nothing() {
        let disabled = Repetition {
            min_repeats: 0,
            ..Repetition::default()
        };
        assert!(!disabled.enabled());
        let text = "stuck stuck stuck stuck stuck stuck stuck stuck ";
        assert_eq!(detect(text, &disabled), None);
    }

    #[test]
    fn detects_loop_only_at_the_tail_after_good_content() {
        // A long, varied answer that only degenerates at the very end.
        let good = "First, consider the tradeoffs between latency and throughput. \
                    Then weigh the operational cost of each option carefully.\n";
        let text = format!("{good}{}", "no ".repeat(20));
        let det = detect(&text, &cfg()).expect("tail loop should be detected");
        let kept = truncated(&text, &det);
        assert!(kept.starts_with("First, consider"));
        // The de-looped answer keeps the substance and drops the runaway.
        assert!(kept.len() < text.len());
        assert!(kept.matches("no no").count() <= 1);
    }

    #[test]
    fn run_start_and_boundary_are_consistent() {
        let text = format!("prefix {}", "xy".repeat(40));
        let det = detect(&text, &cfg()).unwrap();
        assert_eq!(det.run_start + det.repeats as usize * det.unit_len, text.chars().count());
        assert_eq!(det.keep_boundary(), det.run_start + det.unit_len);
    }

    #[test]
    fn handles_multibyte_characters() {
        // Non-ASCII repeated phrase — indexing must stay char-based, not byte.
        let text = "résultat: ".to_string() + &"café café café café café café ".repeat(3);
        let det = detect(&text, &cfg()).expect("unicode loop should be detected");
        let kept = truncated(&text, &det);
        assert!(kept.starts_with("résultat: "));
        assert!(kept.chars().count() < text.chars().count());
    }

    #[test]
    fn empty_and_tiny_text_are_safe() {
        assert_eq!(detect("", &cfg()), None);
        assert_eq!(detect("hello", &cfg()), None);
    }
}
