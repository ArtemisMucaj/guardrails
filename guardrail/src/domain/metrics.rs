//! Failure-metrics recording.
//!
//! Every guarded request that flows through the guardrail loop ends in exactly
//! one terminal [`Outcome`]. The loop builds an [`OutcomeRecord`] at each such
//! point and hands it to a [`Recorder`]. The recorder is a sink abstraction:
//! the default [`NoopRecorder`] discards records (metrics off), while
//! [`SqliteRecorder`] persists them to a local SQLite database for later
//! aggregate querying (totals per model, success/error proportions by category,
//! and the list of errors the guardrails could not fix).
//!
//! Keeping the sink behind a trait means an OpenTelemetry / OTLP exporter can be
//! added later as a second `Recorder` implementation without touching the loop.

use std::sync::Arc;

use crate::domain::validate::ErrorCategory;

/// Terminal classification of a single guarded request.
///
/// Variants map one-to-one onto the `return` points of the guardrail loop. The
/// snake_case [`Outcome::as_str`] tag is what gets stored and grouped on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Valid native `tool_calls`, forwarded unchanged.
    NativeValid,
    /// Tool calls recovered from model text by a rescue parser.
    Rescued,
    /// Tool calls made valid by deterministic argument repair (name / type).
    Repaired,
    /// Initially invalid, then valid after one or more corrective retries.
    RecoveredAfterRetry,
    /// The synthetic `respond` tool carried the model's final text answer.
    RespondIntercept,
    /// Retries exhausted and the call was still invalid — the guardrails could
    /// not fix it. This is the population worth triaging.
    RetriesExhausted,
    /// A write-only tool was called on a file that already exists. The model
    /// was instructed to read the file first and use the edit tool instead.
    WriteRefused,
    /// The model returned plain text with no tool call to validate.
    PassthroughNoCalls,
    /// A streaming request that declared no tools, forwarded live and unguarded.
    /// With no declared tool there is no tool call to validate; recording it
    /// keeps streamed chat traffic visible. (Streaming requests that *do* declare
    /// tools are buffered and guarded like any other tool request — see the
    /// proxy's dispatch — so they never land here.)
    StreamedPassthrough,
    /// A non-streaming request that declared no tools, forwarded unguarded.
    /// There was no tool call to check, but it is recorded so the report
    /// reflects all chat traffic rather than only the guarded slice.
    NonToolPassthrough,
    /// The backend response was not JSON and was forwarded unverified.
    NonJson,
    /// The backend request itself failed (connection refused, timeout, …); the
    /// proxy never received a response to guard.
    BackendError,
    /// The proxy could not serialize the (re)built request — an internal error.
    InternalError,
}

impl Outcome {
    /// Stable snake_case tag for storage and grouping.
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::NativeValid => "native_valid",
            Outcome::Rescued => "rescued",
            Outcome::Repaired => "repaired",
            Outcome::RecoveredAfterRetry => "recovered_after_retry",
            Outcome::RespondIntercept => "respond_intercept",
            Outcome::RetriesExhausted => "retries_exhausted",
            Outcome::WriteRefused => "write_refused",
            Outcome::PassthroughNoCalls => "passthrough_no_calls",
            Outcome::StreamedPassthrough => "streamed_passthrough",
            Outcome::NonToolPassthrough => "non_tool_passthrough",
            Outcome::NonJson => "non_json",
            Outcome::BackendError => "backend_error",
            Outcome::InternalError => "internal_error",
        }
    }

    /// Outcome tags that represent a real tool call against the client's
    /// declared tools. Excludes `respond_intercept` (the synthetic `respond`
    /// path is the model's final *text* answer, not a tool call) and the
    /// no-call outcomes. This is the denominator for the success rate, and is
    /// also formatted into the stats SQL so the two never drift.
    pub const TOOL_CALL_TAGS: [&'static str; 6] = [
        "native_valid",
        "rescued",
        "repaired",
        "recovered_after_retry",
        "retries_exhausted",
        "write_refused",
    ];

    /// Whether this outcome is a real tool call against a client-declared tool
    /// (see [`Outcome::TOOL_CALL_TAGS`]). Used to count "tool calls total".
    pub fn is_tool_call(self) -> bool {
        Self::TOOL_CALL_TAGS.contains(&self.as_str())
    }

    /// Whether the guardrails produced a usable result. `RetriesExhausted` and
    /// `WriteRefused` both represent failures the guardrails could not resolve.
    pub fn fixed(self) -> bool {
        !matches!(self, Outcome::RetriesExhausted | Outcome::WriteRefused)
    }
}

/// Token usage reported by the backend for a guarded request.
///
/// Accumulated across *every* backend attempt a request made, not just the one
/// that produced the delivered answer: a corrective retry is a second billed
/// call, and hiding it would understate what the guardrails cost. `attempts`
/// records how many backend calls the totals span so the overhead stays
/// legible.
///
/// The cached counts are the read side of prompt caching — the portion of
/// `prompt_tokens` that was served from cache and so billed at a discount.
/// Both APIs report it in a nested object (`prompt_tokens_details.cached_tokens`
/// for chat, `input_tokens_details.cached_tokens` for responses), and some
/// backends omit the object entirely; a missing count reads as zero rather than
/// as an error, since "no cache hit" and "no cache reporting" are indistinguish-
/// able from the proxy's side.
///
/// # Prompt tokens do not add up across a conversation
///
/// Every chat turn resends the whole transcript, so turn 5's prompt *contains*
/// turns 1–4. Summing `prompt_tokens` over the requests of one conversation
/// therefore counts the early turns once per later turn — growth that is
/// quadratic in turn count, not a measurement of distinct tokens.
///
/// That makes the sum a faithful answer to "what did the provider bill" and a
/// wrong answer to "how many tokens did this conversation contain". Only
/// [`completion_tokens`](Self::completion_tokens) is generated once and so sums
/// cleanly. Every rendered figure is therefore labelled *billed* rather than
/// *total*, and [`uncached_prompt_tokens`](Self::uncached_prompt_tokens) is
/// shown alongside because resent prefixes are exactly what a prompt cache
/// serves — the inflation and its discount are the same tokens.
///
/// Deduplicating properly needs per-conversation grouping (take the maximum
/// prompt over a chain, not the sum), which needs a conversation key the Chat
/// Completions API does not carry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Tokens in the prompt, cached and uncached together. Not additive across
    /// a conversation — see the type docs.
    pub prompt_tokens: i64,
    /// Tokens the model generated. Generated once and never resent, so this is
    /// the one figure that sums cleanly.
    pub completion_tokens: i64,
    /// Of `prompt_tokens`, the portion served from the prompt cache.
    pub cached_tokens: i64,
    /// Backend calls these totals span. `1` for a request answered on the first
    /// attempt; higher when the guardrails retried.
    pub attempts: i64,
}

impl Usage {
    /// Whether the backend reported anything at all. A request whose backend
    /// sent no usage block records zeroes, and counting those rows as "0 tokens"
    /// in an average would drag it down; callers use this to skip them.
    pub fn is_empty(self) -> bool {
        self.prompt_tokens == 0 && self.completion_tokens == 0 && self.cached_tokens == 0
    }

    /// Prompt tokens that missed the cache and were billed at full rate.
    pub fn uncached_prompt_tokens(self) -> i64 {
        (self.prompt_tokens - self.cached_tokens).max(0)
    }

    /// Fold another usage total into this one.
    ///
    /// Adds `other.attempts` rather than a hard `1`, so folding an already
    /// aggregated total keeps the backend-call count right. Every caller today
    /// passes a single attempt straight from [`extract_usage`] (which reports
    /// `attempts: 1`), making the two equivalent in practice — but a hard `1`
    /// would silently undercount the moment anything folded a running total in,
    /// and `calls_per_request` would understate the retry multiplier it exists
    /// to expose.
    pub fn add(&mut self, other: Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.cached_tokens += other.cached_tokens;
        self.attempts += other.attempts;
    }
}

/// Extract token usage from a backend response body or terminal SSE chunk.
///
/// Reads both API dialects: Chat Completions names the fields
/// `prompt_tokens` / `completion_tokens`, the Responses API names them
/// `input_tokens` / `output_tokens`. Returns `None` when the value carries no
/// `usage` object, which is the common case for every non-terminal chunk.
pub fn extract_usage(value: &serde_json::Value) -> Option<Usage> {
    let usage = value.get("usage")?;
    let field = |names: [&str; 2]| -> i64 {
        names
            .iter()
            .find_map(|n| usage.get(*n).and_then(serde_json::Value::as_i64))
            .unwrap_or(0)
    };
    // Both dialects nest the cache read count one level down, under a details
    // object named after their own input field.
    let cached = ["prompt_tokens_details", "input_tokens_details"]
        .iter()
        .find_map(|d| {
            usage
                .get(*d)
                .and_then(|d| d.get("cached_tokens"))
                .and_then(serde_json::Value::as_i64)
        })
        .unwrap_or(0);
    Some(Usage {
        prompt_tokens: field(["prompt_tokens", "input_tokens"]),
        completion_tokens: field(["completion_tokens", "output_tokens"]),
        cached_tokens: cached,
        attempts: 1,
    })
}

/// One row of failure metrics: the terminal outcome of a guarded request.
#[derive(Debug, Clone)]
pub struct OutcomeRecord {
    /// RFC3339 UTC timestamp, stamped when the outcome occurs (not when it is
    /// written), so a backed-up writer queue does not skew the recorded time.
    pub ts: String,
    /// Name of the provider that served the request. Recorded alongside the
    /// model because the same id can be served by several providers, and
    /// merging their outcomes would hide which upstream is actually failing.
    pub provider: String,
    pub model: String,
    pub outcome: Outcome,
    /// Failure category, present only on `RetriesExhausted`.
    pub error_category: Option<ErrorCategory>,
    /// Rescue parser name, present only on `Rescued`.
    pub parser: Option<String>,
    /// The primary tool involved (offending call on failure).
    pub tool_name: Option<String>,
    /// Corrective retries issued before this outcome.
    pub retries: u32,
    /// Triage detail: the last nudge plus a redacted argument snippet, on
    /// failure outcomes only.
    pub detail: Option<String>,
    /// Token usage summed over every backend attempt this request made. Absent
    /// when the request never reached a backend (an internal error) or the
    /// backend reported no usage.
    pub usage: Option<Usage>,
    /// Identity of this turn within a conversation, when the API provides one.
    /// Present for chained Responses traffic; always absent on Chat
    /// Completions, which carries no conversation key. See [`Conversation`].
    pub conversation: Option<Conversation>,
}

/// Where a request sits in a conversation.
///
/// Only the Responses API can supply this: it is stateful, so a client
/// continues an exchange by naming the previous response rather than resending
/// the transcript. `id` is this turn's response, `parent` the one it continues.
/// Together they form the edges of a chain whose root identifies the
/// conversation.
///
/// This is what makes prompt tokens summable. Because each turn's prompt
/// contains every earlier turn, the *last* turn of a chain already accounts for
/// the whole conversation — so a conversation contributes the maximum prompt
/// over its turns, not the sum (see [`Usage`]). Without these edges there is no
/// way to tell two turns of one conversation from two unrelated requests, and
/// the shared prefix is counted twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversation {
    /// This turn's response id, as the backend assigned it.
    pub id: String,
    /// The response this turn continues; `None` on the first turn, which makes
    /// this row the root of its chain.
    pub parent: Option<String>,
}

/// A sink for terminal outcome records.
///
/// `record` is called on the request hot path; implementations must not block
/// (the SQLite sink hands the row to a background writer thread and returns).
pub trait Recorder: Send + Sync {
    fn record(&self, record: OutcomeRecord);
}

/// Default recorder: drops every record. Used when metrics are not configured.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopRecorder;

impl Recorder for NoopRecorder {
    fn record(&self, _record: OutcomeRecord) {}
}

/// Shared handle to whichever recorder the proxy is running with.
pub type SharedRecorder = Arc<dyn Recorder>;

/// Build a privacy-preserving, single-line snippet of a tool call's arguments
/// for triage.
///
/// Argument *values* can carry secrets or PII, and metrics are always on, so
/// values are never stored verbatim. A JSON object is reduced to its keys with
/// each value replaced by a type/size tag (`<str:LEN>`, `<number>`, `<array:N>`,
/// …); anything that does not parse as a JSON object becomes a bare
/// `<non-object: N chars>` marker. Knowing which fields were present and their
/// shape is what makes a fallback row actionable — the concrete values are not.
pub fn redact_args(arguments: &str) -> String {
    const MAX: usize = 200;
    let snippet = match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(serde_json::Value::Object(map)) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(key, value)| format!("{key}: {}", redact_value(value)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        // Not an object (or unparseable): keep only a length marker, never the
        // raw text — the `bad_arguments` case routinely lands here.
        _ => format!("<non-object: {} chars>", arguments.chars().count()),
    };
    if snippet.chars().count() > MAX {
        let head: String = snippet.chars().take(MAX).collect();
        format!("{head}…")
    } else {
        snippet
    }
}

/// A non-revealing tag describing a JSON value's type and size (never its
/// contents).
fn redact_value(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(_) => "<bool>".to_string(),
        Value::Number(_) => "<number>".to_string(),
        Value::String(s) => format!("<str:{}>", s.chars().count()),
        Value::Array(a) => format!("<array:{}>", a.len()),
        Value::Object(o) => format!("<object:{}>", o.len()),
    }
}

/// Current time as an RFC3339 UTC timestamp, without pulling in a date library.
pub fn now_rfc3339() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Howard Hinnant's days-to-civil-date algorithm (days since 1970-01-01).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_rfc3339_is_well_formed_utc() {
        let ts = now_rfc3339();
        assert_eq!(ts.len(), 20, "expected YYYY-MM-DDThh:mm:ssZ, got {ts}");
        assert!(ts.ends_with('Z'));
        assert_eq!(&ts[4..5], "-");
        // Epoch and a known leap-aware date anchor the civil-date math.
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
    }

    #[test]
    fn respond_intercept_is_not_counted_as_a_tool_call() {
        assert!(!Outcome::RespondIntercept.is_tool_call());
    }

    #[test]
    fn tool_call_outcomes_are_distinguished_from_passthrough() {
        assert!(Outcome::NativeValid.is_tool_call());
        assert!(Outcome::RetriesExhausted.is_tool_call());
        assert!(!Outcome::PassthroughNoCalls.is_tool_call());
        assert!(!Outcome::StreamedPassthrough.is_tool_call());
        assert!(!Outcome::NonToolPassthrough.is_tool_call());
        assert!(!Outcome::NonJson.is_tool_call());
        assert!(!Outcome::BackendError.is_tool_call());
        assert!(!Outcome::InternalError.is_tool_call());
    }

    #[test]
    fn forwarded_passthroughs_are_not_errors() {
        // Streaming / non-tool requests are forwarded unguarded; they are
        // recorded for visibility, not as failures.
        assert!(Outcome::StreamedPassthrough.fixed());
        assert!(Outcome::NonToolPassthrough.fixed());
    }

    #[test]
    fn only_fallback_is_unfixed() {
        assert!(Outcome::Rescued.fixed());
        assert!(Outcome::Repaired.fixed());
        assert!(Outcome::RecoveredAfterRetry.fixed());
        assert!(!Outcome::RetriesExhausted.fixed());
    }

    #[test]
    fn usage_is_read_from_both_api_dialects() {
        // Chat Completions names them prompt/completion and nests the cache
        // read under `prompt_tokens_details`.
        let chat = serde_json::json!({
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 80}
            }
        });
        let u = extract_usage(&chat).expect("chat usage");
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.cached_tokens, 80);
        assert_eq!(u.uncached_prompt_tokens(), 20);

        // The Responses API names them input/output, under `input_tokens_details`.
        let responses = serde_json::json!({
            "usage": {
                "input_tokens": 100, "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 80}
            }
        });
        assert_eq!(extract_usage(&responses), Some(u));
    }

    #[test]
    fn a_backend_reporting_no_cache_details_reads_as_zero_cached() {
        // Most local backends omit the details object entirely. That must read
        // as "nothing cached", not as a parse failure that drops the usage.
        let value = serde_json::json!({
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}
        });
        let u = extract_usage(&value).expect("usage without cache details");
        assert_eq!(u.cached_tokens, 0);
        assert_eq!(u.prompt_tokens, 7);
        assert_eq!(u.uncached_prompt_tokens(), 7);
    }

    #[test]
    fn a_chunk_without_usage_reports_none() {
        // Every non-terminal chunk lands here; it must not be mistaken for a
        // zeroed report.
        let chunk = serde_json::json!({"choices": [{"delta": {"content": "hi"}}]});
        assert_eq!(extract_usage(&chunk), None);
    }

    #[test]
    fn usage_sums_across_attempts_so_retries_show_their_cost() {
        // The reason usage is accumulated rather than overwritten: a request
        // the guardrails retried twice was billed three times, and a report
        // showing only the last attempt would understate it threefold.
        let mut billed = Usage::default();
        assert!(billed.is_empty());

        for _ in 0..3 {
            billed.add(Usage {
                prompt_tokens: 100,
                completion_tokens: 10,
                cached_tokens: 40,
                attempts: 1,
            });
        }
        assert_eq!(billed.prompt_tokens, 300);
        assert_eq!(billed.completion_tokens, 30);
        assert_eq!(billed.cached_tokens, 120);
        assert_eq!(billed.attempts, 3, "three backend calls were billed");
        assert!(!billed.is_empty());
    }

    #[test]
    fn folding_an_aggregated_total_preserves_the_call_count() {
        // `add` takes the other side's attempts rather than assuming one, so
        // combining two running totals gives the same count as adding every
        // attempt individually. Callers all pass single attempts today; this
        // keeps that from being load-bearing.
        let attempt = Usage { prompt_tokens: 100, completion_tokens: 10, cached_tokens: 40, attempts: 1 };

        let mut one_at_a_time = Usage::default();
        for _ in 0..4 {
            one_at_a_time.add(attempt);
        }

        let mut left = Usage::default();
        left.add(attempt);
        left.add(attempt);
        let mut right = Usage::default();
        right.add(attempt);
        right.add(attempt);
        let mut combined = left;
        combined.add(right);

        assert_eq!(combined, one_at_a_time);
        assert_eq!(combined.attempts, 4, "four backend calls, however grouped");
    }

    #[test]
    fn cached_tokens_never_exceed_the_prompt_in_the_uncached_split() {
        // A backend reporting more cached than prompt tokens would otherwise
        // yield a negative "new tokens" figure in the report.
        let u = Usage { prompt_tokens: 10, completion_tokens: 0, cached_tokens: 25, attempts: 1 };
        assert_eq!(u.uncached_prompt_tokens(), 0);
    }

    #[test]
    fn redact_args_keeps_shape_but_never_values() {
        // Object: keys are kept (sorted by serde_json's BTreeMap), values become
        // type/size tags. "/etc/secret" is 11 chars.
        assert_eq!(
            redact_args("{\"filePath\":\"/etc/secret\",\"count\":3}"),
            "{count: <number>, filePath: <str:11>}"
        );
        // The raw secret value never appears in the snippet.
        assert!(!redact_args("{\"token\":\"sk-abc123xyz\"}").contains("sk-abc123xyz"));
        // Non-object (or unparseable) input is reduced to a length marker, with
        // no raw content and no newlines.
        let r = redact_args("not json, has a secret\nvalue");
        assert!(r.starts_with("<non-object:"));
        assert!(!r.contains("secret"));
        assert!(!r.contains('\n'));
        // Output stays bounded.
        let long = format!("{{\"k\":\"{}\"}}", "x".repeat(500));
        assert!(redact_args(&long).chars().count() <= 201);
    }
}

pub use sqlite::{default_db_path, ErrorGroup, ModelStats, SqliteRecorder, Stats};


mod sqlite {
    use std::path::Path;
    use std::sync::mpsc::{self, SyncSender, TrySendError};
    use std::thread::{self, JoinHandle};

    use rusqlite::Connection;
    use tracing::{error, info, warn};

    use super::{OutcomeRecord, Recorder, Usage};

    /// Bound on records buffered for the writer thread. `record` never blocks the
    /// request path: if the writer falls this far behind (e.g. a slow disk under
    /// a burst), further records are dropped rather than growing memory without
    /// limit — shedding a metric is preferable to stalling a proxied request.
    const QUEUE_CAPACITY: usize = 8192;

    /// Persists outcome records to a local SQLite database.
    ///
    /// A dedicated writer thread owns the connection; `record` only enqueues onto
    /// a bounded channel and returns immediately, so a database write never
    /// blocks the proxy's response path. On drop the channel is closed and the
    /// writer thread is joined, so queued rows are flushed before exit.
    pub struct SqliteRecorder {
        sender: Option<SyncSender<OutcomeRecord>>,
        writer: Option<JoinHandle<()>>,
    }

    impl SqliteRecorder {
        /// Open (or create) the database at `path`, ensure the schema exists, and
        /// spawn the background writer thread.
        pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
            let conn = Connection::open(path.as_ref())?;
            // WAL keeps the writer from blocking concurrent readers (e.g. an
            // analyst querying the file while the proxy runs).
            conn.pragma_update(None, "journal_mode", "WAL")?;
            reset_if_stale(&conn)?;
            conn.execute_batch(SCHEMA_TABLE)?;

            let (sender, receiver) = mpsc::sync_channel::<OutcomeRecord>(QUEUE_CAPACITY);
            let writer = thread::Builder::new()
                .name("guardrail-metrics".into())
                .spawn(move || writer_loop(conn, receiver))?;

            info!(path = %path.as_ref().display(), "metrics enabled (sqlite)");
            Ok(Self {
                sender: Some(sender),
                writer: Some(writer),
            })
        }
    }

    impl Recorder for SqliteRecorder {
        fn record(&self, record: OutcomeRecord) {
            let Some(sender) = self.sender.as_ref() else {
                return;
            };
            match sender.try_send(record) {
                Ok(()) => {}
                // Queue full: shed the metric rather than block the request path.
                Err(TrySendError::Full(_)) => {
                    warn!("metrics queue full; dropping outcome record")
                }
                // Writer gone: not worth failing a request over.
                Err(TrySendError::Disconnected(_)) => {}
            }
        }
    }

    impl Drop for SqliteRecorder {
        fn drop(&mut self) {
            // Close the channel (drop the sender) so the writer loop ends, then
            // wait for it to drain any queued rows before the process exits.
            self.sender = None;
            if let Some(writer) = self.writer.take() {
                let _ = writer.join();
            }
        }
    }

    /// Default database path: `~/.guardrails/guardrails.sql`. The
    /// `.guardrails` directory is created if absent. Falls back to the current
    /// directory when no home directory can be determined.
    pub fn default_db_path() -> std::path::PathBuf {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let dir = home.join(".guardrails");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            error!(error = %e, dir = %dir.display(), "failed to create metrics directory");
        }
        dir.join("guardrails.sql")
    }

    /// How far the conversation walk follows `parent_id` links before giving
    /// up. Bounds the recursive query against a cycle, which nothing in the
    /// proxy produces but a corrupted database could; well beyond any real
    /// conversation length, so it never truncates honest data.
    const MAX_CHAIN_DEPTH: u32 = 1024;

    /// Per-provider-and-model rollup, in the total → tool calls → errors
    /// hierarchy.
    ///
    /// Keyed on the pair, not the model alone: the same id served by two
    /// providers is two rows, so a provider degrading is visible instead of
    /// being averaged away against a healthy one.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ModelStats {
        pub provider: String,
        pub model: String,
        /// All guarded requests for this model (the denominator for how often it
        /// even attempts a tool call versus answering in text).
        pub total: i64,
        /// Of `total`, the requests that were a real tool call (see
        /// [`super::Outcome::TOOL_CALL_TAGS`]).
        pub tool_calls: i64,
        /// Of `tool_calls`, the ones the guardrails could not fix.
        pub errors: i64,
        /// Counts per outcome tag, summing to `total`.
        pub by_outcome: Vec<(String, i64)>,
        /// Token usage summed over the requests that reported any. Rows where
        /// the backend reported nothing are excluded, so this is the cost of the
        /// measured traffic rather than of all of it.
        pub usage: Usage,
        /// Of `total`, the requests that carried a usage report — the
        /// denominator `usage` is actually over.
        pub usage_requests: i64,
        /// Prompt tokens with resent transcript prefixes counted once.
        ///
        /// A conversation contributes the largest prompt among its turns rather
        /// than the sum of them, because the last turn's prompt already
        /// contains every earlier one. Unchained requests each count in full,
        /// being conversations of one turn.
        ///
        /// `None` when no request carried a conversation key — every Chat
        /// Completions deployment — since there is then no basis to tell a
        /// second turn from an unrelated request, and reporting the plain sum
        /// as if it were deduplicated would be a lie. See [`Conversation`].
        pub distinct_prompt_tokens: Option<i64>,
        /// Conversations the measured requests span, when chains are known.
        pub conversations: Option<i64>,
    }

    impl ModelStats {
        /// Tool calls the guardrails delivered as valid.
        pub fn succeeded(&self) -> i64 {
            self.tool_calls - self.errors
        }

        /// Tokens billed across prompt and completion.
        ///
        /// Deliberately *billed*, not *total*: resent transcripts mean the
        /// prompt side counts shared prefixes once per turn (see [`Usage`]).
        /// This is what the provider charged for, not a count of distinct
        /// tokens.
        pub fn billed_tokens(&self) -> i64 {
            self.usage.prompt_tokens + self.usage.completion_tokens
        }

        /// Tokens with resent transcript prefixes counted once — the honest
        /// answer to "how large was this traffic", as opposed to
        /// [`billed_tokens`](Self::billed_tokens)'s "what did it cost".
        ///
        /// `None` when conversations cannot be reconstructed; see
        /// [`distinct_prompt_tokens`](Self::distinct_prompt_tokens).
        pub fn distinct_tokens(&self) -> Option<i64> {
            self.distinct_prompt_tokens
                .map(|p| p + self.usage.completion_tokens)
        }

        /// Share of prompt tokens served from the cache, in `[0, 1]`, or `None`
        /// when no prompt tokens were reported (so the report shows `n/a`
        /// rather than a misleading `0%`).
        pub fn cache_hit_rate(&self) -> Option<f64> {
            if self.usage.prompt_tokens == 0 {
                None
            } else {
                Some(self.usage.cached_tokens as f64 / self.usage.prompt_tokens as f64)
            }
        }

        /// Backend calls per client request — `1.0` when nothing retried, higher
        /// when the guardrails had to ask again. This is the multiplier the
        /// guardrails apply to the bill, and `None` without any usage report.
        pub fn calls_per_request(&self) -> Option<f64> {
            if self.usage_requests == 0 {
                None
            } else {
                Some(self.usage.attempts as f64 / self.usage_requests as f64)
            }
        }

        /// Success rate over tool calls, or `None` when the model made no tool
        /// call (so the report shows `n/a` rather than a misleading `0%`).
        pub fn success_rate(&self) -> Option<f64> {
            if self.tool_calls == 0 {
                None
            } else {
                Some(self.succeeded() as f64 / self.tool_calls as f64)
            }
        }
    }

    /// One group of identical errors the guardrails could not fix, awaiting
    /// triage.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ErrorGroup {
        pub provider: String,
        pub model: String,
        pub error_category: Option<String>,
        pub tool_name: Option<String>,
        pub detail: Option<String>,
        pub count: i64,
    }

    /// A full read of the guardrails database for the `stats` command.
    #[derive(Debug, Clone, Default)]
    pub struct Stats {
        pub per_model: Vec<ModelStats>,
        pub errors: Vec<ErrorGroup>,
    }

    impl Stats {
        /// Read and aggregate metrics from the database at `path`. A missing
        /// database (proxy never run) reads as empty stats rather than an error.
        pub fn read(path: impl AsRef<Path>) -> anyhow::Result<Self> {
            if !path.as_ref().exists() {
                return Ok(Self::default());
            }
            let conn = Connection::open(path.as_ref())?;

            // The proxy may have created the file but not yet committed the
            // schema (it writes on a background thread). Treat an absent table as
            // empty rather than failing the command.
            let has_table: bool = conn
                .query_row(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'outcomes'",
                    [],
                    |_| Ok(true),
                )
                .unwrap_or(false);
            if !has_table {
                return Ok(Self::default());
            }

            // `stats` reads a database the proxy owns and may not have opened
            // this run, so the table can predate any column added since. Reading
            // is not the place to migrate — the CLI may not even be able to
            // write the file — so a column that is not there is selected as
            // NULL, which every aggregate below already handles as "not
            // recorded". Without this the command fails outright on an older
            // database instead of reporting what it does have.
            let column = |name: &str| -> String {
                let present = conn
                    .prepare("SELECT 1 FROM pragma_table_info('outcomes') WHERE name = ?1")
                    .and_then(|mut s| s.exists([name]))
                    .unwrap_or(false);
                if present { name.to_string() } else { "NULL".to_string() }
            };
            let provider_col = column("provider");
            let (prompt, completion) = (column("prompt_tokens"), column("completion_tokens"));
            let (cached, billed) = (column("cached_tokens"), column("billed_calls"));

            // Per-model totals. The tool-call set is formatted from the single
            // source of truth in `Outcome` so it can never drift from the Rust
            // classification; the tags are static literals, so this is not a
            // SQL-injection surface.
            let in_list = super::Outcome::TOOL_CALL_TAGS
                .iter()
                .map(|t| format!("'{t}'"))
                .collect::<Vec<_>>()
                .join(",");
            // `COUNT(prompt_tokens)` counts only non-NULL rows, which is exactly
            // the set the token sums cover — requests whose backend reported
            // usage. A database written before the token columns existed has
            // NULL throughout and simply reports no usage.
            let query = format!(
                "SELECT COALESCE({provider_col}, 'unknown'), model, \
                    COUNT(*), \
                    SUM(CASE WHEN outcome IN ({in_list}) THEN 1 ELSE 0 END), \
                    SUM(CASE WHEN fixed = 0 THEN 1 ELSE 0 END), \
                    SUM({prompt}), SUM({completion}), SUM({cached}), \
                    SUM({billed}), COUNT({prompt}) \
                 FROM outcomes GROUP BY 1, model ORDER BY 1, model"
            );
            let mut stmt = conn.prepare(&query)?;
            let rows = stmt.query_map([], |r| {
                Ok(ModelStats {
                    provider: r.get(0)?,
                    model: r.get(1)?,
                    total: r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                    tool_calls: r.get::<_, Option<i64>>(3)?.unwrap_or(0),
                    errors: r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                    by_outcome: Vec::new(),
                    usage: Usage {
                        prompt_tokens: r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                        completion_tokens: r.get::<_, Option<i64>>(6)?.unwrap_or(0),
                        cached_tokens: r.get::<_, Option<i64>>(7)?.unwrap_or(0),
                        attempts: r.get::<_, Option<i64>>(8)?.unwrap_or(0),
                    },
                    usage_requests: r.get::<_, Option<i64>>(9)?.unwrap_or(0),
                    // Filled by the deduplication pass below.
                    distinct_prompt_tokens: None,
                    conversations: None,
                })
            })?;
            let mut per_model: Vec<ModelStats> = rows.collect::<rusqlite::Result<_>>()?;

            Self::fold_distinct_prompts(&conn, &column, &provider_col, &mut per_model)?;

            // Outcome breakdown per provider and model, folded into the rows
            // above.
            let mut stmt = conn.prepare(&format!(
                "SELECT COALESCE({provider_col}, 'unknown'), model, outcome, COUNT(*) \
                 FROM outcomes GROUP BY 1, model, outcome ORDER BY 1, model, outcome",
            ))?;
            let breakdown = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                ))
            })?;
            for row in breakdown {
                let (provider, model, outcome, count) = row?;
                if let Some(m) = per_model
                    .iter_mut()
                    .find(|m| m.provider == provider && m.model == model)
                {
                    m.by_outcome.push((outcome, count));
                }
            }

            // Errors the guardrails could not fix, grouped for triage, most
            // frequent first.
            let mut stmt = conn.prepare(&format!(
                "SELECT COALESCE({provider_col}, 'unknown'), model, error_category, tool_name, \
                    detail, COUNT(*) AS n \
                 FROM outcomes WHERE fixed = 0 \
                 GROUP BY 1, model, error_category, tool_name, detail \
                 ORDER BY n DESC, 1, model",
            ))?;
            let errors = stmt.query_map([], |r| {
                Ok(ErrorGroup {
                    provider: r.get(0)?,
                    model: r.get(1)?,
                    error_category: r.get(2)?,
                    tool_name: r.get(3)?,
                    detail: r.get(4)?,
                    count: r.get(5)?,
                })
            })?;
            let errors: Vec<ErrorGroup> = errors.collect::<rusqlite::Result<_>>()?;

            Ok(Self { per_model, errors })
        }

        /// Fold per-conversation deduplicated prompt totals into `per_model`.
        ///
        /// Each turn's prompt contains every earlier turn of its conversation,
        /// so summing them counts shared prefixes once per turn. This walks the
        /// `parent_id` edges to each chain's root, groups the turns by that
        /// root, and takes the largest prompt in each group — the last turn,
        /// which already accounts for the whole exchange. Requests with no
        /// chain are conversations of one and count in full.
        ///
        /// Leaves the fields `None` when the database has no conversation keys
        /// at all (a Chat-Completions-only deployment, or a table predating the
        /// columns): with nothing to group on, the deduplicated figure would
        /// just be the inflated sum wearing a better name.
        fn fold_distinct_prompts(
            conn: &Connection,
            column: &dyn Fn(&str) -> String,
            provider_col: &str,
            per_model: &mut [ModelStats],
        ) -> anyhow::Result<()> {
            let (response_col, parent_col) = (column("response_id"), column("parent_id"));
            if response_col == "NULL" || parent_col == "NULL" {
                return Ok(());
            }
            // Nothing chained: leave the fields absent rather than reporting a
            // deduplication that did not happen.
            let any_chain: bool = conn
                .prepare("SELECT 1 FROM outcomes WHERE response_id IS NOT NULL")?
                .exists([])?;
            if !any_chain {
                return Ok(());
            }

            // `chain` walks each turn up its parent edges, carrying the depth
            // travelled. The anchor maps every turn to itself at depth 0; the
            // recursive step steps one link further towards the first turn. So
            // a turn appears once per ancestor, and the *deepest* of those rows
            // — the one furthest from the turn — names the root of its chain.
            //
            // Depth is what picks the root, not the id. Response ids are opaque
            // strings, so ordering them lexically would let a turn whose id
            // happens to sort before its parent's keep itself as root, splitting
            // one conversation into several and restoring the double counting
            // this whole query exists to remove.
            //
            // `depth < MAX_CHAIN_DEPTH` bounds the walk. `UNION ALL` does not
            // deduplicate, so a `parent_id` cycle — which nothing in the proxy
            // should produce, but which a corrupted or hand-edited database
            // could — would otherwise recurse until the process died. A chain
            // longer than the bound simply groups from as far back as the walk
            // reached, which is a partial grouping rather than a wrong one.
            //
            // A turn whose parent was never recorded (metrics enabled
            // mid-conversation) roots at the earliest turn that *was* seen,
            // which is the best grouping the data supports.
            let query = format!(
                "WITH RECURSIVE chain(response_id, root, depth) AS (\
                     SELECT response_id, response_id, 0 FROM outcomes \
                        WHERE response_id IS NOT NULL \
                   UNION ALL \
                     SELECT o.response_id, c.root, c.depth + 1 FROM outcomes o \
                        JOIN chain c ON o.parent_id = c.response_id \
                        WHERE o.response_id IS NOT NULL AND c.depth < {MAX_CHAIN_DEPTH}\
                 ), \
                 deepest AS (\
                     SELECT response_id, root, \
                            ROW_NUMBER() OVER (\
                                PARTITION BY response_id ORDER BY depth DESC\
                            ) AS rn \
                     FROM chain\
                 ), \
                 rooted AS (\
                     SELECT o.rowid AS rid, \
                            COALESCE({provider_col}, 'unknown') AS provider, \
                            o.model AS model, \
                            o.prompt_tokens AS prompt_tokens, \
                            COALESCE(\
                                (SELECT d.root FROM deepest d \
                                 WHERE d.response_id = o.response_id AND d.rn = 1), \
                                o.response_id\
                            ) AS root \
                     FROM outcomes o \
                     WHERE o.prompt_tokens IS NOT NULL\
                 ), \
                 per_conversation AS (\
                     SELECT provider, model, \
                            COALESCE(root, 'row:' || rid) AS conversation, \
                            MAX(prompt_tokens) AS prompt_tokens \
                     FROM rooted GROUP BY provider, model, conversation\
                 ) \
                 SELECT provider, model, SUM(prompt_tokens), COUNT(*) \
                 FROM per_conversation GROUP BY provider, model"
            );

            let mut stmt = conn.prepare(&query)?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                    r.get::<_, i64>(3)?,
                ))
            })?;
            for row in rows {
                let (provider, model, distinct, conversations) = row?;
                if let Some(m) = per_model
                    .iter_mut()
                    .find(|m| m.provider == provider && m.model == model)
                {
                    m.distinct_prompt_tokens = Some(distinct);
                    m.conversations = Some(conversations);
                }
            }
            Ok(())
        }

        /// Render a plain-text report for the CLI.
        pub fn render(&self) -> String {
            use std::fmt::Write;
            let mut out = String::new();

            if self.per_model.is_empty() {
                return "No metrics recorded yet.\n".to_string();
            }

            out.push_str("Requests by provider and model\n");
            out.push_str("==============================\n");
            for m in &self.per_model {
                let rate = match m.success_rate() {
                    Some(r) => format!("{:.1}%", r * 100.0),
                    None => "n/a".to_string(),
                };
                let _ = writeln!(
                    out,
                    "\n{} / {}\n  total: {}  |  tool calls: {}  |  succeeded: {}  |  errors: {}  |  success rate: {}",
                    m.provider,
                    m.model,
                    m.total,
                    m.tool_calls,
                    m.succeeded(),
                    m.errors,
                    rate,
                );
                for (outcome, count) in &m.by_outcome {
                    let _ = writeln!(out, "    {outcome:<22} {count}");
                }
                // Only models with a usage report get a token line; printing
                // zeroes for a backend that does not report usage would read as
                // "this model is free" rather than "this is unmeasured".
                if m.usage_requests > 0 {
                    let cache = match m.cache_hit_rate() {
                        Some(r) => format!("{:.1}%", r * 100.0),
                        None => "n/a".to_string(),
                    };
                    // "new" (uncached prompt) leads because it is the additive
                    // figure: resent transcript prefixes are what the cache
                    // serves, so the cached share is the part that would double
                    // count if this were read as distinct tokens.
                    let _ = writeln!(
                        out,
                        "  tokens billed: {}  |  prompt: {} ({} new, {} cached)  |  completion: {}",
                        m.billed_tokens(),
                        m.usage.prompt_tokens,
                        m.usage.uncached_prompt_tokens(),
                        m.usage.cached_tokens,
                        m.usage.completion_tokens,
                    );
                    let calls = match m.calls_per_request() {
                        Some(c) => format!("{c:.2}"),
                        None => "n/a".to_string(),
                    };
                    let _ = writeln!(
                        out,
                        "  cache hit rate: {cache}  |  backend calls per request: {calls}  \
                         |  measured over {} of {} requests",
                        m.usage_requests, m.total,
                    );
                    // Shown only where conversations are reconstructible, so the
                    // absence of the line says "cannot dedupe" rather than
                    // implying the billed figure is already distinct.
                    if let (Some(distinct), Some(conversations)) =
                        (m.distinct_tokens(), m.conversations)
                    {
                        let _ = writeln!(
                            out,
                            "  distinct tokens: {distinct} over {conversations} conversation(s)",
                        );
                    }
                }
            }

            out.push_str("\nErrors (triage list)\n");
            out.push_str("====================\n");
            if self.errors.is_empty() {
                out.push_str("  none — every tool call was delivered valid.\n");
            } else {
                for e in &self.errors {
                    let _ = writeln!(
                        out,
                        "\n  [{}x] {} / {} / {} / {}",
                        e.count,
                        e.provider,
                        e.model,
                        e.error_category.as_deref().unwrap_or("?"),
                        e.tool_name.as_deref().unwrap_or("?"),
                    );
                    if let Some(detail) = &e.detail {
                        let _ = writeln!(out, "        {detail}");
                    }
                }
            }
            out
        }
    }

    fn writer_loop(conn: Connection, receiver: mpsc::Receiver<OutcomeRecord>) {
        for record in receiver {
            if let Err(e) = insert(&conn, &record) {
                error!(error = %e, "failed to write metrics row");
            }
        }
    }

    fn insert(conn: &Connection, record: &OutcomeRecord) -> rusqlite::Result<()> {
        // A request with no reported usage stores NULLs rather than zeroes, so
        // the aggregates can tell "the backend said nothing" apart from "the
        // backend said zero" and leave those rows out of the totals.
        let usage = record.usage.filter(|u| !u.is_empty());
        conn.execute(
            "INSERT INTO outcomes \
             (ts, provider, model, outcome, error_category, parser, tool_name, retries, fixed, detail, \
              prompt_tokens, completion_tokens, cached_tokens, billed_calls, response_id, parent_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            rusqlite::params![
                record.ts,
                record.provider,
                record.model,
                record.outcome.as_str(),
                record.error_category.map(|c| c.as_str()),
                record.parser,
                record.tool_name,
                record.retries,
                record.outcome.fixed() as i64,
                record.detail,
                usage.map(|u| u.prompt_tokens),
                usage.map(|u| u.completion_tokens),
                usage.map(|u| u.cached_tokens),
                usage.map(|u| u.attempts),
                record.conversation.as_ref().map(|c| c.id.as_str()),
                record.conversation.as_ref().and_then(|c| c.parent.as_deref()),
            ],
        )?;
        Ok(())
    }

    /// Drop an `outcomes` table predating the `provider` column.
    ///
    /// Deliberately a reset rather than a migration: these are regenerable
    /// usage metrics, not user data, and the rows carry no honest provider to
    /// backfill. Without this the table survives, `CREATE INDEX` fails on the
    /// missing column, and metrics are silently disabled for good — a far worse
    /// outcome than losing a local stats history, and one that is invisible
    /// until someone runs `stats` and finds it empty.
    fn reset_if_stale(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
        let exists = conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'outcomes'")?
            .exists([])?;
        if !exists {
            return Ok(());
        }
        let has_provider = conn
            .prepare("SELECT 1 FROM pragma_table_info('outcomes') WHERE name = 'provider'")?
            .exists([])?;
        if !has_provider {
            warn!(
                "metrics database predates per-provider stats; recreating the \
                 outcomes table (previous request history is discarded)"
            );
            conn.execute_batch(
                "DROP INDEX IF EXISTS idx_outcomes_model;\
                 DROP INDEX IF EXISTS idx_outcomes_unfixed;\
                 DROP TABLE IF EXISTS outcomes;",
            )?;
            return Ok(());
        }
        // A table that predates the token columns is reset for the same reason:
        // these are regenerable local metrics, and leaving the old table in
        // place would make every insert fail on the unknown columns, silently
        // disabling metrics for good.
        let has_tokens = conn
            .prepare("SELECT 1 FROM pragma_table_info('outcomes') WHERE name = 'response_id'")?
            .exists([])?;
        if !has_tokens {
            warn!(
                "metrics database predates token usage; recreating the outcomes \
                 table (previous request history is discarded)"
            );
            conn.execute_batch(
                "DROP INDEX IF EXISTS idx_outcomes_provider_model;\
                 DROP INDEX IF EXISTS idx_outcomes_unfixed;\
                 DROP TABLE IF EXISTS outcomes;",
            )?;
        }
        Ok(())
    }

    const SCHEMA_TABLE: &str = "\
        CREATE TABLE IF NOT EXISTS outcomes (\
            id             INTEGER PRIMARY KEY,\
            ts             TEXT NOT NULL,\
            provider       TEXT NOT NULL DEFAULT 'unknown',\
            model          TEXT NOT NULL,\
            outcome        TEXT NOT NULL,\
            error_category TEXT,\
            parser         TEXT,\
            tool_name      TEXT,\
            retries        INTEGER NOT NULL DEFAULT 0,\
            fixed          INTEGER NOT NULL,\
            detail         TEXT,\
            prompt_tokens     INTEGER,\
            completion_tokens INTEGER,\
            cached_tokens     INTEGER,\
            billed_calls      INTEGER,\
            response_id       TEXT,\
            parent_id         TEXT\
        );\
        CREATE INDEX IF NOT EXISTS idx_outcomes_provider_model \
            ON outcomes(provider, model);\
        CREATE INDEX IF NOT EXISTS idx_outcomes_unfixed \
            ON outcomes(provider, model, error_category) WHERE fixed = 0;\
        CREATE INDEX IF NOT EXISTS idx_outcomes_response \
            ON outcomes(response_id) WHERE response_id IS NOT NULL;\
        CREATE INDEX IF NOT EXISTS idx_outcomes_parent \
            ON outcomes(parent_id) WHERE parent_id IS NOT NULL;";

    #[cfg(test)]
    mod tests {
        use super::super::{now_rfc3339, Outcome, OutcomeRecord, Recorder, Usage};
        use super::{SqliteRecorder, Stats};
        use crate::domain::validate::ErrorCategory;

        fn rec(model: &str, outcome: Outcome) -> OutcomeRecord {
            rec_from("default", model, outcome)
        }

        fn rec_from(provider: &str, model: &str, outcome: Outcome) -> OutcomeRecord {
            OutcomeRecord {
                ts: now_rfc3339(),
                provider: provider.into(),
                model: model.into(),
                outcome,
                error_category: None,
                parser: None,
                tool_name: None,
                retries: 0,
                detail: None,
                usage: None,
                conversation: None,
            }
        }

        #[test]
        fn records_round_trip_to_the_database() {
            let dir =
                std::env::temp_dir().join(format!("guardrail-metrics-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("metrics.sqlite");
            let recorder = SqliteRecorder::open(&db).unwrap();

            recorder.record(OutcomeRecord {
                ts: now_rfc3339(),
                provider: "default".into(),
                model: "qwen2.5".into(),
                outcome: Outcome::NativeValid,
                error_category: None,
                parser: None,
                tool_name: Some("get_weather".into()),
                retries: 0,
                detail: None,
                usage: None,
                conversation: None,
            });
            recorder.record(OutcomeRecord {
                ts: now_rfc3339(),
                provider: "default".into(),
                model: "qwen2.5".into(),
                outcome: Outcome::RetriesExhausted,
                error_category: Some(ErrorCategory::MissingArgument),
                parser: None,
                tool_name: Some("Edit".into()),
                retries: 2,
                detail: Some("missing filePath | args: {}".into()),
                usage: None,
                conversation: None,
            });
            // Drop closes the channel and joins the writer; rows are flushed.
            drop(recorder);

            // Reopen read-only and assert the rows landed with the right shape.
            let conn = rusqlite::Connection::open(&db).unwrap();
            // Spin briefly in case the writer thread is mid-drain.
            let mut total = 0i64;
            for _ in 0..50 {
                total = conn
                    .query_row("SELECT COUNT(*) FROM outcomes", [], |r| r.get(0))
                    .unwrap();
                if total == 2 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            assert_eq!(total, 2);

            let (outcome, category, fixed): (String, Option<String>, i64) = conn
                .query_row(
                    "SELECT outcome, error_category, fixed FROM outcomes WHERE outcome = 'retries_exhausted'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .unwrap();
            assert_eq!(outcome, "retries_exhausted");
            assert_eq!(category.as_deref(), Some("missing_argument"));
            assert_eq!(fixed, 0);

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn a_database_predating_providers_is_recreated_rather_than_disabling_metrics() {
            // The pre-provider schema, as shipped before this change.
            let dir = std::env::temp_dir().join(format!("guardrail-stale-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("stale.sqlite");
            let _ = std::fs::remove_file(&db);
            {
                let conn = rusqlite::Connection::open(&db).unwrap();
                conn.execute_batch(
                    "CREATE TABLE outcomes (\
                        id INTEGER PRIMARY KEY, ts TEXT NOT NULL, model TEXT NOT NULL,\
                        outcome TEXT NOT NULL, error_category TEXT, parser TEXT,\
                        tool_name TEXT, retries INTEGER NOT NULL DEFAULT 0,\
                        fixed INTEGER NOT NULL, detail TEXT);\
                     CREATE INDEX idx_outcomes_model ON outcomes(model);\
                     CREATE INDEX idx_outcomes_unfixed \
                        ON outcomes(model, error_category) WHERE fixed = 0;\
                     INSERT INTO outcomes (ts, model, outcome, retries, fixed) \
                        VALUES ('t', 'old-model', 'native_valid', 0, 1);",
                )
                .unwrap();
            }

            // Opening must succeed. Before the reset this failed on the missing
            // column and left metrics silently off for every later run.
            let recorder = SqliteRecorder::open(&db).expect("stale schema must not disable metrics");
            recorder.record(rec_from("copilot", "gpt-4o", Outcome::NativeValid));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            assert_eq!(stats.per_model.len(), 1);
            assert_eq!(stats.per_model[0].provider, "copilot");

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn the_same_model_on_two_providers_stays_separate() {
            // The reason the provider column exists. `gpt-4o` is reachable
            // through several vendors; merging them would average a degrading
            // provider against a healthy one and hide which upstream to fix.
            let dir =
                std::env::temp_dir().join(format!("guardrail-providers-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("providers.sqlite");

            let recorder = SqliteRecorder::open(&db).unwrap();
            recorder.record(rec_from("copilot", "gpt-4o", Outcome::NativeValid));
            recorder.record(rec_from("copilot", "gpt-4o", Outcome::RetriesExhausted));
            recorder.record(rec_from("azure", "gpt-4o", Outcome::NativeValid));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            assert_eq!(
                stats.per_model.len(),
                2,
                "one row per (provider, model), got: {:?}",
                stats
                    .per_model
                    .iter()
                    .map(|m| (&m.provider, &m.model))
                    .collect::<Vec<_>>()
            );

            let copilot = stats
                .per_model
                .iter()
                .find(|m| m.provider == "copilot")
                .expect("copilot row");
            assert_eq!(copilot.total, 2);
            assert_eq!(copilot.errors, 1);

            let azure = stats
                .per_model
                .iter()
                .find(|m| m.provider == "azure")
                .expect("azure row");
            assert_eq!(azure.total, 1);
            assert_eq!(
                azure.errors, 0,
                "the failing provider's errors must not be attributed here"
            );

            // The triage list must name the provider too, or an error is not
            // actionable when two providers serve the same id.
            assert_eq!(stats.errors.len(), 1);
            assert_eq!(stats.errors[0].provider, "copilot");

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn stats_separate_total_tool_calls_and_errors() {
            let dir = std::env::temp_dir().join(format!("guardrail-stats-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("stats.sqlite");

            let recorder = SqliteRecorder::open(&db).unwrap();
            // 2 real tool calls (1 of which is unfixed), plus a respond and a
            // plain-text passthrough that must NOT count as tool calls.
            recorder.record(rec("m", Outcome::NativeValid));
            recorder.record(rec("m", Outcome::RetriesExhausted));
            recorder.record(rec("m", Outcome::RespondIntercept));
            recorder.record(rec("m", Outcome::PassthroughNoCalls));
            drop(recorder); // flushes

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];
            assert_eq!(m.total, 4);
            assert_eq!(m.tool_calls, 2); // respond + passthrough excluded
            assert_eq!(m.errors, 1);
            assert_eq!(m.succeeded(), 1);
            assert_eq!(m.success_rate(), Some(0.5));

            let _ = std::fs::remove_dir_all(&dir);
        }

        fn rec_with_usage(model: &str, outcome: Outcome, usage: Usage) -> OutcomeRecord {
            OutcomeRecord {
                usage: Some(usage),
                ..rec_from("default", model, outcome)
            }
        }

        #[test]
        fn token_usage_is_summed_per_provider_and_model() {
            let dir = std::env::temp_dir().join(format!("guardrail-tokens-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("tokens.sqlite");

            let recorder = SqliteRecorder::open(&db).unwrap();
            recorder.record(rec_with_usage(
                "m",
                Outcome::NativeValid,
                Usage { prompt_tokens: 100, completion_tokens: 10, cached_tokens: 60, attempts: 1 },
            ));
            // A retried request: two backend calls folded into one row.
            recorder.record(rec_with_usage(
                "m",
                Outcome::RecoveredAfterRetry,
                Usage { prompt_tokens: 300, completion_tokens: 30, cached_tokens: 140, attempts: 2 },
            ));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];
            assert_eq!(m.usage.prompt_tokens, 400);
            assert_eq!(m.usage.completion_tokens, 40);
            assert_eq!(m.usage.cached_tokens, 200);
            assert_eq!(m.billed_tokens(), 440);
            assert_eq!(m.usage.uncached_prompt_tokens(), 200);
            assert_eq!(m.cache_hit_rate(), Some(0.5));
            // Three backend calls over two client requests.
            assert_eq!(m.usage.attempts, 3);
            assert_eq!(m.usage_requests, 2);
            assert_eq!(m.calls_per_request(), Some(1.5));

            let _ = std::fs::remove_dir_all(&dir);
        }

        fn rec_chained(
            model: &str,
            usage: Usage,
            id: &str,
            parent: Option<&str>,
        ) -> OutcomeRecord {
            OutcomeRecord {
                usage: Some(usage),
                conversation: Some(super::super::Conversation {
                    id: id.to_string(),
                    parent: parent.map(str::to_string),
                }),
                ..rec_from("default", model, Outcome::NativeValid)
            }
        }

        fn usage_of(prompt: i64, completion: i64) -> Usage {
            Usage { prompt_tokens: prompt, completion_tokens: completion, cached_tokens: 0, attempts: 1 }
        }

        #[test]
        fn a_conversation_counts_its_resent_prefix_once() {
            // The whole point of fix 2. Three turns of one conversation, each
            // resending the transcript: prompts of 100, 300, 600. The billed
            // sum is 1000, but the conversation only ever contained 600 distinct
            // prompt tokens — turn 3's prompt already holds turns 1 and 2.
            let dir = std::env::temp_dir().join(format!("guardrail-dedupe-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("dedupe.sqlite");
            let _ = std::fs::remove_file(&db);

            let recorder = SqliteRecorder::open(&db).unwrap();
            recorder.record(rec_chained("m", usage_of(100, 10), "resp_1", None));
            recorder.record(rec_chained("m", usage_of(300, 20), "resp_2", Some("resp_1")));
            recorder.record(rec_chained("m", usage_of(600, 30), "resp_3", Some("resp_2")));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];

            // Billed is still the honest sum of what the provider charged.
            assert_eq!(m.usage.prompt_tokens, 1000);
            assert_eq!(m.billed_tokens(), 1060);

            // Deduplicated: the largest prompt in the chain, plus all output.
            assert_eq!(m.distinct_prompt_tokens, Some(600));
            assert_eq!(m.distinct_tokens(), Some(660));
            assert_eq!(m.conversations, Some(1), "three turns, one conversation");
            // 1000 billed over 600 distinct.
            // Billed and distinct diverge by exactly the resent prefixes: 1000
            // charged for 600 distinct tokens.
            assert!(m.usage.prompt_tokens > m.distinct_prompt_tokens.unwrap());

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn a_conversations_cache_hits_are_recorded_against_its_resent_prefixes() {
            // Resent prefixes are what a prompt cache serves, so the cached
            // count is what says whether resending was expensive. The proxy
            // records the measured tokens and leaves pricing to whoever knows
            // the rates — it cannot see them.
            let dir = std::env::temp_dir().join(format!("guardrail-cost-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("cost.sqlite");
            let _ = std::fs::remove_file(&db);

            let cached = |prompt: i64, cached: i64| Usage {
                prompt_tokens: prompt,
                completion_tokens: 0,
                cached_tokens: cached,
                attempts: 1,
            };

            let recorder = SqliteRecorder::open(&db).unwrap();
            // Turn 1 is all new; each later turn's prefix is served from cache.
            recorder.record(rec_chained("m", cached(1000, 0), "c1", None));
            recorder.record(rec_chained("m", cached(2500, 1500), "c2", Some("c1")));
            recorder.record(rec_chained("m", cached(4200, 3200), "c3", Some("c2")));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];

            assert_eq!(m.usage.prompt_tokens, 7700);
            assert_eq!(m.usage.cached_tokens, 4700);
            assert_eq!(m.distinct_prompt_tokens, Some(4200));

            // 4700 of the 7700 prompt tokens were cache reads, so most of what
            // the resends re-sent was served from cache rather than reprocessed.
            let hit = m.cache_hit_rate().unwrap();
            assert!((hit - 4700.0 / 7700.0).abs() < 1e-9, "got {hit}");
            assert!(hit > 0.6, "the resends are mostly cache hits");
            // Only 3000 were genuinely new work.
            assert_eq!(m.usage.uncached_prompt_tokens(), 3000);

            let report = stats.render();
            assert!(report.contains("cache hit rate"));
            assert!(report.contains("distinct tokens"));

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn unrelated_requests_each_count_in_full() {
            // Deduplication must not collapse traffic that merely happens to
            // share a model. Three unchained requests are three conversations
            // of one turn, so distinct equals billed.
            let dir = std::env::temp_dir().join(format!("guardrail-unrel-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("unrelated.sqlite");
            let _ = std::fs::remove_file(&db);

            let recorder = SqliteRecorder::open(&db).unwrap();
            recorder.record(rec_chained("m", usage_of(100, 10), "resp_a", None));
            recorder.record(rec_chained("m", usage_of(200, 10), "resp_b", None));
            recorder.record(rec_chained("m", usage_of(300, 10), "resp_c", None));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];
            assert_eq!(m.usage.prompt_tokens, 600);
            assert_eq!(m.distinct_prompt_tokens, Some(600), "nothing to dedupe");
            assert_eq!(m.conversations, Some(3));
            assert_eq!(
                m.distinct_prompt_tokens,
                Some(m.usage.prompt_tokens),
                "with nothing resent, distinct equals billed"
            );

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn two_conversations_are_deduplicated_independently() {
            // Sum of per-conversation maxima, not one global maximum: collapsing
            // across conversations would undercount as badly as summing
            // overcounts.
            let dir = std::env::temp_dir().join(format!("guardrail-twoconv-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("twoconv.sqlite");
            let _ = std::fs::remove_file(&db);

            let recorder = SqliteRecorder::open(&db).unwrap();
            // Conversation A: 100 → 250.
            recorder.record(rec_chained("m", usage_of(100, 5), "a1", None));
            recorder.record(rec_chained("m", usage_of(250, 5), "a2", Some("a1")));
            // Conversation B: 400 → 900.
            recorder.record(rec_chained("m", usage_of(400, 5), "b1", None));
            recorder.record(rec_chained("m", usage_of(900, 5), "b2", Some("b1")));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];
            assert_eq!(m.usage.prompt_tokens, 1650, "billed sum");
            // 250 + 900, not 900 alone and not 1650.
            assert_eq!(m.distinct_prompt_tokens, Some(1150));
            assert_eq!(m.conversations, Some(2));

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn chat_completions_traffic_reports_no_deduplication_rather_than_a_false_one() {
            // Chat Completions carries no conversation key, so resends are
            // invisible. Reporting the inflated sum as "distinct" would be a
            // lie; absence is the honest answer.
            let dir = std::env::temp_dir().join(format!("guardrail-nochain-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("nochain.sqlite");
            let _ = std::fs::remove_file(&db);

            let recorder = SqliteRecorder::open(&db).unwrap();
            recorder.record(rec_with_usage("m", Outcome::NativeValid, usage_of(100, 10)));
            recorder.record(rec_with_usage("m", Outcome::NativeValid, usage_of(300, 20)));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];
            assert_eq!(m.usage.prompt_tokens, 400, "billed is still reported");
            assert_eq!(m.distinct_prompt_tokens, None);
            assert_eq!(m.distinct_tokens(), None);
            assert_eq!(m.conversations, None);
            // And the report must not claim a deduplication it did not do.
            assert!(!stats.render().contains("distinct tokens"));

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn a_chain_groups_regardless_of_how_its_ids_happen_to_sort() {
            // Response ids are opaque strings, so chain order and lexical order
            // are unrelated. Here every later turn's id sorts *before* its
            // parent's; picking a root by string comparison would leave each
            // turn as its own root, split one conversation into three, and add
            // the resent prefixes right back up. The root must come from the
            // walk, not the ordering.
            let dir = std::env::temp_dir().join(format!("guardrail-idsort-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("idsort.sqlite");
            let _ = std::fs::remove_file(&db);

            let recorder = SqliteRecorder::open(&db).unwrap();
            recorder.record(rec_chained("m", usage_of(1000, 10), "zzz_first", None));
            recorder.record(rec_chained("m", usage_of(2500, 20), "mmm_second", Some("zzz_first")));
            recorder.record(rec_chained("m", usage_of(4200, 30), "aaa_third", Some("mmm_second")));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];

            assert_eq!(m.usage.prompt_tokens, 7700, "billed is unaffected");
            // One conversation, so only the last turn's prompt counts.
            assert_eq!(
                m.conversations,
                Some(1),
                "lexically-descending ids must not split the chain"
            );
            assert_eq!(m.distinct_prompt_tokens, Some(4200));

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn a_cycle_in_the_parent_links_terminates_instead_of_hanging() {
            // Nothing in the proxy writes a cycle, but the recursive walk uses
            // UNION ALL and would not terminate on one. A corrupted or
            // hand-edited database must still let `stats` return.
            let dir = std::env::temp_dir().join(format!("guardrail-cycle-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("cycle.sqlite");
            let _ = std::fs::remove_file(&db);

            {
                let recorder = SqliteRecorder::open(&db).unwrap();
                recorder.record(rec_chained("m", usage_of(100, 5), "x", Some("y")));
                recorder.record(rec_chained("m", usage_of(200, 5), "y", Some("x")));
            }

            // The assertion is simply that this returns at all.
            let stats = Stats::read(&db).expect("a cycle must not hang or error");
            let m = &stats.per_model[0];
            assert_eq!(m.usage.prompt_tokens, 300);
            assert!(m.distinct_prompt_tokens.is_some());

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn a_chain_whose_first_turn_was_never_recorded_still_groups() {
            // Metrics enabled mid-conversation: turn 2 names a parent that is
            // not in the database. Those turns must still group together rather
            // than each counting in full.
            let dir = std::env::temp_dir().join(format!("guardrail-orphan-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("orphan.sqlite");
            let _ = std::fs::remove_file(&db);

            let recorder = SqliteRecorder::open(&db).unwrap();
            recorder.record(rec_chained("m", usage_of(300, 10), "resp_2", Some("resp_missing")));
            recorder.record(rec_chained("m", usage_of(700, 10), "resp_3", Some("resp_2")));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];
            assert_eq!(m.usage.prompt_tokens, 1000);
            // Rooted at the earliest turn actually seen, so still one group.
            assert_eq!(m.distinct_prompt_tokens, Some(700));
            assert_eq!(m.conversations, Some(1));

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn requests_without_a_usage_report_are_left_out_of_the_totals() {
            // A backend that reports no usage must not be averaged in as a
            // zero-token request — that would drag every per-request figure
            // down and make the numbers look like a measurement rather than
            // the absence of one.
            let dir =
                std::env::temp_dir().join(format!("guardrail-nousage-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("nousage.sqlite");

            let recorder = SqliteRecorder::open(&db).unwrap();
            recorder.record(rec_with_usage(
                "m",
                Outcome::NativeValid,
                Usage { prompt_tokens: 50, completion_tokens: 5, cached_tokens: 0, attempts: 1 },
            ));
            recorder.record(rec("m", Outcome::NativeValid)); // no usage at all
            // An all-zero report is indistinguishable from no report and is
            // stored as NULL too.
            recorder.record(rec_with_usage("m", Outcome::NativeValid, Usage::default()));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];
            assert_eq!(m.total, 3, "every request is still counted");
            assert_eq!(m.usage_requests, 1, "only one carried a usage report");
            assert_eq!(m.billed_tokens(), 55);
            assert_eq!(m.calls_per_request(), Some(1.0));

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn reading_a_database_older_than_the_current_schema_degrades_instead_of_failing() {
            // `stats` reads a database the proxy owns and may not have opened
            // since an upgrade, so the table can be missing any column added
            // since. It must report what it has rather than erroring out — the
            // whole history is otherwise unreadable until the proxy runs again.
            let dir = std::env::temp_dir().join(format!("guardrail-oldread-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("old.sqlite");
            let _ = std::fs::remove_file(&db);
            {
                // The oldest shipped schema: no provider, no token columns.
                let conn = rusqlite::Connection::open(&db).unwrap();
                conn.execute_batch(
                    "CREATE TABLE outcomes (\
                        id INTEGER PRIMARY KEY, ts TEXT NOT NULL, model TEXT NOT NULL,\
                        outcome TEXT NOT NULL, error_category TEXT, parser TEXT,\
                        tool_name TEXT, retries INTEGER NOT NULL DEFAULT 0,\
                        fixed INTEGER NOT NULL, detail TEXT);\
                     INSERT INTO outcomes (ts, model, outcome, retries, fixed) \
                        VALUES ('t', 'old-model', 'native_valid', 0, 1);\
                     INSERT INTO outcomes (ts, model, outcome, retries, fixed, error_category) \
                        VALUES ('t', 'old-model', 'retries_exhausted', 2, 0, 'missing_argument');",
                )
                .unwrap();
            }

            let stats = Stats::read(&db).expect("an old database must still be readable");
            assert_eq!(stats.per_model.len(), 1);
            let m = &stats.per_model[0];
            assert_eq!(m.model, "old-model");
            assert_eq!(m.provider, "unknown", "no provider was recorded back then");
            assert_eq!(m.total, 2);
            assert_eq!(m.errors, 1);
            // The outcome history is intact; only the tokens are unknown.
            assert_eq!(m.usage_requests, 0);
            assert_eq!(m.billed_tokens(), 0);
            assert_eq!(stats.errors.len(), 1, "triage list still reads");

            // And it renders without panicking.
            assert!(stats.render().contains("old-model"));

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn a_database_predating_tokens_is_recreated_rather_than_disabling_metrics() {
            // Same reasoning as the provider reset: these are regenerable local
            // metrics, and leaving the old table would make every insert fail on
            // the unknown columns — metrics silently off until someone runs
            // `stats` and finds it empty.
            let dir =
                std::env::temp_dir().join(format!("guardrail-pretokens-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("pretokens.sqlite");
            let _ = std::fs::remove_file(&db);
            {
                let conn = rusqlite::Connection::open(&db).unwrap();
                conn.execute_batch(
                    "CREATE TABLE outcomes (\
                        id INTEGER PRIMARY KEY, ts TEXT NOT NULL,\
                        provider TEXT NOT NULL DEFAULT 'unknown', model TEXT NOT NULL,\
                        outcome TEXT NOT NULL, error_category TEXT, parser TEXT,\
                        tool_name TEXT, retries INTEGER NOT NULL DEFAULT 0,\
                        fixed INTEGER NOT NULL, detail TEXT);\
                     INSERT INTO outcomes (ts, provider, model, outcome, retries, fixed) \
                        VALUES ('t', 'copilot', 'gpt-4o', 'native_valid', 0, 1);",
                )
                .unwrap();
            }

            let recorder = SqliteRecorder::open(&db).expect("stale schema must not disable metrics");
            recorder.record(rec_with_usage(
                "gpt-4o",
                Outcome::NativeValid,
                Usage { prompt_tokens: 9, completion_tokens: 1, cached_tokens: 0, attempts: 1 },
            ));
            drop(recorder);

            // The old row is gone with the table; the new one records tokens.
            let stats = Stats::read(&db).unwrap();
            assert_eq!(stats.per_model.len(), 1);
            let m = &stats.per_model[0];
            assert_eq!(m.total, 1);
            assert_eq!(m.billed_tokens(), 10);

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn success_rate_is_none_without_tool_calls() {
            let dir = std::env::temp_dir().join(format!("guardrail-norate-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("stats.sqlite");

            let recorder = SqliteRecorder::open(&db).unwrap();
            recorder.record(rec("text-only", Outcome::PassthroughNoCalls));
            drop(recorder);

            let stats = Stats::read(&db).unwrap();
            let m = &stats.per_model[0];
            assert_eq!(m.total, 1);
            assert_eq!(m.tool_calls, 0);
            assert_eq!(m.success_rate(), None);

            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
