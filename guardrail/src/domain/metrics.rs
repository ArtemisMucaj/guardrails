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
    FallbackUnfixed,
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
            Outcome::FallbackUnfixed => "fallback_unfixed",
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
    pub const TOOL_CALL_TAGS: [&'static str; 5] = [
        "native_valid",
        "rescued",
        "repaired",
        "recovered_after_retry",
        "fallback_unfixed",
    ];

    /// Whether this outcome is a real tool call against a client-declared tool
    /// (see [`Outcome::TOOL_CALL_TAGS`]). Used to count "tool calls total".
    pub fn is_tool_call(self) -> bool {
        Self::TOOL_CALL_TAGS.contains(&self.as_str())
    }

    /// Whether the guardrails produced a usable result. Only `FallbackUnfixed`
    /// represents an error the guardrails failed to resolve.
    pub fn fixed(self) -> bool {
        !matches!(self, Outcome::FallbackUnfixed)
    }
}

/// One row of failure metrics: the terminal outcome of a guarded request.
#[derive(Debug, Clone)]
pub struct OutcomeRecord {
    /// RFC3339 UTC timestamp, stamped when the outcome occurs (not when it is
    /// written), so a backed-up writer queue does not skew the recorded time.
    pub ts: String,
    pub model: String,
    pub outcome: Outcome,
    /// Failure category, present only on `FallbackUnfixed`.
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
        assert!(Outcome::FallbackUnfixed.is_tool_call());
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
        assert!(!Outcome::FallbackUnfixed.fixed());
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

    use super::{OutcomeRecord, Recorder};

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
            conn.execute_batch(SCHEMA)?;

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

    /// Per-model rollup, in the total → tool calls → errors hierarchy.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ModelStats {
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
    }

    impl ModelStats {
        /// Tool calls the guardrails delivered as valid.
        pub fn succeeded(&self) -> i64 {
            self.tool_calls - self.errors
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

            // Per-model totals. The tool-call set is formatted from the single
            // source of truth in `Outcome` so it can never drift from the Rust
            // classification; the tags are static literals, so this is not a
            // SQL-injection surface.
            let in_list = super::Outcome::TOOL_CALL_TAGS
                .iter()
                .map(|t| format!("'{t}'"))
                .collect::<Vec<_>>()
                .join(",");
            let query = format!(
                "SELECT model, \
                    COUNT(*), \
                    SUM(CASE WHEN outcome IN ({in_list}) THEN 1 ELSE 0 END), \
                    SUM(CASE WHEN outcome = 'fallback_unfixed' THEN 1 ELSE 0 END) \
                 FROM outcomes GROUP BY model ORDER BY model"
            );
            let mut stmt = conn.prepare(&query)?;
            let rows = stmt.query_map([], |r| {
                Ok(ModelStats {
                    model: r.get(0)?,
                    total: r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                    tool_calls: r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                    errors: r.get::<_, Option<i64>>(3)?.unwrap_or(0),
                    by_outcome: Vec::new(),
                })
            })?;
            let mut per_model: Vec<ModelStats> = rows.collect::<rusqlite::Result<_>>()?;

            // Outcome breakdown per model, folded into the rows above.
            let mut stmt = conn.prepare(
                "SELECT model, outcome, COUNT(*) FROM outcomes \
                 GROUP BY model, outcome ORDER BY model, outcome",
            )?;
            let breakdown = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })?;
            for row in breakdown {
                let (model, outcome, count) = row?;
                if let Some(m) = per_model.iter_mut().find(|m| m.model == model) {
                    m.by_outcome.push((outcome, count));
                }
            }

            // Errors the guardrails could not fix, grouped for triage, most
            // frequent first.
            let mut stmt = conn.prepare(
                "SELECT model, error_category, tool_name, detail, COUNT(*) AS n \
                 FROM outcomes WHERE fixed = 0 \
                 GROUP BY model, error_category, tool_name, detail \
                 ORDER BY n DESC, model",
            )?;
            let errors = stmt.query_map([], |r| {
                Ok(ErrorGroup {
                    model: r.get(0)?,
                    error_category: r.get(1)?,
                    tool_name: r.get(2)?,
                    detail: r.get(3)?,
                    count: r.get(4)?,
                })
            })?;
            let errors: Vec<ErrorGroup> = errors.collect::<rusqlite::Result<_>>()?;

            Ok(Self { per_model, errors })
        }

        /// Render a plain-text report for the CLI.
        pub fn render(&self) -> String {
            use std::fmt::Write;
            let mut out = String::new();

            if self.per_model.is_empty() {
                return "No metrics recorded yet.\n".to_string();
            }

            out.push_str("Requests by model\n");
            out.push_str("=================\n");
            for m in &self.per_model {
                let rate = match m.success_rate() {
                    Some(r) => format!("{:.1}%", r * 100.0),
                    None => "n/a".to_string(),
                };
                let _ = writeln!(
                    out,
                    "\n{}\n  total: {}  |  tool calls: {}  |  succeeded: {}  |  errors: {}  |  success rate: {}",
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
            }

            out.push_str("\nErrors (triage list)\n");
            out.push_str("====================\n");
            if self.errors.is_empty() {
                out.push_str("  none — every tool call was delivered valid.\n");
            } else {
                for e in &self.errors {
                    let _ = writeln!(
                        out,
                        "\n  [{}x] {} / {} / {}",
                        e.count,
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
        conn.execute(
            "INSERT INTO outcomes \
             (ts, model, outcome, error_category, parser, tool_name, retries, fixed, detail) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                record.ts,
                record.model,
                record.outcome.as_str(),
                record.error_category.map(|c| c.as_str()),
                record.parser,
                record.tool_name,
                record.retries,
                record.outcome.fixed() as i64,
                record.detail,
            ],
        )?;
        Ok(())
    }

    const SCHEMA: &str = "\
        CREATE TABLE IF NOT EXISTS outcomes (\
            id             INTEGER PRIMARY KEY,\
            ts             TEXT NOT NULL,\
            model          TEXT NOT NULL,\
            outcome        TEXT NOT NULL,\
            error_category TEXT,\
            parser         TEXT,\
            tool_name      TEXT,\
            retries        INTEGER NOT NULL DEFAULT 0,\
            fixed          INTEGER NOT NULL,\
            detail         TEXT\
        );\
        CREATE INDEX IF NOT EXISTS idx_outcomes_model ON outcomes(model);\
        CREATE INDEX IF NOT EXISTS idx_outcomes_unfixed \
            ON outcomes(model, error_category) WHERE fixed = 0;";

    #[cfg(test)]
    mod tests {
        use super::super::{now_rfc3339, Outcome, OutcomeRecord, Recorder};
        use super::{SqliteRecorder, Stats};
        use crate::domain::validate::ErrorCategory;

        fn rec(model: &str, outcome: Outcome) -> OutcomeRecord {
            OutcomeRecord {
                ts: now_rfc3339(),
                model: model.into(),
                outcome,
                error_category: None,
                parser: None,
                tool_name: None,
                retries: 0,
                detail: None,
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
                model: "qwen2.5".into(),
                outcome: Outcome::NativeValid,
                error_category: None,
                parser: None,
                tool_name: Some("get_weather".into()),
                retries: 0,
                detail: None,
            });
            recorder.record(OutcomeRecord {
                ts: now_rfc3339(),
                model: "qwen2.5".into(),
                outcome: Outcome::FallbackUnfixed,
                error_category: Some(ErrorCategory::MissingArgument),
                parser: None,
                tool_name: Some("Edit".into()),
                retries: 2,
                detail: Some("missing filePath | args: {}".into()),
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
                    "SELECT outcome, error_category, fixed FROM outcomes WHERE outcome = 'fallback_unfixed'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .unwrap();
            assert_eq!(outcome, "fallback_unfixed");
            assert_eq!(category.as_deref(), Some("missing_argument"));
            assert_eq!(fixed, 0);

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
            recorder.record(rec("m", Outcome::FallbackUnfixed));
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
