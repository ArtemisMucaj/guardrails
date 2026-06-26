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
    /// The backend response was not JSON and was forwarded unverified.
    NonJson,
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
            Outcome::NonJson => "non_json",
        }
    }

    /// Whether this outcome carries an actual tool call (as opposed to a request
    /// that produced no call to validate). Used to count "tool calls total".
    pub fn is_tool_call(self) -> bool {
        !matches!(self, Outcome::PassthroughNoCalls | Outcome::NonJson)
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

/// Truncate a tool-call argument string to a bounded, single-line snippet so the
/// stored `detail` stays small and never carries unbounded model output.
pub fn redact_args(arguments: &str) -> String {
    const MAX: usize = 200;
    let flattened: String = arguments
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    let trimmed = flattened.trim();
    if trimmed.chars().count() > MAX {
        let head: String = trimmed.chars().take(MAX).collect();
        format!("{head}…")
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_outcomes_are_distinguished_from_passthrough() {
        assert!(Outcome::NativeValid.is_tool_call());
        assert!(Outcome::FallbackUnfixed.is_tool_call());
        assert!(!Outcome::PassthroughNoCalls.is_tool_call());
        assert!(!Outcome::NonJson.is_tool_call());
    }

    #[test]
    fn only_fallback_is_unfixed() {
        assert!(Outcome::Rescued.fixed());
        assert!(Outcome::Repaired.fixed());
        assert!(Outcome::RecoveredAfterRetry.fixed());
        assert!(!Outcome::FallbackUnfixed.fixed());
    }

    #[test]
    fn redact_args_flattens_and_bounds() {
        assert_eq!(redact_args("  {\"a\":1}\n "), "{\"a\":1}");
        assert!(!redact_args("line1\nline2").contains('\n'));
        let long = "x".repeat(500);
        let out = redact_args(&long);
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 201); // 200 + ellipsis
    }
}

pub use sqlite::{default_db_path, ModelStats, SqliteRecorder, Stats, UnfixedError};

mod sqlite {
    use std::path::Path;
    use std::sync::mpsc::{self, Sender};
    use std::thread;

    use rusqlite::Connection;
    use tracing::{error, info};

    use super::{OutcomeRecord, Recorder};

    /// Persists outcome records to a local SQLite database.
    ///
    /// A dedicated writer thread owns the connection; `record` only enqueues onto
    /// an unbounded channel and returns immediately, so a database write never
    /// blocks the proxy's response path. When the recorder is dropped the channel
    /// closes and the writer thread drains and exits.
    pub struct SqliteRecorder {
        sender: Sender<OutcomeRecord>,
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

            let (sender, receiver) = mpsc::channel::<OutcomeRecord>();
            thread::Builder::new()
                .name("guardrail-metrics".into())
                .spawn(move || writer_loop(conn, receiver))?;

            info!(path = %path.as_ref().display(), "metrics enabled (sqlite)");
            Ok(Self { sender })
        }
    }

    impl Recorder for SqliteRecorder {
        fn record(&self, record: OutcomeRecord) {
            // A closed channel (writer gone) is not worth failing a request over.
            let _ = self.sender.send(record);
        }
    }

    /// Default metrics database path: `~/.guardrails/stats.sqlite`. The
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
        dir.join("stats.sqlite")
    }

    /// Per-model rollup of tool-call outcomes.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ModelStats {
        pub model: String,
        /// Requests that produced a tool call (excludes plain text / non-JSON).
        pub tool_calls: i64,
        /// Tool calls the guardrails delivered as valid.
        pub succeeded: i64,
        /// Tool calls the guardrails could not fix (`fallback_unfixed`).
        pub unfixed: i64,
        /// Counts per outcome tag, for the breakdown line.
        pub by_outcome: Vec<(String, i64)>,
    }

    impl ModelStats {
        pub fn success_rate(&self) -> f64 {
            if self.tool_calls == 0 {
                0.0
            } else {
                self.succeeded as f64 / self.tool_calls as f64
            }
        }
    }

    /// One group of unfixed errors awaiting triage.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct UnfixedError {
        pub model: String,
        pub error_category: Option<String>,
        pub tool_name: Option<String>,
        pub detail: Option<String>,
        pub count: i64,
    }

    /// A full read of the metrics database for the `stats` command.
    #[derive(Debug, Clone, Default)]
    pub struct Stats {
        pub per_model: Vec<ModelStats>,
        pub unfixed: Vec<UnfixedError>,
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

            // Per-model totals. `tool_calls` excludes requests with no call.
            let mut stmt = conn.prepare(
                "SELECT model, \
                    SUM(CASE WHEN outcome NOT IN ('passthrough_no_calls','non_json') THEN 1 ELSE 0 END), \
                    SUM(CASE WHEN outcome NOT IN ('passthrough_no_calls','non_json') AND fixed = 1 THEN 1 ELSE 0 END), \
                    SUM(CASE WHEN outcome = 'fallback_unfixed' THEN 1 ELSE 0 END) \
                 FROM outcomes GROUP BY model ORDER BY model",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok(ModelStats {
                    model: r.get(0)?,
                    tool_calls: r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                    succeeded: r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                    unfixed: r.get::<_, Option<i64>>(3)?.unwrap_or(0),
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

            // Unfixed errors, grouped for triage, most frequent first.
            let mut stmt = conn.prepare(
                "SELECT model, error_category, tool_name, detail, COUNT(*) AS n \
                 FROM outcomes WHERE fixed = 0 \
                 GROUP BY model, error_category, tool_name, detail \
                 ORDER BY n DESC, model",
            )?;
            let unfixed = stmt.query_map([], |r| {
                Ok(UnfixedError {
                    model: r.get(0)?,
                    error_category: r.get(1)?,
                    tool_name: r.get(2)?,
                    detail: r.get(3)?,
                    count: r.get(4)?,
                })
            })?;
            let unfixed: Vec<UnfixedError> = unfixed.collect::<rusqlite::Result<_>>()?;

            Ok(Self { per_model, unfixed })
        }

        /// Render a plain-text report for the CLI.
        pub fn render(&self) -> String {
            use std::fmt::Write;
            let mut out = String::new();

            if self.per_model.is_empty() {
                return "No metrics recorded yet.\n".to_string();
            }

            out.push_str("Tool calls by model\n");
            out.push_str("===================\n");
            for m in &self.per_model {
                let _ = writeln!(
                    out,
                    "\n{}\n  tool calls: {}  |  succeeded: {}  |  unfixed: {}  |  success rate: {:.1}%",
                    m.model,
                    m.tool_calls,
                    m.succeeded,
                    m.unfixed,
                    m.success_rate() * 100.0,
                );
                for (outcome, count) in &m.by_outcome {
                    let _ = writeln!(out, "    {outcome:<22} {count}");
                }
            }

            out.push_str("\nUnfixed errors (triage list)\n");
            out.push_str("============================\n");
            if self.unfixed.is_empty() {
                out.push_str("  none — every tool call was delivered valid.\n");
            } else {
                for e in &self.unfixed {
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
                now_rfc3339(),
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

    /// RFC3339 UTC timestamp without pulling in a date library.
    fn now_rfc3339() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Civil-date conversion from a Unix timestamp (days since 1970-01-01).
        let days = (secs / 86_400) as i64;
        let rem = secs % 86_400;
        let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
        let (y, m, d) = civil_from_days(days);
        format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
    }

    /// Howard Hinnant's days-to-civil-date algorithm.
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
        use super::super::{Outcome, OutcomeRecord, Recorder};
        use super::SqliteRecorder;
        use crate::domain::validate::ErrorCategory;

        #[test]
        fn records_round_trip_to_the_database() {
            let dir =
                std::env::temp_dir().join(format!("guardrail-metrics-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let db = dir.join("metrics.sqlite");
            let recorder = SqliteRecorder::open(&db).unwrap();

            recorder.record(OutcomeRecord {
                model: "qwen2.5".into(),
                outcome: Outcome::NativeValid,
                error_category: None,
                parser: None,
                tool_name: Some("get_weather".into()),
                retries: 0,
                detail: None,
            });
            recorder.record(OutcomeRecord {
                model: "qwen2.5".into(),
                outcome: Outcome::FallbackUnfixed,
                error_category: Some(ErrorCategory::MissingArgument),
                parser: None,
                tool_name: Some("Edit".into()),
                retries: 2,
                detail: Some("missing filePath | args: {}".into()),
            });
            // Drop closes the channel; the writer thread drains then exits.
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
    }
}
