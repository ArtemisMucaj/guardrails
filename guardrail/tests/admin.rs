//! The admin server is a separate, read-only HTTP port for operators and
//! embedding UIs (a desktop app). These tests spin it up on an ephemeral port
//! and assert its JSON contract: a liveness probe, a description of the running
//! proxy, and the failure-metrics rollup read straight from the SQLite database.

use std::net::SocketAddr;

use guardrail::admin::{build_admin_app, AdminInfo, AdminState};
use guardrail::domain::metrics::{now_rfc3339, Outcome, OutcomeRecord, Recorder, SqliteRecorder};
use guardrail::domain::validate::ErrorCategory;

/// Spawn the admin server reading from `db_path`, return its base URL.
async fn spawn_admin(db_path: std::path::PathBuf) -> String {
    let info = AdminInfo {
        version: "9.9.9".into(),
        backend: "http://127.0.0.1:1234".into(),
        proxy_listen: "127.0.0.1:8080".into(),
        admin_listen: "127.0.0.1:8081".into(),
        max_retries: 2,
        metrics_db: db_path.display().to_string(),
    };
    let app = build_admin_app(AdminState::new(db_path, info));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

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

/// A temp database path unique to this test process and a given label.
fn temp_db(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("guardrail-admin-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("metrics.sqlite")
}

#[tokio::test]
async fn healthz_reports_ok() {
    let admin = spawn_admin(temp_db("health")).await;
    let body: serde_json::Value = reqwest::get(format!("{admin}/healthz"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn info_describes_the_proxy() {
    let admin = spawn_admin(temp_db("info")).await;
    let body: serde_json::Value = reqwest::get(format!("{admin}/info"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["version"], "9.9.9");
    assert_eq!(body["backend"], "http://127.0.0.1:1234");
    assert_eq!(body["max_retries"], 2);
}

#[tokio::test]
async fn stats_for_a_missing_database_is_empty() {
    // No proxy has run, so the database does not exist: the endpoint must read
    // as empty rather than error (mirrors `Stats::read`).
    let admin = spawn_admin(temp_db("missing").with_file_name("nope.sqlite")).await;
    let resp = reqwest::get(format!("{admin}/stats")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["per_model"].as_array().unwrap().len(), 0);
    assert_eq!(body["unfixed"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn stats_returns_the_metrics_rollup_as_json() {
    let db = temp_db("rollup");
    let recorder = SqliteRecorder::open(&db).unwrap();
    // 2 real tool calls (1 unfixed), plus a respond and a plain-text passthrough
    // that must NOT count as tool calls — matches the metrics unit test.
    recorder.record(rec("m", Outcome::NativeValid));
    recorder.record(OutcomeRecord {
        tool_name: Some("Edit".into()),
        error_category: Some(ErrorCategory::MissingArgument),
        detail: Some("missing filePath | args: {}".into()),
        retries: 2,
        ..rec("m", Outcome::FallbackUnfixed)
    });
    recorder.record(rec("m", Outcome::RespondIntercept));
    recorder.record(rec("m", Outcome::PassthroughNoCalls));
    drop(recorder); // flushes the background writer

    let admin = spawn_admin(db).await;
    let body: serde_json::Value = reqwest::get(format!("{admin}/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let model = &body["per_model"][0];
    assert_eq!(model["model"], "m");
    assert_eq!(model["total"], 4);
    assert_eq!(model["tool_calls"], 2); // respond + passthrough excluded
    assert_eq!(model["succeeded"], 1);
    assert_eq!(model["errors"], 1);
    assert_eq!(model["success_rate"], 0.5);

    // Outcome breakdown is a list of named {outcome, count} objects.
    let by_outcome = model["by_outcome"].as_array().unwrap();
    assert!(by_outcome
        .iter()
        .any(|o| o["outcome"] == "native_valid" && o["count"] == 1));

    // The single unfixed error is surfaced for triage.
    let unfixed = &body["unfixed"][0];
    assert_eq!(unfixed["model"], "m");
    assert_eq!(unfixed["tool_name"], "Edit");
    assert_eq!(unfixed["error_category"], "missing_argument");
    assert_eq!(unfixed["count"], 1);
}
