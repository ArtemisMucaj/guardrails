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
        providers: vec!["default=http://127.0.0.1:1234".into()],
        proxy_listen: "127.0.0.1:8080".into(),
        admin_listen: "127.0.0.1:8081".into(),
        max_retries: 2,
        database: db_path.display().to_string(),
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
        provider: "default".into(),
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
    assert_eq!(
        body["providers"],
        serde_json::json!(["default=http://127.0.0.1:1234"])
    );
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
    assert_eq!(body["errors"].as_array().unwrap().len(), 0);
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
        ..rec("m", Outcome::RetriesExhausted)
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
    let err = &body["errors"][0];
    assert_eq!(err["model"], "m");
    assert_eq!(err["tool_name"], "Edit");
    assert_eq!(err["error_category"], "missing_argument");
    assert_eq!(err["count"], 1);
}

#[tokio::test]
async fn copilot_login_routes_are_absent_without_a_copilot_provider() {
    // The mutable surface should not exist at all unless it is needed.
    let admin = spawn_admin(temp_db("no-copilot")).await;

    let status = reqwest::get(format!("{admin}/copilot/login")).await.unwrap();
    assert_eq!(status.status(), 404);

    let started = reqwest::Client::new()
        .post(format!("{admin}/copilot/login"))
        .send()
        .await
        .unwrap();
    assert_eq!(started.status(), 404);

    // And discovery must not advertise it.
    let index: serde_json::Value = reqwest::get(format!("{admin}/"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let endpoints = index["endpoints"].as_array().unwrap();
    assert!(
        !endpoints.iter().any(|e| e == "/copilot/login"),
        "got: {endpoints:?}"
    );
}

#[tokio::test]
async fn copilot_login_status_is_idle_before_any_attempt_and_carries_no_token() {
    let db = temp_db("copilot-idle");
    let store = db.with_file_name("copilot-token-idle");
    let _ = std::fs::remove_file(&store);

    let login = guardrail::copilot::CopilotLogin::new(store.clone()).unwrap();
    let state = AdminState::new(
        db,
        AdminInfo {
            version: "9.9.9".into(),
            providers: vec!["copilot=https://api.githubcopilot.com".into()],
            proxy_listen: "127.0.0.1:8080".into(),
            admin_listen: "127.0.0.1:8081".into(),
            max_retries: 2,
            database: "/tmp/db".into(),
        },
    )
    .with_login(login);

    let app = guardrail::build_admin_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let response = reqwest::get(format!("http://{addr}/copilot/login"))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();

    assert_eq!(body, r#"{"status":"idle"}"#);
    // The guarantee that makes this endpoint safe: CopilotToken is
    // serde-transparent, so a credential in a serialized type would be emitted
    // in full rather than redacted.
    assert!(!body.contains("ghu_"), "a response body must never carry a token");

    // The route is listed once it exists.
    let index: serde_json::Value = reqwest::get(format!("http://{addr}/"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(index["endpoints"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e == "/copilot/login"));

    let _ = std::fs::remove_file(&store);
}
