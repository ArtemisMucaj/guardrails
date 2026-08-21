//! Multi-provider routing: a request reaches the provider that serves its
//! model, and its outcome is recorded against that provider.

use std::sync::Arc;

use guardrail::application::AppState;
use guardrail::connector::Backend;
use guardrail::domain::metrics::{SqliteRecorder, Stats};
use guardrail::domain::provider::Provider;
use guardrail::domain::registry::Registry;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A backend answering chat completions with plain text, tagged so the test can
/// tell which one replied.
async fn backend_named(tag: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(format!(
            r#"{{"id":"{tag}","object":"chat.completion","choices":[{{"index":0,"message":{{"role":"assistant","content":"from {tag}"}},"finish_reason":"stop"}}]}}"#
        )))
        .mount(&server)
        .await;
    server
}

async fn spawn_proxy(state: AppState) -> String {
    let app = guardrail::build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn temp_db(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("guardrail-routing-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.sqlite"))
}

/// Ask the proxy to complete `model` and return the responding backend's tag.
async fn ask(proxy: &str, model: &str) -> String {
    let body: serde_json::Value = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    body["id"].as_str().unwrap_or_default().to_string()
}

#[tokio::test]
async fn each_model_reaches_the_provider_that_serves_it() {
    let alpha = backend_named("alpha").await;
    let beta = backend_named("beta").await;

    let mut registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        Provider::new("beta", beta.uri()),
    ])
    .unwrap();
    assert!(registry.route("model-a", "alpha"));
    assert!(registry.route("model-b", "beta"));

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    assert_eq!(ask(&proxy, "model-a").await, "alpha");
    assert_eq!(ask(&proxy, "model-b").await, "beta");
}

#[tokio::test]
async fn an_unrouted_model_falls_back_to_the_first_provider() {
    // Discovery can miss a model a backend loaded after startup. Falling back
    // keeps those working instead of failing a request the single-backend
    // proxy would have served.
    let alpha = backend_named("alpha").await;
    let beta = backend_named("beta").await;

    let mut registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        Provider::new("beta", beta.uri()),
    ])
    .unwrap();
    registry.route("model-b", "beta");

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    assert_eq!(ask(&proxy, "never-discovered").await, "alpha");
}

#[tokio::test]
async fn outcomes_are_recorded_against_the_serving_provider() {
    // The point of the provider column: the same model id on two providers must
    // not merge into one row.
    let alpha = backend_named("alpha").await;
    let beta = backend_named("beta").await;

    let mut registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        Provider::new("beta", beta.uri()),
    ])
    .unwrap();
    // Same model id, served by different providers is impossible through
    // routing alone (first wins), so give each provider its own id and assert
    // both land under their own provider.
    registry.route("model-a", "alpha");
    registry.route("model-b", "beta");

    let db = temp_db("outcomes");
    let _ = std::fs::remove_file(&db);
    let recorder = Arc::new(SqliteRecorder::open(&db).unwrap());

    let proxy = spawn_proxy(
        AppState::with_registry(Backend::new(reqwest::Client::new()), registry)
            .with_recorder(recorder.clone()),
    )
    .await;

    ask(&proxy, "model-a").await;
    ask(&proxy, "model-b").await;

    // Close the recorder so the background writer flushes.
    drop(recorder);
    // The proxy holds a clone; give the writer a moment to drain.
    for _ in 0..50 {
        if Stats::read(&db).map(|s| s.per_model.len()).unwrap_or(0) >= 2 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    let stats = Stats::read(&db).unwrap();
    let mut rows: Vec<(String, String)> = stats
        .per_model
        .iter()
        .map(|m| (m.provider.clone(), m.model.clone()))
        .collect();
    rows.sort();

    assert_eq!(
        rows,
        vec![
            ("alpha".to_string(), "model-a".to_string()),
            ("beta".to_string(), "model-b".to_string()),
        ],
        "each outcome must be attributed to the provider that served it"
    );

    let _ = std::fs::remove_file(&db);
}
