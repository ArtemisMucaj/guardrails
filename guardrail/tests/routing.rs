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

/// A backend answering chat completions and embeddings, tagged so the test can
/// tell which one replied.
///
/// Both replies echo the `model` the backend was actually asked for, which is
/// how a test sees that a provider qualifier was stripped before the hop.
async fn backend_named(tag: &str) -> MockServer {
    let server = MockServer::start().await;
    let chat_tag = tag.to_string();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            let tag = chat_tag.clone();
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": tag,
                "object": "chat.completion",
                "model": received_model(req),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": format!("from {tag}")},
                    "finish_reason": "stop",
                }],
            }))
        })
        .mount(&server)
        .await;
    let embedding_tag = tag.to_string();
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(move |req: &wiremock::Request| {
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": embedding_tag.clone(),
                "object": "list",
                "model": received_model(req),
                "data": [{"object": "embedding", "index": 0, "embedding": [0.0]}],
            }))
        })
        .mount(&server)
        .await;
    server
}

/// The `model` field of a request the backend received.
fn received_model(req: &wiremock::Request) -> String {
    serde_json::from_slice::<serde_json::Value>(&req.body)
        .ok()
        .and_then(|b| b["model"].as_str().map(str::to_string))
        .unwrap_or_default()
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
    chat(proxy, model).await["id"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

/// Ask the proxy to complete `model` and return the backend's whole reply.
async fn chat(proxy: &str, model: &str) -> serde_json::Value {
    post(
        proxy,
        "/v1/chat/completions",
        serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await
}

/// Ask the proxy to embed with `model` and return the backend's whole reply.
async fn embed(proxy: &str, model: &str) -> serde_json::Value {
    post(
        proxy,
        "/v1/embeddings",
        serde_json::json!({"model": model, "input": "hi"}),
    )
    .await
}

async fn post(proxy: &str, path: &str, body: serde_json::Value) -> serde_json::Value {
    reqwest::Client::new()
        .post(format!("{proxy}{path}"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
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
async fn an_embeddings_request_reaches_the_provider_that_serves_its_model() {
    // The bug this guards: routing read the model only on the chat path, so an
    // embeddings request went to the default provider — a different upstream,
    // which either 404s the model or, worse, answers with vectors from some
    // other model that silently do not match the stored ones.
    let alpha = backend_named("alpha").await;
    let beta = backend_named("beta").await;

    let mut registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        Provider::new("beta", beta.uri()),
    ])
    .unwrap();
    registry.route("text-embedding-3-small", "beta");

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    assert_eq!(embed(&proxy, "text-embedding-3-small").await["id"], "beta");
}

#[tokio::test]
async fn a_qualified_model_reaches_that_provider_and_arrives_bare() {
    // `provider/model` is the proxy's own addressing, and the only way to name
    // the loser of a duplicate id. The upstream published the bare id, so the
    // qualifier must not survive the hop.
    let alpha = backend_named("alpha").await;
    let beta = backend_named("beta").await;

    let mut registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        Provider::new("beta", beta.uri()),
    ])
    .unwrap();
    // alpha claimed the bare id, so beta's copy is reachable only qualified.
    registry.route("shared", "alpha");

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    let reply = chat(&proxy, "beta/shared").await;
    assert_eq!(reply["id"], "beta", "the qualifier decides the provider");
    assert_eq!(reply["model"], "shared", "the qualifier is stripped");

    // The same on the embeddings path, which reads the model from the body the
    // proxy rewrites rather than from a typed request.
    let reply = embed(&proxy, "beta/shared").await;
    assert_eq!(reply["id"], "beta");
    assert_eq!(reply["model"], "shared");

    // Unqualified still goes where routing says, unchanged.
    let reply = chat(&proxy, "shared").await;
    assert_eq!(reply["id"], "alpha");
    assert_eq!(reply["model"], "shared");
}

#[tokio::test]
async fn a_hidden_model_is_refused_on_the_embeddings_path_too() {
    // Hiding has to mean the same thing on every path, or it is a suggestion.
    let alpha = backend_named("alpha").await;
    let mut registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        Provider::new("beta", backend_named("beta").await.uri()),
    ])
    .unwrap();
    registry.route("text-embedding-3-small", "beta");
    registry.hide("text-embedding-3-small", "beta");

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    let response = reqwest::Client::new()
        .post(format!("{proxy}/v1/embeddings"))
        .json(&serde_json::json!({"model": "text-embedding-3-small", "input": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn hiding_a_model_on_one_provider_leaves_another_serving_it() {
    // The case an operator actually hits: the same id on a local server and on
    // Copilot, hidden on the local one. It used to 404 everywhere, because
    // hiding was a single global set — and the listing dropped both copies.
    let alpha = backend_with_models("alpha", &["gpt-4o"]).await;
    let beta = backend_with_models("beta", &["gpt-4o"]).await;

    let mut registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        Provider::new("beta", beta.uri()),
    ])
    .unwrap();
    // What discovery records when alpha's config hides the id and beta's
    // exposes it.
    registry.hide("gpt-4o", "alpha");
    registry.route("gpt-4o", "beta");

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    // Served, by the provider that still exposes it.
    let reply = chat(&proxy, "gpt-4o").await;
    assert_eq!(reply["id"], "beta");
    assert_eq!(reply["model"], "gpt-4o");

    // Listed once, bare, under that provider: alpha's copy is gone, beta's is
    // not pushed to a qualified id by a rival that no longer claims the name.
    assert_eq!(
        ids_with_providers(&list_models(&proxy).await),
        vec![("gpt-4o".to_string(), "beta".to_string())]
    );

    // And naming alpha explicitly is still refused — hiding is addressed now,
    // not global, so it holds exactly where it was set.
    let response = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "alpha/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
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

/// A backend whose `/v1/models` lists `models`, and which also answers chat.
async fn backend_with_models(tag: &str, models: &[&str]) -> MockServer {
    let server = backend_named(tag).await;
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|id| serde_json::json!({"id": id, "object": "model"}))
        .collect();
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": data,
        })))
        .mount(&server)
        .await;
    server
}

async fn list_models(proxy: &str) -> serde_json::Value {
    reqwest::get(format!("{proxy}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

fn ids_with_providers(body: &serde_json::Value) -> Vec<(String, String)> {
    let mut rows: Vec<(String, String)> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["id"].as_str().unwrap_or_default().to_string(),
                m["provider"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect();
    rows.sort();
    rows
}

#[tokio::test]
async fn models_lists_the_union_across_providers() {
    // A client must see every routable model, not just the default provider's,
    // or it cannot name one that would route elsewhere.
    let alpha = backend_with_models("alpha", &["model-a"]).await;
    let beta = backend_with_models("beta", &["model-b"]).await;

    let registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        Provider::new("beta", beta.uri()),
    ])
    .unwrap();

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    let body = list_models(&proxy).await;
    assert_eq!(body["object"], "list");
    assert_eq!(
        ids_with_providers(&body),
        vec![
            ("model-a".to_string(), "alpha".to_string()),
            ("model-b".to_string(), "beta".to_string()),
        ]
    );
}

#[tokio::test]
async fn a_duplicate_id_is_listed_qualified_so_it_stays_reachable() {
    // Routing gives a contested bare id to the first provider listed, so
    // listing it bare twice would tell the client something routing will not
    // honour. Listing it once dropped beta's model from the catalogue and left
    // it with no name a client could ask for, so it appears qualified — the
    // form routing accepts to reach beta specifically.
    let alpha = backend_with_models("alpha", &["shared"]).await;
    let beta = backend_with_models("beta", &["shared", "only-beta"]).await;

    let mut registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        Provider::new("beta", beta.uri()),
    ])
    .unwrap();
    // The routes discovery would have recorded: alpha listed `shared` first, so
    // beta's claim on it loses.
    registry.route("shared", "alpha");
    registry.route("only-beta", "beta");

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    assert_eq!(
        ids_with_providers(&list_models(&proxy).await),
        vec![
            ("beta/shared".to_string(), "beta".to_string()),
            ("only-beta".to_string(), "beta".to_string()),
            ("shared".to_string(), "alpha".to_string()),
        ]
    );

    // And every listed id resolves to the provider it is listed under.
    assert_eq!(ask(&proxy, "shared").await, "alpha");
    assert_eq!(ask(&proxy, "beta/shared").await, "beta");
    assert_eq!(ask(&proxy, "only-beta").await, "beta");
}

#[tokio::test]
async fn an_unreachable_provider_does_not_empty_the_list() {
    // A local server that is not running yet must not hide the models of one
    // that is.
    let alpha = backend_with_models("alpha", &["model-a"]).await;
    let registry = Registry::new(vec![
        Provider::new("alpha", alpha.uri()),
        // A port nothing is listening on.
        Provider::new("down", "http://127.0.0.1:1"),
    ])
    .unwrap();

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    assert_eq!(
        ids_with_providers(&list_models(&proxy).await),
        vec![("model-a".to_string(), "alpha".to_string())]
    );
}

#[tokio::test]
async fn a_single_provider_response_is_forwarded_untouched() {
    // With nothing to merge there is nothing to disambiguate, so the
    // byte-for-byte passthrough single-backend users have today is preserved —
    // no `provider` tag is added.
    let alpha = backend_with_models("alpha", &["model-a", "model-b"]).await;
    let registry = Registry::single(Provider::new("default", alpha.uri()));

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    let body = list_models(&proxy).await;
    assert_eq!(
        body,
        serde_json::json!({
            "object": "list",
            "data": [
                {"id": "model-a", "object": "model"},
                {"id": "model-b", "object": "model"},
            ]
        })
    );
}

#[tokio::test]
async fn every_provider_failing_is_an_error_not_an_empty_list() {
    // An empty list would read as "this proxy serves no models", which is a
    // different and misleading claim from "nothing could be reached".
    let registry = Registry::new(vec![
        Provider::new("down-one", "http://127.0.0.1:1"),
        Provider::new("down-two", "http://127.0.0.1:2"),
    ])
    .unwrap();

    let proxy = spawn_proxy(AppState::with_registry(
        Backend::new(reqwest::Client::new()),
        registry,
    ))
    .await;

    let response = reqwest::get(format!("{proxy}/v1/models")).await.unwrap();
    assert_eq!(response.status(), 502);
}
