//! The management API: list providers and their models, choose which are
//! exposed, and have that take effect on the live proxy without a restart.

use std::sync::Arc;
use std::time::Duration;

use guardrail::admin::manage::Management;
use guardrail::admin::{build_admin_app, AdminInfo, AdminState};
use guardrail::application::{AppState, SharedRegistry};
use guardrail::connector::Backend;
use guardrail::domain::config::{Config, ProviderConfig};
use guardrail::domain::provider::Provider;
use guardrail::domain::registry::Registry;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A backend serving `models` and answering chat completions.
async fn backend_with(tag: &str, models: &[&str]) -> MockServer {
    let server = MockServer::start().await;
    serve_models(&server, tag, models).await;
    server
}

/// Mount the catalogue and completion routes on `server`.
///
/// Separate from [`backend_with`] so a test can restate what a provider serves
/// — after `MockServer::reset` — and model a backend that loaded a model after
/// the proxy started.
async fn serve_models(server: &MockServer, tag: &str, models: &[&str]) {
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|id| serde_json::json!({"id": id, "object": "model"}))
        .collect();
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list", "data": data,
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(format!(
            r#"{{"id":"{tag}","object":"chat.completion","choices":[{{"index":0,"message":{{"role":"assistant","content":"ok"}},"finish_reason":"stop"}}]}}"#
        )))
        .mount(server)
        .await;
}

fn temp_config(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("guardrail-mgmt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{label}.json"));
    let _ = std::fs::remove_file(&path);
    path
}

struct Harness {
    proxy: String,
    admin: String,
    config_path: std::path::PathBuf,
}

/// Spin up a proxy and an admin server sharing one live registry.
async fn harness(label: &str, servers: &[(&str, &MockServer, &[&str])]) -> Harness {
    let config_path = temp_config(label);
    let config = Config {
        providers: servers
            .iter()
            .map(|(name, server, _)| ProviderConfig::new(*name, server.uri()))
            .collect(),
    };
    config.save(&config_path).unwrap();

    let mut registry = Registry::new(
        servers
            .iter()
            .map(|(name, server, _)| Provider::new(*name, server.uri()))
            .collect(),
    )
    .unwrap();
    for (name, _, models) in servers {
        for model in *models {
            registry.route(model.to_string(), name);
        }
    }

    let shared: SharedRegistry = Arc::new(tokio::sync::RwLock::new(Arc::new(registry)));
    // Discovery goes through the same `Backend` the proxy forwards with, so a
    // refresh asks the mock servers themselves.
    let management = Arc::new(
        Management::new(shared.clone(), config, config_path.clone())
            .with_discovery(Arc::new(Backend::new(reqwest::Client::new()))),
    );
    for (name, _, models) in servers {
        management
            .set_discovered(name, models.iter().map(|m| openai_rs::Model::new(*m)).collect())
            .await;
    }

    let proxy_app = guardrail::build_app(AppState::with_shared_registry(
        Backend::new(reqwest::Client::new()),
        shared,
    ));
    let proxy_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_addr = proxy_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(proxy_listener, proxy_app).await.unwrap();
    });

    let admin_app = build_admin_app(
        AdminState::new(
            config_path.clone(),
            AdminInfo {
                version: "test".into(),
                providers: vec![],
                proxy_listen: proxy_addr.to_string(),
                admin_listen: "127.0.0.1:0".into(),
                max_retries: 2,
                database: "/tmp/none".into(),
            },
        )
        .with_management(management),
    );
    let admin_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let admin_addr = admin_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(admin_listener, admin_app).await.unwrap();
    });

    Harness {
        proxy: format!("http://{proxy_addr}"),
        admin: format!("http://{admin_addr}"),
        config_path,
    }
}

async fn get_json(url: String) -> serde_json::Value {
    reqwest::get(url).await.unwrap().json().await.unwrap()
}

async fn post_json(url: String) -> (u16, serde_json::Value) {
    let response = reqwest::Client::new().post(url).send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// The discovery entry for `provider`.
fn discovered<'a>(body: &'a serde_json::Value, provider: &str) -> &'a serde_json::Value {
    body["discovery"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == provider)
        .unwrap_or_else(|| panic!("no discovery entry for {provider}"))
}

/// Whether the live registry routes `model`, as `GET /providers` reports it.
fn routes(body: &serde_json::Value, provider: &str, model: &str) -> bool {
    body["providers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["name"] == provider)
        .flat_map(|p| p["models"].as_array().unwrap())
        .any(|m| m["id"] == model && m["routed"] == true)
}

/// Ask for a completion; returns (status, responding backend tag).
async fn ask(proxy: &str, model: &str) -> (u16, String) {
    let response = reqwest::Client::new()
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    let body: serde_json::Value = response.json().await.unwrap();
    (status, body["id"].as_str().unwrap_or_default().to_string())
}

fn listed_ids(body: &serde_json::Value) -> Vec<String> {
    let mut ids: Vec<String> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_string())
        .collect();
    ids.sort();
    ids
}

#[tokio::test]
async fn providers_and_their_models_are_listed() {
    let alpha = backend_with("alpha", &["model-a", "model-b"]).await;
    let beta = backend_with("beta", &["model-c"]).await;
    let h = harness(
        "list",
        &[
            ("alpha", &alpha, &["model-a", "model-b"]),
            ("beta", &beta, &["model-c"]),
        ],
    )
    .await;

    let body = get_json(format!("{}/providers", h.admin)).await;
    let providers = body["providers"].as_array().unwrap();
    assert_eq!(providers.len(), 2);
    assert_eq!(providers[0]["name"], "alpha");
    assert_eq!(providers[0]["models"].as_array().unwrap().len(), 2);
    // Everything is exposed until a choice is made.
    assert_eq!(providers[0]["models"][0]["exposed"], true);
    assert_eq!(providers[1]["name"], "beta");
}

#[tokio::test]
async fn hiding_a_model_removes_it_from_the_listing_and_refuses_it() {
    // The whole point: exposure is not merely cosmetic. What /v1/models
    // advertises and what the proxy will serve must agree.
    let alpha = backend_with("alpha", &["keep", "hide-me"]).await;
    let beta = backend_with("beta", &["other"]).await;
    let h = harness(
        "hide",
        &[
            ("alpha", &alpha, &["keep", "hide-me"]),
            ("beta", &beta, &["other"]),
        ],
    )
    .await;

    // Both work to start with.
    assert_eq!(ask(&h.proxy, "hide-me").await.0, 200);
    assert_eq!(
        listed_ids(&get_json(format!("{}/v1/models", h.proxy)).await),
        vec!["hide-me", "keep", "other"]
    );

    let response = reqwest::Client::new()
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({"models": {"hide-me": false}}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // No restart: the change is live immediately.
    assert_eq!(
        listed_ids(&get_json(format!("{}/v1/models", h.proxy)).await),
        vec!["keep", "other"]
    );
    let (status, _) = ask(&h.proxy, "hide-me").await;
    assert_eq!(status, 404, "a hidden model must be refused, not routed");
    assert_eq!(ask(&h.proxy, "keep").await.0, 200);
}

#[tokio::test]
async fn a_hidden_model_is_refused_rather_than_falling_back() {
    // Without the hidden/unknown distinction this would route to the default
    // provider and quietly serve what the user hid.
    let alpha = backend_with("alpha", &["shared-name"]).await;
    let beta = backend_with("beta", &["other"]).await;
    let h = harness(
        "no-fallback",
        &[
            ("alpha", &alpha, &["shared-name"]),
            ("beta", &beta, &["other"]),
        ],
    )
    .await;

    reqwest::Client::new()
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({"models": {"shared-name": false}}))
        .send()
        .await
        .unwrap();

    let (status, tag) = ask(&h.proxy, "shared-name").await;
    assert_eq!(status, 404);
    assert_ne!(tag, "alpha");
    assert_ne!(tag, "beta", "must not fall back to the default provider");
}

#[tokio::test]
async fn hiding_a_shared_id_on_one_provider_leaves_the_other_serving_it() {
    // Exposure is stored per provider, so it has to behave per provider. It
    // used to collapse into one global set: hiding the local `gpt-4o` also
    // refused Copilot's, and the listing lost both copies — the operator's only
    // way to prefer one vendor's build of a model silently disabled it
    // everywhere.
    let alpha = backend_with("alpha", &["gpt-4o"]).await;
    let beta = backend_with("beta", &["gpt-4o", "only-beta"]).await;
    let h = harness(
        "shared-hide",
        &[
            ("alpha", &alpha, &["gpt-4o"]),
            ("beta", &beta, &["gpt-4o", "only-beta"]),
        ],
    )
    .await;

    // alpha listed it first, so it holds the bare name and beta's copy is
    // qualified.
    assert_eq!(ask(&h.proxy, "gpt-4o").await, (200, "alpha".to_string()));
    assert_eq!(
        listed_ids(&get_json(format!("{}/v1/models", h.proxy)).await),
        vec!["beta/gpt-4o", "gpt-4o", "only-beta"]
    );

    let response = reqwest::Client::new()
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({"models": {"gpt-4o": false}}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // beta inherits the bare name, and serving it is the point: the request
    // must succeed, not 404.
    assert_eq!(ask(&h.proxy, "gpt-4o").await, (200, "beta".to_string()));
    assert_eq!(
        listed_ids(&get_json(format!("{}/v1/models", h.proxy)).await),
        vec!["gpt-4o", "only-beta"],
        "alpha's copy is gone and beta's is no longer crowded onto a qualifier"
    );

    // The hide still holds exactly where it was set.
    assert_eq!(ask(&h.proxy, "alpha/gpt-4o").await.0, 404);
}

#[tokio::test]
async fn a_change_is_persisted_so_it_survives_a_restart() {
    let alpha = backend_with("alpha", &["a", "b"]).await;
    let h = harness("persist", &[("alpha", &alpha, &["a", "b"])]).await;

    reqwest::Client::new()
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({"models": {"b": false}}))
        .send()
        .await
        .unwrap();

    // Read the file the proxy would load on its next start.
    let saved = Config::load(&h.config_path).unwrap().unwrap();
    let provider = saved.provider("alpha").unwrap();
    assert!(provider.exposes("a"));
    assert!(!provider.exposes("b"), "the choice must be on disk");
}

#[tokio::test]
async fn disabling_a_provider_hides_everything_it_serves() {
    let alpha = backend_with("alpha", &["a"]).await;
    let beta = backend_with("beta", &["b"]).await;
    let h = harness(
        "disable",
        &[("alpha", &alpha, &["a"]), ("beta", &beta, &["b"])],
    )
    .await;

    reqwest::Client::new()
        .patch(format!("{}/providers/beta", h.admin))
        .json(&serde_json::json!({"enabled": false}))
        .send()
        .await
        .unwrap();

    assert_eq!(
        listed_ids(&get_json(format!("{}/v1/models", h.proxy)).await),
        vec!["a"]
    );
    // The provider keeps its configuration so re-enabling restores it.
    let saved = Config::load(&h.config_path).unwrap().unwrap();
    assert!(saved.provider("beta").is_some());
    assert!(!saved.provider("beta").unwrap().enabled);
}

#[tokio::test]
async fn expose_by_default_false_curates_instead_of_excludes() {
    // The "expose only what I pick" workflow for a remote server with a large
    // catalogue.
    let alpha = backend_with("alpha", &["a", "b", "c"]).await;
    let h = harness("curate", &[("alpha", &alpha, &["a", "b", "c"])]).await;

    reqwest::Client::new()
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({
            "expose_by_default": false,
            "models": {"b": true},
        }))
        .send()
        .await
        .unwrap();

    let body = get_json(format!("{}/providers", h.admin)).await;
    let models = body["providers"][0]["models"].as_array().unwrap();
    let exposed: Vec<&str> = models
        .iter()
        .filter(|m| m["exposed"] == true)
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(exposed, vec!["b"]);
}

#[tokio::test]
async fn adding_and_removing_a_provider_round_trips() {
    let alpha = backend_with("alpha", &["a"]).await;
    let h = harness("add-remove", &[("alpha", &alpha, &["a"])]).await;

    let added = reqwest::Client::new()
        .post(format!("{}/providers", h.admin))
        .json(&serde_json::json!({
            "name": "remote",
            "base_url": "https://example.com",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(added.status(), 200);
    let body: serde_json::Value = added.json().await.unwrap();
    assert_eq!(body["providers"].as_array().unwrap().len(), 2);

    // A base URL may embed credentials, so it comes back reduced.
    assert_eq!(body["providers"][1]["base_url"], "https://example.com");

    let removed = reqwest::Client::new()
        .delete(format!("{}/providers/remote", h.admin))
        .send()
        .await
        .unwrap();
    assert_eq!(removed.status(), 200);
    assert_eq!(
        Config::load(&h.config_path).unwrap().unwrap().providers.len(),
        1
    );
}

#[tokio::test]
async fn a_duplicate_provider_name_is_refused() {
    let alpha = backend_with("alpha", &["a"]).await;
    let h = harness("duplicate", &[("alpha", &alpha, &["a"])]).await;

    let response = reqwest::Client::new()
        .post(format!("{}/providers", h.admin))
        .json(&serde_json::json!({"name": "alpha", "base_url": "http://other"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 409);
}

#[tokio::test]
async fn disabling_the_last_provider_is_refused_and_rolled_back() {
    // Leaving the proxy with nothing to route to would break every request;
    // better to refuse the change than to accept an unusable state.
    let alpha = backend_with("alpha", &["a"]).await;
    let h = harness("last-provider", &[("alpha", &alpha, &["a"])]).await;

    let response = reqwest::Client::new()
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({"enabled": false}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);

    // And the proxy still works.
    assert_eq!(ask(&h.proxy, "a").await.0, 200);
    assert!(
        Config::load(&h.config_path).unwrap().unwrap().provider("alpha").unwrap().enabled,
        "the refused change must not persist"
    );
}

#[tokio::test]
async fn an_unknown_provider_is_a_404() {
    let alpha = backend_with("alpha", &["a"]).await;
    let h = harness("unknown", &[("alpha", &alpha, &["a"])]).await;

    let response = reqwest::Client::new()
        .patch(format!("{}/providers/nope", h.admin))
        .json(&serde_json::json!({"enabled": false}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
}

#[tokio::test]
async fn clearing_decisions_releases_models_a_caller_can_no_longer_name() {
    // Exposure decisions outlive the model they were made for -- deliberately,
    // so a model that disappears and returns keeps its setting. The cost is
    // that a caller working from the *currently offered* list cannot undo a
    // decision for a model that has since gone: it sets what it can see and
    // leaves the rest stranded hidden. That is what stranded 13 of Copilot's
    // models after a "None" taken when the catalogue was larger.
    let alpha = backend_with("alpha", &["a", "b", "c"]).await;
    let h = harness("clear", &[("alpha", &alpha, &["a", "b", "c"])]).await;
    let client = reqwest::Client::new();

    // Hide all three, as a "None" button does.
    client
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({"models": {"a": false, "b": false, "c": false}}))
        .send()
        .await
        .unwrap();

    // Now only `a` and `b` can be named -- `c` is no longer offered. Setting
    // those two leaves `c` stored false: invisible, and unreachable.
    client
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({"models": {"a": true, "b": true}}))
        .send()
        .await
        .unwrap();
    let saved = Config::load(&h.config_path).unwrap().unwrap();
    assert_eq!(
        saved.provider("alpha").unwrap().models.get("c"),
        Some(&false),
        "c is stranded: the caller could not name it"
    );

    // Clearing drops every stored decision, so anything not named falls back to
    // `expose_by_default` -- which is how "expose everything" is expressed when
    // the caller cannot enumerate what it is undoing.
    client
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({"clear_models": true}))
        .send()
        .await
        .unwrap();
    let saved = Config::load(&h.config_path).unwrap().unwrap();
    assert!(
        saved.provider("alpha").unwrap().models.is_empty(),
        "no stored decision should remain"
    );
    assert!(
        saved.provider("alpha").unwrap().exposes("c"),
        "c inherits expose_by_default again"
    );

    // And the live listing agrees: all three are served.
    let body = get_json(format!("{}/providers", h.admin)).await;
    let exposed = body["providers"][0]["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["exposed"] == true)
        .count();
    assert_eq!(exposed, 3);
}

#[tokio::test]
async fn a_model_loaded_after_startup_becomes_routable_without_a_restart() {
    // The whole point of the endpoint. Discovery is a startup snapshot, so an
    // id a provider picks up later is advertised by the live `/v1/models` and
    // refused by routing — until something re-asks.
    let alpha = backend_with("alpha", &["a"]).await;
    let beta = backend_with("beta", &["b"]).await;
    let h = harness(
        "refresh-late-model",
        &[("alpha", &alpha, &["a"]), ("beta", &beta, &["b"])],
    )
    .await;

    let (status, body) = ask(&h.proxy, "c").await;
    assert_eq!(status, 404, "nothing discovered it: {body}");

    // beta loads it.
    beta.reset().await;
    serve_models(&beta, "beta", &["b", "c"]).await;

    let (status, body) = post_json(format!("{}/discovery", h.admin)).await;
    assert_eq!(status, 200);
    assert_eq!(discovered(&body, "beta")["refreshed"], true);
    assert_eq!(discovered(&body, "beta")["models"], 2);
    assert!(routes(&body, "beta", "c"), "the new id is routed: {body}");

    let (status, tag) = ask(&h.proxy, "c").await;
    assert_eq!(status, 200, "no restart needed");
    assert_eq!(tag, "beta", "and it reaches the provider that loaded it");
}

#[tokio::test]
async fn an_unreachable_provider_keeps_the_catalogue_it_had() {
    // The failure mode a refresh must not have: emptying a good catalogue on a
    // transient error turns every model that provider serves into a 404, which
    // is worse than the staleness the refresh set out to fix.
    let alpha = backend_with("alpha", &["a"]).await;
    let beta = backend_with("beta", &["b"]).await;
    let h = harness(
        "refresh-unreachable",
        &[("alpha", &alpha, &["a"]), ("beta", &beta, &["b"])],
    )
    .await;

    // beta's catalogue endpoint fails, while beta itself keeps answering
    // completions — a restarting server, a rate limit, a blip.
    beta.reset().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&beta)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"id":"beta","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
        ))
        .mount(&beta)
        .await;

    let (status, body) = post_json(format!("{}/discovery", h.admin)).await;
    assert_eq!(status, 200, "one provider failing is not a failed refresh");
    assert_eq!(discovered(&body, "alpha")["refreshed"], true);
    assert_eq!(discovered(&body, "beta")["refreshed"], false);
    assert!(
        discovered(&body, "beta")["error"].is_string(),
        "the caller is told why: {body}"
    );
    assert_eq!(discovered(&body, "beta")["models"], 1, "kept, not emptied");
    assert!(routes(&body, "beta", "b"), "still routed: {body}");
    assert!(routes(&body, "alpha", "a"));
    assert_eq!(
        ask(&h.proxy, "b").await,
        (200, "beta".to_string()),
        "a live model stays served through its provider's blip"
    );
}

#[tokio::test]
async fn a_provider_reporting_nothing_keeps_the_catalogue_it_had() {
    // A local server answering `200 []` while it loads its index is the same
    // transient as an unreachable one, so it is treated the same way — and
    // reported distinctly, with no error, so a caller can tell them apart.
    let alpha = backend_with("alpha", &["a"]).await;
    let beta = backend_with("beta", &["b"]).await;
    let h = harness(
        "refresh-empty",
        &[("alpha", &alpha, &["a"]), ("beta", &beta, &["b"])],
    )
    .await;

    beta.reset().await;
    serve_models(&beta, "beta", &[]).await;

    let (status, body) = post_json(format!("{}/discovery", h.admin)).await;
    assert_eq!(status, 200);
    assert_eq!(discovered(&body, "beta")["refreshed"], false);
    assert!(
        discovered(&body, "beta")["error"].is_null(),
        "answered, so not an error: {body}"
    );
    assert_eq!(discovered(&body, "beta")["models"], 1, "kept, not emptied");
    assert_eq!(ask(&h.proxy, "b").await, (200, "beta".to_string()));
}

#[tokio::test]
async fn a_refresh_picks_up_a_model_a_provider_stopped_hiding() {
    // Exposure decisions outlive discovery, so a refresh has to apply them to
    // whatever comes back rather than routing everything it is told about.
    let alpha = backend_with("alpha", &["a", "secret"]).await;
    let beta = backend_with("beta", &["b"]).await;
    let h = harness(
        "refresh-honours-exposure",
        &[("alpha", &alpha, &["a"]), ("beta", &beta, &["b"])],
    )
    .await;

    let hidden = reqwest::Client::new()
        .patch(format!("{}/providers/alpha", h.admin))
        .json(&serde_json::json!({"models": {"secret": false}}))
        .send()
        .await
        .unwrap();
    assert_eq!(hidden.status(), 200);

    let (status, body) = post_json(format!("{}/discovery", h.admin)).await;
    assert_eq!(status, 200);
    assert_eq!(discovered(&body, "alpha")["models"], 2, "both discovered");
    assert!(
        !routes(&body, "alpha", "secret"),
        "the decision survives the refresh: {body}"
    );
    assert_eq!(ask(&h.proxy, "secret").await.0, 404);
    assert!(routes(&body, "alpha", "a"));
}

#[tokio::test]
async fn a_removed_provider_forgets_what_it_reported() {
    // The catalogue is keyed by name and would otherwise outlive the entry, so
    // re-adding a provider under the same name would resurrect models it may no
    // longer serve — routed immediately, before it has been asked anything.
    let alpha = backend_with("alpha", &["a"]).await;
    let beta = backend_with("beta", &["b"]).await;
    let h = harness(
        "refresh-forgets",
        &[("alpha", &alpha, &["a"]), ("beta", &beta, &["b"])],
    )
    .await;

    let removed = reqwest::Client::new()
        .delete(format!("{}/providers/beta", h.admin))
        .send()
        .await
        .unwrap();
    assert_eq!(removed.status(), 200);

    let added = reqwest::Client::new()
        .post(format!("{}/providers", h.admin))
        .json(&serde_json::json!({"name": "beta", "base_url": beta.uri()}))
        .send()
        .await
        .unwrap();
    assert_eq!(added.status(), 200);
    let body: serde_json::Value = added.json().await.unwrap();
    assert!(
        !routes(&body, "beta", "b"),
        "nothing is claimed before it is asked: {body}"
    );

    // And a refresh is what puts it back.
    let (status, body) = post_json(format!("{}/discovery", h.admin)).await;
    assert_eq!(status, 200);
    assert!(routes(&body, "beta", "b"), "{body}");
}

#[tokio::test]
async fn a_removal_during_discovery_is_not_undone_by_the_reply() {
    // Discovery snapshots the providers, spends a round trip per provider, and
    // only then writes what came back. A `DELETE` landing inside that window
    // used to be undone by the reply: the removed provider was written back
    // into the catalogue, where the next rebuild ignored it — until it was
    // added again under the same name and its stale models were routed at once,
    // which is the exact failure removal clears the entry to prevent.
    let alpha = backend_with("alpha", &["a"]).await;
    let beta = MockServer::start().await;
    // Slow enough that the removal below is issued while beta is still being
    // asked. It has to wait its turn, rather than interleaving.
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({
                    "object": "list", "data": [{"id": "b", "object": "model"}],
                }))
                .set_delay(Duration::from_millis(400)),
        )
        .mount(&beta)
        .await;

    let h = harness(
        "discovery-vs-removal",
        &[("alpha", &alpha, &["a"]), ("beta", &beta, &["b"])],
    )
    .await;

    let admin = h.admin.clone();
    let discovering =
        tokio::spawn(async move { post_json(format!("{admin}/discovery")).await });
    tokio::time::sleep(Duration::from_millis(80)).await;

    let removed = reqwest::Client::new()
        .delete(format!("{}/providers/beta", h.admin))
        .send()
        .await
        .unwrap();
    assert_eq!(removed.status(), 200);
    let (status, _) = discovering.await.unwrap();
    assert_eq!(status, 200, "the discovery itself still succeeds");

    // Adding it back must start from nothing known about it.
    let added = reqwest::Client::new()
        .post(format!("{}/providers", h.admin))
        .json(&serde_json::json!({"name": "beta", "base_url": beta.uri()}))
        .send()
        .await
        .unwrap();
    assert_eq!(added.status(), 200);
    let body: serde_json::Value = added.json().await.unwrap();
    assert!(
        !routes(&body, "beta", "b"),
        "the reply for a removed provider must not survive as a catalogue: {body}"
    );
}
