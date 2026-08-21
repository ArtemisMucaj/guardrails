//! The management API: list providers and their models, choose which are
//! exposed, and have that take effect on the live proxy without a restart.

use std::sync::Arc;

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
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|id| serde_json::json!({"id": id, "object": "model"}))
        .collect();
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list", "data": data,
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(format!(
            r#"{{"id":"{tag}","object":"chat.completion","choices":[{{"index":0,"message":{{"role":"assistant","content":"ok"}},"finish_reason":"stop"}}]}}"#
        )))
        .mount(&server)
        .await;
    server
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
    let management = Management::new(shared.clone(), config, config_path.clone());
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
