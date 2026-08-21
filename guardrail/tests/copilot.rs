//! The Copilot provider end to end: gating headers and the credential reach the
//! upstream on every path, a client cannot displace them, and the proxy's
//! `/v1/...` surface maps onto Copilot's root-level routes.
//!
//! Runs against `wiremock` on localhost — no credentials, no network.

use std::time::Duration;

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use gh_copilot_rs::CopilotToken;
use guardrail::application::{AppState, BackendPort};
use guardrail::connector::Backend;
use guardrail::domain::provider::Provider;
use guardrail::domain::registry::Registry;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The six headers GitHub gates Copilot access on.
const GATING_HEADERS: &[&str] = &[
    "copilot-integration-id",
    "editor-version",
    "editor-plugin-version",
    "x-github-api-version",
    "openai-intent",
    "user-agent",
];

const TOKEN: &str = "ghu_test_token";

/// The shipped Copilot provider, pointed at a mock instead of GitHub.
fn copilot(base_url: &str) -> (Provider, Backend) {
    let built = guardrail::copilot::provider(
        CopilotToken::new(TOKEN),
        Duration::from_secs(10),
        Duration::from_secs(300),
    )
    .expect("provider builds with a valid token");

    // `with_base_url` is not on the built provider, so rebuild the same shape
    // against the mock: unversioned routes, same reserved names.
    let provider = Provider::new(built.provider.name(), base_url)
        .unversioned()
        .owning_credential()
        .reserving(built.provider.reserved_headers().collect::<Vec<_>>());

    let backend = Backend::new(reqwest::Client::new())
        .with_client_for(provider.name(), built.client);
    (provider, backend)
}

fn client_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    headers
}

fn assert_copilot_request(received: &HeaderMap) {
    for name in GATING_HEADERS {
        assert!(
            received.get(*name).is_some(),
            "gating header `{name}` never reached the upstream; Copilot would 403"
        );
    }
    assert_eq!(
        received.get("authorization").map(|v| v.to_str().unwrap()),
        Some(format!("Bearer {TOKEN}").as_str()),
        "the credential must reach the upstream"
    );
}

#[tokio::test]
async fn chat_completions_reaches_the_root_route_with_the_credential() {
    let server = MockServer::start().await;
    // Copilot serves at the root, not under /v1 — this path is the assertion.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"id":"c","choices":[]}"#))
        .mount(&server)
        .await;

    let (provider, backend) = copilot(&server.uri());
    let target = provider.target("/v1/chat/completions");
    assert_eq!(target, format!("{}/chat/completions", server.uri()));

    let (status, _h, _b) = backend
        .post(&provider, &target, &client_headers(), b"{}".to_vec())
        .await
        .expect("request reaches the mock");
    assert_eq!(status, 200);

    let requests = server.received_requests().await.unwrap();
    assert_copilot_request(&requests[0].headers);
}

#[tokio::test]
async fn the_streaming_path_carries_the_credential_too() {
    // Guarded requests always go upstream with stream: true, so this path
    // matters as much as the buffered one.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "data: {\"choices\":[]}\n\ndata: [DONE]\n\n",
            "text/event-stream",
        ))
        .mount(&server)
        .await;

    let (provider, backend) = copilot(&server.uri());
    let target = provider.target("/v1/chat/completions");

    let (status, _h, _rx, is_sse) = backend
        .stream_post(&provider, &target, &client_headers(), b"{}".to_vec())
        .await
        .map_err(|_| "stream_post failed")
        .expect("stream request reaches the mock");
    assert_eq!(status, 200);
    assert!(is_sse);

    let requests = server.received_requests().await.unwrap();
    assert_copilot_request(&requests[0].headers);
}

#[tokio::test]
async fn a_clients_own_credential_cannot_displace_copilots() {
    // The bug the reserved-header mechanism exists for. OpenAI-compatible
    // clients routinely send `Bearer no-key`; without reservation reqwest's
    // per-name replacement would hand that to GitHub and every request would
    // 401 — reading as an expired token rather than a precedence bug.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&server)
        .await;

    let mut headers = client_headers();
    headers.insert(
        HeaderName::from_static("authorization"),
        HeaderValue::from_static("Bearer no-key"),
    );
    // A client's user-agent is a gating header too, and just as commonly sent.
    headers.insert(
        HeaderName::from_static("user-agent"),
        HeaderValue::from_static("some-client/1.0"),
    );

    let (provider, backend) = copilot(&server.uri());
    let target = provider.target("/v1/chat/completions");
    let _ = backend
        .post(&provider, &target, &headers, b"{}".to_vec())
        .await;

    let requests = server.received_requests().await.unwrap();
    let received = &requests[0].headers;
    assert_copilot_request(received);
    assert_ne!(
        received.get("user-agent").map(|v| v.to_str().unwrap()),
        Some("some-client/1.0"),
        "the client's user-agent must not reach GitHub as the client identity"
    );
}

#[tokio::test]
async fn the_models_catalogue_is_fetched_from_the_root_route() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [{"id": "gpt-4o", "object": "model"}],
        })))
        .mount(&server)
        .await;

    let (provider, backend) = copilot(&server.uri());
    let target = provider.target("/v1/models");
    assert_eq!(target, format!("{}/models", server.uri()));

    let response = backend
        .forward(
            &provider,
            axum::http::Method::GET,
            &target,
            &client_headers(),
            bytes::Bytes::new(),
        )
        .await;
    assert_eq!(response.status(), 200);

    let requests = server.received_requests().await.unwrap();
    assert_copilot_request(&requests[0].headers);
}

#[tokio::test]
async fn a_copilot_model_routes_through_the_proxy_end_to_end() {
    // The whole path: client -> proxy -> Copilot-shaped upstream, with routing
    // by model and the credential applied.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"id":"copilot","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#,
        ))
        .mount(&server)
        .await;

    let (provider, backend) = copilot(&server.uri());
    let mut registry = Registry::new(vec![provider]).unwrap();
    registry.route("gpt-4o", "copilot");

    let app = guardrail::build_app(AppState::with_registry(backend, registry));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let body: serde_json::Value = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        // The placeholder key a client would send.
        .header("authorization", "Bearer no-key")
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(body["id"], "copilot");

    let requests = server.received_requests().await.unwrap();
    assert_copilot_request(&requests[0].headers);
}
