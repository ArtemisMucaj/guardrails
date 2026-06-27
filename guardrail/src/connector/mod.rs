//! HTTP connector — POST, stream, and filter headers for the backend.
//!
//! This is the infrastructure layer: it depends inward on the application's
//! `BackendPort` and provides the concrete `reqwest`-backed adapter that
//! implements it.

use axum::{
    body::Body,
    http::{header::CONNECTION, HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
};

use crate::application::BackendPort;

/// Concrete backend adapter that delegates to a `reqwest::Client`.
#[derive(Clone)]
pub struct Backend {
    client: reqwest::Client,
}

impl Backend {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

#[async_trait::async_trait]
impl BackendPort for Backend {
    async fn post(
        &self,
        target: &str,
        headers: &HeaderMap,
        body: Vec<u8>,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>), Response> {
        post_backend(&self.client, target, headers, body).await
    }

    async fn stream_post(
        &self,
        target: &str,
        headers: &HeaderMap,
        body: Vec<u8>,
    ) -> Result<(StatusCode, HeaderMap, tokio::sync::mpsc::Receiver<Option<String>>, bool), Response> {
        use futures_util::StreamExt;

        let resp = self
            .client
            .post(target)
            .headers(forward_headers(headers))
            .body(body)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(error = %e, target = %target, "backend stream request failed");
                (StatusCode::BAD_GATEWAY, "backend request failed").into_response()
            })?;

        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let out_headers = copy_response_headers(resp.headers());

        // Peek at the content-type to decide whether to stream or buffer.
        let is_sse = out_headers
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.contains("text/event-stream"))
            .unwrap_or(false);

        let (tx, rx) = tokio::sync::mpsc::channel::<Option<String>>(256);

        if is_sse {
            // True streaming path: read bytes as they arrive, split into lines,
            // send each line down the channel immediately.
            let mut byte_stream = resp.bytes_stream();
            tokio::spawn(async move {
                let mut buf = String::new();
                while let Some(chunk) = byte_stream.next().await {
                    let Ok(bytes) = chunk else { break };
                    let Ok(text) = std::str::from_utf8(&bytes) else { break };
                    buf.push_str(text);
                    // Emit complete lines as they accumulate.
                    while let Some(pos) = buf.find('\n') {
                        let end = if pos > 0 && buf.as_bytes()[pos - 1] == b'\r' { pos - 1 } else { pos };
                        let line = buf[..end].to_string();
                        buf.drain(..=pos);
                        if tx.send(Some(line)).await.is_err() {
                            return;
                        }
                    }
                }
                // Flush any remaining partial line.
                if !buf.is_empty() {
                    let _ = tx.send(Some(buf)).await;
                }
                let _ = tx.send(None).await; // EOF sentinel
            });
        } else {
            // Backend returned JSON or non-SSE — buffer it, convert, then feed
            // the synthetic lines into the channel from a background task.
            let text = resp.text().await.map_err(|e| {
                tracing::error!(error = %e, "failed to read backend response body");
                (StatusCode::BAD_GATEWAY, "failed to read backend response").into_response()
            })?;
            let trimmed = text.trim_start();
            let sse = if trimmed.starts_with('{') || trimmed.starts_with('[') {
                json_to_sse(&text)
            } else {
                // Non-JSON, non-SSE — forward verbatim via the error path.
                return Err(crate::connector::bytes_response(
                    status,
                    out_headers,
                    text.into_bytes(),
                ));
            };
            tokio::spawn(async move {
                for line in sse.lines() {
                    if tx.send(Some(line.to_string())).await.is_err() {
                        return;
                    }
                }
                let _ = tx.send(None).await;
            });
        }

        Ok((status, out_headers, rx, is_sse))
    }

    async fn forward(
        &self,
        method: Method,
        target: &str,
        headers: &HeaderMap,
        body: bytes::Bytes,
    ) -> Response {
        // Forward verbatim: preserve the body for every method (some APIs send
        // entity bodies with GET/DELETE), matching the proxy's transparency.
        let resp = match self
            .client
            .request(method, target)
            .headers(forward_headers(headers))
            .body(body.to_vec())
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(error = %e, target = %target, "backend request failed");
                return (StatusCode::BAD_GATEWAY, "backend request failed").into_response();
            }
        };
        relay_response(resp)
    }
}

/// Headers that are connection-specific and must not be forwarded across a
/// proxy hop (RFC 9110 §7.6.1).
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
];

/// Wrap a plain JSON chat-completion body into a minimal synthetic SSE stream.
/// Used when a backend ignores `stream: true` and returns JSON directly.
fn json_to_sse(json: &str) -> String {
    // Convert `chat.completion` → `chat.completion.chunk` and
    // `message` → `delta` so the assembler sees a normal chunk stream.
    let mut sse = String::new();
    if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(json) {
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "object".to_string(),
                serde_json::Value::String("chat.completion.chunk".to_string()),
            );
            if let Some(choices) = obj.get_mut("choices").and_then(serde_json::Value::as_array_mut) {
                for choice in choices.iter_mut() {
                    if let Some(c) = choice.as_object_mut() {
                        if let Some(msg) = c.remove("message") {
                            c.insert("delta".to_string(), msg);
                        }
                    }
                }
            }
        }
        if let Ok(s) = serde_json::to_string(&value) {
            sse.push_str("data: ");
            sse.push_str(&s);
            sse.push_str("\n\n");
        }
    }
    sse.push_str("data: [DONE]\n\n");
    sse
}

/// POST a body to the backend and return the status, filtered response headers,
/// and the fully-buffered body. Errors are mapped to a client-facing response.
pub async fn post_backend(
    client: &reqwest::Client,
    target: &str,
    headers: &HeaderMap,
    body: Vec<u8>,
) -> Result<(StatusCode, HeaderMap, Vec<u8>), Response> {
    let resp = client
        .post(target)
        .headers(forward_headers(headers))
        .body(body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!(error = %e, target = %target, "backend request failed");
            (StatusCode::BAD_GATEWAY, "backend request failed").into_response()
        })?;

    let status =
        StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let out_headers = copy_response_headers(resp.headers());
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "failed to read backend response body");
            (StatusCode::BAD_GATEWAY, "failed to read backend response").into_response()
        })?
        .to_vec();
    Ok((status, out_headers, bytes))
}

/// Stream the backend response back to the client, preserving status and headers.
pub fn relay_response(resp: reqwest::Response) -> Response {
    let status =
        StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let headers = copy_response_headers(resp.headers());

    let mut response = Response::new(Body::from_stream(resp.bytes_stream()));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

/// Build a response from already-buffered bytes, preserving status and headers.
pub fn bytes_response(status: StatusCode, headers: HeaderMap, bytes: Vec<u8>) -> Response {
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

/// Copy client → backend headers, dropping hop-by-hop headers (both the static
/// set and any header named in this message's own `Connection` header).
pub fn forward_headers(src: &HeaderMap) -> HeaderMap {
    let connection = connection_header_names(src);
    let mut out = HeaderMap::with_capacity(src.len());
    for (name, value) in src.iter() {
        if should_strip_header(name, &connection) || name == "content-length" {
            continue;
        }
        out.append(name.clone(), value.clone());
    }
    out
}

/// Copy backend → client response headers, dropping hop-by-hop (static +
/// Connection-named) and length/framing headers.
fn copy_response_headers(src: &HeaderMap) -> HeaderMap {
    let connection = connection_header_names(src);
    let mut headers = HeaderMap::with_capacity(src.len());
    for (name, value) in src.iter() {
        if should_strip_header(name, &connection) || name == "content-length" {
            continue;
        }
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_ref()),
            HeaderValue::from_bytes(value.as_ref()),
        ) {
            headers.append(n, v);
        }
    }
    headers
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    HOP_BY_HOP
        .iter()
        .any(|h| name.as_str().eq_ignore_ascii_case(h))
}

fn connection_header_names(headers: &HeaderMap) -> Vec<HeaderName> {
    headers
        .get_all(CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect()
}

fn should_strip_header(name: &HeaderName, connection: &[HeaderName]) -> bool {
    is_hop_by_hop(name) || connection.iter().any(|h| h == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_map(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut m = HeaderMap::new();
        for (k, v) in pairs {
            m.append(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        m
    }

    #[test]
    fn forward_headers_strips_static_hop_by_hop() {
        let src = header_map(&[
            ("host", "example.com"),
            ("connection", "keep-alive"),
            ("authorization", "Bearer t"),
            ("content-type", "application/json"),
        ]);
        let out = forward_headers(&src);
        assert!(out.get("host").is_none());
        assert!(out.get("connection").is_none());
        assert_eq!(out.get("authorization").unwrap(), "Bearer t");
        assert_eq!(out.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn forward_headers_strips_content_length() {
        // The guardrail loop rewrites (and usually grows) the request body before
        // forwarding — it injects the synthetic `respond` tool and may re-emit
        // repaired calls. Forwarding the client's original Content-Length would
        // truncate the rewritten body at the backend, so it must be dropped and
        // recomputed from the actual bytes by the HTTP client.
        let src = header_map(&[
            ("content-length", "42"),
            ("content-type", "application/json"),
        ]);
        let out = forward_headers(&src);
        assert!(out.get("content-length").is_none());
        assert_eq!(out.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn forward_headers_strips_connection_named_headers() {
        let src = header_map(&[
            ("connection", "x-internal, x-trace"),
            ("x-internal", "secret"),
            ("x-trace", "abc"),
            ("x-keep", "kept"),
        ]);
        let out = forward_headers(&src);
        assert!(out.get("x-internal").is_none());
        assert!(out.get("x-trace").is_none());
        assert_eq!(out.get("x-keep").unwrap(), "kept");
    }

    #[test]
    fn connection_token_matching_is_case_insensitive() {
        let headers = header_map(&[("connection", "X-Internal")]);
        let names = connection_header_names(&headers);
        let lower = HeaderName::from_static("x-internal");
        assert!(should_strip_header(&lower, &names));
    }
}
