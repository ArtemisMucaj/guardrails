//! HTTP connector — POST, stream, and filter headers for the backend.

use axum::{
    body::Body,
    http::{header::CONNECTION, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};

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
            (
                StatusCode::BAD_GATEWAY,
                format!("backend request failed: {e}"),
            )
                .into_response()
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
        if should_strip_header(name, &connection) {
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
