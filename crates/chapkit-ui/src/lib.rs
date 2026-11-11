//! The chapkit-rs web UI, embedded in the binary.
//!
//! `ui/` (React on sokkel-ui) builds into this crate's `dist/`, which is embedded with rust-embed:
//! the files are compiled into the binary (in debug builds too, with `debug-embed`), so a model
//! binary serves its UI with no files beside it and no network at build or run time. [`router`]
//! serves it at `/ui/`, with `index.html` for every path that is not a file (the UI routes on the
//! client), and redirects `/` there.

use axum::Router;
use axum::extract::Path;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use rust_embed::Embed;

/// Where the UI is served.
pub const PREFIX: &str = "/ui/";

const INDEX: &str = "index.html";

#[derive(Embed)]
#[folder = "dist/"]
struct Dist;

/// `GET /` (redirect to `/ui/`), `GET /ui` (likewise) and `GET /ui/*` (the UI's files).
pub fn router() -> Router {
    Router::new()
        .route("/", get(|| async { Redirect::temporary(PREFIX) }))
        .route("/ui", get(|| async { Redirect::temporary(PREFIX) }))
        .route("/ui/", get(|h: HeaderMap| async move { serve("", &h) }))
        .route(
            "/ui/{*path}",
            get(|Path(path): Path<String>, h: HeaderMap| async move { serve(&path, &h) }),
        )
}

/// `Cache-Control` of a file: content-hashed build output (Vite's `assets/`) never changes under
/// its name; anything else, `index.html` first, is revalidated on every load.
pub fn cache_control(path: &str) -> &'static str {
    if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    }
}

/// The `Content-Type` of a file, with a charset for text.
fn content_type(mime: &str) -> String {
    let text = mime.starts_with("text/")
        || matches!(
            mime,
            "application/javascript" | "application/json" | "image/svg+xml"
        );
    if text {
        format!("{mime}; charset=utf-8")
    } else {
        mime.to_owned()
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from(DIGITS[usize::from(b >> 4)]));
        out.push(char::from(DIGITS[usize::from(b & 0xf)]));
    }
    out
}

/// A file of the build, or `index.html` for any other path.
fn serve(path: &str, headers: &HeaderMap) -> Response {
    let (path, file) = match Dist::get(path).filter(|_| !path.is_empty()) {
        Some(file) => (path, file),
        None => match Dist::get(INDEX) {
            Some(file) => (INDEX, file),
            None => return StatusCode::NOT_FOUND.into_response(),
        },
    };
    let etag = format!("\"{}\"", hex(&file.metadata.sha256_hash()));
    let cache = HeaderValue::from_static(cache_control(path));
    let fresh = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|t| t.trim() == etag || t.trim() == "*"));
    let mut res = if fresh {
        StatusCode::NOT_MODIFIED.into_response()
    } else {
        let mut res = file.data.into_owned().into_response();
        if let Ok(v) = HeaderValue::from_str(&content_type(file.metadata.mimetype())) {
            res.headers_mut().insert(header::CONTENT_TYPE, v);
        }
        res
    };
    res.headers_mut().insert(header::CACHE_CONTROL, cache);
    if let Ok(v) = HeaderValue::from_str(&etag) {
        res.headers_mut().insert(header::ETAG, v);
    }
    res
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn get(uri: &str, etag: Option<&str>) -> Response {
        let mut req = Request::get(uri);
        if let Some(e) = etag {
            req = req.header(header::IF_NONE_MATCH, e);
        }
        router()
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn serves_index_for_client_routes_and_redirects_the_root() {
        for uri in ["/", "/ui"] {
            let res = get(uri, None).await;
            assert_eq!(res.status(), StatusCode::TEMPORARY_REDIRECT);
            assert_eq!(res.headers()[header::LOCATION], "/ui/");
        }
        let index = get("/ui/", None).await;
        assert_eq!(index.status(), StatusCode::OK);
        assert_eq!(
            index.headers()[header::CONTENT_TYPE],
            "text/html; charset=utf-8"
        );
        assert_eq!(index.headers()[header::CACHE_CONTROL], "no-cache");
        let etag = index.headers()[header::ETAG].to_str().unwrap().to_owned();
        let body = index.into_body().collect().await.unwrap().to_bytes();
        assert!(body.starts_with(b"<!doctype html>") || body.starts_with(b"<!DOCTYPE html>"));
        for uri in ["/ui/datasets/01ABC", "/ui/index.html"] {
            let res = get(uri, None).await;
            assert_eq!(res.status(), StatusCode::OK);
            assert_eq!(res.headers()[header::ETAG], etag.as_str());
        }
        let cached = get("/ui/", Some(&etag)).await;
        assert_eq!(cached.status(), StatusCode::NOT_MODIFIED);
        assert!(
            cached
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .is_empty()
        );
    }

    #[test]
    fn hashed_assets_are_immutable_and_types_carry_a_charset() {
        assert_eq!(
            cache_control("assets/index-BkX3a9fq.js"),
            "public, max-age=31536000, immutable"
        );
        assert_eq!(cache_control("index.html"), "no-cache");
        assert_eq!(cache_control("favicon.svg"), "no-cache");
        assert_eq!(content_type("text/css"), "text/css; charset=utf-8");
        assert_eq!(content_type("image/png"), "image/png");
        assert_eq!(hex(&[0, 171, 255]), "00abff");
    }

    #[test]
    fn every_embedded_file_is_served_with_its_type() {
        for path in Dist::iter() {
            let file = Dist::get(&path).unwrap();
            assert!(!file.metadata.mimetype().is_empty(), "{path}");
        }
    }
}
