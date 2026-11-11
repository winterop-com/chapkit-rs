//! Request ids and API-key auth.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Request, State};
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::problem::Problem;

/// Tag every response with a fresh `X-Request-ID` (a ULID) and log the request.
pub async fn request_id(req: Request, next: Next) -> Response {
    let id = ulid::Ulid::generate().to_string();
    let (method, path) = (req.method().clone(), req.uri().path().to_owned());
    let route = req
        .extensions()
        .get::<axum::extract::MatchedPath>()
        .map_or_else(|| "unmatched".to_owned(), |m| m.as_str().to_owned());
    let start = Instant::now();
    let mut res = next.run(req).await;
    tracing::info!(
        request_id = %id, %method, %path, status_code = res.status().as_u16(),
        duration_ms = start.elapsed().as_secs_f64() * 1000.0, "http.request.complete"
    );
    if route != "/metrics" {
        crate::metrics::observe_http(
            method.as_str(),
            &route,
            res.status().as_u16(),
            start.elapsed().as_secs_f64(),
        );
    }
    if let Ok(v) = HeaderValue::from_str(&id) {
        res.headers_mut().insert("x-request-id", v);
    }
    res
}

/// API-key auth settings.
#[derive(Debug, Clone)]
pub struct ApiKeys {
    /// Accepted keys.
    pub keys: HashSet<String>,
    /// Header carrying the key.
    pub header: String,
    /// Paths that need no key (exact match).
    pub open_paths: HashSet<String>,
}

impl ApiKeys {
    /// Keys from `SERVICEKIT_API_KEYS` (comma separated). `None` when unset or empty.
    pub fn from_env() -> Option<Self> {
        let keys: HashSet<String> = std::env::var("SERVICEKIT_API_KEYS")
            .ok()?
            .split(',')
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(str::to_owned)
            .collect();
        (!keys.is_empty()).then(|| Self::new(keys))
    }

    /// Keys with servicekit's defaults: header `X-API-Key`, docs and health open.
    pub fn new(keys: HashSet<String>) -> Self {
        let open_paths = ["/docs", "/redoc", "/openapi.json", "/health", "/"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        Self {
            keys,
            header: "X-API-Key".into(),
            open_paths,
        }
    }
}

/// Reject requests without a valid API key (401 problem+json).
pub async fn require_api_key(
    State(cfg): State<Arc<ApiKeys>>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_owned();
    // The web UI's files are open like the docs; its API calls are not.
    if cfg.open_paths.contains(&path) || path == "/ui" || path.starts_with("/ui/") {
        return next.run(req).await;
    }
    match req
        .headers()
        .get(cfg.header.as_str())
        .and_then(|v| v.to_str().ok())
    {
        None => Problem::unauthorized(format!("Missing authentication header: {}", cfg.header))
            .instance(path)
            .into_response(),
        Some(k) if cfg.keys.contains(k) => next.run(req).await,
        Some(k) => {
            tracing::warn!(key_prefix = %k.chars().take(7).collect::<String>(), "auth.invalid_key");
            Problem::unauthorized("Invalid API key")
                .instance(path)
                .into_response()
        }
    }
}
