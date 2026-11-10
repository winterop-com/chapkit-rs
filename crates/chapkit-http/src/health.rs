//! `GET /health`: aggregated health checks, as servicekit.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

/// Health of one check or of the whole service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum HealthState {
    /// Working.
    Healthy,
    /// Working with problems.
    Degraded,
    /// Not working.
    Unhealthy,
}

/// One check's result.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct CheckResult {
    /// State.
    pub state: HealthState,
    /// Optional explanation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// The health response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct HealthStatus {
    /// Worst state over all checks.
    pub status: HealthState,
    /// Per-check results; omitted when there are no checks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checks: Option<BTreeMap<String, CheckResult>>,
}

/// A named async health check.
pub type Check = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = CheckResult> + Send>> + Send + Sync>;

/// The configured checks.
#[derive(Clone, Default)]
pub struct Checks(pub Vec<(String, Check)>);

async fn status_of(checks: &Checks) -> (StatusCode, HealthStatus) {
    let mut results = BTreeMap::new();
    for (name, check) in &checks.0 {
        results.insert(name.clone(), check().await);
    }
    let status = results
        .values()
        .map(|r| r.state)
        .max()
        .unwrap_or(HealthState::Healthy);
    let code = if status == HealthState::Unhealthy {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };
    (
        code,
        HealthStatus {
            status,
            checks: (!results.is_empty()).then_some(results),
        },
    )
}

/// Service health. 503 when any check is unhealthy; degraded still returns 200.
#[utoipa::path(get, path = "/health", tag = "Observability",
    responses((status = OK, body = HealthStatus), (status = SERVICE_UNAVAILABLE, body = HealthStatus)))]
async fn health(State(checks): State<Checks>) -> Response {
    let (code, body) = status_of(&checks).await;
    (code, axum::Json(body)).into_response()
}

/// Stream the health status as server-sent events, forever.
#[utoipa::path(get, path = "/health/$stream", tag = "Observability", params(crate::sse::PollParams),
    responses((status = OK, content_type = "text/event-stream", body = String)))]
async fn stream_health(
    State(checks): State<Checks>,
    crate::extract::Query(p): crate::extract::Query<crate::sse::PollParams>,
) -> Result<Response, crate::problem::Problem> {
    let interval = p.interval(1.0)?;
    let stream = futures_util::stream::unfold((checks, true), move |(checks, first)| async move {
        if !first {
            tokio::time::sleep(interval).await;
        }
        let (_, body) = status_of(&checks).await;
        Some((Ok(crate::sse::event(&body)), (checks, false)))
    });
    Ok(crate::sse::response(stream))
}

/// The health router (mounted at the root).
pub fn router(checks: Checks) -> OpenApiRouter {
    OpenApiRouter::new()
        .routes(routes!(health))
        .routes(routes!(stream_health))
        .with_state(checks)
}

/// Check a running service's `/health` at `base` (`http://127.0.0.1:8000`): `Ok` only for HTTP 200
/// with `"status": "healthy"`. Used by `<binary> healthcheck` (chapkit's `Service::run`), which
/// container health checks call, since minimal images have no curl.
///
/// The error is one line: `no service at <base>`, or `<base>/health answered 503 (unhealthy)`.
pub async fn probe(base: &str) -> std::io::Result<()> {
    let base = base.trim_end_matches('/');
    let url = format!("{base}/health");
    let res = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(std::io::Error::other)?
        .get(&url)
        .send()
        .await
        .map_err(|e| {
            std::io::Error::other(if e.is_connect() {
                format!("no service at {base}")
            } else if e.is_timeout() {
                format!("{url} did not answer in 5 s")
            } else {
                format!("{url}: {e}")
            })
        })?;
    let status = res.status();
    let body: serde_json::Value = res.json().await.unwrap_or_default();
    let state = body
        .get("status")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("no status");
    if status.is_success() && state == "healthy" {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "{url} answered {} ({state})",
            status.as_u16()
        )))
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;

    #[tokio::test]
    async fn probe_reports_healthy_and_unreachable() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (router, _) = router(Checks::default()).split_for_parts();
        tokio::spawn(async move { axum::serve(listener, router).await });
        probe(&format!("http://127.0.0.1:{port}")).await.unwrap();
        // Nothing listens on the port after this one is dropped.
        let free = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let unused = free.local_addr().unwrap().port();
        drop(free);
        let err = probe(&format!("http://127.0.0.1:{unused}"))
            .await
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("no service at http://127.0.0.1:{unused}")
        );
        // Something answers, but not a healthy chapkit service.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let sick = axum::Router::new().route(
            "/health",
            axum::routing::get(|| async {
                (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    axum::Json(serde_json::json!({"status": "unhealthy"})),
                )
            }),
        );
        tokio::spawn(async move { axum::serve(listener, sick).await });
        let err = probe(&format!("http://127.0.0.1:{port}"))
            .await
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("http://127.0.0.1:{port}/health answered 503 (unhealthy)")
        );
    }
}
