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
