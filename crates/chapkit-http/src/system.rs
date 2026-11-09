//! `GET /api/v1/system` and `GET /api/v1/info`.

use axum::extract::State;
use serde::Serialize;
use serde_json::Value;
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use crate::time::rfc3339_now;

/// Host information, shaped like servicekit's system response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct SystemInfo {
    /// Current UTC time.
    pub current_time: String,
    /// Server timezone (always UTC).
    pub timezone: String,
    /// Always null: this is a Rust service. Kept for shape compatibility.
    pub python_version: Option<String>,
    /// chapkit-rs version.
    pub chapkit_rs_version: String,
    /// `os-arch`.
    pub platform: String,
    /// Host name.
    pub hostname: String,
}

/// System information.
#[utoipa::path(get, path = "/api/v1/system", tag = "Service", responses((status = OK, body = SystemInfo)))]
async fn system() -> axum::Json<SystemInfo> {
    axum::Json(SystemInfo {
        current_time: rfc3339_now(),
        timezone: "UTC".into(),
        python_version: None,
        chapkit_rs_version: env!("CARGO_PKG_VERSION").into(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        hostname: gethostname::gethostname().to_string_lossy().into_owned(),
    })
}

/// Bundled static web apps. chapkit-rs serves none; kept for API compatibility.
#[utoipa::path(get, path = "/api/v1/system/apps", tag = "Service", responses((status = OK, body = Vec<Object>)))]
async fn apps() -> axum::Json<Vec<Value>> {
    axum::Json(Vec::new())
}

/// JSON schema of the app list.
#[utoipa::path(get, path = "/api/v1/system/apps/$schema", tag = "Service", responses((status = OK, body = Object)))]
async fn apps_schema() -> axum::Json<Value> {
    axum::Json(
        serde_json::json!({"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "version": {"type": "string"}, "prefix": {"type": "string"},
        "description": {"type": ["string", "null"]}, "author": {"type": ["string", "null"]},
        "entry": {"type": "string"}, "is_package": {"type": "boolean"}}}}),
    )
}

/// Service information (the service's `ServiceInfo` / `MLServiceInfo` JSON).
#[utoipa::path(get, path = "/api/v1/info", tag = "Service", responses((status = OK, body = Object)))]
async fn info(State(info): State<Value>) -> axum::Json<Value> {
    axum::Json(info)
}

/// System and info routes.
pub fn router(service_info: Value) -> OpenApiRouter {
    OpenApiRouter::new()
        .routes(routes!(system))
        .routes(routes!(apps))
        .routes(routes!(apps_schema))
        .merge(
            OpenApiRouter::new()
                .routes(routes!(info))
                .with_state(service_info),
        )
}
