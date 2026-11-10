//! RFC 9457 problem details, matching servicekit's error types and titles.

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::{Map, Value};
use utoipa::ToSchema;

/// Error type URIs used by servicekit.
pub mod types {
    pub const NOT_FOUND: &str = "urn:servicekit:error:not-found";
    pub const VALIDATION_FAILED: &str = "urn:servicekit:error:validation-failed";
    pub const CONFLICT: &str = "urn:servicekit:error:conflict";
    pub const INVALID_ULID: &str = "urn:servicekit:error:invalid-ulid";
    pub const INTERNAL: &str = "urn:servicekit:error:internal";
    pub const DATABASE: &str = "urn:servicekit:error:database";
    pub const UNAUTHORIZED: &str = "urn:servicekit:error:unauthorized";
    pub const FORBIDDEN: &str = "urn:servicekit:error:forbidden";
    pub const BAD_REQUEST: &str = "urn:servicekit:error:bad-request";
    pub const PAYLOAD_TOO_LARGE: &str = "urn:servicekit:error:payload-too-large";
}

/// A problem+json response body. `None` fields are omitted, extensions are
/// flattened into the top level.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Problem {
    /// Error type URI.
    #[serde(rename = "type")]
    pub r#type: String,
    /// Short summary.
    pub title: String,
    /// HTTP status code.
    pub status: u16,
    /// Human-readable explanation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// The request path or URL the problem occurred on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// Extension members.
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub extensions: Map<String, Value>,
}

const RESERVED: [&str; 5] = ["type", "title", "status", "detail", "instance"];

impl Problem {
    /// A problem with the given type, title and status.
    pub fn new(r#type: &str, title: &str, status: StatusCode) -> Self {
        Self {
            r#type: r#type.to_owned(),
            title: title.to_owned(),
            status: status.as_u16(),
            detail: None,
            instance: None,
            extensions: Map::new(),
        }
    }

    /// Set the detail.
    #[must_use]
    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Set the instance.
    #[must_use]
    pub fn instance(mut self, instance: impl Into<String>) -> Self {
        self.instance = Some(instance.into());
        self
    }

    /// Add an extension member. Names clashing with reserved fields are dropped.
    #[must_use]
    pub fn extension(mut self, key: &str, value: Value) -> Self {
        if !RESERVED.contains(&key) {
            self.extensions.insert(key.to_owned(), value);
        }
        self
    }

    /// 404 "Resource Not Found".
    pub fn not_found(detail: impl Into<String>) -> Self {
        Self::new(
            types::NOT_FOUND,
            "Resource Not Found",
            StatusCode::NOT_FOUND,
        )
        .detail(detail)
    }

    /// 400 "Invalid ULID Format".
    pub fn invalid_ulid(raw: &str) -> Self {
        Self::new(
            types::INVALID_ULID,
            "Invalid ULID Format",
            StatusCode::BAD_REQUEST,
        )
        .detail(format!("Invalid ULID format: {raw}"))
    }

    /// 400 "Bad Request".
    pub fn bad_request(detail: impl Into<String>) -> Self {
        Self::new(types::BAD_REQUEST, "Bad Request", StatusCode::BAD_REQUEST).detail(detail)
    }

    /// 413 "Payload Too Large".
    pub fn payload_too_large(detail: impl Into<String>) -> Self {
        Self::new(
            types::PAYLOAD_TOO_LARGE,
            "Payload Too Large",
            StatusCode::PAYLOAD_TOO_LARGE,
        )
        .detail(detail)
    }

    /// 409 "Resource Conflict".
    pub fn conflict(detail: impl Into<String>) -> Self {
        Self::new(types::CONFLICT, "Resource Conflict", StatusCode::CONFLICT).detail(detail)
    }

    /// 422 "Validation Error", with an `errors` list.
    pub fn validation(errors: Vec<Value>) -> Self {
        Self::new(
            types::VALIDATION_FAILED,
            "Validation Error",
            StatusCode::UNPROCESSABLE_ENTITY,
        )
        .detail("Request validation failed")
        .extension("errors", Value::Array(errors))
        .extension(
            "trace_id",
            Value::String(ulid::Ulid::generate().to_string()),
        )
    }

    /// 401 "Unauthorized".
    pub fn unauthorized(detail: impl Into<String>) -> Self {
        Self::new(
            types::UNAUTHORIZED,
            "Unauthorized",
            StatusCode::UNAUTHORIZED,
        )
        .detail(detail)
    }

    /// 500 "Internal Server Error". The detail is generic; log the cause instead.
    pub fn internal() -> Self {
        Self::new(
            types::INTERNAL,
            "Internal Server Error",
            StatusCode::INTERNAL_SERVER_ERROR,
        )
        .detail("An internal error occurred")
    }

    /// 500 database error. SQL and driver messages are never exposed.
    pub fn database() -> Self {
        Self::new(
            types::DATABASE,
            "Database Error",
            StatusCode::INTERNAL_SERVER_ERROR,
        )
        .detail("A database error occurred")
    }
}

impl IntoResponse for Problem {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut res = (status, Json(self)).into_response();
        res.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/problem+json"),
        );
        res
    }
}

/// Result alias for handlers.
pub type ApiResult<T> = Result<T, Problem>;
