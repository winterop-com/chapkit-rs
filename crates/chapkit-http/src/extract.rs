//! Extractors that fail with servicekit-style problem responses.

use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::extract::{FromRequest, FromRequestParts, Request};
use axum::http::request::Parts;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use ulid::Ulid;
use utoipa::{IntoParams, ToSchema};

use crate::problem::Problem;

/// JSON body extractor: malformed or invalid bodies give a 422 problem.
#[derive(Debug, Clone, Copy, Default)]
pub struct Json<T>(pub T);

impl<S: Send + Sync, T: DeserializeOwned> FromRequest<S> for Json<T> {
    type Rejection = Problem;

    async fn from_request(req: Request, state: &S) -> Result<Self, Problem> {
        let instance = req.uri().path().to_owned();
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(v)) => Ok(Json(v)),
            Err(e) if e.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE => {
                Err(Problem::payload_too_large(
                    "Request body is larger than the service accepts (CHAPKIT_MAX_BODY_MB)",
                )
                .instance(instance))
            }
            Err(e) => Err(Problem::validation(vec![
                json!({"loc": ["body"], "msg": body_error(&e), "type": "value_error"}),
            ])
            .instance(instance)),
        }
    }
}

fn body_error(e: &JsonRejection) -> String {
    e.body_text()
}

/// Query extractor: invalid query parameters give a 422 problem.
#[derive(Debug, Clone, Copy, Default)]
pub struct Query<T>(pub T);

impl<S: Send + Sync, T: DeserializeOwned> FromRequestParts<S> for Query<T> {
    type Rejection = Problem;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Problem> {
        match axum::extract::Query::<T>::from_request_parts(parts, state).await {
            Ok(axum::extract::Query(v)) => Ok(Query(v)),
            Err(e) => Err(query_problem(&e).instance(parts.uri.path().to_owned())),
        }
    }
}

fn query_problem(e: &QueryRejection) -> Problem {
    Problem::validation(vec![
        json!({"loc": ["query"], "msg": e.body_text(), "type": "value_error"}),
    ])
}

/// Parse a ULID path segment, or fail with a 400 "Invalid ULID Format" problem.
pub fn parse_ulid(raw: &str) -> Result<Ulid, Problem> {
    Ulid::from_string(raw).map_err(|_| Problem::invalid_ulid(raw))
}

/// Optional `page` and `size` query parameters. Pagination applies only when both are given.
#[derive(Debug, Clone, Copy, Default, Deserialize, IntoParams)]
pub struct PageParams {
    /// Page number, starting at 1.
    pub page: Option<i64>,
    /// Page size, 1-100.
    pub size: Option<i64>,
}

impl PageParams {
    /// `(page, size)` when both are given and in range; `None` when either is
    /// missing; a 422 problem when out of range.
    pub fn resolve(self) -> Result<Option<(u64, u64)>, Problem> {
        let mut errors = Vec::new();
        if self.page.is_some_and(|p| p < 1) {
            errors.push(json!({"loc": ["query", "page"], "msg": "Input should be greater than or equal to 1", "type": "greater_than_equal"}));
        }
        if self.size.is_some_and(|s| !(1..=100).contains(&s)) {
            errors.push(json!({"loc": ["query", "size"], "msg": "Input should be between 1 and 100", "type": "value_error"}));
        }
        if !errors.is_empty() {
            return Err(Problem::validation(errors));
        }
        Ok(match (self.page, self.size) {
            // Both are validated positive above, so the conversions cannot fail.
            (Some(p), Some(s)) => {
                Some((u64::try_from(p).unwrap_or(1), u64::try_from(s).unwrap_or(1)))
            }
            _ => None,
        })
    }
}

/// Row offset of a page: `(page - 1) * size`, saturating, and capped at `i64::MAX` because
/// SQLite's OFFSET is a signed 64-bit integer. A page past the end is simply empty.
pub fn page_offset(page: u64, size: u64) -> u64 {
    page.saturating_sub(1)
        .saturating_mul(size)
        .min(i64::MAX.unsigned_abs())
}

/// A page of results.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Paginated<T> {
    /// Items on this page.
    pub items: Vec<T>,
    /// Total item count.
    pub total: u64,
    /// This page number.
    pub page: u64,
    /// Page size.
    pub size: u64,
    /// Number of pages (0 when there are no items).
    pub pages: u64,
}

impl<T> Paginated<T> {
    /// Build a page; `pages` is `ceil(total / size)`.
    pub fn new(items: Vec<T>, total: u64, page: u64, size: u64) -> Self {
        let pages = if size == 0 { 0 } else { total.div_ceil(size) };
        Self {
            items,
            total,
            page,
            size,
            pages,
        }
    }
}

/// A list response: a plain array, or a [`Paginated`] page when both `page` and `size`
/// are given. Used for OpenAPI documentation of list endpoints.
#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(untagged)]
pub enum ListOrPage<T> {
    /// No pagination requested.
    List(Vec<T>),
    /// A page.
    Page(Paginated<T>),
}

impl<T: Serialize> axum::response::IntoResponse for ListOrPage<T> {
    fn into_response(self) -> axum::response::Response {
        axum::Json(self).into_response()
    }
}

impl<T> ListOrPage<T> {
    /// A list, or a page when `paging` is `Some((page, size))`.
    pub fn new(items: Vec<T>, total: u64, paging: Option<(u64, u64)>) -> Self {
        match paging {
            Some((page, size)) => ListOrPage::Page(Paginated::new(items, total, page, size)),
            None => ListOrPage::List(items),
        }
    }
}

/// `{"total": n}` for `$stats` endpoints.
#[derive(Debug, Clone, Copy, Serialize, ToSchema)]
pub struct Stats {
    /// Entity count.
    pub total: u64,
}
