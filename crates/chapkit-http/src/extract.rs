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
    pub fn resolve(self) -> Result<Option<(u32, u32)>, Problem> {
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
            (Some(p), Some(s)) => Some((p as u32, s as u32)),
            _ => None,
        })
    }
}

/// A page of results.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Paginated<T> {
    /// Items on this page.
    pub items: Vec<T>,
    /// Total item count.
    pub total: u64,
    /// This page number.
    pub page: u32,
    /// Page size.
    pub size: u32,
    /// Number of pages (0 when there are no items).
    pub pages: u64,
}

impl<T> Paginated<T> {
    /// Build a page; `pages` is `ceil(total / size)`.
    pub fn new(items: Vec<T>, total: u64, page: u32, size: u32) -> Self {
        let pages = if size == 0 {
            0
        } else {
            total.div_ceil(u64::from(size))
        };
        Self {
            items,
            total,
            page,
            size,
            pages,
        }
    }
}

/// `{"total": n}` for `$stats` endpoints.
#[derive(Debug, Clone, Copy, Serialize, ToSchema)]
pub struct Stats {
    /// Entity count.
    pub total: u64,
}
