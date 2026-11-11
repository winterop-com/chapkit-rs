//! HTTP layer for chapkit-rs: the servicekit-compatible parts of a Chap service
//! that have nothing to do with models. RFC 9457 problem responses with
//! servicekit's error types, an in-memory job scheduler and `/api/v1/jobs`,
//! `/health`, `/api/v1/system`, `/api/v1/info`, API keys, request ids,
//! orchestrator registration, OpenAPI at `/openapi.json` and docs at `/docs`.

mod app;
pub mod extract;
pub mod health;
pub mod ids;
pub mod jobs;
pub mod metrics;
pub mod middleware;
pub mod problem;
pub mod registration;
pub mod sse;
mod system;
pub mod time;

pub use app::{App, init_tracing, max_body_from_env};
pub use extract::{Json, ListOrPage, PageParams, Paginated, Query, Stats, parse_ulid};
pub use health::{CheckResult, HealthState};
pub use jobs::{JobError, JobRecord, JobStatus, Scheduler};
pub use problem::{ApiResult, Problem};

/// Re-exports for building routers.
pub use utoipa;
pub use utoipa_axum;
