//! In-memory job scheduler and the `/api/v1/jobs` router, matching servicekit.
//!
//! Jobs are queued behind a semaphore (default concurrency 1, as in chapkit).
//! Records live in memory only and are lost on restart.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, Semaphore};
use tokio::task::AbortHandle;
use ulid::Ulid;
use utoipa::{IntoParams, ToSchema};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use crate::extract::{Json, Query, parse_ulid};
use crate::problem::{ApiResult, Problem};
use crate::time::job_now;

/// Job lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Queued.
    Pending,
    /// Running.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Cancel requested while running on a blocking thread.
    Canceling,
    /// Canceled.
    Canceled,
}

impl JobStatus {
    /// True for completed, failed and canceled.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Canceled
        )
    }
}

/// A job record, as servicekit's `JobRecord`.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct JobRecord {
    /// Job id (ULID).
    #[schema(value_type = String)]
    pub id: Ulid,
    /// Current status.
    pub status: JobStatus,
    /// When the job was submitted.
    pub submitted_at: String,
    /// When it started running.
    pub started_at: Option<String>,
    /// When it reached a terminal state.
    pub finished_at: Option<String>,
    /// Short error, `"Kind: message"`.
    pub error: Option<String>,
    /// Longer error description.
    pub error_traceback: Option<String>,
}

/// A job's failure: a short kind (like a Python exception name) and a message.
#[derive(Debug, Clone)]
pub struct JobError {
    /// Error kind, e.g. `ValueError`.
    pub kind: String,
    /// Message.
    pub message: String,
}

impl JobError {
    /// A `ValueError`-style failure (bad input or config).
    pub fn value(message: impl Into<String>) -> Self {
        Self {
            kind: "ValueError".into(),
            message: message.into(),
        }
    }

    /// A `RuntimeError`-style failure (anything else).
    pub fn runtime(message: impl Into<String>) -> Self {
        Self {
            kind: "RuntimeError".into(),
            message: message.into(),
        }
    }
}

/// The work a job runs.
pub type JobFuture = Pin<Box<dyn Future<Output = Result<(), JobError>> + Send>>;

struct Entry {
    record: JobRecord,
    abort: Option<AbortHandle>,
    done: Arc<Notify>,
}

#[derive(Default)]
struct Inner {
    jobs: HashMap<Ulid, Entry>,
}

/// The scheduler. Cheap to clone.
#[derive(Clone)]
pub struct Scheduler {
    inner: Arc<Mutex<Inner>>,
    permits: Arc<Semaphore>,
}

impl Scheduler {
    /// A scheduler running at most `max_concurrency` jobs at once.
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            inner: Arc::default(),
            permits: Arc::new(Semaphore::new(max_concurrency.max(1))),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn update(&self, id: Ulid, f: impl FnOnce(&mut JobRecord)) {
        if let Some(e) = self.lock().jobs.get_mut(&id) {
            f(&mut e.record);
        }
    }

    /// Submit a job; returns its id immediately.
    pub fn submit(&self, work: JobFuture) -> Ulid {
        let id = Ulid::generate();
        let record = JobRecord {
            id,
            status: JobStatus::Pending,
            submitted_at: job_now(),
            started_at: None,
            finished_at: None,
            error: None,
            error_traceback: None,
        };
        let done = Arc::new(Notify::new());
        self.lock().jobs.insert(
            id,
            Entry {
                record,
                abort: None,
                done: done.clone(),
            },
        );

        let this = self.clone();
        let handle = tokio::spawn(async move {
            let Ok(_permit) = this.permits.clone().acquire_owned().await else {
                return;
            };
            this.update(id, |r| {
                r.status = JobStatus::Running;
                r.started_at = Some(job_now());
            });
            let result = work.await;
            this.update(id, |r| {
                r.finished_at = Some(job_now());
                match result {
                    _ if r.status == JobStatus::Canceling => r.status = JobStatus::Canceled,
                    Ok(()) => r.status = JobStatus::Completed,
                    Err(e) => {
                        r.status = JobStatus::Failed;
                        r.error = Some(format!("{}: {}", e.kind, e.message));
                        r.error_traceback = Some(format!("{}: {}", e.kind, e.message));
                    }
                }
            });
            done.notify_waiters();
        });
        if let Some(e) = self.lock().jobs.get_mut(&id) {
            e.abort = Some(handle.abort_handle());
        }
        id
    }

    /// A job's record.
    pub fn get(&self, id: Ulid) -> Option<JobRecord> {
        self.lock().jobs.get(&id).map(|e| e.record.clone())
    }

    /// All records, newest submission first, optionally filtered by status.
    pub fn list(&self, status: Option<JobStatus>) -> Vec<JobRecord> {
        let mut v: Vec<JobRecord> = self
            .lock()
            .jobs
            .values()
            .map(|e| e.record.clone())
            .filter(|r| status.is_none_or(|s| r.status == s))
            .collect();
        v.sort_by_key(|r| std::cmp::Reverse(r.id));
        v
    }

    /// Cancel a job if it has not finished, then forget it. Returns false if unknown.
    pub fn delete(&self, id: Ulid) -> bool {
        let entry = self.lock().jobs.remove(&id);
        match entry {
            Some(e) => {
                if !e.record.status.is_terminal()
                    && let Some(a) = e.abort
                {
                    a.abort();
                }
                true
            }
            None => false,
        }
    }

    /// Wait until a job reaches a terminal state. `None` if the job is unknown or deleted.
    pub async fn wait(&self, id: Ulid) -> Option<JobRecord> {
        loop {
            let done = self.lock().jobs.get(&id).map(|e| e.done.clone())?;
            let notified = done.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let r = self.get(id)?;
            if r.status.is_terminal() {
                return Some(r);
            }
            // Wake on completion, or re-check periodically in case the job was deleted.
            let _ = tokio::time::timeout(std::time::Duration::from_millis(250), notified).await;
        }
    }

    /// Wait up to `budget` for a job to finish; returns its record either way.
    pub async fn wait_for(&self, id: Ulid, budget: std::time::Duration) -> Option<JobRecord> {
        match tokio::time::timeout(budget, self.wait(id)).await {
            Ok(r) => r,
            Err(_) => self.get(id),
        }
    }
}

#[derive(Debug, Deserialize, IntoParams)]
struct ListParams {
    /// Only jobs with this status.
    status_filter: Option<JobStatus>,
}

/// List jobs, newest first.
#[utoipa::path(get, path = "", tag = "Jobs", params(ListParams),
    responses((status = OK, body = Vec<JobRecord>)))]
async fn list_jobs(
    State(s): State<Scheduler>,
    Query(p): Query<ListParams>,
) -> axum::Json<Vec<JobRecord>> {
    axum::Json(s.list(p.status_filter))
}

/// Get a job.
#[utoipa::path(get, path = "/{job_id}", tag = "Jobs", params(("job_id" = String, Path)),
    responses((status = OK, body = JobRecord), (status = NOT_FOUND, body = Problem), (status = BAD_REQUEST, body = Problem)))]
async fn get_job(
    State(s): State<Scheduler>,
    Path(job_id): Path<String>,
) -> ApiResult<axum::Json<JobRecord>> {
    let id = Ulid::from_string(&job_id)
        .map_err(|_| Problem::bad_request(format!("Invalid job ID format: {job_id}")))?;
    s.get(id).map(axum::Json).ok_or_else(|| {
        Problem::not_found(format!("Job with id {job_id} not found"))
            .instance(format!("/api/v1/jobs/{job_id}"))
    })
}

/// Cancel (if running) and delete a job.
#[utoipa::path(delete, path = "/{job_id}", tag = "Jobs", params(("job_id" = String, Path)),
    responses((status = NO_CONTENT), (status = NOT_FOUND, body = Problem)))]
async fn delete_job(
    State(s): State<Scheduler>,
    Path(job_id): Path<String>,
) -> ApiResult<StatusCode> {
    let id = parse_ulid(&job_id)?;
    if s.delete(id) {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(Problem::not_found(format!(
            "Job with id {job_id} not found"
        )))
    }
}

/// JSON schema of the job list.
#[utoipa::path(get, path = "/$schema", tag = "Jobs", responses((status = OK, body = Object)))]
async fn job_schema() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "running", "completed", "failed", "canceling", "canceled"]},
                "submitted_at": {"type": ["string", "null"]},
                "started_at": {"type": ["string", "null"]},
                "finished_at": {"type": ["string", "null"]},
                "error": {"type": ["string", "null"]},
                "error_traceback": {"type": ["string", "null"]}
            },
            "required": ["id"]
        }
    }))
}

impl axum::response::IntoResponse for Json<serde_json::Value> {
    fn into_response(self) -> axum::response::Response {
        axum::Json(self.0).into_response()
    }
}

/// The `/api/v1/jobs` router (mount under that prefix).
pub fn router(scheduler: Scheduler) -> OpenApiRouter {
    OpenApiRouter::new()
        .routes(routes!(list_jobs))
        .routes(routes!(job_schema))
        .routes(routes!(get_job, delete_job))
        .with_state(scheduler)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn runs_jobs_and_records_failures() {
        let s = Scheduler::new(1);
        let ok = s.submit(Box::pin(async { Ok(()) }));
        let bad = s.submit(Box::pin(async {
            Err(JobError::value("Config x not found"))
        }));
        assert_eq!(s.wait(ok).await.unwrap().status, JobStatus::Completed);
        let r = s.wait(bad).await.unwrap();
        assert_eq!(r.status, JobStatus::Failed);
        assert_eq!(r.error.as_deref(), Some("ValueError: Config x not found"));
        assert_eq!(s.list(None).first().map(|r| r.id), Some(bad));
        assert_eq!(s.list(Some(JobStatus::Failed)).len(), 1);
    }

    #[tokio::test]
    async fn delete_cancels_queued_jobs() {
        let s = Scheduler::new(1);
        let slow = s.submit(Box::pin(async {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            Ok(())
        }));
        let queued = s.submit(Box::pin(async { Ok(()) }));
        assert!(s.delete(slow));
        assert!(s.get(slow).is_none());
        assert_eq!(s.wait(queued).await.unwrap().status, JobStatus::Completed);
        assert!(!s.delete(slow));
    }
}
