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
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
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

tokio::task_local! {
    /// The running job's concurrency permit, shared with its blocking calls.
    static JOB_PERMIT: Arc<OwnedSemaphorePermit>;
}

/// Run blocking work (a model call) on the blocking pool. Inside a job, the work holds the job's
/// concurrency permit until it returns: deleting a running job stops waiting for it, but the slot
/// stays taken until the model call actually finishes, so `max_concurrency` is never exceeded.
/// A panic in `f` becomes a `RuntimeError`.
pub async fn blocking<F, R>(f: F) -> Result<R, JobError>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let permit = JOB_PERMIT.try_with(Arc::clone).ok();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        f()
    })
    .await
    .map_err(|e| {
        if e.is_panic() {
            JobError::runtime(format!("panicked: {}", panic_message(&*e.into_panic())))
        } else {
            JobError::runtime(e.to_string())
        }
    })
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

struct Entry {
    record: JobRecord,
    abort: Option<AbortHandle>,
    done: Arc<Notify>,
    /// Submission order (ULIDs from the same millisecond are not ordered).
    seq: u64,
}

#[derive(Default)]
struct Inner {
    jobs: HashMap<Ulid, Entry>,
    next_seq: u64,
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
        {
            let mut inner = self.lock();
            inner.next_seq += 1;
            let seq = inner.next_seq;
            inner.jobs.insert(
                id,
                Entry {
                    record,
                    abort: None,
                    done: done.clone(),
                    seq,
                },
            );
        }

        let this = self.clone();
        let handle = tokio::spawn(async move {
            let Ok(permit) = this.permits.clone().acquire_owned().await else {
                return;
            };
            this.update(id, |r| {
                r.status = JobStatus::Running;
                r.started_at = Some(job_now());
            });
            // A panicking job fails instead of staying "running" forever.
            let result = JOB_PERMIT
                .scope(
                    Arc::new(permit),
                    std::panic::AssertUnwindSafe(work).catch_unwind(),
                )
                .await
                .unwrap_or_else(|p| {
                    Err(JobError::runtime(format!(
                        "panicked: {}",
                        panic_message(&*p)
                    )))
                });
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
            if let Some(r) = this.get(id) {
                let status = serde_json::to_value(r.status)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .unwrap_or_default();
                crate::metrics::inc(
                    "jobs_finished_total",
                    "Jobs that reached a terminal state.",
                    &[("status", &status)],
                );
            }
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
        let inner = self.lock();
        let mut v: Vec<(u64, JobRecord)> = inner
            .jobs
            .values()
            .filter(|e| status.is_none_or(|s| e.record.status == s))
            .map(|e| (e.seq, e.record.clone()))
            .collect();
        v.sort_by_key(|(seq, _)| std::cmp::Reverse(*seq));
        v.into_iter().map(|(_, r)| r).collect()
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

/// Stream a job's record as server-sent events until it finishes. Sends
/// `{"status": "deleted"}` and stops if the job is deleted.
#[utoipa::path(get, path = "/{job_id}/$stream", tag = "Jobs", params(("job_id" = String, Path), crate::sse::PollParams),
    responses((status = OK, content_type = "text/event-stream", body = String), (status = NOT_FOUND, body = Problem)))]
async fn stream_job(
    State(s): State<Scheduler>,
    Path(job_id): Path<String>,
    Query(p): Query<crate::sse::PollParams>,
) -> ApiResult<axum::response::Response> {
    let id = Ulid::from_string(&job_id)
        .map_err(|_| Problem::bad_request(format!("Invalid job ID format: {job_id}")))?;
    let interval = p.interval(0.5)?;
    if s.get(id).is_none() {
        return Err(Problem::not_found(format!(
            "Job with id {job_id} not found"
        )));
    }
    let stream = futures_util::stream::unfold(Some((s, id, true)), move |state| async move {
        let (s, id, first) = state?;
        if !first {
            tokio::time::sleep(interval).await;
        }
        match s.get(id) {
            None => Some((
                Ok(crate::sse::event(&serde_json::json!({"status": "deleted"}))),
                None,
            )),
            Some(r) => {
                let next = (!r.status.is_terminal()).then_some((s, id, false));
                Some((Ok(crate::sse::event(&r)), next))
            }
        }
    });
    Ok(crate::sse::response(stream))
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
        .routes(routes!(stream_job))
        .with_state(scheduler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[tokio::test]
    async fn a_panicking_job_fails_instead_of_hanging() {
        let s = Scheduler::new(1);
        let async_panic = s.submit(Box::pin(async { panic!("bad period") }));
        let r = s
            .wait_for(async_panic, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(r.status, JobStatus::Failed);
        assert!(r.error.unwrap().contains("bad period"));

        let blocking_panic = s.submit(Box::pin(async {
            blocking(|| -> () { panic!("in the model") }).await
        }));
        let r = s
            .wait_for(blocking_panic, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(r.status, JobStatus::Failed);
        assert!(r.error.unwrap().contains("in the model"));

        // The scheduler still runs jobs afterwards.
        let ok = s.submit(Box::pin(async { Ok(()) }));
        assert_eq!(s.wait(ok).await.unwrap().status, JobStatus::Completed);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deleting_a_running_job_keeps_its_slot_until_the_model_returns() {
        let s = Scheduler::new(1);
        let first_done = Arc::new(AtomicBool::new(false));
        let flag = first_done.clone();
        let first = s.submit(Box::pin(async move {
            blocking(move || {
                std::thread::sleep(Duration::from_millis(400));
                flag.store(true, Ordering::SeqCst);
            })
            .await
        }));
        // Wait until the first job is running inside its blocking call.
        for _ in 0..100 {
            if s.get(first).is_some_and(|r| r.status == JobStatus::Running) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(s.delete(first));

        let seen_first_done = Arc::new(AtomicBool::new(false));
        let seen = seen_first_done.clone();
        let flag = first_done.clone();
        let second = s.submit(Box::pin(async move {
            seen.store(flag.load(Ordering::SeqCst), Ordering::SeqCst);
            Ok(())
        }));
        let r = s.wait_for(second, Duration::from_secs(5)).await.unwrap();
        assert_eq!(r.status, JobStatus::Completed);
        assert!(
            seen_first_done.load(Ordering::SeqCst),
            "second job started while the deleted job's model call was still running"
        );
    }

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
    async fn runs_at_most_max_concurrency_jobs_at_once() {
        use std::sync::atomic::AtomicUsize;
        let s = Scheduler::new(2);
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let ids: Vec<Ulid> = (0..4)
            .map(|_| {
                let (running, peak) = (running.clone(), peak.clone());
                s.submit(Box::pin(async move {
                    // The model call runs on the blocking pool, as in the service.
                    blocking(move || {
                        let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(200));
                        running.fetch_sub(1, Ordering::SeqCst);
                    })
                    .await
                }))
            })
            .collect();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let count = |status| {
            ids.iter()
                .filter(|id| s.get(**id).map(|r| r.status) == Some(status))
                .count()
        };
        assert_eq!(count(JobStatus::Running), 2, "two jobs run");
        assert_eq!(count(JobStatus::Pending), 2, "the others wait");
        for id in ids {
            assert_eq!(s.wait(id).await.unwrap().status, JobStatus::Completed);
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
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
