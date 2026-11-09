//! Shared service state.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chapkit_http::{Problem, Scheduler};
use chapkit_store::{Store, StoreError};
use lru::LruCache;
use serde_json::Value;

use crate::erased::{DynModel, Trained};

/// State shared by every handler. Cheap to clone.
#[derive(Clone)]
pub struct AppState {
    pub(crate) inner: Arc<Inner>,
}

pub(crate) struct Inner {
    pub model: Arc<dyn DynModel>,
    pub store: Arc<dyn Store>,
    pub jobs: Scheduler,
    /// Warm trained models by training-artifact id.
    pub cache: Mutex<LruCache<String, Trained>>,
    pub service_id: String,
    pub version: String,
    pub eager_wait: Duration,
    /// Bounds concurrent synchronous model calls (`$predict-sync`) to the CPU budget.
    pub sync_permits: tokio::sync::Semaphore,
}

impl AppState {
    pub(crate) fn new(
        model: Arc<dyn DynModel>,
        store: Arc<dyn Store>,
        jobs: Scheduler,
        service_id: String,
        version: String,
        eager_wait: Duration,
        cache_size: usize,
    ) -> Self {
        let cap = NonZeroUsize::new(cache_size.max(1)).unwrap_or(NonZeroUsize::MIN);
        Self {
            inner: Arc::new(Inner {
                model,
                store,
                jobs,
                cache: Mutex::new(LruCache::new(cap)),
                service_id,
                version,
                eager_wait,
                sync_permits: tokio::sync::Semaphore::new(sync_concurrency()),
            }),
        }
    }

    pub(crate) fn model(&self) -> &Arc<dyn DynModel> {
        &self.inner.model
    }

    pub(crate) fn store(&self) -> &Arc<dyn Store> {
        &self.inner.store
    }

    pub(crate) fn cached(&self, artifact_id: &str) -> Option<Trained> {
        self.inner
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(artifact_id)
            .cloned()
    }

    pub(crate) fn cache(&self, artifact_id: String, trained: Trained) {
        self.inner
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .put(artifact_id, trained);
    }
}

/// Concurrent `$predict-sync` calls: `CHAPKIT_SYNC_CONCURRENCY`, else the number of cores.
fn sync_concurrency() -> usize {
    std::env::var("CHAPKIT_SYNC_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
        })
        .max(1)
}

/// Map store errors onto problem responses. Database details are logged, not returned.
pub(crate) fn store_problem(e: StoreError) -> Problem {
    match e {
        StoreError::Conflict(m) => Problem::conflict(m),
        StoreError::NotFound(m) => Problem::not_found(m),
        StoreError::Invalid(m) => Problem::bad_request(m),
        StoreError::Database(m) => {
            tracing::error!(error = %m, "store.error");
            Problem::database()
        }
    }
}

/// Validate tags as servicekit does.
pub(crate) fn validate_tags(tags: &[String]) -> Result<(), Problem> {
    let mut errors = Vec::new();
    if tags.len() > 50 {
        errors.push("At most 50 tags are allowed".to_owned());
    }
    for t in tags {
        if t.is_empty() {
            errors.push("Tags must not be empty".to_owned());
        } else if t.len() > 100 {
            errors.push(format!("Tag '{t}' is longer than 100 characters"));
        } else if !t
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            errors.push(format!(
                "Tag '{t}' may only contain letters, digits, '_' and '-'"
            ));
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    let dups: std::collections::BTreeSet<&String> =
        tags.iter().filter(|t| !seen.insert(*t)).collect();
    if !dups.is_empty() {
        errors.push(format!(
            "Duplicate tags: {:?}",
            dups.into_iter().collect::<Vec<_>>()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(Problem::validation(vec![
            serde_json::json!({"loc": ["body", "tags"], "msg": errors.join("; "), "type": "value_error"}),
        ]))
    }
}

/// `Value` helper: a JSON object's string field.
pub(crate) fn str_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}
