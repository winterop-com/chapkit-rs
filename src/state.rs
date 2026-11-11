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
    pub sync_permits: Arc<tokio::sync::Semaphore>,
    /// Stored datasets, parsed (datasets never change once stored).
    pub frames: Mutex<FrameCache>,
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
                sync_permits: Arc::new(tokio::sync::Semaphore::new(sync_concurrency())),
                frames: Mutex::new(FrameCache::new(FRAME_CACHE_BYTES)),
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

    /// The parsed-dataset cache.
    pub(crate) fn frames(&self) -> std::sync::MutexGuard<'_, FrameCache> {
        self.inner
            .frames
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn cache(&self, artifact_id: String, trained: Trained) {
        self.inner
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .put(artifact_id, trained);
    }
}

/// The memory parsed datasets may take in [`FrameCache`]: 1 GiB. A dataset is parsed once, then
/// projected (`$data`), backtested or forecast from memory; 100,000 rows of a few columns take
/// about 10 MB, so many fit, while a dataset larger than the whole budget is parsed per request.
pub(crate) const FRAME_CACHE_BYTES: u64 = 1 << 30;

/// Parsed stored datasets by id, the least recently used dropped first, their estimated memory
/// (rows times column widths) at most a budget.
pub(crate) struct FrameCache {
    frames: LruCache<String, (Arc<chapkit_core::DataFrame>, u64)>,
    bytes: u64,
    max: u64,
}

/// A frame's estimated memory: per row, 16 bytes a number, 4 a text code, 2 a boolean, 32 a JSON
/// value.
fn frame_bytes(frame: &chapkit_core::DataFrame) -> u64 {
    use chapkit_core::Column;
    let row: u64 = frame
        .iter()
        .map(|(_, c)| match c {
            Column::Float(_) | Column::Int(_) => 16,
            Column::Str(_) => 4,
            Column::Bool(_) => 2,
            Column::Json(_) => 32,
        })
        .sum();
    u64::try_from(frame.n_rows())
        .unwrap_or(u64::MAX)
        .saturating_mul(row)
}

impl FrameCache {
    pub(crate) fn new(max: u64) -> Self {
        Self {
            frames: LruCache::unbounded(),
            bytes: 0,
            max,
        }
    }

    pub(crate) fn get(&mut self, id: &str) -> Option<Arc<chapkit_core::DataFrame>> {
        self.frames.get(id).map(|(f, _)| f.clone())
    }

    /// Keep `frame` as `id`'s, unless it alone is over the budget; drops the least recently used
    /// until the rest fit.
    pub(crate) fn put(&mut self, id: String, frame: Arc<chapkit_core::DataFrame>) {
        let size = frame_bytes(&frame);
        if size > self.max {
            return;
        }
        if let Some((_, old)) = self.frames.put(id, (frame, size)) {
            self.bytes = self.bytes.saturating_sub(old);
        }
        self.bytes = self.bytes.saturating_add(size);
        while self.bytes > self.max {
            match self.frames.pop_lru() {
                Some((_, (_, b))) => self.bytes = self.bytes.saturating_sub(b),
                None => break,
            }
        }
    }

    pub(crate) fn remove(&mut self, id: &str) {
        if let Some((_, b)) = self.frames.pop(id) {
            self.bytes = self.bytes.saturating_sub(b);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use chapkit_core::{Column, DataFrame};

    fn frame(rows: usize) -> Arc<DataFrame> {
        Arc::new(
            DataFrame::from_columns(vec![("x", Column::Float(vec![Some(1.0); rows]))]).unwrap(),
        )
    }

    /// A parsed dataset is kept and shared; the least recently used goes first; one over the
    /// whole budget is never kept.
    #[test]
    fn frame_cache_is_bounded() {
        let mut c = FrameCache::new(16 * 10);
        let a = frame(4);
        c.put("a".into(), a.clone());
        assert!(Arc::ptr_eq(&c.get("a").unwrap(), &a));
        c.put("b".into(), frame(4));
        assert!(c.get("a").is_some());
        c.put("c".into(), frame(4));
        assert!(c.get("b").is_none(), "b was the least recently used");
        assert!(c.get("a").is_some() && c.get("c").is_some());
        assert_eq!(c.bytes, 16 * 8);
        c.put("big".into(), frame(11));
        assert!(c.get("big").is_none());
        c.remove("a");
        assert!(c.get("a").is_none());
        assert_eq!(c.bytes, 16 * 4);
    }
}
