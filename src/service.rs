//! [`Service`]: turn a [`Model`] into a chapkit-compatible HTTP service.

use std::sync::Arc;
use std::time::Duration;

use chapkit_core::Model;
use chapkit_http::health::CheckResult;
use chapkit_http::{App, HealthState, Scheduler};
use chapkit_store::{Store, StoreError};
use serde_json::{Value, json};

use crate::erased::{DynModel, Erased};
use crate::routes;
use crate::state::AppState;

/// Builder for an ML service. Mirrors Python chapkit's `MLServiceBuilder`: one model,
/// configs, artifacts, jobs, health, system, info, and the ML endpoints.
pub struct Service {
    id: String,
    display_name: String,
    version: String,
    description: Option<String>,
    model: Arc<dyn DynModel>,
    database_url: Option<String>,
    max_concurrency: usize,
    eager_wait: Duration,
    cache_size: usize,
}

impl Service {
    /// A service for `model`. `id` must be a slug (`^[a-z][a-z0-9]*(-[a-z0-9]+)*$`).
    ///
    /// Defaults read from the environment: `DATABASE_URL` (in-memory when unset),
    /// `CHAPKIT_EAGER_WAIT_MS` (1000), `CHAPKIT_MAX_CONCURRENCY` (1).
    pub fn new<M: Model>(id: impl Into<String>, display_name: impl Into<String>, model: M) -> Self {
        let env_u64 = |k: &str, d: u64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        Self {
            id: id.into(),
            display_name: display_name.into(),
            version: "1.0.0".into(),
            description: None,
            model: Arc::new(Erased(model)),
            database_url: std::env::var("DATABASE_URL").ok(),
            max_concurrency: usize::try_from(env_u64("CHAPKIT_MAX_CONCURRENCY", 1)).unwrap_or(1),
            eager_wait: Duration::from_millis(env_u64("CHAPKIT_EAGER_WAIT_MS", 1000)),
            cache_size: 16,
        }
    }

    /// Service version (default `1.0.0`).
    #[must_use]
    pub fn version(mut self, v: impl Into<String>) -> Self {
        self.version = v.into();
        self
    }

    /// Description.
    #[must_use]
    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }

    /// Database URL (`:memory:`, a path, `sqlite:///path`). Overrides `DATABASE_URL`.
    #[must_use]
    pub fn database(mut self, url: impl Into<String>) -> Self {
        self.database_url = Some(url.into());
        self
    }

    /// Jobs allowed to run at once.
    #[must_use]
    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    /// How long `$train`/`$predict` wait for a job before answering 202.
    /// Jobs finished within this budget are already complete at the client's first poll.
    #[must_use]
    pub fn eager_wait(mut self, d: Duration) -> Self {
        self.eager_wait = d;
        self
    }

    /// Number of trained models kept warm in memory.
    #[must_use]
    pub fn cache_size(mut self, n: usize) -> Self {
        self.cache_size = n.max(1);
        self
    }

    /// The `/api/v1/info` document (chapkit's `MLServiceInfo`).
    pub fn info(&self) -> Value {
        let m = self.model.info();
        json!({
            "id": self.id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "model_metadata": m.model_metadata,
            "period_type": m.period_type,
            "min_prediction_periods": m.min_prediction_periods,
            "max_prediction_periods": m.max_prediction_periods,
            "allow_free_additional_continuous_covariates": m.allow_free_additional_continuous_covariates,
            "required_covariates": m.required_covariates,
            "requires_geo": m.requires_geo,
            "target": m.target,
            "hpo_search_space": self.model.search_space(),
            "git_revision": std::env::var("GIT_REVISION").ok().filter(|s| !s.is_empty()),
            "chapkit_version": format!("chapkit-rs {}", env!("CARGO_PKG_VERSION")),
            "servicekit_version": null,
            "extensions": ["$predict-sync", "$explain"],
        })
    }

    /// Build the app (routers, state, store). Use [`App::serve`] to run it, or
    /// [`App::build`] for an in-process router (tests, benchmarks).
    pub fn app(self) -> Result<App, StoreError> {
        let store: Arc<dyn Store> = Arc::new(chapkit_store::open(self.database_url.as_deref())?);
        let info = self.info();
        let jobs = Scheduler::new(self.max_concurrency);
        let state = AppState::new(
            self.model,
            store.clone(),
            jobs.clone(),
            self.id.clone(),
            self.version.clone(),
            self.eager_wait,
            self.cache_size,
        );
        let check: chapkit_http::health::Check = Arc::new(move || {
            let store = store.clone();
            Box::pin(async move {
                match store.ping().await {
                    Ok(()) => CheckResult {
                        state: HealthState::Healthy,
                        message: None,
                    },
                    Err(e) => CheckResult {
                        state: HealthState::Unhealthy,
                        message: Some(format!("Check failed: {e}")),
                    },
                }
            })
        });
        Ok(App::new(self.id, self.display_name, self.version, info)
            .health_check("database", check)
            .nest("/api/v1/configs", routes::configs::router(state.clone()))
            .nest(
                "/api/v1/artifacts",
                routes::artifacts::router(state.clone()),
            )
            .nest("/api/v1/ml", routes::ml::router(state))
            .nest("/api/v1/jobs", chapkit_http::jobs::router(jobs)))
    }

    /// Build and serve on `HOST:PORT` until shutdown.
    pub async fn serve(self) -> std::io::Result<()> {
        self.app().map_err(std::io::Error::other)?.serve().await
    }
}
