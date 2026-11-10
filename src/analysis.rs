//! The Rust-only analysis jobs: `$backtest` and `$hpo`, each stored as an artifact.

use std::time::Instant;

use chapkit_core::DataFrame;
use chapkit_eval::{Metric, Options, Prepare};
use chapkit_http::time::{entity_now, job_now};
use chapkit_http::{JobError, Problem, parse_ulid};
use chapkit_store::NewArtifact;
use serde_json::{Value, json};

use crate::erased::DEFAULT_PREDICTION_PERIODS;
use crate::evaluate::{backtest_config, check, evaluate};
use crate::state::{AppState, store_problem};
use crate::wire::{BacktestSettings, HpoRequest};

/// Artifact `data.type` of backtest artifacts.
pub const BACKTEST_TYPE: &str = "ml_backtest";
/// Content type of a backtest's or search's result.
pub const JSON: &str = "application/json";

/// What a backtest runs with, checked before the job is submitted.
pub(crate) struct Plan {
    pub config_id: Option<String>,
    pub config: Value,
    pub options: Options,
    pub weather: String,
}

/// Resolve the config (stored, given or default) and the settings, and check them.
pub(crate) async fn plan(
    state: &AppState,
    config_id: Option<&str>,
    config: Option<Value>,
    settings: &BacktestSettings,
) -> Result<Plan, Problem> {
    let (config_id, base) = match (config_id, config) {
        (Some(_), Some(_)) => {
            return Err(Problem::bad_request("give config_id or config, not both"));
        }
        (Some(id), None) => {
            let id = parse_ulid(id)?.to_string();
            let row = state
                .store()
                .get_config(&id)
                .await
                .map_err(store_problem)?
                .ok_or_else(|| Problem::not_found(format!("Config {id} not found")))?;
            (Some(id), Some(row.data))
        }
        (None, config) => (None, config),
    };
    let periods = settings.periods.unwrap_or_else(|| {
        base.as_ref()
            .and_then(|c| c.get("prediction_periods"))
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_PREDICTION_PERIODS) as usize
    });
    let defaults = Options::default();
    let options = Options {
        periods,
        splits: settings.splits.unwrap_or(defaults.splits),
        stride: settings.stride.unwrap_or(defaults.stride),
        retrain: settings.retrain.unwrap_or(defaults.retrain),
        prepare: Prepare {
            fill_gaps: settings.fill_gaps.unwrap_or(defaults.prepare.fill_gaps),
            pad: settings.pad.unwrap_or(defaults.prepare.pad),
        },
        drop_untrained: settings.drop_untrained.unwrap_or(defaults.drop_untrained),
    };
    let weather = settings
        .weather
        .clone()
        .unwrap_or_else(|| "climatology".into());
    let model = state.model().as_ref();
    check(model, &options, &weather).map_err(Problem::bad_request)?;
    let config = backtest_config(model, base, periods).map_err(Problem::bad_request)?;
    Ok(Plan {
        config_id,
        config,
        options,
        weather,
    })
}

pub(crate) fn store_error(e: &chapkit_store::StoreError) -> JobError {
    tracing::error!(error = %e, "store.error");
    JobError::runtime("A database error occurred")
}

/// Store a root artifact of `kind` with `content`.
pub(crate) async fn store_result(
    state: &AppState,
    artifact_id: String,
    kind: &str,
    config_id: Option<String>,
    started_at: String,
    start: Instant,
    content: Value,
) -> Result<(), JobError> {
    let data = json!({
        "type": kind,
        "metadata": {
            "status": "success",
            "config_id": config_id,
            "started_at": started_at,
            "completed_at": job_now(),
            "duration_seconds": start.elapsed().as_secs_f64(),
        },
        "content": content,
        "content_type": JSON,
        "content_size": null,
    });
    state
        .store()
        .insert_artifact(NewArtifact {
            id: artifact_id,
            parent_id: None,
            data,
            tags: vec![],
            content: None,
            created_at: entity_now(),
        })
        .await
        .map(|_| ())
        .map_err(|e| store_error(&e))
}

/// Run a backtest job: backtest, score, store the result (as `<bin> eval --json` prints it).
pub(crate) async fn backtest(
    state: AppState,
    artifact_id: String,
    plan: Plan,
    data: DataFrame,
) -> Result<(), JobError> {
    let started_at = job_now();
    let start = Instant::now();
    let model = state.model().clone();
    let id = state.inner.service_id.clone();
    let Plan {
        config_id,
        config,
        options,
        weather,
    } = plan;
    let report = chapkit_http::jobs::blocking(move || {
        evaluate(
            model.as_ref(),
            &id,
            &config,
            &data,
            "request".into(),
            &options,
            &weather,
        )
    })
    .await?
    .map_err(JobError::value)?;
    let content = serde_json::to_value(&report).map_err(|e| JobError::runtime(e.to_string()))?;
    store_result(
        &state,
        artifact_id,
        BACKTEST_TYPE,
        config_id,
        started_at,
        start,
        content,
    )
    .await
}

/// Artifact `data.type` of hyperparameter search artifacts.
pub const HPO_TYPE: &str = "ml_hpo";

/// Most trials one `$hpo` job may run.
pub const MAX_TRIALS: usize = 10_000;

/// A search's settings, checked before the job is submitted.
pub(crate) struct Search {
    pub plan: Plan,
    pub metric: Metric,
    pub options: chapkit_hpo::Options,
    /// The name of the config to create with the best options, if any.
    pub create: Option<String>,
}

/// Check a `$hpo` request.
pub(crate) async fn search_plan(
    state: &AppState,
    req: &mut HpoRequest,
    artifact_id: &str,
) -> Result<Search, Problem> {
    let plan = plan(
        state,
        req.config_id.as_deref(),
        req.config.take(),
        &req.settings,
    )
    .await?;
    if state.model().search_space().is_none() {
        return Err(Problem::bad_request(
            "the model has no search space (Model::search_space)",
        ));
    }
    let metric_id = req.metric.as_deref().unwrap_or("crps");
    let metric = Metric::from_id(metric_id)
        .ok_or_else(|| Problem::bad_request(format!("unknown metric {metric_id}")))?;
    chapkit_hpo::direction(metric).map_err(|e| Problem::bad_request(e.to_string()))?;
    let method = match req.method.as_deref() {
        None | Some("random") => chapkit_hpo::Method::Random,
        Some("grid") => chapkit_hpo::Method::Grid,
        Some(other) => {
            return Err(Problem::bad_request(format!(
                "method must be 'random' or 'grid', got '{other}'"
            )));
        }
    };
    let defaults = chapkit_hpo::Options::default();
    let trials = req.trials.unwrap_or(defaults.trials);
    if trials == 0 || trials > MAX_TRIALS {
        return Err(Problem::bad_request(format!(
            "trials must be between 1 and {MAX_TRIALS}"
        )));
    }
    let create = req.create_config.then(|| {
        req.config_name
            .clone()
            .unwrap_or_else(|| format!("hpo {artifact_id}"))
    });
    Ok(Search {
        plan,
        metric,
        options: chapkit_hpo::Options {
            method,
            trials,
            seed: req.seed.unwrap_or(defaults.seed),
            grid_points: req
                .grid_points
                .unwrap_or(defaults.grid_points)
                .clamp(2, 1000),
        },
        create,
    })
}

/// Run a search job: search, optionally store the best config, store the result.
pub(crate) async fn hpo(
    state: AppState,
    artifact_id: String,
    search: Search,
    data: DataFrame,
) -> Result<(), JobError> {
    let started_at = job_now();
    let start = Instant::now();
    let model = state.model().clone();
    let id = state.inner.service_id.clone();
    let Search {
        plan,
        metric,
        options,
        create,
    } = search;
    let config_id = plan.config_id.clone();
    let report = chapkit_http::jobs::blocking(move || {
        crate::evaluate::hpo(
            model.as_ref(),
            &id,
            &plan.config,
            &data,
            "request".into(),
            &plan.options,
            &plan.weather,
            metric,
            &options,
        )
    })
    .await?
    .map_err(JobError::value)?;
    let mut content =
        serde_json::to_value(&report).map_err(|e| JobError::runtime(e.to_string()))?;
    if let Some(name) = create {
        let now = entity_now();
        let best_id = ulid::Ulid::generate().to_string();
        state
            .store()
            .insert_config(chapkit_store::ConfigRow {
                id: best_id.clone(),
                name,
                data: report.best_config.clone(),
                tags: vec!["hpo".into()],
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .map_err(|e| store_error(&e))?;
        content["best_config_id"] = json!(best_id);
    }
    store_result(
        &state,
        artifact_id,
        HPO_TYPE,
        config_id,
        started_at,
        start,
        content,
    )
    .await
}
