//! `/api/v1/ml`: `$train`, `$predict`, `$validate`, plus the Rust-only fast path `$predict-sync`.

use axum::extract::State;
use axum::http::StatusCode;
use chapkit_core::{DataFrame, cols};
use chapkit_http::{ApiResult, Json, Problem, parse_ulid};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use crate::state::AppState;
use crate::wire::{
    BacktestRequest, Diagnostic, ExplainRequest, ExplainResponse, HpoRequest, JobSubmitted,
    PredictRequest, TrainRequest, ValidateRequest, ValidateResponse,
};
use crate::{analysis, ml};

fn ulid_field(field: &str, raw: &str) -> ApiResult<String> {
    ulid::Ulid::from_string(raw).map(|u| u.to_string()).map_err(|_| {
        Problem::validation(vec![serde_json::json!({"loc": ["body", field], "msg": format!("Invalid ULID: {raw}"), "type": "value_error"})])
    })
}

/// Submit, then wait up to the eager budget so fast jobs are already finished
/// when the client first polls (chap-core polls before it sleeps).
async fn submit_eager(s: &AppState, work: chapkit_http::jobs::JobFuture) -> ulid::Ulid {
    let id = s.inner.jobs.submit(work);
    if !s.inner.eager_wait.is_zero() {
        let _ = s.inner.jobs.wait_for(id, s.inner.eager_wait).await;
    }
    id
}

/// Train a model. Returns 202 with the job id and the artifact id the model will be stored under.
#[utoipa::path(post, path = "/$train", tag = "ML", request_body = TrainRequest,
    responses((status = ACCEPTED, body = JobSubmitted), (status = UNPROCESSABLE_ENTITY, body = Problem)))]
async fn train(
    State(s): State<AppState>,
    Json(req): Json<TrainRequest>,
) -> ApiResult<(StatusCode, axum::Json<JobSubmitted>)> {
    let config_id = ulid_field("config_id", &req.config_id)?;
    chapkit_http::metrics::inc("ml_train_jobs_total", "Training jobs submitted.", &[]);
    let artifact_id = ulid::Ulid::generate().to_string();
    let work = Box::pin(ml::train(
        s.clone(),
        artifact_id.clone(),
        config_id,
        req.data,
        req.geo,
        req.run_info,
    ));
    let job_id = submit_eager(&s, work).await;
    Ok((
        StatusCode::ACCEPTED,
        axum::Json(JobSubmitted {
            job_id: job_id.to_string(),
            artifact_id,
            message: format!("Training job submitted. Job ID: {job_id}"),
        }),
    ))
}

/// Predict with a trained model. Returns 202; the prediction artifact is a child of the training artifact.
#[utoipa::path(post, path = "/$predict", tag = "ML", request_body = PredictRequest,
    responses((status = ACCEPTED, body = JobSubmitted), (status = UNPROCESSABLE_ENTITY, body = Problem)))]
async fn predict(
    State(s): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> ApiResult<(StatusCode, axum::Json<JobSubmitted>)> {
    let artifact_id = ulid_field("artifact_id", &req.artifact_id)?;
    chapkit_http::metrics::inc("ml_predict_jobs_total", "Prediction jobs submitted.", &[]);
    let prediction_id = ulid::Ulid::generate().to_string();
    let work = Box::pin(ml::predict(
        s.clone(),
        prediction_id.clone(),
        artifact_id,
        req.historic,
        req.future,
        req.geo,
        req.run_info,
    ));
    let job_id = submit_eager(&s, work).await;
    Ok((
        StatusCode::ACCEPTED,
        axum::Json(JobSubmitted {
            job_id: job_id.to_string(),
            artifact_id: prediction_id,
            message: format!("Prediction job submitted. Job ID: {job_id}"),
        }),
    ))
}

/// Predict and return the forecast directly: no job, no stored artifact. The trained
/// model is kept warm in memory, so repeated calls cost only the model itself.
/// chapkit-rs extension; not part of Python chapkit.
#[utoipa::path(post, path = "/$predict-sync", tag = "ML", request_body = PredictRequest,
    responses((status = OK, body = Object), (status = BAD_REQUEST, body = Problem), (status = NOT_FOUND, body = Problem)))]
async fn predict_sync(
    State(s): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> ApiResult<axum::Json<DataFrame>> {
    let artifact_id = ulid_field("artifact_id", &req.artifact_id)?;
    let _permit = s
        .inner
        .sync_permits
        .acquire()
        .await
        .map_err(|_| Problem::internal())?;
    let (frame, _) = ml::predict_frame(
        &s,
        &artifact_id,
        req.historic,
        req.future,
        req.geo,
        req.run_info.as_ref(),
    )
    .await
    .map_err(|e| {
        if e.message.contains("not found") {
            Problem::not_found(e.message)
        } else {
            Problem::bad_request(format!("{}: {}", e.kind, e.message))
        }
    })?;
    Ok(axum::Json(frame))
}

/// Explain one forecast cell with LIME or KernelSHAP, computed in-process: every
/// perturbation is a direct model call, run in parallel across cores when that is faster.
/// Features are the cell's covariates, their earlier values and the target's history.
/// chapkit-rs extension; not part of Python chapkit.
#[utoipa::path(post, path = "/$explain", tag = "ML", request_body = ExplainRequest,
    responses((status = OK, body = ExplainResponse), (status = BAD_REQUEST, body = Problem), (status = NOT_FOUND, body = Problem)))]
async fn explain(
    State(s): State<AppState>,
    Json(req): Json<ExplainRequest>,
) -> ApiResult<axum::Json<ExplainResponse>> {
    let artifact_id = ulid_field("artifact_id", &req.artifact_id)?;
    let _permit = s
        .inner
        .sync_permits
        .acquire()
        .await
        .map_err(|_| Problem::internal())?;
    let job_problem = |e: chapkit_http::JobError| {
        if e.message.contains("not found") {
            Problem::not_found(e.message)
        } else {
            Problem::bad_request(format!("{}: {}", e.kind, e.message))
        }
    };
    let (trained, config, _) = ml::load_trained(&s, &artifact_id)
        .await
        .map_err(job_problem)?;
    let run = ml::resolve_run(&s, req.run_info.as_ref(), &config, Some(&req.future))
        .map_err(job_problem)?;
    let model = s.model().clone();
    let target_col = model.info().target;
    let opts = chapkit_xai::Options {
        method: req.method,
        n_samples: req.n_samples.unwrap_or(500).clamp(1, 100_000),
        seed: req.seed.unwrap_or(0),
        kernel_width: req.kernel_width,
        parallel: chapkit_xai::Parallel::Auto,
    };
    let spec = chapkit_xai::FeatureSpec {
        covariates: req.features,
        scope: req.scope.unwrap_or_default(),
        lags: req.lags.unwrap_or(chapkit_xai::DEFAULT_LAGS).min(52),
    };
    let target = chapkit_xai::ForecastTarget {
        location: req.location,
        time_period: req.time_period,
    };
    let start = std::time::Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        let predict = |historic: &DataFrame, future: &DataFrame| {
            model
                .predict(
                    &trained,
                    &config,
                    chapkit_core::PredictInput::new(historic, future, &run, req.geo.as_ref()),
                )
                .map_err(|e| e.to_string())
        };
        chapkit_xai::explain_forecast(
            &predict,
            &req.historic,
            &req.future,
            &target,
            &spec,
            &target_col,
            &opts,
        )
    })
    .await
    .map_err(|_| Problem::internal())?;
    let explanation = result.map_err(|e| match e {
        chapkit_xai::Error::InvalidInput(m) => Problem::bad_request(m),
        other => Problem::bad_request(other.to_string()),
    })?;
    Ok(axum::Json(ExplainResponse {
        explanation,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

/// Backtest the model on a dataset in-process, scored as chap-core scores it. Returns 202; the
/// result (as `<bin> eval --json` prints it) is stored as an `ml_backtest` artifact.
/// chapkit-rs extension; not part of Python chapkit.
#[utoipa::path(post, path = "/$backtest", tag = "ML", request_body = BacktestRequest,
    responses((status = ACCEPTED, body = JobSubmitted), (status = BAD_REQUEST, body = Problem), (status = NOT_FOUND, body = Problem)))]
async fn backtest(
    State(s): State<AppState>,
    Json(req): Json<BacktestRequest>,
) -> ApiResult<(StatusCode, axum::Json<JobSubmitted>)> {
    let plan = analysis::plan(&s, req.config_id.as_deref(), req.config, &req.settings).await?;
    chapkit_http::metrics::inc("ml_backtest_jobs_total", "Backtest jobs submitted.", &[]);
    let artifact_id = ulid::Ulid::generate().to_string();
    let work = Box::pin(analysis::backtest(
        s.clone(),
        artifact_id.clone(),
        plan,
        req.data,
    ));
    let job_id = submit_eager(&s, work).await;
    Ok((
        StatusCode::ACCEPTED,
        axum::Json(JobSubmitted {
            job_id: job_id.to_string(),
            artifact_id,
            message: format!("Backtest job submitted. Job ID: {job_id}"),
        }),
    ))
}

/// Search the model's hyperparameters (random or grid over its `hpo_search_space`), each trial
/// an in-process backtest. Returns 202; the trials, the best config and its backtest are stored
/// as an `ml_hpo` artifact, and with `create_config` the best config is created.
/// chapkit-rs extension; not part of Python chapkit.
#[utoipa::path(post, path = "/$hpo", tag = "ML", request_body = HpoRequest,
    responses((status = ACCEPTED, body = JobSubmitted), (status = BAD_REQUEST, body = Problem), (status = NOT_FOUND, body = Problem)))]
async fn hpo(
    State(s): State<AppState>,
    Json(mut req): Json<HpoRequest>,
) -> ApiResult<(StatusCode, axum::Json<JobSubmitted>)> {
    let artifact_id = ulid::Ulid::generate().to_string();
    let search = analysis::search_plan(&s, &mut req, &artifact_id).await?;
    chapkit_http::metrics::inc(
        "ml_hpo_jobs_total",
        "Hyperparameter search jobs submitted.",
        &[],
    );
    let work = Box::pin(analysis::hpo(
        s.clone(),
        artifact_id.clone(),
        search,
        req.data,
    ));
    let job_id = submit_eager(&s, work).await;
    Ok((
        StatusCode::ACCEPTED,
        axum::Json(JobSubmitted {
            job_id: job_id.to_string(),
            artifact_id,
            message: format!("Hyperparameter search job submitted. Job ID: {job_id}"),
        }),
    ))
}

#[derive(Debug, serde::Deserialize, utoipa::IntoParams)]
struct SampleParams {
    /// `train` (default) or `predict`.
    kind: Option<String>,
    /// Include this config's `additional_continuous_covariates`.
    config_id: Option<String>,
    num_locations: Option<usize>,
    num_periods: Option<usize>,
    num_features: Option<usize>,
    /// `monthly` or `weekly` (default: the model's period type, else monthly).
    period_type: Option<String>,
    /// `polygon` (default) or `point`.
    geo_type: Option<String>,
    /// Default: whether the model requires geo.
    include_geo: Option<bool>,
    /// Negative (default) for fresh data each call.
    seed: Option<i64>,
}

/// A ready-to-submit sample `$train` or `$predict` payload from the synthetic data
/// generator (same columns and options as Python chapkit's).
#[utoipa::path(get, path = "/$generate-sample-data", tag = "ML", params(SampleParams),
    responses((status = OK, body = Object), (status = BAD_REQUEST, body = Problem), (status = NOT_FOUND, body = Problem)))]
async fn generate_sample_data(
    State(s): State<AppState>,
    chapkit_http::Query(p): chapkit_http::Query<SampleParams>,
) -> ApiResult<axum::Json<serde_json::Value>> {
    use chapkit_core::synth::{Generator, Resolution, Spec};
    let info = s.model().info();
    let mut config_covariates = Vec::new();
    if let Some(cid) = &p.config_id {
        parse_ulid(cid).map_err(|_| Problem::bad_request(format!("Invalid config_id: '{cid}'")))?;
        let cfg = s
            .store()
            .get_config(cid)
            .await
            .map_err(crate::state::store_problem)?
            .ok_or_else(|| Problem::not_found(format!("Config {cid} not found")))?;
        config_covariates = cfg
            .data
            .get("additional_continuous_covariates")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
    }
    let resolution = match p.period_type.as_deref() {
        Some("weekly") => Resolution::Weekly,
        Some("monthly") => Resolution::Monthly,
        Some(other) => {
            return Err(Problem::bad_request(format!(
                "period_type must be 'monthly' or 'weekly', got '{other}'"
            )));
        }
        None if info.period_type == chapkit_core::PeriodType::Weekly => Resolution::Weekly,
        None => Resolution::Monthly,
    };
    let num_periods = p.num_periods.unwrap_or(50).clamp(1, 10_000);
    let mut spec = Spec {
        num_locations: p.num_locations.unwrap_or(5).clamp(1, 10_000),
        num_periods,
        num_features: p.num_features.unwrap_or(3).min(100),
        required_covariates: info.required_covariates.clone(),
        additional_covariates: config_covariates,
        resolution,
        start_year: 2020,
    };
    let seed = p.seed.filter(|s| *s >= 0).map(i64::cast_unsigned);
    let mut generator = Generator::new(seed);
    let want_geo = p.include_geo.unwrap_or(info.requires_geo);
    let geo = want_geo
        .then(|| Generator::geo(spec.num_locations, p.geo_type.as_deref() == Some("point")));
    let gen_err = |e: chapkit_core::Error| Problem::bad_request(e.to_string());
    let mut payload = match p.kind.as_deref().unwrap_or("train") {
        "predict" => {
            let lower = info.min_prediction_periods.max(1) as usize;
            let upper = (info.max_prediction_periods as usize).max(lower);
            spec.num_periods = num_periods.clamp(lower, upper);
            let (historic, future) = generator
                .prediction_data(&spec, num_periods)
                .map_err(gen_err)?;
            serde_json::json!({"historic": historic, "future": future})
        }
        "train" => {
            let data = generator.training_data(&spec).map_err(gen_err)?;
            let mut v = serde_json::json!({"data": data});
            if let Some(cid) = &p.config_id {
                v["config_id"] = serde_json::json!(cid);
            }
            v
        }
        other => {
            return Err(Problem::bad_request(format!(
                "kind must be 'train' or 'predict', got '{other}'"
            )));
        }
    };
    if let Some(g) = geo {
        payload["geo"] = g;
    }
    Ok(axum::Json(payload))
}

/// Run the model's `validate_train` / `validate_predict` on a blocking thread: they are model code,
/// possibly slow, and a panic in them must not take the request down.
async fn model_checks(
    model: std::sync::Arc<dyn crate::erased::DynModel>,
    req: ValidateRequest,
    config: serde_json::Value,
    run: chapkit_core::RunInfo,
) -> Vec<Diagnostic> {
    let found = tokio::task::spawn_blocking(move || match req.kind.as_str() {
        "train" => req.data.as_ref().map(|data| {
            model.validate_train(
                &config,
                chapkit_core::TrainInput::new(data, &run, req.geo.as_ref()),
            )
        }),
        _ => match (&req.historic, &req.future) {
            (Some(h), Some(f)) => Some(model.validate_predict(
                &config,
                chapkit_core::PredictInput::new(h, f, &run, req.geo.as_ref()),
            )),
            _ => None,
        },
    })
    .await;
    match found {
        Ok(None) => Vec::new(),
        Ok(Some(Ok(found))) => found
            .into_iter()
            .map(|x| Diagnostic {
                severity: x.severity.as_str().into(),
                code: x.code,
                message: x.message,
                field: x.field,
            })
            .collect(),
        Ok(Some(Err(e))) => vec![diag("error", "model_check_failed", e.to_string(), None)],
        Err(_) => vec![diag(
            "error",
            "model_check_failed",
            "the model's validation check panicked",
            None,
        )],
    }
}

/// `frame` is present, has rows, and has `time_period` and `location` columns.
fn check_frame(d: &mut Vec<Diagnostic>, frame: Option<&DataFrame>, code: &str, field: &str) {
    match frame {
        None => d.push(diag(
            "error",
            code,
            format!("{field} is required"),
            Some(field),
        )),
        Some(f) if f.is_empty() => d.push(diag(
            "error",
            code,
            format!("{field} has no rows"),
            Some(field),
        )),
        Some(f) => {
            for c in [cols::TIME_PERIOD, cols::LOCATION] {
                if !f.has(c) {
                    d.push(diag(
                        "error",
                        "missing_column",
                        format!("{field} has no '{c}' column"),
                        Some(field),
                    ));
                }
            }
        }
    }
}

fn diag(severity: &str, code: &str, message: impl Into<String>, field: Option<&str>) -> Diagnostic {
    Diagnostic {
        severity: severity.into(),
        code: code.into(),
        message: message.into(),
        field: field.map(Into::into),
    }
}

/// Check a train or predict request without running it.
#[utoipa::path(post, path = "/$validate", tag = "ML", request_body = ValidateRequest,
    responses((status = OK, body = ValidateResponse)))]
async fn validate(
    State(s): State<AppState>,
    Json(req): Json<ValidateRequest>,
) -> ApiResult<axum::Json<ValidateResponse>> {
    let mut d = Vec::new();
    let config = match req.kind.as_str() {
        "train" => {
            check_frame(&mut d, req.data.as_ref(), "data_empty", "data");
            match req.config_id.as_deref().map(|c| parse_ulid(c).map(|_| c)) {
                Some(Ok(cid)) => s
                    .store()
                    .get_config(cid)
                    .await
                    .map_err(crate::state::store_problem)?
                    .map(|c| c.data),
                _ => None,
            }
        }
        "predict" => {
            check_frame(&mut d, req.historic.as_ref(), "historic_empty", "historic");
            check_frame(&mut d, req.future.as_ref(), "future_empty", "future");
            let found = match req.artifact_id.as_deref() {
                Some(aid) => ml::load_trained(&s, aid)
                    .await
                    .map(|(_, config, _)| config)
                    .map_err(|e| ("training_artifact_not_found", e.message)),
                None => Err((
                    "artifact_id_missing",
                    "artifact_id is required for a prediction".to_owned(),
                )),
            };
            found
                .map_err(|(code, msg)| d.push(diag("error", code, msg, Some("artifact_id"))))
                .ok()
        }
        other => {
            d.push(diag(
                "error",
                "invalid_type",
                format!("type must be 'train' or 'predict', got '{other}'"),
                Some("type"),
            ));
            None
        }
    };
    let mut run = None;
    match (&config, req.kind.as_str()) {
        (None, "train") => d.push(diag(
            "error",
            "config_not_found",
            "Config not found",
            Some("config_id"),
        )),
        (Some(cfg), _) => {
            match ml::resolve_run(&s, req.run_info.as_ref(), cfg, req.future.as_ref()) {
                Ok(r) => run = Some(r),
                Err(e) => d.push(diag(
                    "error",
                    "prediction_periods_out_of_bounds",
                    e.message,
                    Some("run_info.prediction_periods"),
                )),
            }
        }
        _ => {}
    }
    // The model's own checks, only when chapkit's passed (as in Python chapkit).
    if !d.iter().any(|x| x.severity == "error")
        && let (Some(cfg), Some(run)) = (config, run)
    {
        d.extend(model_checks(s.model().clone(), req, cfg, run).await);
    }
    let valid = !d.iter().any(|x| x.severity == "error");
    Ok(axum::Json(ValidateResponse {
        valid,
        diagnostics: d,
    }))
}

/// The router (mount at `/api/v1/ml`).
pub fn router(state: AppState) -> OpenApiRouter {
    OpenApiRouter::new()
        .routes(routes!(train))
        .routes(routes!(predict))
        .routes(routes!(predict_sync))
        .routes(routes!(explain))
        .routes(routes!(backtest))
        .routes(routes!(hpo))
        .routes(routes!(generate_sample_data))
        .routes(routes!(validate))
        .with_state(state)
}
