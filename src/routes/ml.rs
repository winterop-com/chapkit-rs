//! `/api/v1/ml`: `$train`, `$predict`, `$validate`, plus the Rust-only fast path `$predict-sync`.

use axum::extract::State;
use axum::http::StatusCode;
use chapkit_core::{DataFrame, cols};
use chapkit_http::{ApiResult, Json, Problem, parse_ulid};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use crate::ml;
use crate::state::AppState;
use crate::wire::{
    Diagnostic, JobSubmitted, PredictRequest, TrainRequest, ValidateRequest, ValidateResponse,
};

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
    let check_frame =
        |d: &mut Vec<Diagnostic>, frame: Option<&DataFrame>, code: &str, field: &str| match frame {
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
        };
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
            match req.artifact_id.as_deref() {
                Some(aid) => match ml::load_trained(&s, aid).await {
                    Ok((_, config, _)) => Some(config),
                    Err(e) => {
                        d.push(diag(
                            "error",
                            "training_artifact_not_found",
                            e.message,
                            Some("artifact_id"),
                        ));
                        None
                    }
                },
                None => None,
            }
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
    match (&config, req.kind.as_str()) {
        (None, "train") => d.push(diag(
            "error",
            "config_not_found",
            "Config not found",
            Some("config_id"),
        )),
        (Some(cfg), _) => {
            if let Err(e) = ml::resolve_run(&s, req.run_info.as_ref(), cfg, req.future.as_ref()) {
                d.push(diag(
                    "error",
                    "prediction_periods_out_of_bounds",
                    e.message,
                    Some("run_info.prediction_periods"),
                ));
            }
        }
        _ => {}
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
        .routes(routes!(validate))
        .with_state(state)
}
