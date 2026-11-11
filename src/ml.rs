//! Train and predict logic, shared by the job endpoints and the synchronous fast paths.

use std::collections::BTreeSet;
use std::time::Instant;

use chapkit_core::{
    Checkpoint, DataFrame, Geo, Period, PredictInput, Provenance, RunInfo, TrainInput, cols,
};
use chapkit_http::JobError;
use chapkit_http::time::{entity_now, job_now};
use chapkit_store::{ArtifactRow, NewArtifact};
use serde_json::{Value, json};

use crate::erased::{DEFAULT_PREDICTION_PERIODS, Trained};
use crate::state::{AppState, str_field};
use crate::wire::RunInfoIn;

/// Artifact `data.type` of training artifacts.
pub const TRAINING_TYPE: &str = "ml_training";
/// Artifact `data.type` of prediction artifacts.
pub const PREDICTION_TYPE: &str = "ml_prediction";
/// Content type of training checkpoints.
pub const SAFETENSORS: &str = "application/vnd.safetensors";
/// Artifact `data.type` of `$predict-splits` artifacts: `content` is the list of prediction frames.
pub const PREDICTION_SPLITS_TYPE: &str = "ml_prediction_splits";
/// Content type of prediction frames (chapkit's).
pub const DATAFRAME_JSON: &str = "application/vnd.chapkit.dataframe+json";

/// Number of distinct periods per location in `future` (the largest), if any.
fn future_horizon(future: &DataFrame) -> Option<u32> {
    let tp = future.column(cols::TIME_PERIOD)?;
    let groups = future.group_indices(cols::LOCATION).ok()?;
    groups
        .iter()
        .map(|(_, idx)| {
            idx.iter()
                .filter_map(|&i| tp.str_at(i))
                .collect::<BTreeSet<_>>()
                .len() as u32
        })
        .max()
}

/// Resolve the run settings: horizon from the request, else (predict) the future frame,
/// else the config; covariates from the request, else the config. Checks the model's bounds.
pub(crate) fn resolve_run(
    state: &AppState,
    run_info: Option<&RunInfoIn>,
    config: &Value,
    future: Option<&DataFrame>,
) -> Result<RunInfo, JobError> {
    let from_config = config
        .get("prediction_periods")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_PREDICTION_PERIODS);
    let periods = run_info
        .and_then(|r| r.prediction_periods)
        .or_else(|| future.and_then(future_horizon))
        .unwrap_or(u32::try_from(from_config).unwrap_or(u32::MAX));
    let info = state.model().info();
    if periods < info.min_prediction_periods || periods > info.max_prediction_periods {
        return Err(JobError::value(format!(
            "prediction_periods {periods} is outside the model's bounds [{}, {}]",
            info.min_prediction_periods, info.max_prediction_periods
        )));
    }
    let covariates = run_info
        .and_then(|r| r.additional_continuous_covariates.clone())
        .unwrap_or_else(|| {
            config
                .get("additional_continuous_covariates")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default()
        });
    Ok(RunInfo {
        prediction_periods: periods,
        additional_continuous_covariates: covariates,
        future_covariate_origin: run_info.and_then(|r| r.future_covariate_origin.clone()),
    })
}

fn model_error(e: &chapkit_core::Error) -> JobError {
    match e {
        chapkit_core::Error::InvalidInput(_)
        | chapkit_core::Error::Config(_)
        | chapkit_core::Error::Frame(_) => JobError::value(e.to_string()),
        _ => JobError::runtime(e.to_string()),
    }
}

fn store_error(e: &chapkit_store::StoreError) -> JobError {
    tracing::error!(error = %e, "store.error");
    JobError::runtime("A database error occurred")
}

/// Run a training job: train, checkpoint with provenance, store, link, cache.
pub(crate) async fn train(
    state: AppState,
    artifact_id: String,
    config_id: String,
    data: DataFrame,
    geo: Option<Geo>,
    run_info: Option<RunInfoIn>,
) -> Result<(), JobError> {
    let started_at = job_now();
    let start = Instant::now();
    let config = state
        .store()
        .get_config(&config_id)
        .await
        .map_err(|e| store_error(&e))?
        .ok_or_else(|| JobError::value(format!("Config {config_id} not found")))?;
    let run = resolve_run(&state, run_info.as_ref(), &config.data, None)?;

    let model = state.model().clone();
    let cfg = config.data.clone();
    let (trained, checkpoint, data) = chapkit_http::jobs::blocking(move || {
        let result = model.train(&cfg, TrainInput::new(&data, &run, geo.as_ref()));
        result.map(|(t, c)| (t, c, data))
    })
    .await?
    .map_err(|e| model_error(&e))?;

    let checkpoint = Checkpoint {
        provenance: Provenance {
            model: state.inner.service_id.clone(),
            model_version: state.inner.version.clone(),
            chapkit_version: format!("chapkit-rs {}", env!("CARGO_PKG_VERSION")),
            created_at: Some(chapkit_http::time::rfc3339_now()),
            config: config.data.clone(),
            ..checkpoint.provenance
        }
        .with_training_data(&data),
        ..checkpoint
    };
    let bytes = checkpoint.to_bytes().map_err(|e| model_error(&e))?;
    let duration = start.elapsed().as_secs_f64();
    let artifact_data = json!({
        "type": TRAINING_TYPE,
        "metadata": {
            "status": "success",
            "config_id": config_id,
            "started_at": started_at,
            "completed_at": job_now(),
            "duration_seconds": duration,
            "provenance": checkpoint.provenance,
        },
        "content": null,
        "content_type": SAFETENSORS,
        "content_size": bytes.len(),
    });
    state
        .store()
        .insert_artifact(NewArtifact {
            id: artifact_id.clone(),
            parent_id: None,
            data: artifact_data,
            tags: vec![],
            content: Some(bytes),
            created_at: entity_now(),
        })
        .await
        .map_err(|e| store_error(&e))?;
    state
        .store()
        .link(&config_id, &artifact_id)
        .await
        .map_err(|e| store_error(&e))?;
    state.cache(artifact_id, trained);
    Ok(())
}

/// A training artifact's trained state (warm cache first) and its config data.
pub(crate) async fn load_trained(
    state: &AppState,
    artifact_id: &str,
) -> Result<(Trained, Value, ArtifactRow), JobError> {
    let artifact = state
        .store()
        .get_artifact(artifact_id)
        .await
        .map_err(|e| store_error(&e))?
        .ok_or_else(|| JobError::value(format!("Training artifact {artifact_id} not found")))?;
    if str_field(&artifact.data, "type") != Some(TRAINING_TYPE) {
        return Err(JobError::value(format!(
            "Artifact {artifact_id} is not a training artifact"
        )));
    }
    let config_id = artifact
        .data
        .get("metadata")
        .and_then(|m| str_field(m, "config_id"))
        .unwrap_or_default()
        .to_owned();
    let config = match state
        .store()
        .get_config(&config_id)
        .await
        .map_err(|e| store_error(&e))?
    {
        Some(c) => c.data,
        None => state
            .store()
            .config_for_artifact(artifact_id)
            .await
            .map_err(|e| store_error(&e))?
            .map(|c| c.data)
            .ok_or_else(|| {
                JobError::value(format!(
                    "Config for training artifact {artifact_id} not found"
                ))
            })?,
    };
    if let Some(t) = state.cached(artifact_id) {
        return Ok((t, config, artifact));
    }
    let bytes = state
        .store()
        .artifact_content(artifact_id)
        .await
        .map_err(|e| store_error(&e))?
        .ok_or_else(|| {
            JobError::value(format!("Training artifact {artifact_id} has no checkpoint"))
        })?;
    let model = state.model().clone();
    let trained = chapkit_http::jobs::blocking(move || {
        Checkpoint::from_bytes(&bytes).and_then(|c| model.load(&c))
    })
    .await?
    .map_err(|e| model_error(&e))?;
    state.cache(artifact_id.to_owned(), trained.clone());
    Ok((trained, config, artifact))
}

/// Predict with a trained artifact; no storage. Used by `$predict` and `$predict-sync`.
pub(crate) async fn predict_frame(
    state: &AppState,
    artifact_id: &str,
    historic: DataFrame,
    future: DataFrame,
    geo: Option<Geo>,
    run_info: Option<&RunInfoIn>,
) -> Result<(DataFrame, Value), JobError> {
    let (trained, config, _) = load_trained(state, artifact_id).await?;
    let run = resolve_run(state, run_info, &config, Some(&future))?;
    let model = state.model().clone();
    let cfg = config.clone();
    let out = chapkit_http::jobs::blocking(move || {
        model.predict(
            &trained,
            &cfg,
            PredictInput::new(&historic, &future, &run, geo.as_ref()),
        )
    })
    .await?
    .map_err(|e| model_error(&e))?;
    check_output(&out)?;
    Ok((out, config))
}

fn check_output(out: &DataFrame) -> Result<(), JobError> {
    if !out.has(cols::TIME_PERIOD) || !out.has(cols::LOCATION) {
        return Err(JobError::runtime(
            "model output is missing time_period or location columns",
        ));
    }
    Ok(())
}

/// Run a prediction job: predict and store the prediction artifact under the training artifact.
pub(crate) async fn predict(
    state: AppState,
    prediction_id: String,
    artifact_id: String,
    historic: DataFrame,
    future: DataFrame,
    geo: Option<Geo>,
    run_info: Option<RunInfoIn>,
) -> Result<(), JobError> {
    let started_at = job_now();
    let start = Instant::now();
    let (frame, _) = predict_frame(
        &state,
        &artifact_id,
        historic,
        future,
        geo,
        run_info.as_ref(),
    )
    .await?;
    let training = state
        .store()
        .get_artifact(&artifact_id)
        .await
        .map_err(|e| store_error(&e))?;
    let config_id = training
        .as_ref()
        .and_then(|a| a.data.get("metadata"))
        .and_then(|m| str_field(m, "config_id"))
        .unwrap_or_default()
        .to_owned();
    let content = serde_json::to_value(&frame).map_err(|e| JobError::runtime(e.to_string()))?;
    let data = json!({
        "type": PREDICTION_TYPE,
        "metadata": {
            "status": "success",
            "config_id": config_id,
            "started_at": started_at,
            "completed_at": job_now(),
            "duration_seconds": start.elapsed().as_secs_f64(),
        },
        "content": content,
        "content_type": DATAFRAME_JSON,
        "content_size": null,
    });
    state
        .store()
        .insert_artifact(NewArtifact {
            id: prediction_id,
            parent_id: Some(artifact_id),
            data,
            tags: vec![],
            content: None,
            created_at: entity_now(),
        })
        .await
        .map_err(|e| store_error(&e))?;
    Ok(())
}

/// The rows of `historic` up to and including `end`, each row's period parsed once in `periods`.
pub(crate) fn history_until(
    historic: &DataFrame,
    periods: &[Period],
    end: Period,
) -> Result<DataFrame, String> {
    let mask: Vec<bool> = periods.iter().map(|p| *p <= end).collect();
    let cut = historic.filter(&mask).map_err(|e| e.to_string())?;
    if cut.is_empty() {
        return Err(format!("historic has no rows up to {end}"));
    }
    Ok(cut)
}

/// The periods of a frame's `time_period` column, all parsed.
pub(crate) fn frame_periods(frame: &DataFrame) -> Result<Vec<Period>, String> {
    let tp = frame
        .column(cols::TIME_PERIOD)
        .ok_or("historic has no 'time_period' column")?;
    (0..frame.n_rows())
        .map(|i| {
            tp.str_at(i)
                .and_then(Period::parse)
                .ok_or_else(|| format!("historic row {i}: time_period is not a period"))
        })
        .collect()
}

/// A `$predict-splits` split's history: its own, or the shared one up to a period.
pub(crate) enum History {
    Own(DataFrame),
    Until(Period),
}

/// A checked `$predict-splits` split: its inputs and run settings.
pub(crate) struct Split {
    pub history: History,
    pub future: DataFrame,
    pub run: RunInfo,
}

/// Predict every split with the trained model, in order, in one blocking call.
pub(crate) async fn predict_split_frames(
    state: &AppState,
    trained: Trained,
    config: Value,
    shared: Option<DataFrame>,
    splits: Vec<Split>,
    geo: Option<Geo>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Result<Vec<DataFrame>, JobError> {
    let model = state.model().clone();
    let max_cells = chapkit_core::max_prediction_cells();
    // The permit goes with the work: held until it returns, even when the caller is gone.
    chapkit_http::jobs::blocking(move || {
        let _permit = permit;
        predict_splits_blocking(
            model.as_ref(),
            &trained,
            &config,
            shared.as_ref(),
            &splits,
            geo.as_ref(),
            max_cells,
        )
    })
    .await?
}

/// [`predict_split_frames`]' work. A split's history cut from `shared` is made when its turn comes
/// and dropped after, so the request's frames plus one cut are all the input held, however many
/// splits share the history. The forecasts together hold at most `max_cells` cells
/// (`CHAPKIT_MAX_PREDICTION_CELLS`, as one forecast does), checked as each arrives.
fn predict_splits_blocking(
    model: &dyn crate::erased::DynModel,
    trained: &Trained,
    config: &Value,
    shared: Option<&DataFrame>,
    splits: &[Split],
    geo: Option<&Geo>,
    max_cells: usize,
) -> Result<Vec<DataFrame>, JobError> {
    let periods = shared
        .map(frame_periods)
        .transpose()
        .map_err(JobError::value)?;
    let mut frames = Vec::with_capacity(splits.len());
    let mut cells = 0usize;
    for (i, split) in splits.iter().enumerate() {
        let cut;
        let historic = match (&split.history, shared, &periods) {
            (History::Own(h), _, _) => h,
            (History::Until(end), Some(h), Some(periods)) => {
                cut = history_until(h, periods, *end)
                    .map_err(|m| JobError::value(format!("splits[{i}]: {m}")))?;
                &cut
            }
            (History::Until(_), _, _) => {
                return Err(JobError::value(format!(
                    "splits[{i}]: historic_end needs a shared historic"
                )));
            }
        };
        let frame = model
            .predict(
                trained,
                config,
                PredictInput::new(historic, &split.future, &split.run, geo),
            )
            .map_err(|e| model_error(&e))?;
        check_output(&frame)?;
        cells = cells.saturating_add(frame.n_rows().saturating_mul(frame.n_cols()));
        if cells > max_cells {
            return Err(JobError::value(format!(
                "the forecasts of the first {} splits have {cells} cells together, more than the \
                 limit of {max_cells} (CHAPKIT_MAX_PREDICTION_CELLS); ask for fewer splits",
                i + 1
            )));
        }
        frames.push(frame);
    }
    Ok(frames)
}

/// Store `$predict-splits` forecasts as one artifact under the training artifact.
pub(crate) async fn store_split_predictions(
    state: &AppState,
    prediction_id: String,
    artifact_id: String,
    config_id: &str,
    frames: &[DataFrame],
    started_at: String,
    start: Instant,
) -> Result<(), JobError> {
    let content = serde_json::to_value(frames).map_err(|e| JobError::runtime(e.to_string()))?;
    let data = json!({
        "type": PREDICTION_SPLITS_TYPE,
        "metadata": {
            "status": "success",
            "config_id": config_id,
            "started_at": started_at,
            "completed_at": job_now(),
            "duration_seconds": start.elapsed().as_secs_f64(),
            "n_splits": frames.len(),
        },
        "content": content,
        "content_type": "application/json",
        "content_size": null,
    });
    state
        .store()
        .insert_artifact(NewArtifact {
            id: prediction_id,
            parent_id: Some(artifact_id),
            data,
            tags: vec![],
            content: None,
            created_at: entity_now(),
        })
        .await
        .map_err(|e| store_error(&e))?;
    Ok(())
}

/// Hierarchy labels for the artifact tree.
pub(crate) fn level_label(level: i64) -> String {
    match level {
        0 => TRAINING_TYPE.to_owned(),
        1 => PREDICTION_TYPE.to_owned(),
        n => format!("level_{n}"),
    }
}

/// Hierarchy name.
pub(crate) const HIERARCHY: &str = "ml_pipeline";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erased::Erased;
    use crate::eval_cmd::tests::Last;
    use chapkit_core::Column;

    fn frame(periods: &[&str], cases: bool) -> DataFrame {
        let mut df = DataFrame::new();
        df.push_column(
            cols::TIME_PERIOD,
            Column::from_strs(periods.iter().copied()),
        )
        .unwrap();
        df.push_column(
            cols::LOCATION,
            Column::from_strs(periods.iter().map(|_| "A")),
        )
        .unwrap();
        if cases {
            let y = (1..=periods.len()).map(|v| Some(v as f64)).collect();
            df.push_column(cols::DISEASE_CASES, Column::Float(y))
                .unwrap();
        }
        df
    }

    /// Splits cut from the shared history forecast as with their own cut history, and the
    /// forecasts together are bounded.
    #[test]
    fn shared_history_splits_are_cut_lazily_and_bounded() {
        let model = Erased(Last);
        let trained: Trained = std::sync::Arc::new(());
        let config = json!({"spread": 1.0});
        let shared = frame(&["2024-01", "2024-02", "2024-03", "2024-04"], true);
        let run = RunInfo::with_periods(1);
        let split = |end: &str, next: &str| Split {
            history: History::Until(Period::parse(end).unwrap()),
            future: frame(&[next], false),
            run: run.clone(),
        };
        let splits = [split("2024-02", "2024-03"), split("2024-03", "2024-04")];
        let got =
            predict_splits_blocking(&model, &trained, &config, Some(&shared), &splits, None, 100)
                .unwrap();
        let own = Split {
            history: History::Own(frame(&["2024-01", "2024-02"], true)),
            ..split("2024-02", "2024-03")
        };
        let want =
            predict_splits_blocking(&model, &trained, &config, None, &[own], None, 100).unwrap();
        assert_eq!(
            serde_json::to_value(&got[0]).unwrap(),
            serde_json::to_value(&want[0]).unwrap()
        );
        assert_eq!(got[1].f64_column("sample_1").unwrap(), vec![3.0]);
        // Each forecast is 1 row x 5 columns: two fit 10 cells, not 9.
        let e = predict_splits_blocking(&model, &trained, &config, Some(&shared), &splits, None, 9)
            .unwrap_err();
        assert!(e.message.contains("more than the limit"), "{}", e.message);
        let early = [split("2023-12", "2024-01")];
        let e =
            predict_splits_blocking(&model, &trained, &config, Some(&shared), &early, None, 100)
                .unwrap_err();
        assert!(e.message.contains("no rows up to"), "{}", e.message);
    }
}
