//! A backtest of the service's model with its result as JSON: shared by `<bin> eval`, `<bin> hpo`,
//! `$backtest` and `$hpo`.

use std::collections::BTreeMap;
use std::time::Instant;

use chapkit_core::{DataFrame, Period, PredictInput, RunInfo, TrainInput, cols};
use chapkit_eval::{
    ForecastSummary, Metric, Options, Prepare, Quantiles, Scores, weather_provider,
};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::erased::DynModel;

/// The model's config for a backtest: `options` (flat, or chap-core's `user_option_values`) with
/// `prediction_periods` set to `periods`, normalized as `POST /api/v1/configs` stores a config.
pub(crate) fn backtest_config(
    model: &dyn DynModel,
    options: Option<Value>,
    periods: usize,
) -> Result<Value, String> {
    let mut obj = match options {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m,
        Some(_) => return Err("the config must be a JSON object".into()),
    };
    obj.insert("prediction_periods".into(), periods.into());
    model.normalize_config(Value::Object(obj))
}

/// Check the backtest settings and the forecast length against the model's bounds.
pub(crate) fn check(model: &dyn DynModel, options: &Options, weather: &str) -> Result<(), String> {
    options.validate().map_err(|e| e.to_string())?;
    let info = model.info();
    let periods = u32::try_from(options.periods).unwrap_or(u32::MAX);
    if periods < info.min_prediction_periods || periods > info.max_prediction_periods {
        return Err(format!(
            "periods {periods} is outside the model's bounds [{}, {}]",
            info.min_prediction_periods, info.max_prediction_periods
        ));
    }
    if weather_provider(weather).is_none() {
        return Err(format!("unknown weather provider {weather}"));
    }
    Ok(())
}

/// The backtest settings as reported.
#[derive(Debug, Clone, Serialize)]
pub struct OptionsReport {
    pub periods: usize,
    pub splits: usize,
    pub stride: usize,
    pub retrain: usize,
    pub weather: String,
    pub fill_gaps: bool,
    pub pad: bool,
    pub drop_untrained: bool,
}

/// The dataset as reported.
#[derive(Debug, Clone, Serialize)]
pub struct DataReport {
    /// Where it came from (a path, `generated (seed N)`, `request`).
    pub source: String,
    pub locations: usize,
    /// Distinct periods.
    pub periods: usize,
    /// `months` or `weeks`.
    pub unit: &'static str,
    pub first: Option<String>,
    pub last: Option<String>,
    pub rows: usize,
    /// Locations left out: no observed target before the test windows (chap-core's rule).
    pub dropped: Vec<String>,
}

/// Metrics by chap-core `metric_id`, as its `aggregateMetrics` (`null` when undefined).
pub type Metrics = BTreeMap<&'static str, Option<f64>>;

#[derive(Debug, Clone, Serialize)]
pub struct HorizonScores {
    pub horizon: usize,
    pub metrics: Metrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocationScores {
    pub location: String,
    pub metrics: Metrics,
}

/// A finished backtest, as `<bin> eval --json` prints it and `$backtest` stores it.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub model: String,
    pub options: OptionsReport,
    /// The model config the backtest ran with.
    pub config: Value,
    pub data: DataReport,
    /// Last period of the first training.
    pub train_last: String,
    /// Each split's last historic period.
    pub split_last: Vec<String>,
    /// The splits at which the model was trained.
    pub trained_at: Vec<usize>,
    /// Forecast rows scored.
    pub rows: usize,
    pub elapsed_ms: f64,
    pub aggregate: Metrics,
    pub by_horizon: Vec<HorizonScores>,
    pub by_location: Vec<LocationScores>,
    /// The stored dataset backtested (`$backtest` with `dataset_id`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
    /// Every forecast with the observed target and the quantiles of its samples, split by split.
    pub forecasts: Vec<ForecastSummary>,
    #[serde(skip)]
    pub scores: Scores,
}

fn metrics(values: &[f64]) -> Metrics {
    Metric::ALL
        .iter()
        .zip(values)
        .map(|(m, v)| (m.id(), (!v.is_nan()).then_some(*v)))
        .collect()
}

/// Describe `data`.
pub(crate) fn describe(data: &DataFrame, source: String) -> Result<DataReport, String> {
    let locations = data
        .group_indices(cols::LOCATION)
        .map_err(|e| e.to_string())?
        .len();
    let tp = data.require(cols::TIME_PERIOD).map_err(|e| e.to_string())?;
    let mut periods: BTreeMap<Period, String> = BTreeMap::new();
    for row in 0..data.n_rows() {
        if let Some(s) = tp.str_at(row)
            && let Some(p) = Period::parse(s)
        {
            periods.entry(p).or_insert_with(|| s.to_owned());
        }
    }
    let unit = match periods.keys().next() {
        Some(Period::Week { .. }) => "weeks",
        _ => "months",
    };
    Ok(DataReport {
        source,
        locations,
        periods: periods.len(),
        unit,
        first: periods.values().next().cloned(),
        last: periods.values().next_back().cloned(),
        rows: data.n_rows(),
        dropped: Vec::new(),
    })
}

/// The run settings of a backtest with `config`.
pub(crate) fn run_info(config: &Value, periods: usize) -> RunInfo {
    RunInfo {
        prediction_periods: u32::try_from(periods).unwrap_or(u32::MAX),
        additional_continuous_covariates: config
            .get("additional_continuous_covariates")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default(),
        future_covariate_origin: None,
    }
}

/// Backtest `model` with `config` (from [`backtest_config`]) on `data` and score it.
pub(crate) fn evaluate(
    model: &dyn DynModel,
    id: &str,
    config: &Value,
    data: &DataFrame,
    source: String,
    options: &Options,
    weather: &str,
) -> Result<Report, String> {
    check(model, options, weather)?;
    let provider =
        weather_provider(weather).ok_or_else(|| format!("unknown weather provider {weather}"))?;
    let run = run_info(config, options.periods);
    let started = Instant::now();
    let backtest = chapkit_eval::backtest(
        data,
        options,
        provider.as_ref(),
        |d| {
            model
                .train(config, TrainInput::new(d, &run, None))
                .map(|(trained, _)| trained)
        },
        |t, h, f| model.predict(t, config, PredictInput::new(h, f, &run, None)),
    )
    .map_err(|e| e.to_string())?;
    let scores = backtest.score(data).map_err(|e| e.to_string())?;
    let forecasts =
        chapkit_eval::summarize(&backtest.forecasts, data).map_err(|e| e.to_string())?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Report {
        model: id.to_owned(),
        options: OptionsReport {
            periods: options.periods,
            splits: options.splits,
            stride: options.stride,
            retrain: options.retrain,
            weather: weather.to_owned(),
            fill_gaps: options.prepare.fill_gaps,
            pad: options.prepare.pad,
            drop_untrained: options.drop_untrained,
        },
        config: config.clone(),
        data: DataReport {
            dropped: backtest.dropped,
            ..describe(data, source)?
        },
        train_last: backtest.train_last,
        split_last: backtest.split_last,
        trained_at: backtest.trained_at,
        rows: scores.rows,
        elapsed_ms,
        aggregate: metrics(&scores.aggregate),
        by_horizon: scores
            .by_horizon
            .iter()
            .map(|(h, v)| HorizonScores {
                horizon: *h,
                metrics: metrics(v),
            })
            .collect(),
        by_location: scores
            .by_location
            .iter()
            .map(|(l, v)| LocationScores {
                location: l.clone(),
                metrics: metrics(v),
            })
            .collect(),
        dataset_id: None,
        forecasts,
        scores,
    })
}

/// One location and period of a forecast ahead: the quantiles of its samples.
#[derive(Debug, Clone, Serialize)]
pub struct AheadSummary {
    pub location: String,
    pub time_period: String,
    #[serde(flatten)]
    pub quantiles: Quantiles,
}

/// A forecast past the end of a dataset.
#[derive(Debug, Clone)]
pub(crate) struct Ahead {
    pub forecasts: Vec<AheadSummary>,
    /// The model's prediction frame.
    pub frame: DataFrame,
}

/// chap-core's `forecast_ahead`: train `model` with `config` (from [`backtest_config`]) on the
/// whole of `data` (prepared as for a backtest) and forecast the `periods` periods after it, with
/// the future covariates from `weather`.
pub(crate) fn forecast(
    model: &dyn DynModel,
    config: &Value,
    data: &DataFrame,
    periods: usize,
    weather: &str,
) -> Result<Ahead, String> {
    let provider =
        weather_provider(weather).ok_or_else(|| format!("unknown weather provider {weather}"))?;
    let ahead = chapkit_eval::forecast_ahead(data, periods, &Prepare::default(), provider.as_ref())
        .map_err(|e| e.to_string())?;
    let run = run_info(config, periods);
    let (trained, _) = model
        .train(config, TrainInput::new(&ahead.historic, &run, None))
        .map_err(|e| e.to_string())?;
    let frame = model
        .predict(
            &trained,
            config,
            PredictInput::new(&ahead.historic, &ahead.future, &run, None),
        )
        .map_err(|e| e.to_string())?;
    let (tp, loc) = (
        frame
            .require(cols::TIME_PERIOD)
            .map_err(|e| e.to_string())?,
        frame.require(cols::LOCATION).map_err(|e| e.to_string())?,
    );
    let samples: Vec<Vec<f64>> = frame
        .iter()
        .filter(|(name, _)| name.starts_with("sample_"))
        .map(|(name, col)| {
            col.to_f64_vec()
                .ok_or_else(|| format!("{name} is not numeric"))
        })
        .collect::<Result<_, _>>()?;
    if samples.is_empty() {
        return Err("the forecast has no sample columns".into());
    }
    let forecasts = (0..frame.n_rows())
        .map(|row| {
            let values: Vec<f64> = samples.iter().filter_map(|c| c.get(row).copied()).collect();
            AheadSummary {
                location: loc.str_at(row).unwrap_or_default().to_owned(),
                time_period: tp.str_at(row).unwrap_or_default().to_owned(),
                quantiles: Quantiles::of(&values),
            }
        })
        .collect();
    Ok(Ahead { forecasts, frame })
}

/// A finished hyperparameter search, as `<bin> hpo --json` prints it and `$hpo` stores it.
#[derive(Debug, Clone, Serialize)]
pub struct HpoReport {
    pub model: String,
    /// The objective: a chap-core `metric_id`.
    pub metric: &'static str,
    pub backtest: OptionsReport,
    pub data: DataReport,
    /// The options every candidate is merged into.
    pub config: Value,
    pub search: chapkit_hpo::Search,
    /// `config` with the best candidate, normalized as a stored config.
    pub best_config: Value,
    /// The backtest of `best_config`.
    pub best: Report,
}

/// Search the model's hyperparameters: every candidate merged into `base` (from
/// [`backtest_config`]) and backtested on `data`, split once, scored by `metric`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hpo(
    model: &dyn DynModel,
    id: &str,
    base: &Value,
    data: &DataFrame,
    source: String,
    options: &Options,
    weather: &str,
    metric: Metric,
    search: &chapkit_hpo::Options,
) -> Result<HpoReport, String> {
    check(model, options, weather)?;
    let space = model
        .search_space()
        .ok_or("the model has no search space (Model::search_space)")?;
    let direction = chapkit_hpo::direction(metric).map_err(|e| e.to_string())?;
    let provider =
        weather_provider(weather).ok_or_else(|| format!("unknown weather provider {weather}"))?;
    let splits = chapkit_eval::train_test_splits(data, options, provider.as_ref())
        .map_err(|e| e.to_string())?;
    let config_of = |params: &chapkit_hpo::Params| {
        backtest_config(
            model,
            Some(chapkit_hpo::with_params(base, params)),
            options.periods,
        )
    };
    let found = chapkit_hpo::search(&space, search, direction, |params| {
        let config = config_of(params)?;
        let run = run_info(&config, options.periods);
        let bt = chapkit_eval::backtest_splits(
            &splits,
            options,
            |d| {
                model
                    .train(&config, TrainInput::new(d, &run, None))
                    .map(|(trained, _)| trained)
            },
            |t, h, f| model.predict(t, &config, PredictInput::new(h, f, &run, None)),
        )
        .map_err(|e| e.to_string())?;
        Ok(bt.score(data).map_err(|e| e.to_string())?.get(metric))
    })
    .map_err(|e| e.to_string())?;
    let best_config = config_of(&found.best_params)?;
    let best = evaluate(model, id, &best_config, data, source, options, weather)?;
    Ok(HpoReport {
        model: id.to_owned(),
        metric: metric.id(),
        backtest: best.options.clone(),
        data: best.data.clone(),
        config: base.clone(),
        search: found,
        best_config,
        best,
    })
}
