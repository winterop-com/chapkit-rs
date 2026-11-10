//! chap-core's `backtest`: train at the retraining points, forecast every split, check the
//! forecasts are complete, and flatten them for scoring.

use std::collections::HashMap;

use chapkit_core::{
    DataFrame, Error, Model, Period, PredictInput, Result, RunInfo, TrainInput, cols,
};
use rayon::prelude::*;
use serde::Serialize;

use crate::metrics::{Scores, score};
use crate::split::{Options, Split, Splits, retrain_points, train_test_splits};
use crate::weather::WeatherProvider;

/// One forecast cell: a location and period of a split, with its samples.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Forecast {
    /// Location.
    pub location: String,
    /// Forecast period, as written in the data.
    pub time_period: String,
    /// Periods ahead of the split's last historic period (1 is the first forecast period).
    pub horizon: usize,
    /// The model's samples.
    pub samples: Vec<f64>,
}

/// A finished backtest.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Backtest {
    /// Last period of the first training.
    pub train_last: String,
    /// Each split's last historic period.
    pub split_last: Vec<String>,
    /// The splits at which the model was trained.
    pub trained_at: Vec<usize>,
    /// Every split's forecasts, split by split.
    pub forecasts: Vec<Forecast>,
    /// Locations left out ([`Options::drop_untrained`]).
    pub dropped: Vec<String>,
}

impl Backtest {
    /// Score the forecasts against the observed targets of `data` (the backtest's dataset).
    pub fn score(&self, data: &DataFrame) -> Result<Scores> {
        score(&self.forecasts, data)
    }
}

/// The forecast cells of one split: a forecast for every observed row of the window, with finite
/// samples (chap-core's `validate_split_forecasts`).
fn flatten(prediction: &DataFrame, split: &Split, index: usize, n: usize) -> Result<Vec<Forecast>> {
    let at = |what: String| Error::model(format!("split {} of {n}: {what}", index + 1));
    let (tp, loc) = (
        prediction.require(cols::TIME_PERIOD)?,
        prediction.require(cols::LOCATION)?,
    );
    let samples: Vec<Vec<f64>> = prediction
        .iter()
        .filter(|(name, _)| name.starts_with("sample_"))
        .map(|(name, col)| {
            col.to_f64_vec()
                .ok_or_else(|| at(format!("{name} is not numeric")))
        })
        .collect::<Result<_>>()?;
    if samples.is_empty() {
        return Err(at("the forecast has no sample columns".into()));
    }
    let mut rows: HashMap<(&str, Period), usize> = HashMap::with_capacity(prediction.n_rows());
    for row in 0..prediction.n_rows() {
        if let (Some(l), Some(p)) = (loc.str_at(row), tp.str_at(row).and_then(Period::parse)) {
            rows.entry((l, p)).or_insert(row);
        }
    }
    let window: Vec<Option<Period>> = split.periods.iter().map(|p| Period::parse(p)).collect();
    let (ttp, tloc) = (
        split.truth.require(cols::TIME_PERIOD)?,
        split.truth.require(cols::LOCATION)?,
    );
    let mut out = Vec::with_capacity(split.truth.n_rows());
    for row in 0..split.truth.n_rows() {
        let (Some(location), Some(label)) = (tloc.str_at(row), ttp.str_at(row)) else {
            continue;
        };
        let period = Period::parse(label);
        let horizon = window
            .iter()
            .position(|p| *p == period)
            .map_or(0, |h| h + 1);
        let r = period
            .and_then(|p| rows.get(&(location, p)))
            .ok_or_else(|| at(format!("no forecast for {location} in {label}")))?;
        let values: Vec<f64> = samples
            .iter()
            .filter_map(|col| col.get(*r).copied())
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(at(format!("non-finite samples for {location} in {label}")));
        }
        out.push(Forecast {
            location: location.to_owned(),
            time_period: label.to_owned(),
            horizon,
            samples: values,
        });
    }
    Ok(out)
}

/// Backtest with plain `train` and `predict` functions: `train(data)` fits, and
/// `predict(trained, historic, future)` returns a chapkit prediction frame (`time_period`,
/// `location`, `sample_0..N`). Trainings run in parallel, then the splits' forecasts.
pub fn backtest<T, Train, Predict>(
    data: &DataFrame,
    options: &Options,
    weather: &dyn WeatherProvider,
    train: Train,
    predict: Predict,
) -> Result<Backtest>
where
    T: Send + Sync,
    Train: Fn(&DataFrame) -> Result<T> + Sync,
    Predict: Fn(&T, &DataFrame, &DataFrame) -> Result<DataFrame> + Sync,
{
    let splits = train_test_splits(data, options, weather)?;
    backtest_splits(&splits, options, train, predict)
}

/// [`backtest`] on splits made once with [`train_test_splits`]: a search that backtests many
/// configs on one dataset splits it, and forecasts its weather, only once.
pub fn backtest_splits<T, Train, Predict>(
    splits: &Splits,
    options: &Options,
    train: Train,
    predict: Predict,
) -> Result<Backtest>
where
    T: Send + Sync,
    Train: Fn(&DataFrame) -> Result<T> + Sync,
    Predict: Fn(&T, &DataFrame, &DataFrame) -> Result<DataFrame> + Sync,
{
    let points = retrain_points(options.splits, options.retrain);
    // The first training uses the train set; later ones the split's history (chap-core's rule).
    let trained = points
        .par_iter()
        .map(|&i| match i {
            0 => train(&splits.train),
            i => splits
                .splits
                .get(i)
                .ok_or_else(|| Error::invalid("retraining point out of range"))
                .and_then(|s| train(&s.historic)),
        })
        .collect::<Result<Vec<T>>>()?;
    let n = splits.splits.len();
    let forecasts = splits
        .splits
        .par_iter()
        .enumerate()
        .map(|(i, split)| {
            let k = points.iter().rposition(|&p| p <= i).unwrap_or_default();
            let model = trained
                .get(k)
                .ok_or_else(|| Error::invalid("no trained model for split"))?;
            let prediction = predict(model, &split.historic, &split.future)?;
            flatten(&prediction, split, i, n)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Backtest {
        train_last: splits.train_last.clone(),
        split_last: splits
            .splits
            .iter()
            .map(|s| s.historic_last.clone())
            .collect(),
        trained_at: points,
        forecasts: forecasts.into_iter().flatten().collect(),
        dropped: splits.dropped.clone(),
    })
}

/// Backtest a [`Model`] with `config`; each call gets [`RunInfo::with_periods`] of the options'
/// periods, and no GeoJSON.
pub fn backtest_model<M: Model>(
    model: &M,
    config: &M::Config,
    data: &DataFrame,
    options: &Options,
    weather: &dyn WeatherProvider,
) -> Result<Backtest> {
    let run = RunInfo::with_periods(u32::try_from(options.periods).unwrap_or(u32::MAX));
    backtest(
        data,
        options,
        weather,
        |d| model.train(config, TrainInput::new(d, &run, None)),
        |t, h, f| model.predict(t, config, PredictInput::new(h, f, &run, None)),
    )
}
