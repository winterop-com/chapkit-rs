//! Explain one forecast cell (a location and time period) of a model.
//!
//! The features are covariate columns of the `future` frame (for example rainfall
//! and temperature). For each perturbation the target row's covariates are
//! replaced, the model predicts the whole future frame, and the value is the mean of
//! the samples at the target row. The background distribution comes from the
//! `historic` rows (all locations), falling back to `future` for covariates that
//! historic does not carry.
//!
//! Limitation: only the target month's covariates are perturbed. A model that reacts
//! to earlier months (for example rain one to three months before) gets those values
//! from `historic` and earlier `future` rows, which stay fixed, so lagged effects do
//! not show up in the importances. Lagged features are a planned extension.

use chapkit_core::{Column, DataFrame, cols};
use serde::{Deserialize, Serialize};

use crate::{Error, Explanation, Options, Result, explain};

/// The forecast cell to explain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastTarget {
    /// Location id.
    pub location: String,
    /// Period id, as in the frames.
    pub time_period: String,
}

/// An explanation of one forecast cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastExplanation {
    /// The cell explained.
    pub target: ForecastTarget,
    /// The covariates explained, in importance order inside `explanation`.
    pub features: Vec<String>,
    /// The explanation; values are the mean forecast at the cell.
    #[serde(flatten)]
    pub explanation: Explanation,
}

fn is_reserved(name: &str, target_col: &str) -> bool {
    name == cols::TIME_PERIOD
        || name == cols::LOCATION
        || name == target_col
        || name.starts_with("sample_")
}

/// Numeric covariates of `future` that can be explained.
pub fn default_features(future: &DataFrame, target_col: &str) -> Vec<String> {
    future
        .iter()
        .filter(|(n, c)| !is_reserved(n, target_col) && !matches!(c, Column::Str(_)))
        .map(|(n, _)| n.to_owned())
        .collect()
}

fn row_of(df: &DataFrame, t: &ForecastTarget) -> Option<usize> {
    let (loc, tp) = (df.column(cols::LOCATION)?, df.column(cols::TIME_PERIOD)?);
    (0..df.n_rows()).find(|&r| {
        loc.str_at(r) == Some(t.location.as_str()) && tp.str_at(r) == Some(t.time_period.as_str())
    })
}

/// Mean of the `sample_*` values at the target cell of a prediction frame.
fn cell_mean(pred: &DataFrame, t: &ForecastTarget) -> Result<f64> {
    let row = row_of(pred, t).ok_or_else(|| {
        Error::Model(format!(
            "prediction has no row for {} {}",
            t.location, t.time_period
        ))
    })?;
    let (mut sum, mut n) = (0.0, 0usize);
    for (name, col) in pred.iter() {
        if name.starts_with("sample_")
            && let Some(v) = col.f64_at(row).filter(|v| v.is_finite())
        {
            sum += v;
            n += 1;
        }
    }
    if n == 0 {
        return Err(Error::Model(
            "prediction has no finite samples at the target cell".into(),
        ));
    }
    Ok(sum / n as f64)
}

/// Explain the forecast at `target` in terms of `features` (default: every numeric
/// covariate of `future`). `predict` maps a future frame to a prediction frame;
/// it is called once per perturbation, in parallel.
pub fn explain_forecast(
    predict: &(dyn Fn(&DataFrame) -> std::result::Result<DataFrame, String> + Sync),
    historic: &DataFrame,
    future: &DataFrame,
    target: &ForecastTarget,
    features: Option<Vec<String>>,
    target_col: &str,
    opts: &Options,
) -> Result<ForecastExplanation> {
    let row = row_of(future, target).ok_or_else(|| {
        Error::InvalidInput(format!(
            "future has no row for location '{}' and period '{}'",
            target.location, target.time_period
        ))
    })?;
    let features = features.unwrap_or_else(|| default_features(future, target_col));
    if features.is_empty() {
        return Err(Error::InvalidInput(
            "future has no numeric covariates to explain".into(),
        ));
    }

    // Base frame with the explained covariates as float columns, so perturbing a cell is a store.
    let mut base = future.clone();
    let mut instance = Vec::with_capacity(features.len());
    for f in &features {
        let values = future.f64_column(f).map_err(|_| {
            Error::InvalidInput(format!("feature '{f}' is not a numeric column of future"))
        })?;
        instance.push(values.get(row).copied().unwrap_or(f64::NAN));
        base.set_column(f.clone(), Column::from_f64(values))
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
    }

    let background = background_rows(historic, future, &features)?;

    let value = |x: &[f64]| -> Result<f64> {
        let mut frame = base.clone();
        for (f, v) in features.iter().zip(x) {
            frame
                .set_f64(f, row, Some(*v))
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
        }
        let pred = predict(&frame).map_err(Error::Model)?;
        cell_mean(&pred, target)
    };

    let explanation = explain(&value, &features, &instance, &background, opts)?;
    Ok(ForecastExplanation {
        target: target.clone(),
        features,
        explanation,
    })
}

/// Complete background rows for `features`: from historic when it has every feature,
/// else from future. Rows with a missing value are skipped.
fn background_rows(
    historic: &DataFrame,
    future: &DataFrame,
    features: &[String],
) -> Result<Vec<Vec<f64>>> {
    let source = if features.iter().all(|f| historic.has(f)) {
        historic
    } else {
        future
    };
    let cols_: Vec<Vec<f64>> = features
        .iter()
        .map(|f| {
            source
                .f64_column(f)
                .map_err(|_| Error::InvalidInput(format!("feature '{f}' is not numeric")))
        })
        .collect::<Result<_>>()?;
    let rows: Vec<Vec<f64>> = (0..source.n_rows())
        .map(|r| {
            cols_
                .iter()
                .map(|c| c.get(r).copied().unwrap_or(f64::NAN))
                .collect::<Vec<f64>>()
        })
        .filter(|row| row.iter().all(|v| v.is_finite()))
        .collect();
    if rows.len() < 2 {
        return Err(Error::InvalidInput(
            "need at least two complete background rows for the features".into(),
        ));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Method;
    use chapkit_core::prediction_frame;

    fn frame(rows: &[(&str, &str, f64, f64)]) -> DataFrame {
        DataFrame::from_columns([
            ("time_period", Column::from_strs(rows.iter().map(|r| r.0))),
            ("location", Column::from_strs(rows.iter().map(|r| r.1))),
            ("rainfall", Column::from_f64(rows.iter().map(|r| r.2))),
            ("temperature", Column::from_f64(rows.iter().map(|r| r.3))),
        ])
        .unwrap()
    }

    /// cases = 2 * rainfall - 1 * temperature, so SHAP contributions are exact.
    fn predict(future: &DataFrame) -> std::result::Result<DataFrame, String> {
        let rain = future.f64_column("rainfall").map_err(|e| e.to_string())?;
        let temp = future
            .f64_column("temperature")
            .map_err(|e| e.to_string())?;
        prediction_frame(future, 4, |row, out| {
            out.fill(2.0 * rain[row] - temp[row]);
            Ok(())
        })
        .map_err(|e| e.to_string())
    }

    #[test]
    fn explains_a_forecast_cell_with_shap() {
        let historic = frame(&[
            ("2024-01", "A", 10.0, 20.0),
            ("2024-02", "A", 20.0, 22.0),
            ("2024-01", "B", 30.0, 24.0),
        ]);
        let future = frame(&[("2024-03", "A", 40.0, 25.0), ("2024-03", "B", 5.0, 30.0)]);
        let target = ForecastTarget {
            location: "A".into(),
            time_period: "2024-03".into(),
        };
        let opts = Options {
            method: Method::Shap,
            ..Options::default()
        };
        let e = explain_forecast(
            &predict,
            &historic,
            &future,
            &target,
            None,
            "disease_cases",
            &opts,
        )
        .unwrap();
        assert_eq!(e.features, vec!["rainfall", "temperature"]);
        assert!((e.explanation.prediction - (80.0 - 25.0)).abs() < 1e-9);
        let rain = e
            .explanation
            .importances
            .iter()
            .find(|i| i.feature == "rainfall")
            .unwrap();
        assert!((rain.weight - 2.0 * (40.0 - 20.0)).abs() < 1e-9, "{rain:?}");
        let temp = e
            .explanation
            .importances
            .iter()
            .find(|i| i.feature == "temperature")
            .unwrap();
        assert!((temp.weight - -(25.0 - 22.0)).abs() < 1e-9, "{temp:?}");
    }

    #[test]
    fn unknown_cell_is_invalid_input() {
        let f = frame(&[("2024-01", "A", 1.0, 2.0), ("2024-02", "A", 2.0, 3.0)]);
        let target = ForecastTarget {
            location: "Z".into(),
            time_period: "2024-01".into(),
        };
        let r = explain_forecast(
            &predict,
            &f,
            &f,
            &target,
            None,
            "disease_cases",
            &Options::default(),
        );
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }
}
