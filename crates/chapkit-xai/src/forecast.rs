//! Explain one forecast cell (a location and time period) of a model.
//!
//! The features are inputs the forecast at the cell can depend on, named by their offset
//! from the cell's period `t`:
//!
//! - `rainfall`: a covariate at the cell itself (the target row of `future`);
//! - `rainfall@t-1`, `rainfall@t-2`, ...: the same covariate at earlier periods of the
//!   location, from `future` (earlier forecast steps) and then from the last
//!   [`FeatureSpec::lags`] periods of `historic`;
//! - `disease_cases@t-j`: the target's own observed values in those historic periods.
//!
//! [`Scope::Row`] keeps only the first kind. Each perturbation draws every feature
//! around its column's background mean and spread (`historic`, all locations; `future`
//! for covariates historic lacks), writes the values into copies of the frames, and
//! calls the model once on the whole frames; the value is the mean of the samples at
//! the cell.

use std::cmp::Ordering;

use chapkit_core::{Column, DataFrame, Period, cols};
use serde::{Deserialize, Serialize};

use crate::{Background, Error, Explanation, Options, Result, explain_with};

/// Historic periods perturbed by default under [`Scope::Temporal`].
pub const DEFAULT_LAGS: usize = 3;

/// The forecast cell to explain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastTarget {
    /// Location id.
    pub location: String,
    /// Period id, as in the frames.
    pub time_period: String,
}

/// Which inputs of the cell are features.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// The cell's covariates, their values at earlier periods and the target's history.
    #[default]
    Temporal,
    /// Only the cell's own covariates.
    Row,
}

/// The features of a forecast explanation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSpec {
    /// Covariates (default: every numeric covariate of `future`).
    #[serde(default)]
    pub covariates: Option<Vec<String>>,
    /// Which inputs are perturbed.
    #[serde(default)]
    pub scope: Scope,
    /// Historic periods perturbed under [`Scope::Temporal`].
    #[serde(default = "default_lags")]
    pub lags: usize,
}

fn default_lags() -> usize {
    DEFAULT_LAGS
}

impl Default for FeatureSpec {
    fn default() -> Self {
        Self {
            covariates: None,
            scope: Scope::Temporal,
            lags: DEFAULT_LAGS,
        }
    }
}

/// An explanation of one forecast cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastExplanation {
    /// The cell explained.
    pub target: ForecastTarget,
    /// The features explained, in input order (importance order is inside `explanation`).
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

/// Rows of `location`, oldest first (by parsed period, then by id).
fn rows_in_time_order(df: &DataFrame, location: &str) -> Vec<usize> {
    let (Some(loc), Some(tp)) = (df.column(cols::LOCATION), df.column(cols::TIME_PERIOD)) else {
        return Vec::new();
    };
    let mut rows: Vec<(Option<Period>, &str, usize)> = (0..df.n_rows())
        .filter(|&r| loc.str_at(r) == Some(location))
        .map(|r| {
            let id = tp.str_at(r).unwrap_or_default();
            (Period::parse(id), id, r)
        })
        .collect();
    rows.sort_by(|a, b| match a.0.cmp(&b.0) {
        Ordering::Equal => a.1.cmp(b.1),
        o => o,
    });
    rows.into_iter().map(|(_, _, r)| r).collect()
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

/// A perturbed cell: a column of `historic` or `future` at one row.
struct Cell {
    historic: bool,
    column: String,
    row: usize,
}

/// The features: names, the cells they write, their values at the instance.
#[derive(Default)]
struct Features {
    names: Vec<String>,
    cells: Vec<Cell>,
    instance: Vec<f64>,
}

impl Features {
    fn push(&mut self, column: &str, offset: usize, historic: bool, row: usize, value: f64) {
        self.names.push(if offset == 0 {
            column.to_owned()
        } else {
            format!("{column}@t-{offset}")
        });
        self.cells.push(Cell {
            historic,
            column: column.to_owned(),
            row,
        });
        self.instance.push(value);
    }
}

/// Numeric values of `name` in `df`, if it is a numeric column.
fn numeric(df: &DataFrame, name: &str) -> Option<Vec<f64>> {
    df.column(name).and_then(Column::to_f64_vec)
}

/// Mean and spread of a column: over `historic` (all locations) when it has at least two
/// finite values, else over `future`.
fn column_stats(historic: &DataFrame, future: &DataFrame, name: &str) -> Result<(f64, f64)> {
    for df in [historic, future] {
        let values: Vec<f64> = numeric(df, name)
            .unwrap_or_default()
            .into_iter()
            .filter(|v| v.is_finite())
            .collect();
        if values.len() >= 2 {
            return Ok(Background::stats(&values));
        }
    }
    Err(Error::InvalidInput(format!(
        "need at least two values of '{name}' for its background"
    )))
}

/// The features of the cell at `row` of `future`, in input order.
fn features_of(
    historic: &DataFrame,
    future: &DataFrame,
    row: usize,
    target: &ForecastTarget,
    spec: &FeatureSpec,
    target_col: &str,
) -> Result<Features> {
    let covariates = spec
        .covariates
        .clone()
        .unwrap_or_else(|| default_features(future, target_col));
    if covariates.is_empty() {
        return Err(Error::InvalidInput(
            "future has no numeric covariates to explain".into(),
        ));
    }

    let mut future_cols = Vec::with_capacity(covariates.len());
    for c in &covariates {
        future_cols.push(numeric(future, c).ok_or_else(|| {
            Error::InvalidInput(format!("feature '{c}' is not a numeric column of future"))
        })?);
    }
    let mut features = Features::default();
    for (c, values) in covariates.iter().zip(&future_cols) {
        features.push(
            c,
            0,
            false,
            row,
            values.get(row).copied().unwrap_or(f64::NAN),
        );
    }

    if spec.scope == Scope::Temporal {
        // Earlier forecast steps of the location, nearest first.
        let steps = rows_in_time_order(future, &target.location);
        let pos = steps.iter().position(|&r| r == row).unwrap_or(0);
        for (offset, &r) in (1..).zip(steps.iter().take(pos).rev()) {
            for (c, values) in covariates.iter().zip(&future_cols) {
                if let Some(v) = values.get(r).copied().filter(|v| v.is_finite()) {
                    features.push(c, offset, false, r, v);
                }
            }
        }
        // The last `lags` historic periods: covariates, then the target's own values.
        let mut lagged: Vec<&str> = covariates.iter().map(String::as_str).collect();
        lagged.push(target_col);
        let history: Vec<(&str, Vec<f64>)> = lagged
            .into_iter()
            .filter_map(|c| numeric(historic, c).map(|v| (c, v)))
            .collect();
        let past = rows_in_time_order(historic, &target.location);
        for (offset, &r) in (pos + 1..).zip(past.iter().rev().take(spec.lags)) {
            for (c, values) in &history {
                if let Some(v) = values.get(r).copied().filter(|v| v.is_finite()) {
                    features.push(c, offset, true, r, v);
                }
            }
        }
    }
    Ok(features)
}

/// `df` with the columns of its cells (`historic` side or not) as float columns, so writing a
/// cell is a store.
fn float_base(df: &DataFrame, cells: &[Cell], historic: bool) -> Result<DataFrame> {
    let mut base = df.clone();
    let mut done: Vec<&str> = Vec::new();
    for cell in cells.iter().filter(|c| c.historic == historic) {
        if done.contains(&cell.column.as_str()) {
            continue;
        }
        done.push(&cell.column);
        if let Some(values) = numeric(df, &cell.column) {
            base.set_column(cell.column.clone(), Column::from_f64(values))
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
        }
    }
    Ok(base)
}

/// Explain the forecast at `target`. `predict` maps `(historic, future)` frames to a
/// prediction frame; it is called once per perturbation, with both frames whole.
pub fn explain_forecast(
    predict: &(dyn Fn(&DataFrame, &DataFrame) -> std::result::Result<DataFrame, String> + Sync),
    historic: &DataFrame,
    future: &DataFrame,
    target: &ForecastTarget,
    spec: &FeatureSpec,
    target_col: &str,
    opts: &Options,
) -> Result<ForecastExplanation> {
    let row = row_of(future, target).ok_or_else(|| {
        Error::InvalidInput(format!(
            "future has no row for location '{}' and period '{}'",
            target.location, target.time_period
        ))
    })?;
    let features = features_of(historic, future, row, target, spec, target_col)?;

    let mut background = Background {
        mean: Vec::with_capacity(features.cells.len()),
        sd: Vec::with_capacity(features.cells.len()),
    };
    for cell in &features.cells {
        let (m, s) = column_stats(historic, future, &cell.column)?;
        background.mean.push(m);
        background.sd.push(s);
    }

    let future_base = float_base(future, &features.cells, false)?;
    let historic_base = if features.cells.iter().any(|c| c.historic) {
        Some(float_base(historic, &features.cells, true)?)
    } else {
        None
    };

    let value = |x: &[f64]| -> Result<f64> {
        let mut fut = future_base.clone();
        let mut hist = historic_base.clone();
        for (cell, v) in features.cells.iter().zip(x) {
            let frame = match (&mut hist, cell.historic) {
                (Some(h), true) => h,
                _ => &mut fut,
            };
            frame
                .set_f64(&cell.column, cell.row, Some(*v))
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
        }
        let pred = predict(hist.as_ref().unwrap_or(historic), &fut).map_err(Error::Model)?;
        cell_mean(&pred, target)
    };

    let explanation = explain_with(
        &value,
        &features.names,
        &features.instance,
        &background,
        opts,
    )?;
    Ok(ForecastExplanation {
        target: target.clone(),
        features: features.names,
        explanation,
    })
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
    fn predict(_: &DataFrame, future: &DataFrame) -> std::result::Result<DataFrame, String> {
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

    fn weight(e: &ForecastExplanation, name: &str) -> f64 {
        e.explanation
            .importances
            .iter()
            .find(|i| i.feature == name)
            .unwrap_or_else(|| panic!("no feature {name}"))
            .weight
    }

    fn cell(location: &str, time_period: &str) -> ForecastTarget {
        ForecastTarget {
            location: location.into(),
            time_period: time_period.into(),
        }
    }

    #[test]
    fn explains_a_forecast_cell_with_shap() {
        let historic = frame(&[
            ("2024-01", "A", 10.0, 20.0),
            ("2024-02", "A", 20.0, 22.0),
            ("2024-01", "B", 30.0, 24.0),
        ]);
        let future = frame(&[("2024-03", "A", 40.0, 25.0), ("2024-03", "B", 5.0, 30.0)]);
        let opts = Options {
            method: Method::Shap,
            ..Options::default()
        };
        let e = explain_forecast(
            &predict,
            &historic,
            &future,
            &cell("A", "2024-03"),
            &FeatureSpec::default(),
            "disease_cases",
            &opts,
        )
        .unwrap();
        assert_eq!(
            e.features,
            vec![
                "rainfall",
                "temperature",
                "rainfall@t-1",
                "temperature@t-1",
                "rainfall@t-2",
                "temperature@t-2"
            ]
        );
        assert!((e.explanation.prediction - (80.0 - 25.0)).abs() < 1e-9);
        assert!((weight(&e, "rainfall") - 2.0 * (40.0 - 20.0)).abs() < 1e-9);
        assert!((weight(&e, "temperature") - -(25.0 - 22.0)).abs() < 1e-9);
        assert!(weight(&e, "rainfall@t-1").abs() < 1e-9);
        assert!((e.explanation.r2.unwrap() - 1.0).abs() < 1e-9);
    }

    /// Recursive per location: y_t = 2 rain_t + 3 temp_t + 0.5 y_{t-1} + 10, from the last
    /// observed case count. The cell at the third step depends on seven inputs.
    fn recursive(
        historic: &DataFrame,
        future: &DataFrame,
    ) -> std::result::Result<DataFrame, String> {
        let err = |e: chapkit_core::FrameError| e.to_string();
        let cases = historic.f64_column("disease_cases").map_err(err)?;
        let mut prev = std::collections::HashMap::new();
        for r in rows_in_time_order(historic, "A") {
            prev.insert("A", cases[r]);
        }
        let rain = future.f64_column("rainfall").map_err(err)?;
        let temp = future.f64_column("temperature").map_err(err)?;
        let mut values = vec![0.0; future.n_rows()];
        for r in rows_in_time_order(future, "A") {
            let p = prev.get_mut("A").ok_or("no history")?;
            *p = 2.0 * rain[r] + 3.0 * temp[r] + 0.5 * *p + 10.0;
            values[r] = *p;
        }
        prediction_frame(future, 2, |row, out| {
            out.fill(values[row]);
            Ok(())
        })
        .map_err(|e| e.to_string())
    }

    fn recursive_frames() -> (DataFrame, DataFrame) {
        let mut rng = chapkit_core::rng::Rng::new(5);
        let mut h = Vec::new();
        for m in 1..=12 {
            h.push((
                format!("2024-{m:02}"),
                50.0 + 20.0 * rng.normal(),
                25.0 + 2.0 * rng.normal(),
                200.0 + 40.0 * rng.normal(),
            ));
        }
        let historic = DataFrame::from_columns([
            (
                "time_period",
                Column::from_strs(h.iter().map(|r| r.0.as_str())),
            ),
            ("location", Column::from_strs(h.iter().map(|_| "A"))),
            ("rainfall", Column::from_f64(h.iter().map(|r| r.1))),
            ("temperature", Column::from_f64(h.iter().map(|r| r.2))),
            ("disease_cases", Column::from_f64(h.iter().map(|r| r.3))),
        ])
        .unwrap();
        // Rows out of time order, to check the steps are ordered by period.
        let future = frame(&[
            ("2025-03", "A", 70.0, 27.0),
            ("2025-01", "A", 40.0, 24.0),
            ("2025-02", "A", 60.0, 26.0),
        ]);
        (historic, future)
    }

    #[test]
    fn temporal_lime_recovers_every_input_of_a_recursive_model() {
        let (historic, future) = recursive_frames();
        let spec = FeatureSpec {
            lags: 2,
            ..FeatureSpec::default()
        };
        let e = explain_forecast(
            &recursive,
            &historic,
            &future,
            &cell("A", "2025-03"),
            &spec,
            "disease_cases",
            &Options::default(),
        )
        .unwrap();
        assert_eq!(e.features.len(), 2 + 2 + 2 + 2 * 3);
        let sd = |c: &str| Background::stats(&historic.f64_column(c).unwrap()).1;
        let truth = [
            ("rainfall", 2.0 * sd("rainfall")),
            ("rainfall@t-1", 1.0 * sd("rainfall")),
            ("rainfall@t-2", 0.5 * sd("rainfall")),
            ("temperature", 3.0 * sd("temperature")),
            ("temperature@t-1", 1.5 * sd("temperature")),
            ("temperature@t-2", 0.75 * sd("temperature")),
            ("disease_cases@t-3", 0.125 * sd("disease_cases")),
            ("disease_cases@t-4", 0.0),
            ("rainfall@t-3", 0.0),
            ("temperature@t-4", 0.0),
        ];
        for (name, w) in truth {
            assert!(
                (weight(&e, name) - w).abs() < 1e-6,
                "{name}: {}",
                weight(&e, name)
            );
        }
        assert!((e.explanation.r2.unwrap() - 1.0).abs() < 1e-9);
        let n_eff = e.explanation.effective_n.unwrap();
        assert!(n_eff > 1.0 && n_eff < 500.0, "{n_eff}");
    }

    #[test]
    fn row_scope_perturbs_only_the_cell() {
        let (historic, future) = recursive_frames();
        let spec = FeatureSpec {
            scope: Scope::Row,
            ..FeatureSpec::default()
        };
        let e = explain_forecast(
            &recursive,
            &historic,
            &future,
            &cell("A", "2025-03"),
            &spec,
            "disease_cases",
            &Options::default(),
        )
        .unwrap();
        assert_eq!(e.features, vec!["rainfall", "temperature"]);
        assert!((e.explanation.r2.unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn first_step_has_only_history_lags() {
        let (historic, future) = recursive_frames();
        let e = explain_forecast(
            &recursive,
            &historic,
            &future,
            &cell("A", "2025-01"),
            &FeatureSpec {
                lags: 1,
                ..FeatureSpec::default()
            },
            "disease_cases",
            &Options::default(),
        )
        .unwrap();
        assert_eq!(
            e.features,
            vec![
                "rainfall",
                "temperature",
                "rainfall@t-1",
                "temperature@t-1",
                "disease_cases@t-1"
            ]
        );
        let sd = Background::stats(&historic.f64_column("disease_cases").unwrap()).1;
        assert!((weight(&e, "disease_cases@t-1") - 0.5 * sd).abs() < 1e-6);
    }

    #[test]
    fn unknown_cell_is_invalid_input() {
        let f = frame(&[("2024-01", "A", 1.0, 2.0), ("2024-02", "A", 2.0, 3.0)]);
        let r = explain_forecast(
            &predict,
            &f,
            &f,
            &cell("Z", "2024-01"),
            &FeatureSpec::default(),
            "disease_cases",
            &Options::default(),
        );
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }
}
