//! Working location by location.
//!
//! Most Chap models treat every location on its own: split the training table into one
//! time-ordered series per location, compute something per location, and for each future row
//! produce forecast samples. These helpers do the splitting, run the locations in parallel and
//! assemble the prediction table, so a model only writes the per-location calculation.

use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::{Column, DataFrame, Error, Period, Result, cols, prediction_frame};

/// One location's target series, ordered by period.
#[derive(Debug, Clone, Default)]
pub struct Series {
    /// Periods in chronological order.
    pub periods: Vec<Period>,
    /// Target values (NaN when missing), aligned with `periods`.
    pub values: Vec<f64>,
}

impl Series {
    /// Period of year (month 1-12, week 1-52; week 53 counts as 52) of each entry.
    pub fn seasons(&self) -> Vec<u32> {
        self.periods.iter().map(|p| p.period_of_year()).collect()
    }
}

/// The `target` series of each location, sorted by period. Locations are processed in parallel
/// and keyed in a `BTreeMap`, so iteration order is deterministic. Fails on a period that is not
/// monthly or weekly.
pub fn series_by_location(df: &DataFrame, target: &str) -> Result<BTreeMap<String, Series>> {
    let y = df.f64_column(target)?;
    let tp = df.require(cols::TIME_PERIOD)?;
    df.group_indices(cols::LOCATION)?
        .into_par_iter()
        .map(|(loc, idx)| {
            let mut rows: Vec<(Period, f64)> = idx
                .iter()
                .map(|&i| Ok((period_at(tp, i)?, y.get(i).copied().unwrap_or(f64::NAN))))
                .collect::<Result<_>>()?;
            rows.sort_by_key(|r| r.0);
            let (periods, values) = rows.into_iter().unzip();
            Ok((loc, Series { periods, values }))
        })
        .collect()
}

/// The `target` values of each location in row order, for models that do not need periods
/// (any period format is accepted).
pub fn values_by_location(df: &DataFrame, target: &str) -> Result<Vec<(String, Vec<f64>)>> {
    let y = df.f64_column(target)?;
    Ok(df
        .group_indices(cols::LOCATION)?
        .into_iter()
        .map(|(loc, idx)| (loc, idx.iter().filter_map(|&i| y.get(i).copied()).collect()))
        .collect())
}

/// A future row: its row index in the future table, and its period.
pub type FutureRow = (usize, Period);

/// For each location in `future`: its rows sorted by period.
pub fn future_by_location(future: &DataFrame) -> Result<Vec<(String, Vec<FutureRow>)>> {
    let tp = future.require(cols::TIME_PERIOD)?;
    future
        .group_indices(cols::LOCATION)?
        .into_par_iter()
        .map(|(loc, idx)| {
            let mut rows: Vec<FutureRow> = idx
                .into_iter()
                .map(|i| Ok((i, period_at(tp, i)?)))
                .collect::<Result<_>>()?;
            rows.sort_by_key(|r| r.1);
            Ok((loc, rows))
        })
        .collect()
}

/// The prediction table (`time_period`, `location`, `sample_0..`) for `future`, computed location
/// by location in parallel. `forecast` gets a location and its future rows sorted by period, and
/// returns `n_samples` samples for each of those rows, in the same order.
pub fn predict_by_location<F>(
    future: &DataFrame,
    n_samples: usize,
    forecast: F,
) -> Result<DataFrame>
where
    F: Fn(&str, &[FutureRow]) -> Result<Vec<Vec<f64>>> + Sync,
{
    crate::prediction::check_prediction_cells(future.n_rows(), n_samples)?;
    let per_location: Vec<Vec<(usize, Vec<f64>)>> = future_by_location(future)?
        .par_iter()
        .map(|(loc, rows)| {
            let samples = forecast(loc, rows)?;
            if samples.len() != rows.len() {
                return Err(Error::model(format!(
                    "forecast for '{loc}' returned {} rows, expected {}",
                    samples.len(),
                    rows.len()
                )));
            }
            Ok(rows.iter().map(|(row, _)| *row).zip(samples).collect())
        })
        .collect::<Result<_>>()?;
    let mut by_row = vec![Vec::new(); future.n_rows()];
    for (row, samples) in per_location.into_iter().flatten() {
        if let Some(slot) = by_row.get_mut(row) {
            *slot = samples;
        }
    }
    samples_frame(future, n_samples, &by_row)
}

/// The prediction table for `future` from samples computed up front, one list per future row.
pub fn samples_frame(
    future: &DataFrame,
    n_samples: usize,
    by_row: &[Vec<f64>],
) -> Result<DataFrame> {
    prediction_frame(future, n_samples, |row, out| {
        let samples = by_row
            .get(row)
            .ok_or_else(|| Error::model("missing samples for a future row"))?;
        if samples.len() != out.len() {
            return Err(Error::model(format!(
                "{} samples for a future row, expected {}",
                samples.len(),
                out.len()
            )));
        }
        out.copy_from_slice(samples);
        Ok(())
    })
}

fn period_at(col: &Column, row: usize) -> Result<Period> {
    let raw = col.str_at(row).unwrap_or_default();
    Period::parse(raw)
        .ok_or_else(|| Error::invalid(format!("expected monthly or weekly periods, got '{raw}'")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn splits_sorts_and_reassembles_by_location() {
        let df: DataFrame = serde_json::from_value(json!({
            "columns": ["time_period", "location", "disease_cases"],
            "data": [["2024-02", "B", 4], ["2024-01", "A", 1], ["2024-02", "A", null], ["2024-01", "B", 3]]
        }))
        .unwrap();
        let s = series_by_location(&df, "disease_cases").unwrap();
        assert_eq!(s["A"].values[0], 1.0);
        assert!(s["A"].values[1].is_nan());
        assert_eq!(s["B"].values, [3.0, 4.0]);
        assert_eq!(s["B"].seasons(), [1, 2]);

        // Each location forecasts its last value; rows land where they were in `future`.
        let out = predict_by_location(&df, 2, |loc, rows| {
            let last = s[loc]
                .values
                .iter()
                .rev()
                .copied()
                .find(|v| v.is_finite())
                .unwrap_or(0.0);
            Ok(rows.iter().map(|_| vec![last, last]).collect())
        })
        .unwrap();
        assert_eq!(out.f64_column("sample_1").unwrap(), [4.0, 1.0, 1.0, 4.0]);

        let bad: DataFrame = serde_json::from_value(json!({
            "columns": ["time_period", "location", "disease_cases"], "data": [["2024", "A", 1]]
        }))
        .unwrap();
        assert!(series_by_location(&bad, "disease_cases").is_err());
        assert_eq!(
            values_by_location(&bad, "disease_cases").unwrap()[0].1,
            [1.0]
        );
    }
}
