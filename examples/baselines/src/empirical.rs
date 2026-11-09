//! Helpers shared by the baseline examples, which forecast with the empirical
//! distribution of observed values. Port of `chap_core/models/builtin/empirical.py`,
//! plus `GlobalMedianSamples`, which the seasonal median falls back on.
//!
//! Included by each example (and by the parity tests) with
//! `#[path = "../common/empirical.rs"] mod empirical;`.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use chapkit::{
    AssessedStatus, DataFrame, Error, ModelInfo, ModelMetadata, Period, Result, cols,
    prediction_frame,
};
use serde::{Deserialize, Serialize};

/// Samples per forecast, as in chap-core.
pub const N_SAMPLES: usize = 100;

/// 99 quantile levels symmetric around 0.5, plus 0.5 once more, sorted. The two
/// middle samples are then both the median, so the median of the samples is
/// exactly the median of the values whichever median definition a metric uses.
static QUANTILE_LEVELS: LazyLock<[f64; N_SAMPLES]> = LazyLock::new(|| {
    let mut levels = [0.5; N_SAMPLES];
    for (i, l) in levels.iter_mut().take(N_SAMPLES - 1).enumerate() {
        *l = (i as f64 + 0.5) / (N_SAMPLES - 1) as f64;
    }
    levels.sort_by(f64::total_cmp);
    levels
});

/// `N_SAMPLES` deterministic samples spread over `values` (numpy's default linear
/// quantile interpolation). `values` must be finite.
pub fn quantile_samples(values: &[f64]) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(Error::invalid("no values to take quantiles of"));
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let last = (v.len() - 1) as f64;
    Ok(QUANTILE_LEVELS
        .iter()
        .map(|&q| {
            let pos = q * last;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
        })
        .collect())
}

/// Median per position across equally long sample vectors (numpy `median(axis=0)`).
/// `None` when there are no vectors or their lengths differ.
pub fn median_per_level<'a>(samples: impl IntoIterator<Item = &'a Vec<f64>>) -> Option<Vec<f64>> {
    let samples: Vec<&Vec<f64>> = samples.into_iter().collect();
    let first = samples.first()?;
    if samples.iter().any(|s| s.len() != first.len()) {
        return None;
    }
    let mut col = Vec::with_capacity(samples.len());
    Some(
        (0..first.len())
            .map(|i| {
                col.clear();
                col.extend(samples.iter().map(|s| s[i]));
                col.sort_by(f64::total_cmp);
                let n = col.len();
                if n % 2 == 1 {
                    col[n / 2]
                } else {
                    f64::midpoint(col[n / 2 - 1], col[n / 2])
                }
            })
            .collect(),
    )
}

/// One location's target series, ordered by period.
#[derive(Debug, Clone, Default)]
pub struct Series {
    /// Periods in chronological order.
    pub periods: Vec<Period>,
    /// Target values (NaN when missing), aligned with `periods`.
    pub values: Vec<f64>,
}

impl Series {
    /// Period of year of each entry.
    pub fn seasons(&self) -> Vec<u32> {
        self.periods.iter().map(|p| p.period_of_year()).collect()
    }
}

/// Parse a period cell, failing with a "not applicable" error otherwise.
pub fn parse_period(df: &DataFrame, row: usize) -> Result<Period> {
    let raw = df
        .require(cols::TIME_PERIOD)?
        .str_at(row)
        .unwrap_or_default();
    Period::parse(raw).ok_or_else(|| {
        Error::invalid(format!(
            "seasonal baselines need monthly or weekly data, got period '{raw}'"
        ))
    })
}

/// The target series of each location, sorted by period. Locations are keyed in a
/// `BTreeMap` so that iteration (and any pooling) is deterministic.
pub fn series_by_location(df: &DataFrame, target: &str) -> Result<BTreeMap<String, Series>> {
    let y = df.f64_column(target)?;
    let mut out = BTreeMap::new();
    for (loc, idx) in df.group_indices(cols::LOCATION)? {
        let mut rows: Vec<(Period, f64)> = idx
            .iter()
            .map(|&i| Ok((parse_period(df, i)?, y[i])))
            .collect::<Result<_>>()?;
        rows.sort_by_key(|r| r.0);
        let (periods, values) = rows.into_iter().unzip();
        out.insert(loc, Series { periods, values });
    }
    Ok(out)
}

/// One location's future rows: `(location, [(row index, period)])`, sorted by period.
pub type FutureRows = (String, Vec<(usize, Period)>);

/// For each location in `future`: its row indices sorted by period, with the parsed periods.
pub fn future_rows(future: &DataFrame) -> Result<Vec<FutureRows>> {
    future
        .group_indices(cols::LOCATION)?
        .into_iter()
        .map(|(loc, idx)| {
            let mut rows: Vec<(usize, Period)> = idx
                .into_iter()
                .map(|i| Ok((i, parse_period(future, i)?)))
                .collect::<Result<_>>()?;
            rows.sort_by_key(|r| r.1);
            Ok((loc, rows))
        })
        .collect()
}

/// Model info shared by the baselines.
pub fn baseline_info(note: &str) -> ModelInfo {
    ModelInfo {
        model_metadata: ModelMetadata {
            author: Some("Morten Hansen".into()),
            author_note: Some(note.into()),
            author_assessed_status: Some(AssessedStatus::Green),
            contact_email: Some("morten@winterop.com".into()),
            ..ModelMetadata::default()
        },
        ..ModelInfo::default()
    }
}

/// Fill a prediction frame from per-row samples computed up front.
pub fn frame_from_rows(future: &DataFrame, rows: &[Vec<f64>]) -> Result<DataFrame> {
    prediction_frame(future, N_SAMPLES, |row, out| {
        let samples = rows
            .get(row)
            .ok_or_else(|| Error::model("missing samples for a future row"))?;
        if samples.len() != out.len() {
            return Err(Error::model("wrong number of samples for a future row"));
        }
        out.copy_from_slice(samples);
        Ok(())
    })
}

/// Quantile samples of all observed target values of each location.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalMedianSamples {
    by_location: BTreeMap<String, Vec<f64>>,
    /// Median per quantile level over all locations, for unseen locations.
    fallback: Option<Vec<f64>>,
}

impl GlobalMedianSamples {
    /// Fit from training data.
    pub fn from_data(data: &DataFrame, target: &str) -> Result<Self> {
        let mut by_location = BTreeMap::new();
        for (loc, s) in series_by_location_unordered(data, target)? {
            let observed: Vec<f64> = s.into_iter().filter(|v| v.is_finite()).collect();
            if !observed.is_empty() {
                by_location.insert(loc, quantile_samples(&observed)?);
            }
        }
        let fallback = median_per_level(by_location.values());
        Ok(Self {
            by_location,
            fallback,
        })
    }

    /// The location's samples, or the median of all locations' samples when it has none.
    pub fn for_location(&self, location: &str) -> Result<&[f64]> {
        self.by_location
            .get(location)
            .or(self.fallback.as_ref())
            .map(Vec::as_slice)
            .ok_or_else(|| {
                Error::invalid("no observed disease_cases for any location in the training data")
            })
    }
}

/// Global median does not need periods, so it accepts any period format.
fn series_by_location_unordered(data: &DataFrame, target: &str) -> Result<Vec<(String, Vec<f64>)>> {
    let y = data.f64_column(target)?;
    Ok(data
        .group_indices(cols::LOCATION)?
        .into_iter()
        .map(|(loc, idx)| (loc, idx.iter().map(|&i| y[i]).collect()))
        .collect())
}
