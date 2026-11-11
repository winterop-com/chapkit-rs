//! chap-core's metrics (`chap_core.assessment.metrics`), with its definitions and aggregation.
//!
//! Each forecast row (location, period, horizon) with an observed target is scored; rows without
//! an observation are left out. Point metrics (MAE, RMSE, MAPE) score the median of the samples.
//! A metric is aggregated over the rows by its mean, skipping undefined values (MAPE of a zero
//! observation), or for RMSE by the root of the mean square. The aggregate is chap-core's
//! `aggregate_metrics` as its REST API stores it: `crps_norm` divides by the range of every
//! observed target in the dataset, not only the test windows.

use std::collections::{BTreeMap, HashMap};

use chapkit_core::{DataFrame, Error, Period, Result, cols};
use serde::Serialize;

use crate::backtest::Forecast;

/// A chap-core metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(into = "&'static str")]
pub enum Metric {
    /// Continuous ranked probability score (order-statistic estimator).
    Crps,
    /// CRPS of `log1p` of the samples and the observation.
    CrpsLog1p,
    /// CRPS divided by the range of the observations.
    CrpsNorm,
    /// Absolute error of the median.
    Mae,
    /// Root mean squared error of the median.
    Rmse,
    /// Absolute percentage error of the median (undefined for a zero observation).
    Mape,
    /// Share of observations within the 10th-90th percentile of the samples.
    Coverage10_90,
    /// Share of observations within the 25th-75th percentile.
    Coverage25_75,
    /// Winkler score of the 10th-90th percentile interval.
    Winkler10_90,
    /// Winkler score of the 25th-75th percentile interval.
    Winkler25_75,
    /// Winkler score of the 10-90 interval of `log1p` values.
    Winkler10_90Log1p,
    /// Winkler score of the 25-75 interval of `log1p` values.
    Winkler25_75Log1p,
    /// Share of samples above the observation.
    RatioAboveTruth,
}

impl From<Metric> for &'static str {
    fn from(m: Metric) -> Self {
        m.id()
    }
}

impl Metric {
    /// Every metric, in display order.
    pub const ALL: [Metric; 13] = [
        Metric::Crps,
        Metric::CrpsLog1p,
        Metric::CrpsNorm,
        Metric::Mae,
        Metric::Rmse,
        Metric::Mape,
        Metric::Coverage10_90,
        Metric::Coverage25_75,
        Metric::Winkler10_90,
        Metric::Winkler25_75,
        Metric::Winkler10_90Log1p,
        Metric::Winkler25_75Log1p,
        Metric::RatioAboveTruth,
    ];

    /// chap-core's `metric_id`.
    pub fn id(self) -> &'static str {
        match self {
            Metric::Crps => "crps",
            Metric::CrpsLog1p => "crps_log1p",
            Metric::CrpsNorm => "crps_norm",
            Metric::Mae => "mae",
            Metric::Rmse => "rmse",
            Metric::Mape => "mape",
            Metric::Coverage10_90 => "coverage_10_90",
            Metric::Coverage25_75 => "coverage_25_75",
            Metric::Winkler10_90 => "winkler_score_10_90",
            Metric::Winkler25_75 => "winkler_score_25_75",
            Metric::Winkler10_90Log1p => "winkler_score_10_90_log1p",
            Metric::Winkler25_75Log1p => "winkler_score_25_75_log1p",
            Metric::RatioAboveTruth => "ratio_above_truth",
        }
    }

    /// chap-core's `metric_name`.
    pub fn name(self) -> &'static str {
        match self {
            Metric::Crps => "CRPS",
            Metric::CrpsLog1p => "CRPS (log1p)",
            Metric::CrpsNorm => "CRPS Normalized",
            Metric::Mae => "MAE",
            Metric::Rmse => "RMSE",
            Metric::Mape => "MAPE",
            Metric::Coverage10_90 => "Coverage 10-90",
            Metric::Coverage25_75 => "Coverage 25-75",
            Metric::Winkler10_90 => "Winkler Score 10-90",
            Metric::Winkler25_75 => "Winkler Score 25-75",
            Metric::Winkler10_90Log1p => "Winkler 10-90 (log1p)",
            Metric::Winkler25_75Log1p => "Winkler 25-75 (log1p)",
            Metric::RatioAboveTruth => "Ratio Above Truth",
        }
    }

    /// The metric with chap-core's `metric_id`.
    pub fn from_id(id: &str) -> Option<Metric> {
        Metric::ALL.into_iter().find(|m| m.id() == id)
    }

    fn index(self) -> usize {
        Metric::ALL
            .iter()
            .position(|m| *m == self)
            .unwrap_or_default()
    }
}

/// Scores of a backtest: the aggregate and the breakdowns, one value per [`Metric::ALL`] each
/// (`NaN` when undefined, as chap-core leaves such a metric out).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Scores {
    /// Forecast rows scored (those with an observation).
    pub rows: usize,
    /// Over every scored row.
    pub aggregate: Vec<f64>,
    /// By horizon (1 is the first forecast period), ascending.
    pub by_horizon: Vec<(usize, Vec<f64>)>,
    /// By location, sorted by name.
    pub by_location: Vec<(String, Vec<f64>)>,
}

impl Scores {
    /// The aggregate of one metric.
    pub fn get(&self, metric: Metric) -> f64 {
        self.aggregate
            .get(metric.index())
            .copied()
            .unwrap_or(f64::NAN)
    }
}

/// Observed targets by (location, period): non-null `disease_cases` rows, the first if repeated.
fn observations(data: &DataFrame) -> Result<HashMap<(String, Period), f64>> {
    let (tp, loc) = (
        data.require(cols::TIME_PERIOD)?,
        data.require(cols::LOCATION)?,
    );
    let target = data.f64_column(cols::DISEASE_CASES)?;
    let mut out = HashMap::with_capacity(data.n_rows());
    for (row, v) in target.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if let (Some(l), Some(p)) = (loc.str_at(row), tp.str_at(row).and_then(Period::parse)) {
            out.entry((l.to_owned(), p)).or_insert(*v);
        }
    }
    Ok(out)
}

/// numpy's default (linear) percentile of sorted values, computed as numpy does.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let virtual_index = (n - 1) as f64 * (q / 100.0);
    let previous = virtual_index.floor();
    let gamma = virtual_index - previous;
    let i = (previous as usize).min(n - 1);
    let (Some(&a), Some(&b)) = (sorted.get(i), sorted.get((i + 1).min(n - 1))) else {
        return f64::NAN;
    };
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// Order-statistic CRPS of sorted samples against `obs`.
fn crps(sorted: &[f64], obs: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let nf = n as f64;
    let term1 = sorted.iter().map(|x| (x - obs).abs()).sum::<f64>() / nf;
    let term2 = sorted
        .iter()
        .enumerate()
        .map(|(i, x)| (2.0 * (i + 1) as f64 - nf - 1.0) * x)
        .sum::<f64>()
        / (n * n) as f64;
    term1 - term2
}

fn coverage(sorted: &[f64], obs: f64, low: u32, high: u32) -> f64 {
    let (lo, hi) = (
        percentile(sorted, f64::from(low)),
        percentile(sorted, f64::from(high)),
    );
    if lo <= obs && obs <= hi { 1.0 } else { 0.0 }
}

fn winkler(sorted: &[f64], obs: f64, low: u32, high: u32) -> f64 {
    let (lo, hi) = (
        percentile(sorted, f64::from(low)),
        percentile(sorted, f64::from(high)),
    );
    let alpha = 1.0 - f64::from(high - low) / 100.0;
    let penalty = if obs < lo {
        lo - obs
    } else if obs > hi {
        obs - hi
    } else {
        0.0
    };
    (hi - lo) + (2.0 / alpha) * penalty
}

/// Every metric of one forecast row, `crps_norm` still undivided.
fn row_metrics(samples: &[f64], obs: f64) -> [f64; 13] {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = match (
        n % 2,
        sorted.get(n / 2),
        sorted.get((n / 2).wrapping_sub(1)),
    ) {
        (1, Some(m), _) => *m,
        (0, Some(b), Some(a)) => (a + b) / 2.0,
        _ => f64::NAN,
    };
    let logs: Vec<f64> = sorted.iter().map(|x| x.ln_1p()).collect();
    let log_obs = obs.ln_1p();
    let crps = crps(&sorted, obs);
    let abs_error = (median - obs).abs();
    let mut out = [f64::NAN; 13];
    for (m, slot) in Metric::ALL.iter().zip(out.iter_mut()) {
        *slot = match m {
            Metric::Crps | Metric::CrpsNorm => crps,
            Metric::CrpsLog1p => self::crps(&logs, log_obs),
            Metric::Mae | Metric::Rmse => abs_error,
            Metric::Mape if obs == 0.0 => f64::NAN,
            Metric::Mape => (obs - median).abs() / obs.abs() * 100.0,
            Metric::Coverage10_90 => coverage(&sorted, obs, 10, 90),
            Metric::Coverage25_75 => coverage(&sorted, obs, 25, 75),
            Metric::Winkler10_90 => winkler(&sorted, obs, 10, 90),
            Metric::Winkler25_75 => winkler(&sorted, obs, 25, 75),
            Metric::Winkler10_90Log1p => winkler(&logs, log_obs, 10, 90),
            Metric::Winkler25_75Log1p => winkler(&logs, log_obs, 25, 75),
            Metric::RatioAboveTruth => {
                sorted.iter().filter(|x| **x > obs).count() as f64 / n as f64
            }
        };
    }
    out
}

/// Aggregate detailed rows: pandas' mean skipping `NaN`, or the root mean square for RMSE.
fn aggregate<'a>(rows: &(impl Iterator<Item = &'a [f64; 13]> + Clone)) -> Vec<f64> {
    Metric::ALL
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let (mut sum, mut count) = (0.0, 0usize);
            for v in rows
                .clone()
                .filter_map(|r| r.get(i))
                .filter(|v| !v.is_nan())
            {
                sum += if *m == Metric::Rmse { v * v } else { *v };
                count += 1;
            }
            let mean = if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            };
            if *m == Metric::Rmse {
                mean.sqrt()
            } else {
                mean
            }
        })
        .collect()
}

/// Score `forecasts` against the observed targets in `data` (the whole dataset).
pub fn score(forecasts: &[Forecast], data: &DataFrame) -> Result<Scores> {
    let observed = observations(data)?;
    let (min, max) = observed
        .values()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    let range = max - min;
    let norm = Metric::CrpsNorm.index();
    let mut rows: Vec<(&str, usize, [f64; 13])> = Vec::with_capacity(forecasts.len());
    for f in forecasts {
        let period = Period::parse(&f.time_period).ok_or_else(|| {
            Error::invalid(format!(
                "forecast period {:?} is not a month or week",
                f.time_period
            ))
        })?;
        let Some(&obs) = observed.get(&(f.location.clone(), period)) else {
            continue;
        };
        let mut values = row_metrics(&f.samples, obs);
        if range > 0.0
            && let Some(v) = values.get_mut(norm)
        {
            *v /= range;
        }
        rows.push((f.location.as_str(), f.horizon, values));
    }
    let mut by_horizon: BTreeMap<usize, Vec<&[f64; 13]>> = BTreeMap::new();
    let mut by_location: BTreeMap<&str, Vec<&[f64; 13]>> = BTreeMap::new();
    for (location, horizon, values) in &rows {
        by_horizon.entry(*horizon).or_default().push(values);
        by_location.entry(location).or_default().push(values);
    }
    Ok(Scores {
        rows: rows.len(),
        aggregate: aggregate(&rows.iter().map(|(_, _, v)| v)),
        by_horizon: by_horizon
            .into_iter()
            .map(|(h, v)| (h, aggregate(&v.into_iter())))
            .collect(),
        by_location: by_location
            .into_iter()
            .map(|(l, v)| (l.to_owned(), aggregate(&v.into_iter())))
            .collect(),
    })
}

/// The median and the 10th, 25th, 75th and 90th percentiles of a forecast's samples, by numpy's
/// default (linear) method, as the metrics compute them. `NaN` samples sort last.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Quantiles {
    pub median: f64,
    pub q10: f64,
    pub q25: f64,
    pub q75: f64,
    pub q90: f64,
}

impl Quantiles {
    /// The quantiles of `samples` (`NaN` each when there are none).
    pub fn of(samples: &[f64]) -> Quantiles {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        Quantiles {
            median: percentile(&sorted, 50.0),
            q10: percentile(&sorted, 10.0),
            q25: percentile(&sorted, 25.0),
            q75: percentile(&sorted, 75.0),
            q90: percentile(&sorted, 90.0),
        }
    }
}

/// One forecast of a backtest without its samples: enough to draw it against the truth.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ForecastSummary {
    /// The split (0 is the first).
    pub split: usize,
    pub location: String,
    pub time_period: String,
    pub horizon: usize,
    /// The observed target, `None` when missing.
    pub observed: Option<f64>,
    #[serde(flatten)]
    pub quantiles: Quantiles,
}

/// Summarize `forecasts` against the observed targets of `data` (the backtest's dataset), in order.
pub fn summarize(forecasts: &[Forecast], data: &DataFrame) -> Result<Vec<ForecastSummary>> {
    let observed = observations(data)?;
    Ok(forecasts
        .iter()
        .map(|f| ForecastSummary {
            split: f.split,
            location: f.location.clone(),
            time_period: f.time_period.clone(),
            horizon: f.horizon,
            observed: Period::parse(&f.time_period)
                .and_then(|p| observed.get(&(f.location.clone(), p)))
                .copied(),
            quantiles: Quantiles::of(&f.samples),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_match_numpy_and_summaries_carry_the_truth() {
        // np.percentile([4, 1, 3, 2], [50, 10, 25, 75, 90]) = [2.5, 1.3, 1.75, 3.25, 3.7]
        let q = Quantiles::of(&[4.0, 1.0, 3.0, 2.0]);
        let want = [2.5, 1.3, 1.75, 3.25, 3.7];
        for (got, want) in [q.median, q.q10, q.q25, q.q75, q.q90].iter().zip(want) {
            assert!((got - want).abs() < 1e-12);
        }
        let data = DataFrame::from_columns(vec![
            (
                "time_period",
                chapkit_core::Column::from_strs(["2024-01", "2024-02"]),
            ),
            ("location", chapkit_core::Column::from_strs(["A", "A"])),
            (
                "disease_cases",
                chapkit_core::Column::Float(vec![Some(5.0), None]),
            ),
        ])
        .unwrap();
        let f = |p: &str| Forecast {
            split: 1,
            location: "A".into(),
            time_period: p.into(),
            horizon: 1,
            samples: vec![1.0, 2.0],
        };
        let s = summarize(&[f("2024-01"), f("2024-02")], &data).unwrap();
        assert_eq!(s[0].observed, Some(5.0));
        assert_eq!(s[1].observed, None);
        assert_eq!((s[1].split, s[1].quantiles.median), (1, 1.5));
    }

    #[test]
    fn percentile_matches_numpy() {
        // np.percentile([1, 2, 3, 4], [10, 25, 75, 90]) = [1.3, 1.75, 3.25, 3.7]
        let s = [1.0, 2.0, 3.0, 4.0];
        for (q, want) in [(10.0, 1.3), (25.0, 1.75), (75.0, 3.25), (90.0, 3.7)] {
            assert!((percentile(&s, q) - want).abs() < 1e-12, "{q}");
        }
        assert_eq!(percentile(&[5.0], 90.0), 5.0);
    }

    #[test]
    fn crps_of_a_point_forecast_is_the_absolute_error() {
        assert!((crps(&[3.0; 5], 1.0) - 2.0).abs() < 1e-12);
        // E|X - y| - E|X - X'| / 2 for X uniform on {0, 1}, y = 0: 0.5 - 0.25.
        assert!((crps(&[0.0, 1.0], 0.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn ids_round_trip() {
        for m in Metric::ALL {
            assert_eq!(Metric::from_id(m.id()), Some(m));
        }
    }
}
