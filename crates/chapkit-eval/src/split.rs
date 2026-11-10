//! chap-core's expanding-window splits (`train_test_generator`) and retraining points.
//!
//! With `n` periods (with gap filling on, the contiguous range from the first to the last), `split_idx = n - (periods + (splits - 1) * stride + 1)`. The train
//! set is every period up to `split_idx`; split `i` has the history up to `split_idx + i * stride`
//! and forecasts the `periods` periods after it. Rows are in the prepared data's order.

use std::collections::BTreeMap;

use chapkit_core::{DataFrame, Error, Period, Result, cols};
use rayon::prelude::*;

use crate::mask::{MASK_INDICATOR, check_mask_outside_window};
use crate::prepare::{Prepare, prepare};
use crate::weather::{WeatherProvider, masked_future};

/// Backtest settings, as chap-core's `BacktestParams` (same defaults), and how the dataset is
/// prepared (chap-core's way by default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// Periods forecast at each split (`n_periods`).
    pub periods: usize,
    /// Number of splits (`n_splits`).
    pub splits: usize,
    /// Periods between successive splits.
    pub stride: usize,
    /// Trainings, evenly spaced over the splits (`n_retrain`); 1 trains once.
    pub retrain: usize,
    /// Dataset preparation before splitting ([`prepare`]).
    pub prepare: Prepare,
    /// Leave out locations without an observed target before the test windows, as chap-core's
    /// REST backtest does (`validate_and_filter_dataset_for_evaluation`). Their observations
    /// still count in `crps_norm`'s range, as there.
    pub drop_untrained: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            periods: 3,
            splits: 7,
            stride: 1,
            retrain: 1,
            prepare: Prepare::default(),
            drop_untrained: true,
        }
    }
}

impl Options {
    /// Check the settings as chap-core's `BacktestParams` does.
    pub fn validate(&self) -> Result<()> {
        if self.periods == 0 || self.splits == 0 || self.stride == 0 || self.retrain == 0 {
            return Err(Error::invalid(
                "periods, splits, stride and retrain must each be at least 1",
            ));
        }
        if self.retrain > self.splits {
            return Err(Error::invalid("retrain cannot exceed splits"));
        }
        Ok(())
    }

    /// Periods covered by the test windows: `periods + (splits - 1) * stride`.
    pub fn window(&self) -> usize {
        self.periods + (self.splits - 1) * self.stride
    }
}

/// The splits at which the model is (re)trained: `{(j * splits) / retrain}` for `j < retrain`,
/// chap-core's `_retrain_split_indices`. Always includes split 0.
pub fn retrain_points(splits: usize, retrain: usize) -> Vec<usize> {
    let mut points: Vec<usize> = (0..retrain.max(1))
        .map(|j| j * splits / retrain.max(1))
        .collect();
    points.dedup();
    points
}

/// One test window.
#[derive(Debug, Clone)]
pub struct Split {
    /// Last period of the history, as written in the data.
    pub historic_last: String,
    /// The forecast periods, as written in the data.
    pub periods: Vec<String>,
    /// Every row up to and including `historic_last`.
    pub historic: DataFrame,
    /// What the model is given for the window: the covariates from the weather provider, without
    /// the target. Rows by location, then period.
    pub future: DataFrame,
    /// The window's rows as observed, target included.
    pub truth: DataFrame,
}

/// A dataset split for a backtest.
#[derive(Debug, Clone)]
pub struct Splits {
    /// Last training period, as written in the data.
    pub train_last: String,
    /// The first training's data (the same rows as the first split's history).
    pub train: DataFrame,
    /// The test windows, in order.
    pub splits: Vec<Split>,
    /// Locations left out: no observed target before the test windows ([`Options::drop_untrained`]).
    pub dropped: Vec<String>,
}

/// The periods of a dataset: sorted distinct periods and each row's position among them.
pub(crate) struct Panel {
    /// Sorted distinct periods; with `contiguous`, every period from the first to the last.
    pub periods: Vec<Period>,
    /// Each period as first written in the data.
    pub labels: Vec<String>,
    /// Each row's index into `periods`.
    pub row_period: Vec<usize>,
}

impl Panel {
    pub fn new(data: &DataFrame, contiguous: bool) -> Result<Panel> {
        let tp = data.require(cols::TIME_PERIOD)?;
        let mut parsed = Vec::with_capacity(data.n_rows());
        let mut labels: BTreeMap<Period, String> = BTreeMap::new();
        for row in 0..data.n_rows() {
            let s = tp.str_at(row).unwrap_or_default();
            let p = Period::parse(s).ok_or_else(|| {
                Error::invalid(format!(
                    "time_period {s:?} in row {row} is not a month or week"
                ))
            })?;
            labels.entry(p).or_insert_with(|| s.to_owned());
            parsed.push(p);
        }
        let (periods, labels): (Vec<Period>, Vec<String>) = labels.into_iter().unzip();
        if let (Some(first), Some(last)) = (periods.first(), periods.last())
            && std::mem::discriminant(first) != std::mem::discriminant(last)
        {
            return Err(Error::invalid("data mixes monthly and weekly periods"));
        }
        let (periods, labels) = match (contiguous, periods.first(), periods.last()) {
            (true, Some(&first), Some(&last)) => {
                let written: BTreeMap<Period, String> = periods.into_iter().zip(labels).collect();
                let all = crate::prepare::contiguous(first, last)?;
                let labels = all
                    .iter()
                    .map(|p| written.get(p).cloned().unwrap_or_else(|| p.to_string()))
                    .collect();
                (all, labels)
            }
            _ => (periods, labels),
        };
        let row_period = parsed
            .iter()
            .map(|p| periods.binary_search(p).unwrap_or_default())
            .collect();
        Ok(Panel {
            periods,
            labels,
            row_period,
        })
    }
}

/// Rows whose period index satisfies `keep`, with their periods.
fn rows(
    data: &DataFrame,
    panel: &Panel,
    keep: impl Fn(usize) -> bool,
) -> Result<(DataFrame, Vec<Period>)> {
    let idx: Vec<usize> = (0..data.n_rows())
        .filter(|&r| panel.row_period.get(r).is_some_and(|&p| keep(p)))
        .collect();
    let periods = idx
        .iter()
        .filter_map(|&r| panel.row_period.get(r).and_then(|&p| panel.periods.get(p)))
        .copied()
        .collect();
    Ok((data.take(&idx)?, periods))
}

/// Prepare `data` ([`Options::prepare`]) and split it as chap-core's `train_test_generator`,
/// filling each window's covariates from `weather`.
pub fn train_test_splits(
    data: &DataFrame,
    options: &Options,
    weather: &dyn WeatherProvider,
) -> Result<Splits> {
    options.validate()?;
    data.require(cols::LOCATION)?;
    data.require(cols::DISEASE_CASES)?;
    let prepared = prepare(data, &options.prepare)?;
    let data = &prepared;
    let panel = Panel::new(data, options.prepare.fill_gaps)?;
    let n = panel.periods.len();
    let window = options.window();
    let Some(split_idx) = n.checked_sub(window + 1) else {
        return Err(Error::invalid(format!(
            "{n} periods are too few for {} splits of {} periods with stride {}: need at least {}",
            options.splits,
            options.periods,
            options.stride,
            window + 1
        )));
    };
    if data.has(MASK_INDICATOR) {
        check_mask_outside_window(data, &panel, window)?;
    }
    let (kept, panel, dropped) = if options.drop_untrained {
        drop_untrained(data, panel, split_idx, options)?
    } else {
        (None, panel, Vec::new())
    };
    let data = kept.as_ref().unwrap_or(data);
    let label = |i: usize| panel.labels.get(i).cloned().unwrap_or_default();
    let (train, _) = rows(data, &panel, |p| p <= split_idx)?;
    let splits = (0..options.splits)
        .into_par_iter()
        .map(|i| {
            let last = split_idx + i * options.stride;
            let (historic, historic_periods) = rows(data, &panel, |p| p <= last)?;
            let (truth, _) = rows(data, &panel, |p| p > last && p <= last + options.periods)?;
            let window = last + 1..=last + options.periods;
            let labels: Vec<String> = window.clone().map(label).collect();
            let periods: Vec<Period> = window
                .filter_map(|p| panel.periods.get(p))
                .copied()
                .collect();
            let future = masked_future(
                weather,
                &historic,
                &historic_periods,
                &labels,
                &periods,
                &truth,
            )?;
            Ok(Split {
                historic_last: label(last),
                periods: labels,
                historic,
                future,
                truth,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Splits {
        train_last: label(split_idx),
        train,
        splits,
        dropped,
    })
}

/// chap-core's `validate_and_filter_dataset_for_evaluation`: keep the locations with an observed
/// target in a period up to `split_idx`; the rest are returned by name. `None` when every
/// location is kept.
fn drop_untrained(
    data: &DataFrame,
    panel: Panel,
    split_idx: usize,
    options: &Options,
) -> Result<(Option<DataFrame>, Panel, Vec<String>)> {
    let target = data.f64_column(cols::DISEASE_CASES)?;
    let mut dropped = Vec::new();
    let mut keep = Vec::with_capacity(data.n_rows());
    for (location, idx) in data.group_indices(cols::LOCATION)? {
        let trains = idx.iter().any(|&r| {
            panel.row_period.get(r).is_some_and(|&p| p <= split_idx)
                && target.get(r).is_some_and(|v| !v.is_nan())
        });
        if trains {
            keep.extend(idx);
        } else {
            dropped.push(location);
        }
    }
    if dropped.is_empty() {
        return Ok((None, panel, dropped));
    }
    if keep.is_empty() {
        return Err(Error::invalid(format!(
            "No org unit has target data left to train on: the evaluation window \
             (n_periods={}, n_splits={}, stride={}) covers the whole dataset.",
            options.periods, options.splits, options.stride
        )));
    }
    keep.sort_unstable();
    let row_period = keep
        .iter()
        .filter_map(|&r| panel.row_period.get(r))
        .copied()
        .collect();
    let panel = Panel {
        row_period,
        ..panel
    };
    Ok((Some(data.take(&keep)?), panel, dropped))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrain_points_as_chap_core() {
        assert_eq!(retrain_points(7, 1), [0]);
        assert_eq!(retrain_points(7, 2), [0, 3]);
        assert_eq!(retrain_points(7, 7), [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(retrain_points(5, 3), [0, 1, 3]);
    }

    #[test]
    fn options_are_checked() {
        assert!(Options::default().validate().is_ok());
        let o = Options {
            retrain: 8,
            ..Options::default()
        };
        assert!(o.validate().is_err());
        let o = Options {
            stride: 0,
            ..Options::default()
        };
        assert!(o.validate().is_err());
        assert_eq!(Options::default().window(), 9);
    }

    #[test]
    fn no_location_left_to_train_is_chap_cores_error() {
        use crate::weather::Climatology;
        let mut data = DataFrame::new();
        let periods: Vec<String> = (1..=6).map(|m| format!("2023-{m:02}")).collect();
        data.push_column(
            cols::TIME_PERIOD,
            chapkit_core::Column::from_strs(periods.iter().map(String::as_str)),
        )
        .unwrap();
        data.push_column(cols::LOCATION, chapkit_core::Column::from_strs(["A"; 6]))
            .unwrap();
        data.push_column(
            cols::DISEASE_CASES,
            chapkit_core::Column::Float(vec![None, None, None, Some(2.0), Some(3.0), None]),
        )
        .unwrap();
        let options = Options {
            periods: 2,
            splits: 2,
            ..Options::default()
        };
        let e = train_test_splits(&data, &options, &Climatology)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("No org unit has target data left to train on"),
            "{e}"
        );
        let kept = Options {
            drop_untrained: false,
            ..options
        };
        assert!(train_test_splits(&data, &kept, &Climatology).is_ok());
    }
}
