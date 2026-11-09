use std::collections::BTreeMap;

use chapkit::{DataFrame, Error, Model, ModelInfo, NoConfig, PredictInput, Result, TrainInput};
use serde::{Deserialize, Serialize};

use super::empirical::{
    baseline_info, frame_from_rows, future_rows, quantile_samples, series_by_location,
};

/// Ratios `(y_t + 1) / (y_{t-h} + 1)` by horizon `h` and the period of year of `t`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeasonalShifts {
    by_horizon: BTreeMap<usize, BTreeMap<u32, Vec<f64>>>,
}

impl SeasonalShifts {
    fn from_cases(cases: &[f64], seasons: &[u32], max_horizon: usize) -> Self {
        let mut by_horizon = BTreeMap::new();
        for h in 1..=max_horizon.min(cases.len().saturating_sub(1)) {
            let mut by_season: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
            for t in h..cases.len() {
                let r = (cases[t] + 1.0) / (cases[t - h] + 1.0);
                if r.is_finite() {
                    by_season.entry(seasons[t]).or_default().push(r);
                }
            }
            by_horizon.insert(h, by_season);
        }
        Self { by_horizon }
    }

    fn pooled<'a>(all: impl IntoIterator<Item = &'a SeasonalShifts>) -> Self {
        let mut by_horizon: BTreeMap<usize, BTreeMap<u32, Vec<f64>>> = BTreeMap::new();
        for s in all {
            for (h, by_season) in &s.by_horizon {
                let dst = by_horizon.entry(*h).or_default();
                for (season, ratios) in by_season {
                    dst.entry(*season).or_default().extend_from_slice(ratios);
                }
            }
        }
        Self { by_horizon }
    }

    /// Ratios ending in the season, else in the nearest seasons earlier and later in
    /// the year, else in every season. `None` when the horizon has no ratios.
    fn ratios(&self, season: u32, horizon: usize) -> Option<Vec<f64>> {
        let by_season = self.by_horizon.get(&horizon).filter(|m| !m.is_empty())?;
        if let Some(r) = by_season.get(&season) {
            return Some(r.clone());
        }
        let earlier = by_season.range(..season).next_back();
        let later = by_season.range(season + 1..).next();
        Some(match (earlier, later) {
            (Some((_, a)), Some((_, b))) => [a.as_slice(), b.as_slice()].concat(),
            _ => by_season.values().flatten().copied().collect(),
        })
    }
}

/// Fitted [`SeasonalShiftPersistence`] state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeasonalShiftTrained {
    by_location: BTreeMap<String, SeasonalShifts>,
    pooled: SeasonalShifts,
}

impl SeasonalShiftTrained {
    /// The location's ratios, else all locations', at the horizon or else the longest
    /// shorter one. Prefers candidates with more than one ratio; falls back to `[1]`.
    fn ratios(&self, location: &str, season: u32, horizon: usize) -> Vec<f64> {
        let candidates: Vec<&SeasonalShifts> = self
            .by_location
            .get(location)
            .into_iter()
            .chain(std::iter::once(&self.pooled))
            .collect();
        let mut single: Option<Option<Vec<f64>>> = None;
        for h in (1..=horizon).rev() {
            for s in &candidates {
                let r = s.ratios(season, h);
                if r.as_ref().is_some_and(|r| r.len() > 1) {
                    return r.unwrap_or_default();
                }
                if single.is_none() {
                    single = Some(r);
                }
            }
        }
        single.flatten().unwrap_or_else(|| vec![1.0])
    }
}

/// Forecasts the last observed value, scaled by how cases changed over the same
/// horizon and time of year in earlier years. Port of chap-core
/// `seasonal_shift_persistence` (version 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct SeasonalShiftPersistence;

impl Model for SeasonalShiftPersistence {
    type Config = NoConfig;
    type Trained = SeasonalShiftTrained;

    fn info(&self) -> ModelInfo {
        baseline_info(
            "Baseline that forecasts the last observed value for the same location, scaled by how cases changed \
             over the same horizon and time of year in the training data.",
        )
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<SeasonalShiftTrained> {
        let series = series_by_location(input.data, &self.info().target)?;
        let max_horizon = series
            .values()
            .find_map(|s| s.periods.first())
            .ok_or_else(|| Error::invalid("no training data"))?
            .periods_per_year() as usize;
        let by_location: BTreeMap<String, SeasonalShifts> = series
            .into_iter()
            .map(|(loc, s)| {
                let shifts = SeasonalShifts::from_cases(&s.values, &s.seasons(), max_horizon);
                (loc, shifts)
            })
            .collect();
        let pooled = SeasonalShifts::pooled(by_location.values());
        Ok(SeasonalShiftTrained {
            by_location,
            pooled,
        })
    }

    fn predict(
        &self,
        trained: &SeasonalShiftTrained,
        _: &NoConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let historic = series_by_location(input.historic, &self.info().target)?;
        let mut rows = vec![Vec::new(); input.future.n_rows()];
        for (loc, entries) in future_rows(input.future)? {
            let cases = historic
                .get(&loc)
                .map(|s| s.values.as_slice())
                .unwrap_or_default();
            let last = cases.iter().rposition(|v| v.is_finite()).ok_or_else(|| {
                Error::invalid(format!(
                    "no observed disease_cases for location '{loc}' in the historic data"
                ))
            })?;
            let last_value = cases[last];
            let since_last = cases.len() - last;
            for (k, (row, period)) in entries.into_iter().enumerate() {
                let ratios = trained.ratios(&loc, period.period_of_year(), since_last + k);
                rows[row] = quantile_samples(&ratios)?
                    .into_iter()
                    .map(|r| ((last_value + 1.0) * r - 1.0).max(0.0))
                    .collect();
            }
        }
        frame_from_rows(input.future, &rows)
    }
}
