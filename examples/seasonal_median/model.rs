use std::collections::BTreeMap;

use chapkit::{DataFrame, Model, ModelInfo, NoConfig, PredictInput, Result, TrainInput};
use serde::{Deserialize, Serialize};

use super::empirical::{
    GlobalMedianSamples, baseline_info, frame_from_rows, future_rows, median_per_level,
    quantile_samples, series_by_location,
};

/// Fitted [`SeasonalMedian`] state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeasonalMedianTrained {
    /// Quantile samples per location and period of year (only observed ones).
    by_location: BTreeMap<String, BTreeMap<u32, Vec<f64>>>,
    global: GlobalMedianSamples,
}

impl SeasonalMedianTrained {
    /// Median per quantile level of every location's samples for the season.
    fn across_locations(&self, season: u32) -> Option<Vec<f64>> {
        median_per_level(self.by_location.values().filter_map(|m| m.get(&season)))
    }

    fn samples(&self, location: &str, season: u32) -> Result<Vec<f64>> {
        let own = self.by_location.get(location);
        if let Some(s) = own.and_then(|m| m.get(&season)) {
            return Ok(s.clone());
        }
        let observed = own.is_some_and(|m| !m.is_empty());
        if !observed && let Some(s) = self.across_locations(season) {
            return Ok(s);
        }
        self.global.for_location(location).map(<[f64]>::to_vec)
    }
}

/// Forecasts each period with all observations of the same location and period
/// of year. Port of chap-core `seasonal_median` (version 2).
///
/// Week 53 is pooled with week 52. An unobserved period of year falls back to the
/// global median; a location without observations gets the median of the other
/// locations' samples for the same period of year, else the global median.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeasonalMedian;

impl Model for SeasonalMedian {
    type Config = NoConfig;
    type Trained = SeasonalMedianTrained;

    fn info(&self) -> ModelInfo {
        baseline_info(
            "Baseline that uses all observed values for the same location and month or week of the year as \
             forecast samples, without smoothing or model fitting.",
        )
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<SeasonalMedianTrained> {
        let target = self.info().target;
        let mut by_location = BTreeMap::new();
        for (loc, series) in series_by_location(input.data, &target)? {
            let mut per_season: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
            for (p, v) in series.periods.iter().zip(&series.values) {
                if v.is_finite() {
                    per_season.entry(p.period_of_year()).or_default().push(*v);
                }
            }
            let samples = per_season
                .into_iter()
                .map(|(k, v)| Ok((k, quantile_samples(&v)?)))
                .collect::<Result<_>>()?;
            by_location.insert(loc, samples);
        }
        let global = GlobalMedianSamples::from_data(input.data, &target)?;
        Ok(SeasonalMedianTrained {
            by_location,
            global,
        })
    }

    fn predict(
        &self,
        trained: &SeasonalMedianTrained,
        _: &NoConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let mut rows = vec![Vec::new(); input.future.n_rows()];
        for (loc, entries) in future_rows(input.future)? {
            for (row, period) in entries {
                rows[row] = trained.samples(&loc, period.period_of_year())?;
            }
        }
        frame_from_rows(input.future, &rows)
    }
}
