//! Future weather: the covariates a model is given for the periods it forecasts, which do not
//! exist yet at prediction time. chap-core's `chap_core.assessment.weather_providers`.
//!
//! As in chap-core, a provider that does not leak sees only the history, and only the seasonal
//! numeric covariates; `population` and non-numeric columns are carried forward from each
//! location's last observation. A provider that leaks ([`Observed`]) is given the window's own
//! observations instead. [`DampedPersistence`] adds the last anomaly from the climatology, damped
//! with the horizon.
//!
//! Not ported: chap-core's MSTL + ARIMA provider. It forecasts each series with statsforecast's
//! AutoARIMA, a stepwise search over ARIMA orders with its own fitting and selection rules; a
//! port with parity is a large piece of work of its own.

use std::collections::HashMap;

use chapkit_core::{Column, DataFrame, Error, Period, Result, cols};

use crate::mask::MASK_INDICATOR;

/// Numeric covariates without a seasonal signal, carried forward instead of forecast.
const NON_SEASONAL: [&str; 1] = ["population"];

/// What a provider is asked for.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct WeatherInput<'a> {
    /// The history (every column; the target and mask indicator are never asked for).
    pub historic: &'a DataFrame,
    /// Each historic row's period.
    pub historic_periods: &'a [Period],
    /// The covariates to provide, in order.
    pub covariates: &'a [String],
    /// The locations, in output order, with their rows in `historic`.
    pub locations: &'a [(String, Vec<usize>)],
    /// The window's periods.
    pub periods: &'a [Period],
    /// The window's observations, target included. Given only to providers that leak.
    pub observed: Option<&'a DataFrame>,
}

/// A source of future covariates.
pub trait WeatherProvider: Send + Sync {
    /// chap-core's provider id (`climatology`, `damped_persistence`, `observed`).
    fn id(&self) -> &'static str;

    /// True when the provider reads the window's own observations (look-ahead). Only then is it
    /// given [`WeatherInput::observed`], and it is given every covariate rather than only the
    /// seasonal numeric ones.
    fn leaks_future_data(&self) -> bool {
        false
    }

    /// One column per [`WeatherInput::covariates`], with a row per location and period,
    /// location-major (all periods of the first location, then the next).
    fn future_weather(&self, input: &WeatherInput<'_>) -> Result<Vec<Column>>;
}

/// The provider with chap-core's id, if ported.
pub fn weather_provider(id: &str) -> Option<Box<dyn WeatherProvider>> {
    match id {
        "climatology" => Some(Box::new(Climatology)),
        "observed" => Some(Box::new(Observed)),
        "damped_persistence" => Some(Box::new(DampedPersistence::default())),
        _ => None,
    }
}

/// The weather observed in the window: perfect foresight, for diagnostics only. Scores are not
/// comparable to what a model can achieve in production.
#[derive(Debug, Clone, Copy, Default)]
pub struct Observed;

impl WeatherProvider for Observed {
    fn id(&self) -> &'static str {
        "observed"
    }

    fn leaks_future_data(&self) -> bool {
        true
    }

    fn future_weather(&self, input: &WeatherInput<'_>) -> Result<Vec<Column>> {
        let observed = input
            .observed
            .ok_or_else(|| Error::invalid("observed weather needs the window's observations"))?;
        let (tp, loc) = (
            observed.require(cols::TIME_PERIOD)?,
            observed.require(cols::LOCATION)?,
        );
        let mut at: HashMap<(&str, Period), usize> = HashMap::new();
        for row in 0..observed.n_rows() {
            if let (Some(l), Some(p)) = (loc.str_at(row), tp.str_at(row).and_then(Period::parse)) {
                at.entry((l, p)).or_insert(row);
            }
        }
        let mut idx = Vec::with_capacity(input.locations.len() * input.periods.len());
        for (location, _) in input.locations {
            for p in input.periods {
                idx.push(*at.get(&(location.as_str(), *p)).ok_or_else(|| {
                    Error::invalid(format!("no observed weather for {location} in {p}"))
                })?);
            }
        }
        input
            .covariates
            .iter()
            .map(|name| {
                observed
                    .require(name)?
                    .take(&idx)
                    .ok_or_else(|| Error::invalid("observed weather row out of range"))
            })
            .collect()
    }
}

/// chap-core's default: per location and covariate, the mean of each month (or week) of the year
/// in the history. Week 53 counts as week 52. When some month or week was never observed, chap-core
/// fits a least-squares regression on the one-hot seasons instead, whose minimum-norm solution
/// predicts each observed season's mean and, for a season never observed, the unweighted mean of
/// the observed seasons' means; that is what is computed here.
#[derive(Debug, Clone, Copy, Default)]
pub struct Climatology;

/// A location's seasonal fit of one covariate: the mean of each observed season, and for the
/// seasons never observed the unweighted mean of the observed seasons' means.
struct SeasonalFit {
    means: Vec<f64>,
}

impl SeasonalFit {
    fn new(
        input: &WeatherInput<'_>,
        col: &Column,
        name: &str,
        location: &str,
        idx: &[usize],
    ) -> Result<Self> {
        let n_seasons = input
            .historic_periods
            .first()
            .map_or(12, |p| p.periods_per_year() as usize);
        let mut sum = vec![0.0; n_seasons];
        let mut count = vec![0usize; n_seasons];
        for &r in idx {
            let (Some(v), Some(p)) = (col.f64_at(r), input.historic_periods.get(r)) else {
                continue;
            };
            if v.is_nan() {
                continue;
            }
            if let (Some(s), Some(c)) = (sum.get_mut(season(p)), count.get_mut(season(p))) {
                *s += v;
                *c += 1;
            }
        }
        let observed: Vec<f64> = sum
            .iter()
            .zip(&count)
            .filter(|(_, c)| **c > 0)
            .map(|(s, c)| s / *c as f64)
            .collect();
        if observed.is_empty() {
            return Err(Error::invalid(format!(
                "cannot fit a climatology for '{name}' in {location}: every value is missing"
            )));
        }
        let fallback = observed.iter().sum::<f64>() / observed.len() as f64;
        let means = sum
            .iter()
            .zip(&count)
            .map(|(s, c)| if *c > 0 { s / *c as f64 } else { fallback })
            .collect();
        Ok(Self { means })
    }

    fn at(&self, p: &Period) -> f64 {
        self.means.get(season(p)).copied().unwrap_or(f64::NAN)
    }
}

/// A period's season: 0-based month or week of year (week 53 pooled with 52).
fn season(p: &Period) -> usize {
    (p.period_of_year() as usize).saturating_sub(1)
}

impl WeatherProvider for Climatology {
    fn id(&self) -> &'static str {
        "climatology"
    }

    fn future_weather(&self, input: &WeatherInput<'_>) -> Result<Vec<Column>> {
        let rows = input.locations.len() * input.periods.len();
        let mut out = Vec::with_capacity(input.covariates.len());
        for name in input.covariates {
            let col = input.historic.require(name)?;
            let mut values = Vec::with_capacity(rows);
            for (location, idx) in input.locations {
                let fit = SeasonalFit::new(input, col, name, location, idx)?;
                values.extend(input.periods.iter().map(|p| Some(fit.at(p))));
            }
            out.push(Column::Float(values));
        }
        Ok(out)
    }
}

/// chap-core's damped persistence: the climatology plus the last anomaly from it, damped toward
/// zero with the horizon, `climatology(T+h) + phi^h * (y(T) - climatology(T))`. `phi` is the
/// lag-1 autocorrelation of the location's anomaly series, clipped to [0, 1] (0 for fewer than
/// three periods), unless [`DampedPersistence::damping`] fixes it. As in chap-core, a missing
/// value anywhere in the history makes the forecast missing.
#[derive(Debug, Clone, Copy, Default)]
pub struct DampedPersistence {
    /// A fixed `phi` (chap-core's `damping` parameter); `None` estimates it per series.
    pub damping: Option<f64>,
}

/// chap-core's `_lag1_autocorrelation`, `NaN` in, `NaN` out.
fn lag1_autocorrelation(x: &[f64]) -> f64 {
    if x.len() < 3 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let centred: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let denominator: f64 = centred.iter().map(|c| c * c).sum();
    if denominator <= 0.0 {
        return 0.0;
    }
    let numerator: f64 = centred.windows(2).map(|w| w[0] * w[1]).sum();
    let r = numerator / denominator;
    if r.is_nan() { r } else { r.clamp(0.0, 1.0) }
}

impl WeatherProvider for DampedPersistence {
    fn id(&self) -> &'static str {
        "damped_persistence"
    }

    fn future_weather(&self, input: &WeatherInput<'_>) -> Result<Vec<Column>> {
        let rows = input.locations.len() * input.periods.len();
        let mut out = Vec::with_capacity(input.covariates.len());
        for name in input.covariates {
            let col = input.historic.require(name)?;
            let mut values = Vec::with_capacity(rows);
            for (location, idx) in input.locations {
                let fit = SeasonalFit::new(input, col, name, location, idx)?;
                let mut ordered: Vec<(Period, f64)> = idx
                    .iter()
                    .filter_map(|&r| {
                        let p = *input.historic_periods.get(r)?;
                        Some((p, col.f64_at(r).unwrap_or(f64::NAN)))
                    })
                    .collect();
                ordered.sort_by_key(|(p, _)| *p);
                let anomaly: Vec<f64> = ordered.iter().map(|(p, v)| v - fit.at(p)).collect();
                let phi = self
                    .damping
                    .unwrap_or_else(|| lag1_autocorrelation(&anomaly));
                let last = anomaly.last().copied().unwrap_or(f64::NAN);
                values.extend(input.periods.iter().enumerate().map(|(h, p)| {
                    let v = fit.at(p) + phi.powf((h + 1) as f64) * last;
                    (!v.is_nan()).then_some(v)
                }));
            }
            out.push(Column::Float(values));
        }
        Ok(out)
    }
}

fn is_numeric(col: &Column) -> bool {
    matches!(col, Column::Float(_) | Column::Int(_))
}

/// The frame a model is given for a window: `time_period`, `location` and every covariate of the
/// history (target and mask indicator left out), rows by location then period; with an all-zero
/// mask indicator when the history has one, as chap-core gives masked runs.
pub(crate) fn masked_future(
    provider: &dyn WeatherProvider,
    historic: &DataFrame,
    historic_periods: &[Period],
    labels: &[String],
    periods: &[Period],
    truth: &DataFrame,
) -> Result<DataFrame> {
    let locations = historic.group_indices(cols::LOCATION)?;
    let names: Vec<String> = historic
        .names()
        .iter()
        .filter(|n| {
            ![
                cols::TIME_PERIOD,
                cols::LOCATION,
                cols::DISEASE_CASES,
                MASK_INDICATOR,
            ]
            .contains(&n.as_str())
        })
        .cloned()
        .collect();
    let leaks = provider.leaks_future_data();
    let forecast: Vec<String> = if leaks {
        names.clone()
    } else {
        names
            .iter()
            .filter(|n| {
                !NON_SEASONAL.contains(&n.as_str()) && historic.column(n).is_some_and(is_numeric)
            })
            .cloned()
            .collect()
    };
    let input = WeatherInput {
        historic,
        historic_periods,
        covariates: &forecast,
        locations: &locations,
        periods,
        observed: leaks.then_some(truth),
    };
    let columns = provider.future_weather(&input)?;
    let rows = locations.len() * periods.len();
    if columns.len() != forecast.len() || columns.iter().any(|c| c.len() != rows) {
        return Err(Error::model(format!(
            "weather provider {} returned the wrong shape",
            provider.id()
        )));
    }
    let mut forecast_columns: HashMap<&str, Column> =
        forecast.iter().map(String::as_str).zip(columns).collect();
    // Carried forward: each location's last row in time, repeated over the window.
    let last: Vec<usize> = locations
        .iter()
        .flat_map(|(_, idx)| {
            let last = idx
                .iter()
                .copied()
                .max_by_key(|&r| historic_periods.get(r).copied())
                .unwrap_or_default();
            std::iter::repeat_n(last, periods.len())
        })
        .collect();
    let mut out = DataFrame::new();
    out.push_column(
        cols::TIME_PERIOD,
        Column::from_strs(
            locations
                .iter()
                .flat_map(|_| labels.iter().map(String::as_str)),
        ),
    )?;
    out.push_column(
        cols::LOCATION,
        Column::from_strs(
            locations
                .iter()
                .flat_map(|(l, _)| std::iter::repeat_n(l.as_str(), periods.len())),
        ),
    )?;
    for name in &names {
        let col = match forecast_columns.remove(name.as_str()) {
            Some(col) => col,
            None => historic
                .require(name)?
                .take(&last)
                .ok_or_else(|| Error::invalid("carried-forward row out of range"))?,
        };
        out.push_column(name.clone(), col)?;
    }
    if historic.has(MASK_INDICATOR) {
        out.push_column(MASK_INDICATOR, Column::from_f64(vec![0.0; rows]))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lag1_autocorrelation_as_chap_core() {
        assert_eq!(lag1_autocorrelation(&[1.0, 5.0]), 0.0);
        assert_eq!(lag1_autocorrelation(&[2.0, 2.0, 2.0]), 0.0);
        // Centred [-1, 0, 1]: no lag-1 correlation.
        assert_eq!(lag1_autocorrelation(&[1.0, 2.0, 3.0]), 0.0);
        // Alternating: negative, clipped to 0.
        assert_eq!(lag1_autocorrelation(&[1.0, -1.0, 1.0, -1.0]), 0.0);
        // Centred [-1.5, -0.5, 0.5, 1.5]: (0.75 - 0.25 + 0.75) / 5.
        assert!((lag1_autocorrelation(&[1.0, 2.0, 3.0, 4.0]) - 0.25).abs() < 1e-15);
        assert!(lag1_autocorrelation(&[1.0, f64::NAN, 3.0]).is_nan());
    }
}
