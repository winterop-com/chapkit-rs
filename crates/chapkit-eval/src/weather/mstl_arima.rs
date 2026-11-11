//! chap-core's MSTL + ARIMA provider (`chap_core.assessment.weather_providers.mstl_arima`): per
//! location and covariate, the seasonal component from MSTL (one STL pass with statsmodels'
//! defaults and a seasonal window of 11), an ARIMA on the deseasonalised series with `d` from KPSS
//! and `(p, q)` in `0..=3` by AICc (candidates with a root within 1.001 of the unit circle
//! rejected; ARIMA(0,0,0) when none fits), and the seasonal component continued as the mean of
//! every complete cycle.

use chapkit_core::{Column, Error, Period, Result};
use chapkit_ts::statsmodels::SmArima;
use chapkit_ts::stl::Stl;
use chapkit_ts::unitroot::kpss_statsmodels;
use rayon::prelude::*;

use super::{WeatherInput, WeatherProvider};

const MAX_P: usize = 3;
const MAX_Q: usize = 3;
const MAX_D: usize = 2;
const MIN_ROOT_MODULUS: f64 = 1.001;
const MIN_SEASONAL_CYCLES: usize = 2;
/// MSTL's default seasonal window for one period (`7 + 4 * 1`).
const SEASONAL_WINDOW: usize = 11;

/// chap-core's MSTL + ARIMA future weather. Needs more than two full seasonal cycles of history;
/// a series with a missing value gets a missing forecast.
#[derive(Debug, Clone, Copy, Default)]
pub struct MstlArima {
    /// A fixed `(p, d, q)` instead of the search (chap-core's `order` parameter).
    pub order: Option<(usize, usize, usize)>,
    /// The seasonal period; by default 12 for months and 52 for weeks.
    pub seasonal_period: Option<usize>,
}

/// numpy's `allclose(x, x[0])`.
fn allclose(x: &[f64]) -> bool {
    x.first()
        .is_some_and(|b| x.iter().all(|a| (a - b).abs() <= 1e-8 + 1e-5 * b.abs()))
}

/// `_n_differences`: difference until KPSS (statsmodels, automatic lag) does not reject level
/// stationarity at 5%, at most twice; stop early for short or constant series.
pub(crate) fn n_differences(y: &[f64]) -> usize {
    let mut series = y.to_vec();
    for d in 0..MAX_D {
        if series.len() < 10 || allclose(&series) {
            return d;
        }
        match kpss_statsmodels(&series) {
            Some((_, p)) if p > 0.05 => return d,
            None => return d,
            Some(_) => {}
        }
        series = series.windows(2).map(|w| w[1] - w[0]).collect();
    }
    MAX_D
}

/// `_select_model`: `(p, q)` over `0..=3` with `d` from KPSS, by AICc; and whether it is the
/// ARIMA(0,0,0) fallback.
pub(crate) fn select_model(y: &[f64]) -> Result<(SmArima, bool)> {
    let d = n_differences(y);
    let mut best: Option<SmArima> = None;
    for p in 0..=MAX_P {
        for q in 0..=MAX_Q {
            let Ok(fitted) = SmArima::fit(y, p, d, q) else {
                continue;
            };
            if fitted.min_root_modulus() < MIN_ROOT_MODULUS {
                continue;
            }
            if fitted.aicc.is_finite() && best.as_ref().is_none_or(|b| fitted.aicc < b.aicc) {
                best = Some(fitted);
            }
        }
    }
    match best {
        Some(model) => Ok((model, false)),
        None => Ok((
            SmArima::fit(y, 0, 0, 0).map_err(|e| Error::model(e.to_string()))?,
            true,
        )),
    }
}

/// MSTL's seasonal component of `y`.
pub(crate) fn seasonal(y: &[f64], period: usize) -> Result<Vec<f64>> {
    Stl::statsmodels(period, SEASONAL_WINDOW)
        .fit(y)
        .map(|d| d.seasonal)
        .map_err(|e| Error::invalid(e.to_string()))
}

/// `_forecast_series`: the next `h` values of `y`.
pub(crate) fn forecast_series(
    y: &[f64],
    h: usize,
    period: usize,
    order: Option<(usize, usize, usize)>,
) -> Result<Vec<f64>> {
    if y.len() <= MIN_SEASONAL_CYCLES * period {
        return Err(Error::invalid(format!(
            "mstl_arima needs more than {MIN_SEASONAL_CYCLES} full seasonal cycles (more than {} \
             periods) of history, got {}. Use the 'climatology' provider for shorter series.",
            MIN_SEASONAL_CYCLES * period,
            y.len()
        )));
    }
    let seasonal = seasonal(y, period)?;
    let deseasonalised: Vec<f64> = y.iter().zip(&seasonal).map(|(y, s)| y - s).collect();
    let model = match order {
        Some((p, d, q)) => {
            SmArima::fit(&deseasonalised, p, d, q).map_err(|e| Error::model(e.to_string()))?
        }
        None => select_model(&deseasonalised)?.0,
    };
    let trend = model.forecast(h);
    // The seasonal component continued as the mean of every complete cycle, the last of which
    // ends with the history.
    let n_cycles = seasonal.len() / period;
    let tail = &seasonal[seasonal.len() - n_cycles * period..];
    let means: Vec<f64> = (0..period)
        .map(|j| (0..n_cycles).map(|c| tail[c * period + j]).sum::<f64>() / n_cycles as f64)
        .collect();
    Ok(trend
        .iter()
        .enumerate()
        .map(|(i, t)| t + means[i % period])
        .collect())
}

impl MstlArima {
    fn period(&self, input: &WeatherInput<'_>) -> Result<usize> {
        if let Some(p) = self.seasonal_period {
            return Ok(p);
        }
        match input.historic_periods.first() {
            Some(Period::Month { .. }) => Ok(12),
            Some(Period::Week { .. }) => Ok(52),
            _ => Err(Error::invalid(
                "mstl_arima does not know the seasonal period of this data",
            )),
        }
    }
}

impl WeatherProvider for MstlArima {
    fn id(&self) -> &'static str {
        "mstl_arima"
    }

    fn future_weather(&self, input: &WeatherInput<'_>) -> Result<Vec<Column>> {
        let period = self.period(input)?;
        let h = input.periods.len();
        // Each location's history of each covariate, in period order.
        let mut series = Vec::with_capacity(input.covariates.len() * input.locations.len());
        for name in input.covariates {
            let col = input.historic.require(name)?;
            for (_, idx) in input.locations {
                let mut ordered: Vec<(Period, f64)> = idx
                    .iter()
                    .filter_map(|&r| {
                        let p = *input.historic_periods.get(r)?;
                        Some((p, col.f64_at(r).unwrap_or(f64::NAN)))
                    })
                    .collect();
                ordered.sort_by_key(|(p, _)| *p);
                series.push(ordered.into_iter().map(|(_, v)| v).collect::<Vec<f64>>());
            }
        }
        let forecasts: Vec<Vec<Option<f64>>> = series
            .par_iter()
            .map(|y| {
                if y.iter().any(|v| !v.is_finite()) {
                    return Ok(vec![None; h]);
                }
                Ok(forecast_series(y, h, period, self.order)?
                    .into_iter()
                    .map(|v| v.is_finite().then_some(v))
                    .collect())
            })
            .collect::<Result<_>>()?;
        let per_covariate = input.locations.len();
        if per_covariate == 0 {
            return Ok(vec![Column::Float(Vec::new()); input.covariates.len()]);
        }
        Ok(forecasts
            .chunks(per_covariate)
            .take(input.covariates.len())
            .map(|chunk| Column::Float(chunk.concat()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use std::time::Instant;

    use serde_json::Value;

    use super::*;

    fn nums(v: &Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN))
            .collect()
    }

    fn worst(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
            .fold(0.0, f64::max)
    }

    /// Series whose deseasonalised (and differenced) values are almost exactly predictable: the
    /// likelihood grows without bound as `sigma2` goes to 0, statsmodels stops wherever rounding
    /// and `maxiter = 50` leave it, and which candidates pass the root check is down to rounding.
    /// The constant series still forecasts the constant.
    const DEGENERATE: [&str; 4] = [
        "monthly_constant_40",
        "weekly_constant_120",
        "monthly_spike_short_26",
        "monthly_linear_short_26",
    ];

    /// Every stage of chap-core 2.7.0's provider, against `scripts/gen_mstl_arima_fixtures.py`.
    ///
    /// Measured (relative): the MSTL seasonal component to 6e-14 and `d` exactly on every series;
    /// on the others the chosen order exactly, and forecasts to 4e-6 where statsmodels' L-BFGS-B
    /// converged for the chosen model. Where it stopped at `maxiter = 50` instead (two series, near
    /// a unit root), the iterations follow rounding: chosen parameters to 0.12 and forecasts to
    /// 1.6e-3.
    #[test]
    fn stages_match_chap_core() {
        let path = format!("{}/fixtures/mstl_arima.json", env!("CARGO_MANIFEST_DIR"));
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let (mut ours, mut theirs) = (0.0, 0.0);
        for case in fixture["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let y = nums(&case["y"]);
            let period = case["period"].as_u64().unwrap() as usize;
            let h = case["horizon"].as_u64().unwrap() as usize;
            let seas = seasonal(&y, period).unwrap();
            let seas_err = worst(&seas, &nums(&case["seasonal"]));
            let des: Vec<f64> = y.iter().zip(&seas).map(|(y, s)| y - s).collect();
            let d = n_differences(&des);
            assert_eq!(d as u64, case["d"].as_u64().unwrap(), "{name}: d");
            let (mut aicc_err, mut param_err, mut agree) = (0.0f64, 0.0f64, 0);
            let candidates = case["candidates"].as_array().unwrap();
            for c in candidates {
                let (p, q) = (
                    c["p"].as_u64().unwrap() as usize,
                    c["q"].as_u64().unwrap() as usize,
                );
                let got = SmArima::fit(&des, p, d, q);
                if c.get("failed").is_some() {
                    agree += usize::from(got.is_err());
                    continue;
                }
                let Ok(got) = got else { continue };
                agree += 1;
                aicc_err = aicc_err.max(worst(&[got.aicc], &[c["aicc"].as_f64().unwrap()]));
                param_err = param_err.max(worst(&got.params, &nums(&c["params"])));
            }
            let (m, fallback) = select_model(&des).unwrap();
            let order = [m.p as u64, m.d as u64, m.q as u64];
            let want: Vec<u64> = case["order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            let chosen_params = if order.to_vec() == want {
                worst(&m.params, &nums(&case["params"]))
            } else {
                f64::NAN
            };
            let trend_err = worst(&m.forecast(h), &nums(&case["trend_forecast"]));
            let start = Instant::now();
            let fc = forecast_series(&y, h, period, None).unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            let fc_err = worst(&fc, &nums(&case["forecast"]));
            ours += elapsed;
            theirs += case["seconds"].as_f64().unwrap();
            eprintln!(
                "{name}: seasonal {seas_err:.1e} d {d} candidates {agree}/{} aicc {aicc_err:.1e} \
                 params {param_err:.1e} order {order:?} (chap-core {want:?}) chosen params \
                 {chosen_params:.1e} trend {trend_err:.1e} forecast {fc_err:.1e} \
                 {elapsed:.3}s (chap-core {:.2}s)",
                candidates.len(),
                case["seconds"].as_f64().unwrap()
            );
            assert_eq!(fallback, case["fallback"].as_bool().unwrap());
            assert!(seas_err < 1e-12, "{name}: seasonal {seas_err}");
            assert!(fc.iter().all(|v| v.is_finite()), "{name}");
            if name.contains("constant") {
                assert!(fc_err < 1e-6, "{name}: forecast {fc_err}");
            }
            if DEGENERATE.contains(&name) {
                continue;
            }
            assert_eq!(order.to_vec(), want, "{name}: order");
            let chosen = candidates
                .iter()
                .find(|c| c["p"].as_u64() == Some(want[0]) && c["q"].as_u64() == Some(want[2]))
                .unwrap();
            let converged = chosen["converged"].as_bool().unwrap();
            let tol = if converged { 1e-5 } else { 5e-3 };
            assert!(fc_err < tol, "{name}: forecast {fc_err}");
        }
        eprintln!("total {ours:.3}s (chap-core {theirs:.2}s)");
        for e in fixture["errors"].as_array().unwrap() {
            let y = nums(&e["y"]);
            let period = e["period"].as_u64().unwrap() as usize;
            let err = forecast_series(&y, 3, period, None)
                .unwrap_err()
                .to_string();
            assert!(err.contains("more than 2 full seasonal cycles"), "{err}");
        }
    }

    #[test]
    fn missing_values_give_missing_forecasts() {
        use chapkit_core::DataFrame;
        let n = 40;
        let periods: Vec<Period> = (0..n)
            .map(|i| Period::parse(&format!("{}-{:02}", 2018 + i / 12, i % 12 + 1)).unwrap())
            .collect();
        let mut historic = DataFrame::new();
        let values: Vec<f64> = (0..n)
            .map(|i| 10.0 + (f64::from(i) * 0.5).sin() + if i == 5 { f64::NAN } else { 0.0 })
            .collect();
        historic
            .push_column("rainfall", Column::from_f64(values))
            .unwrap();
        let locations = vec![("A".to_owned(), (0..n as usize).collect::<Vec<_>>())];
        let future: Vec<Period> = periods[..2].to_vec();
        let covariates = vec!["rainfall".to_owned()];
        let input = WeatherInput {
            historic: &historic,
            historic_periods: &periods,
            covariates: &covariates,
            locations: &locations,
            periods: &future,
            observed: None,
        };
        let cols = MstlArima::default().future_weather(&input).unwrap();
        assert_eq!(cols, vec![Column::Float(vec![None, None])]);
    }
}
