//! ARIMA as fable fits it: R's `stats::arima` underneath ([`Method`]), fable's
//! `ARIMA(y ~ pdq() + PDQ())` order search on top ([`auto_arima`]), and forecasts with their
//! variance as fable's `forecast()` gives them (R's `predict.Arima`).
//!
//! ```
//! use chapkit_ts::{AutoArima, auto_arima};
//!
//! let y: Vec<f64> = (0..60)
//!     .map(|i| 10.0 + 3.0 * (i as f64 * std::f64::consts::PI / 6.0).sin() + (i % 7) as f64 * 0.3)
//!     .collect();
//! let model = auto_arima(&y, &AutoArima::new(12)).unwrap();
//! let fc = model.forecast(3).unwrap();
//! assert_eq!(fc.mean.len(), 3);
//! // Same orders and coefficients on a longer history (fable's `refit()`):
//! let later = model.filter(&y).unwrap().forecast(3).unwrap();
//! assert_eq!(later.variance.len(), 3);
//! ```

mod auto;
pub(crate) mod kalman;
mod stats;

use serde::{Deserialize, Serialize};

pub use auto::{AutoArima, auto_arima};
use kalman::StateSpace;
pub use stats::Method;

use crate::poly::roots_outside;
use crate::{Error, Result};

/// An ARIMA(p,d,q)(P,D,Q)[period] specification, with or without fable's constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArimaSpec {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub seasonal_p: usize,
    pub seasonal_d: usize,
    pub seasonal_q: usize,
    /// The seasonal period (1 for none).
    pub period: usize,
    /// fable's constant: a mean for `d + D = 0`, a drift for `d + D = 1` (a regressor of
    /// `arima_constant`, R's `diffinv` of ones).
    pub constant: bool,
}

impl ArimaSpec {
    fn arma(&self) -> stats::Arma {
        [
            self.p,
            self.q,
            self.seasonal_p,
            self.seasonal_q,
            self.period,
            self.d,
            self.seasonal_d,
        ]
    }

    fn n_coef(&self) -> usize {
        self.p + self.q + self.seasonal_p + self.seasonal_q + usize::from(self.constant)
    }
}

impl std::fmt::Display for ArimaSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ARIMA({},{},{})({},{},{})[{}]{}",
            self.p,
            self.d,
            self.q,
            self.seasonal_p,
            self.seasonal_d,
            self.seasonal_q,
            self.period,
            if self.constant { " w/ constant" } else { "" }
        )
    }
}

/// A fitted ARIMA model: what fable keeps of it, enough to forecast and to refit. Serializable,
/// so a trained model can be stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Arima {
    pub spec: ArimaSpec,
    /// `ar1.., ma1.., sar1.., sma1..` and, with a constant, the constant's regression coefficient
    /// (fable's `model$coef`).
    #[serde(with = "crate::serde_float::many")]
    pub coef: Vec<f64>,
    /// fable's `sigma2`: the sum of squared (standardised) residuals over `nstar - npar + 1`, with
    /// `nstar = n - d - D * period` and `npar` the estimated coefficients plus one.
    #[serde(with = "crate::serde_float::one")]
    pub sigma2: f64,
    #[serde(with = "crate::serde_float::one")]
    pub loglik: f64,
    /// AIC, AICc and BIC as fable computes them (`f64::INFINITY` when undefined).
    #[serde(with = "crate::serde_float::one")]
    pub aic: f64,
    #[serde(with = "crate::serde_float::one")]
    pub aicc: f64,
    #[serde(with = "crate::serde_float::one")]
    pub bic: f64,
    /// The number of observations of the history.
    pub n: usize,
    /// R's `nobs`: the observations used, less the differencing.
    pub nobs: usize,
    /// The residuals (`y - fitted`), standardised innovations as R's `arima` returns them.
    #[serde(with = "crate::serde_float::many")]
    pub residuals: Vec<f64>,
    pub method: Method,
    state: StateSpace,
}

/// Forecasts: the mean and the variance of each step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Forecast {
    #[serde(with = "crate::serde_float::many")]
    pub mean: Vec<f64>,
    #[serde(with = "crate::serde_float::many")]
    pub variance: Vec<f64>,
}

/// fable's `arima_constant(n, d, D, period)`: ones, integrated `d` times and seasonally `D` times
/// with R's `diffinv(xi = 1)`.
pub(crate) fn arima_constant(n: usize, d: usize, sd: usize, period: usize) -> Vec<f64> {
    let mut c = vec![1.0; n];
    if d > 0 {
        c = diffinv(&c, 1, d, &vec![1.0; d]);
        c.truncate(n);
    }
    if sd > 0 {
        c = diffinv(&c, period, sd, &vec![1.0; period * sd]);
        c.truncate(n);
    }
    c
}

/// R's `diffinv(x, lag, differences, xi)`.
fn diffinv(x: &[f64], lag: usize, differences: usize, xi: &[f64]) -> Vec<f64> {
    if differences == 1 {
        let mut y = xi.to_vec();
        y.resize(x.len() + lag, 0.0);
        for i in 0..x.len() {
            y[i + lag] = x[i] + y[i];
        }
        return y;
    }
    let inner_xi = crate::unitroot::diff(xi, lag, 1);
    let inner = diffinv(x, lag, differences - 1, &inner_xi);
    diffinv(&inner, lag, 1, &xi[..lag])
}

/// The smallest root modulus test fable applies to candidate models: false when an AR or MA root
/// lies within `1.01` (coefficients below `1e-8` in magnitude past the last larger one dropped).
fn roots_ok(phi: &[f64], theta: &[f64]) -> bool {
    let trimmed = |v: &[f64]| -> Vec<f64> {
        let last = v.iter().rposition(|x| x.abs() > 1e-8).map_or(0, |i| i + 1);
        v[..last].to_vec()
    };
    let ar: Vec<f64> = trimmed(&phi.iter().map(|v| -v).collect::<Vec<_>>());
    let ma = trimmed(theta);
    roots_outside(&ar, 1.01) && roots_outside(&ma, 1.01)
}

/// fable's `compare_arima` for one specification: `stats::arima` and fable's information
/// criteria and `sigma2`. `offset` is the CSS log-likelihood offset fable adds when `method` is
/// CSS.
pub(crate) fn fit_spec(
    y: &[f64],
    spec: &ArimaSpec,
    method: Method,
    fixed: Option<&[f64]>,
    offset: f64,
) -> Result<Arima> {
    let n = y.len();
    let xreg = spec
        .constant
        .then(|| arima_constant(n, spec.d, spec.seasonal_d, spec.period));
    let fit = stats::arima(y, spec.arma(), xreg.as_deref(), false, fixed, method)?;
    let nstar = n as f64 - spec.d as f64 - (spec.seasonal_d * spec.period) as f64;
    let npar = fit.mask.iter().filter(|m| **m).count() as f64 + 1.0;
    let aic = match fit.method {
        Method::Css => Some(offset + nstar * fit.sigma2.ln() + 2.0 * npar),
        _ => fit.aic,
    };
    let (aic, bic, aicc) = match aic {
        Some(aic) if !aic.is_nan() => (
            aic,
            aic + npar * (nstar.ln() - 2.0),
            aic + 2.0 * npar * (npar + 1.0) / (nstar - npar - 1.0),
        ),
        _ => (f64::INFINITY, f64::INFINITY, f64::INFINITY),
    };
    let ssr: f64 = fit
        .residuals
        .iter()
        .filter(|r| !r.is_nan())
        .map(|r| r * r)
        .sum();
    Ok(Arima {
        spec: *spec,
        coef: fit.coef,
        sigma2: ssr / (nstar - npar + 1.0),
        loglik: fit.loglik,
        aic,
        aicc,
        bic,
        n,
        nobs: fit.nobs,
        residuals: fit.residuals,
        method: fit.method,
        state: fit.model,
    })
}

fn check_series(y: &[f64]) -> Result<()> {
    if y.iter().all(|v| v.is_nan()) {
        return Err(Error::invalid(
            "all observations are missing, a model cannot be estimated without data",
        ));
    }
    if y.iter().any(|v| v.is_infinite()) {
        return Err(Error::invalid("the series has infinite values"));
    }
    Ok(())
}

/// Estimate the coefficients of `spec` on `y` (`NaN` for missing), as fable's `ARIMA()` with
/// every order given: R's `stats::arima` with `method` (fable uses CSS-ML).
pub fn fit_arima(y: &[f64], spec: &ArimaSpec, method: Method) -> Result<Arima> {
    check_series(y)?;
    auto::train(y, &AutoArima::exact(*spec), None, Some(method), false)
}

/// Filter `y` with `spec` and fixed coefficients (`coef` as in [`Arima::coef`]): no estimation,
/// only the state and `sigma2` from the new data. fable's `refit()` (`reestimate = FALSE`).
pub fn fit_arima_fixed(y: &[f64], spec: &ArimaSpec, coef: &[f64]) -> Result<Arima> {
    check_series(y)?;
    if coef.len() != spec.n_coef() {
        return Err(Error::invalid(format!(
            "{spec} has {} coefficients, got {}",
            spec.n_coef(),
            coef.len()
        )));
    }
    if coef.iter().any(|c| !c.is_finite()) {
        return Err(Error::invalid("coefficients must be finite"));
    }
    auto::train(y, &AutoArima::exact(*spec), Some(coef), None, true)
}

impl Arima {
    /// The same model on new data with the same coefficients: fable's `refit(reestimate = FALSE)`.
    pub fn filter(&self, y: &[f64]) -> Result<Arima> {
        fit_arima_fixed(y, &self.spec, &self.coef)
    }

    /// The same orders with the coefficients re-estimated on `y`: fable's
    /// `refit(reestimate = TRUE)`.
    pub fn reestimate(&self, y: &[f64]) -> Result<Arima> {
        check_series(y)?;
        auto::train(y, &AutoArima::exact(self.spec), None, None, true)
    }

    /// The AR polynomial's coefficients with the seasonal part multiplied out (R's `model$phi`).
    pub fn phi(&self) -> &[f64] {
        &self.state.phi
    }

    /// The MA polynomial's coefficients with the seasonal part multiplied out (`model$theta`,
    /// zero-padded).
    pub fn theta(&self) -> &[f64] {
        &self.state.theta
    }

    /// The next `h` steps' means and variances: R's `predict.Arima` with fable's `sigma2`, and
    /// the constant continued (`arima_constant` over `n + h`).
    pub fn forecast(&self, h: usize) -> Result<Forecast> {
        let (mut mean, var) = self.state.forecast(h);
        if self.spec.constant {
            let b = *self
                .coef
                .last()
                .ok_or_else(|| Error::invalid("a model with a constant has a coefficient"))?;
            let c = arima_constant(
                self.n + h,
                self.spec.d,
                self.spec.seasonal_d,
                self.spec.period,
            );
            for (m, x) in mean.iter_mut().zip(&c[self.n..]) {
                *m += x * b;
            }
        }
        Ok(Forecast {
            mean,
            variance: var.iter().map(|v| v * self.sigma2).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A missing observation gives a `NaN` residual; the model still goes through JSON (as a
    /// checkpoint stores it) and forecasts the same after.
    #[test]
    fn missing_observations_survive_a_json_round_trip() {
        let spec = ArimaSpec {
            p: 0,
            d: 0,
            q: 0,
            seasonal_p: 0,
            seasonal_d: 0,
            seasonal_q: 0,
            period: 1,
            constant: true,
        };
        let model = fit_arima(&[1.0, f64::NAN, 2.0, 3.0], &spec, Method::CssMl).unwrap();
        assert!(model.residuals.iter().any(|r| r.is_nan()));
        let text = serde_json::to_string(&serde_json::to_value(&model).unwrap()).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        let back: Arima = serde_json::from_value(value).unwrap();
        assert_eq!(
            serde_json::to_value(&back).unwrap(),
            serde_json::to_value(&model).unwrap()
        );
        assert_eq!(back.forecast(2).unwrap(), model.forecast(2).unwrap());
        let nan = |v: &[f64]| v.iter().map(|x| x.is_nan()).collect::<Vec<_>>();
        assert_eq!(nan(&back.residuals), nan(&model.residuals));
    }

    #[test]
    fn constants() {
        assert_eq!(arima_constant(4, 0, 0, 12), vec![1.0; 4]);
        assert_eq!(arima_constant(4, 1, 0, 12), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(arima_constant(5, 2, 0, 12), vec![1.0, 1.0, 2.0, 4.0, 7.0]);
        assert_eq!(
            arima_constant(6, 0, 1, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        );
    }
}
