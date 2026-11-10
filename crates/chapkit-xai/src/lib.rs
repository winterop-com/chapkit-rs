//! In-process explainers for chapkit-rs models.
//!
//! [`explain`] runs LIME or KernelSHAP over any batch value function
//! `f(features) -> value`; [`forecast::explain_forecast`] adapts a model's
//! `predict` so one forecast cell (a location and time period) can be explained in
//! terms of its covariates, their earlier values and the target's own history. Both
//! methods call the model hundreds to thousands of times, which is only practical
//! in-process; evaluations run in parallel on all cores (rayon) unless the whole batch
//! is cheap enough that one thread is faster.
//!
//! Ported from `chap-rs/crates/xai`, made generic over the value function.

pub mod forecast;
mod lime;
mod linalg;

mod shap;

use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub use forecast::{
    DEFAULT_LAGS, FeatureSpec, ForecastExplanation, ForecastTarget, Scope, explain_forecast,
};

/// Explainer errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Bad request: no features, no background, missing values.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// The model failed on a perturbed input.
    #[error("model error: {0}")]
    Model(String),
    /// The surrogate regression could not be solved.
    #[error("{0} regression was singular")]
    Singular(&'static str),
}

/// Result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Explanation method.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// Local linear surrogate; importances are coefficients on standardized features.
    #[default]
    Lime,
    /// KernelSHAP; importances are additive contributions summing to `prediction - baseline`.
    Shap,
}

/// Whether model evaluations run on one thread or across cores.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Parallel {
    /// Time the first evaluation; go parallel only if the batch would take longer than
    /// [`PARALLEL_MIN_BATCH`] on one thread.
    #[default]
    Auto,
    /// Always one thread (for callers that parallelize themselves).
    Never,
    /// Always across cores.
    Always,
}

/// Estimated single-thread batch time below which [`Parallel::Auto`] stays on one thread.
/// Measured with `make bench BENCH=xai` (see the explainers design note).
pub const PARALLEL_MIN_BATCH: Duration = Duration::from_micros(PARALLEL_MIN_BATCH_US);
const PARALLEL_MIN_BATCH_US: u64 = 75;

/// Explainer settings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Options {
    /// Method.
    #[serde(default)]
    pub method: Method,
    /// Perturbations (LIME) or sampled coalitions (SHAP, above 12 features).
    #[serde(default = "default_samples")]
    pub n_samples: usize,
    /// RNG seed.
    #[serde(default)]
    pub seed: u64,
    /// LIME kernel width in standardized distance (default `0.75 * sqrt(k)`).
    #[serde(default)]
    pub kernel_width: Option<f64>,
    /// One thread or across cores.
    #[serde(default)]
    pub parallel: Parallel,
}

fn default_samples() -> usize {
    500
}

impl Default for Options {
    fn default() -> Self {
        Self {
            method: Method::Lime,
            n_samples: default_samples(),
            seed: 0,
            kernel_width: None,
            parallel: Parallel::Auto,
        }
    }
}

/// One feature's importance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature name.
    pub feature: String,
    /// LIME: local coefficient. SHAP: additive contribution.
    pub weight: f64,
}

/// An explanation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Explanation {
    /// Method used.
    pub method: Method,
    /// The model's value at the instance.
    pub prediction: f64,
    /// SHAP: `f(reference)`; importances sum to `prediction - baseline`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline: Option<f64>,
    /// LIME: the surrogate's intercept.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intercept: Option<f64>,
    /// Importances, largest magnitude first.
    pub importances: Vec<FeatureImportance>,
    /// How well the surrogate fits the model's values (weighted R2; 1 is exact). LIME: the
    /// linear fit on the perturbations. SHAP: the additive fit on the coalitions. Absent when
    /// the model's values do not vary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r2: Option<f64>,
    /// LIME: effective number of perturbations, `(sum w)^2 / sum w^2` of the kernel weights.
    /// Far below `n_model_calls` means few perturbations land near the instance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_n: Option<f64>,
    /// Model evaluations performed.
    pub n_model_calls: usize,
}

/// A batch value function: maps feature vectors (in `features` order) to values.
pub trait ValueFn: Sync {
    /// Evaluate one feature vector.
    fn eval(&self, x: &[f64]) -> Result<f64>;
}

impl<F: Fn(&[f64]) -> Result<f64> + Sync> ValueFn for F {
    fn eval(&self, x: &[f64]) -> Result<f64> {
        self(x)
    }
}

/// Evaluate `f` on every row, keeping order, on one thread or across cores.
pub(crate) fn eval_batch(f: &dyn ValueFn, rows: &[Vec<f64>], mode: Parallel) -> Result<Vec<f64>> {
    let parallel = |rows: &[Vec<f64>]| {
        rows.par_iter()
            .map(|r| f.eval(r))
            .collect::<Result<Vec<_>>>()
    };
    match mode {
        Parallel::Never => rows.iter().map(|r| f.eval(r)).collect(),
        Parallel::Always => parallel(rows),
        Parallel::Auto => {
            let Some((first, rest)) = rows.split_first() else {
                return Ok(Vec::new());
            };
            let start = Instant::now();
            let mut out = Vec::with_capacity(rows.len());
            out.push(f.eval(first)?);
            let estimate = start
                .elapsed()
                .saturating_mul(u32::try_from(rest.len()).unwrap_or(u32::MAX));
            if estimate < PARALLEL_MIN_BATCH {
                for r in rest {
                    out.push(f.eval(r)?);
                }
            } else {
                out.extend(parallel(rest)?);
            }
            Ok(out)
        }
    }
}

/// Weighted R2 of `fitted` against `y`; `None` when `y` does not vary.
pub(crate) fn weighted_r2(y: &[f64], fitted: &[f64], w: &[f64]) -> Option<f64> {
    let sw: f64 = w.iter().sum();
    if sw <= 0.0 {
        return None;
    }
    let mean = y.iter().zip(w).map(|(y, w)| y * w).sum::<f64>() / sw;
    let (mut ss_res, mut ss_tot) = (0.0, 0.0);
    for ((y, f), w) in y.iter().zip(fitted).zip(w) {
        ss_res += w * (y - f) * (y - f);
        ss_tot += w * (y - mean) * (y - mean);
    }
    let scale = mean.abs().max(1.0);
    (ss_tot > 1e-18 * scale * scale * sw).then(|| 1.0 - ss_res / ss_tot)
}

/// Per-feature distribution: LIME samples around `mean` with spread `sd`; SHAP uses
/// `mean` as the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct Background {
    /// Mean per feature.
    pub mean: Vec<f64>,
    /// Standard deviation per feature (floored at 1e-9).
    pub sd: Vec<f64>,
}

impl Background {
    /// Mean and population standard deviation of each feature over `rows`.
    pub fn from_rows(k: usize, rows: &[Vec<f64>]) -> Self {
        let (mean, sd) = feature_stats(k, rows);
        Self { mean, sd }
    }

    /// Mean and population standard deviation of one feature's values.
    pub fn stats(values: &[f64]) -> (f64, f64) {
        let n = values.len().max(1) as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        (mean, var.sqrt().max(1e-9))
    }
}

/// Explain `f` at `instance`. `background` rows describe the feature distribution
/// (LIME samples around their mean and spread; SHAP uses their mean as the reference).
pub fn explain(
    f: &dyn ValueFn,
    features: &[String],
    instance: &[f64],
    background: &[Vec<f64>],
    opts: &Options,
) -> Result<Explanation> {
    if background.iter().any(|r| r.len() != features.len()) {
        return Err(Error::InvalidInput(
            "instance and background must have one value per feature".into(),
        ));
    }
    if background.is_empty() {
        return Err(Error::InvalidInput(
            "explaining needs at least one background row".into(),
        ));
    }
    let bg = Background::from_rows(features.len(), background);
    explain_with(f, features, instance, &bg, opts)
}

/// [`explain`] with the per-feature background statistics given directly.
pub fn explain_with(
    f: &dyn ValueFn,
    features: &[String],
    instance: &[f64],
    background: &Background,
    opts: &Options,
) -> Result<Explanation> {
    let k = features.len();
    if k == 0 {
        return Err(Error::InvalidInput("no features to explain".into()));
    }
    if instance.len() != k || background.mean.len() != k || background.sd.len() != k {
        return Err(Error::InvalidInput(
            "instance and background must have one value per feature".into(),
        ));
    }
    if instance.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "the instance has missing or non-finite feature values".into(),
        ));
    }
    let mut e = match opts.method {
        Method::Lime => lime::explain(f, features, instance, background, opts)?,
        Method::Shap => shap::explain(f, features, instance, background, opts)?,
    };
    e.importances
        .sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
    Ok(e)
}

/// Per-feature mean and standard deviation (floored at 1e-9) over the background.
pub(crate) fn feature_stats(k: usize, background: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = background.len().max(1) as f64;
    let mut mu = vec![0.0; k];
    for row in background {
        for (m, v) in mu.iter_mut().zip(row) {
            *m += v;
        }
    }
    for m in &mut mu {
        *m /= n;
    }
    let mut var = vec![0.0; k];
    for row in background {
        for ((s, v), m) in var.iter_mut().zip(row).zip(&mu) {
            *s += (v - m) * (v - m);
        }
    }
    let sigma = var.iter().map(|v| (v / n).sqrt().max(1e-9)).collect();
    (mu, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(k: usize) -> Vec<String> {
        (0..k).map(|i| format!("x{i}")).collect()
    }

    fn background() -> Vec<Vec<f64>> {
        let mut r = chapkit_core::rng::Rng::new(3);
        (0..200)
            .map(|_| vec![r.normal() * 2.0 + 1.0, r.normal() + 5.0, r.normal() * 0.5])
            .collect()
    }

    /// For a linear model, SHAP values are exactly `w_j * (x_j - mean_j)`.
    #[test]
    fn shap_recovers_linear_contributions() {
        let f = |x: &[f64]| Ok(3.0 * x[0] - 2.0 * x[1] + 0.5 * x[2] + 7.0);
        let bg = background();
        let (mu, _) = feature_stats(3, &bg);
        let x = [2.0, 4.0, 1.0];
        let e = explain(
            &f,
            &names(3),
            &x,
            &bg,
            &Options {
                method: Method::Shap,
                ..Options::default()
            },
        )
        .unwrap();
        let w = [3.0, -2.0, 0.5];
        for imp in &e.importances {
            let j: usize = imp.feature[1..].parse().unwrap();
            assert!((imp.weight - w[j] * (x[j] - mu[j])).abs() < 1e-9, "{imp:?}");
        }
        let total: f64 = e.importances.iter().map(|i| i.weight).sum();
        assert!((total - (e.prediction - e.baseline.unwrap())).abs() < 1e-9);
        assert_eq!(e.n_model_calls, 2 + 6);
    }

    /// For a linear model, LIME coefficients are `w_j * sigma_j` (standardized features).
    #[test]
    fn lime_recovers_linear_coefficients() {
        let f = |x: &[f64]| Ok(3.0 * x[0] - 2.0 * x[1] + 0.5 * x[2] + 7.0);
        let bg = background();
        let (_, sigma) = feature_stats(3, &bg);
        let e = explain(&f, &names(3), &[2.0, 4.0, 1.0], &bg, &Options::default()).unwrap();
        let w = [3.0, -2.0, 0.5];
        for imp in &e.importances {
            let j: usize = imp.feature[1..].parse().unwrap();
            assert!((imp.weight - w[j] * sigma[j]).abs() < 1e-6, "{imp:?}");
        }
        assert_eq!(e.importances[0].feature, "x0");
        assert_eq!(e.n_model_calls, 500);
    }

    /// R2 is 1 for a linear model and below 1 with an interaction; every thread mode
    /// gives the same answer.
    #[test]
    fn surrogate_quality_and_thread_modes() {
        let linear = |x: &[f64]| Ok(3.0 * x[0] - 2.0 * x[1] + 0.5 * x[2]);
        let bent = |x: &[f64]| Ok(x[0] * x[0] * x[1] + (3.0 * x[1]).sin());
        let bg = background();
        let x = [2.0, 4.0, 1.0];
        let lime = explain(&linear, &names(3), &x, &bg, &Options::default()).unwrap();
        assert!((lime.r2.unwrap() - 1.0).abs() < 1e-9);
        let n_eff = lime.effective_n.unwrap();
        assert!(n_eff > 1.0 && n_eff < 500.0, "{n_eff}");
        let poor = explain(&bent, &names(3), &x, &bg, &Options::default()).unwrap();
        assert!(poor.r2.unwrap() < 0.99, "{poor:?}");
        let shap = |parallel| Options {
            method: Method::Shap,
            parallel,
            ..Options::default()
        };
        let additive = explain(&linear, &names(3), &x, &bg, &shap(Parallel::Auto)).unwrap();
        assert!((additive.r2.unwrap() - 1.0).abs() < 1e-9);
        assert!(additive.effective_n.is_none());
        let modes = [Parallel::Auto, Parallel::Never, Parallel::Always];
        let runs: Vec<Explanation> = modes
            .iter()
            .map(|&p| explain(&bent, &names(3), &x, &bg, &shap(p)).unwrap())
            .collect();
        assert!(runs.iter().all(|e| *e == runs[0]));
        assert!(runs[0].r2.unwrap() < 1.0);
        let constant = |_: &[f64]| Ok(4.0);
        assert!(
            explain(&constant, &names(3), &x, &bg, &Options::default())
                .unwrap()
                .r2
                .is_none()
        );
    }

    #[test]
    fn rejects_bad_input() {
        let f = |_: &[f64]| Ok(0.0);
        assert!(explain(&f, &[], &[], &[vec![]], &Options::default()).is_err());
        assert!(
            explain(
                &f,
                &names(1),
                &[f64::NAN],
                &[vec![1.0]],
                &Options::default()
            )
            .is_err()
        );
        assert!(explain(&f, &names(1), &[1.0], &[], &Options::default()).is_err());
    }

    #[test]
    fn model_errors_propagate() {
        let f = |_: &[f64]| Err(Error::Model("boom".into()));
        assert!(matches!(
            explain(&f, &names(1), &[1.0], &[vec![1.0]], &Options::default()),
            Err(Error::Model(_))
        ));
    }
}
