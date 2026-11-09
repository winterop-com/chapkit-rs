//! In-process explainers for chapkit-rs models.
//!
//! [`explain`] runs LIME or KernelSHAP over any batch value function
//! `f(features) -> value`; [`forecast::explain_forecast`] adapts a model's
//! `predict` so one forecast cell (a location and time period) can be explained in
//! terms of its covariates. Both methods call the model hundreds to thousands of
//! times, which is only practical in-process; evaluations run in parallel on all
//! cores (rayon).
//!
//! Ported from `chap-rs/crates/xai`, made generic over the value function.

pub mod forecast;
mod lime;
mod linalg;
mod rng;
mod shap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub use forecast::{ForecastExplanation, ForecastTarget, explain_forecast};

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

/// Evaluate `f` on every row in parallel, keeping order.
pub(crate) fn eval_batch(f: &dyn ValueFn, rows: &[Vec<f64>]) -> Result<Vec<f64>> {
    rows.par_iter().map(|r| f.eval(r)).collect()
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
    if features.is_empty() {
        return Err(Error::InvalidInput("no features to explain".into()));
    }
    if instance.len() != features.len() || background.iter().any(|r| r.len() != features.len()) {
        return Err(Error::InvalidInput(
            "instance and background must have one value per feature".into(),
        ));
    }
    if background.is_empty() {
        return Err(Error::InvalidInput(
            "explaining needs at least one background row".into(),
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
        let mut r = rng::SplitMix::new(3);
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
