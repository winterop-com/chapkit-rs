//! LIME: a local linear surrogate fit on kernel-weighted perturbations.
//!
//! Perturbations are drawn from a normal with the background's per-feature mean and
//! spread; the surrogate is fit on standardized features, so a coefficient is the
//! change in prediction per standard deviation of that feature.

use crate::linalg::weighted_least_squares;
use crate::rng::SplitMix;
use crate::{
    Error, Explanation, FeatureImportance, Method, Options, Result, ValueFn, eval_batch,
    feature_stats,
};

pub(crate) fn explain(
    f: &dyn ValueFn,
    features: &[String],
    instance: &[f64],
    background: &[Vec<f64>],
    opts: &Options,
) -> Result<Explanation> {
    let k = features.len();
    let (mu, sigma) = feature_stats(k, background);
    let width = opts
        .kernel_width
        .unwrap_or((k as f64).sqrt() * 0.75)
        .max(1e-6);
    let n = opts.n_samples.max(k + 1);
    let mut rng = SplitMix::new(opts.seed);

    // Row 0 is the instance itself (weight 1); the rest are perturbations.
    let inst_z: Vec<f64> = (0..k).map(|j| (instance[j] - mu[j]) / sigma[j]).collect();
    let mut points = vec![instance.to_vec()];
    let mut design = vec![with_intercept(&inst_z)];
    let mut weights = vec![1.0];
    for _ in 1..n {
        let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
        let x: Vec<f64> = (0..k).map(|j| mu[j] + z[j] * sigma[j]).collect();
        let dist_sq: f64 = (0..k)
            .map(|j| ((x[j] - instance[j]) / sigma[j]).powi(2))
            .sum();
        weights.push((-dist_sq / (2.0 * width * width)).exp());
        design.push(with_intercept(&z));
        points.push(x);
    }

    let values = eval_batch(f, &points)?;
    let coef = weighted_least_squares(&design, &values, &weights).ok_or(Error::Singular("LIME"))?;
    Ok(Explanation {
        method: Method::Lime,
        prediction: values.first().copied().unwrap_or(f64::NAN),
        baseline: None,
        intercept: coef.first().copied(),
        importances: features
            .iter()
            .zip(coef.iter().skip(1))
            .map(|(name, w)| FeatureImportance {
                feature: name.clone(),
                weight: *w,
            })
            .collect(),
        n_model_calls: points.len(),
    })
}

fn with_intercept(z: &[f64]) -> Vec<f64> {
    let mut row = Vec::with_capacity(z.len() + 1);
    row.push(1.0);
    row.extend_from_slice(z);
    row
}
