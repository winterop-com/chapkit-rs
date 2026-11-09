//! KernelSHAP with a single reference (the background mean).
//!
//! For each coalition `z` (feature present or absent) the model is evaluated on the
//! instance's values for present features and the reference's for absent ones. The
//! SHAP-kernel-weighted regression is solved under the efficiency constraint
//! `sum(phi) = f(x) - f(reference)`, enforced by eliminating the last coefficient.
//! Up to 12 features every coalition is enumerated; above that they are
//! importance-sampled with probability proportional to the kernel.

use std::collections::BTreeMap;

use crate::linalg::weighted_least_squares;
use crate::rng::SplitMix;
use crate::{
    Error, Explanation, FeatureImportance, Method, Options, Result, ValueFn, eval_batch,
    feature_stats,
};

const EXHAUSTIVE_MAX_FEATURES: usize = 12;

pub(crate) fn explain(
    f: &dyn ValueFn,
    features: &[String],
    instance: &[f64],
    background: &[Vec<f64>],
    opts: &Options,
) -> Result<Explanation> {
    let k = features.len();
    let (reference, _) = feature_stats(k, background);
    let coalitions = coalitions(k, opts.n_samples, opts.seed);

    let mut points = vec![instance.to_vec(), reference.clone()];
    for (z, _) in &coalitions {
        points.push(
            (0..k)
                .map(|j| if z[j] { instance[j] } else { reference[j] })
                .collect(),
        );
    }
    let values = eval_batch(f, &points)?;
    let (f_x, f_ref) = (values[0], values[1]);
    let total = f_x - f_ref;

    let mut x_rows = Vec::with_capacity(coalitions.len());
    let mut y = Vec::with_capacity(coalitions.len());
    let mut w = Vec::with_capacity(coalitions.len());
    for ((z, weight), fz) in coalitions.iter().zip(&values[2..]) {
        let z_last = f64::from(u8::from(z[k - 1]));
        x_rows.push(
            (0..k - 1)
                .map(|j| f64::from(u8::from(z[j])) - z_last)
                .collect::<Vec<f64>>(),
        );
        y.push((fz - f_ref) - z_last * total);
        w.push(*weight);
    }
    let mut phi = if k == 1 {
        Vec::new()
    } else {
        weighted_least_squares(&x_rows, &y, &w).ok_or(Error::Singular("SHAP"))?
    };
    phi.push(total - phi.iter().sum::<f64>());

    Ok(Explanation {
        method: Method::Shap,
        prediction: f_x,
        baseline: Some(f_ref),
        intercept: None,
        importances: features
            .iter()
            .zip(&phi)
            .map(|(name, w)| FeatureImportance {
                feature: name.clone(),
                weight: *w,
            })
            .collect(),
        n_model_calls: points.len(),
    })
}

/// Coalitions with their regression weight (exact kernel, or sampling multiplicity).
fn coalitions(k: usize, n_samples: usize, seed: u64) -> Vec<(Vec<bool>, f64)> {
    if k <= EXHAUSTIVE_MAX_FEATURES {
        (1..(1u32 << k) - 1)
            .map(|mask| {
                let z: Vec<bool> = (0..k).map(|j| mask & (1 << j) != 0).collect();
                let s = z.iter().filter(|b| **b).count();
                (z, shap_kernel(k, s))
            })
            .collect()
    } else {
        sample_coalitions(k, n_samples.max(2 * k), seed)
    }
}

/// Draw coalition sizes with probability proportional to the summed kernel per size,
/// then a uniform subset of that size; each unique mask is weighted by its count.
fn sample_coalitions(k: usize, target: usize, seed: u64) -> Vec<(Vec<bool>, f64)> {
    let size_w: Vec<f64> = (1..k).map(|s| 1.0 / (s as f64 * (k - s) as f64)).collect();
    let total_w: f64 = size_w.iter().sum();
    let mut rng = SplitMix::new(seed);
    let mut counts: BTreeMap<Vec<bool>, f64> = BTreeMap::new();
    for _ in 0..target {
        let mut t = rng.unit() * total_w;
        let mut s = k - 1;
        for (i, w) in size_w.iter().enumerate() {
            t -= w;
            if t <= 0.0 {
                s = i + 1;
                break;
            }
        }
        let mut idx: Vec<usize> = (0..k).collect();
        for i in 0..s {
            let span = k - i;
            let j = i + ((rng.unit() * span as f64) as usize).min(span - 1);
            idx.swap(i, j);
        }
        let mut z = vec![false; k];
        for &i in &idx[..s] {
            z[i] = true;
        }
        *counts.entry(z).or_insert(0.0) += 1.0;
    }
    counts.into_iter().collect()
}

/// `(k - 1) / (C(k, s) * s * (k - s))`.
fn shap_kernel(k: usize, s: usize) -> f64 {
    let binom = (0..s).fold(1.0, |acc, i| acc * (k - i) as f64 / (i + 1) as f64);
    (k as f64 - 1.0) / (binom * s as f64 * (k - s) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerates_and_samples_coalitions() {
        assert_eq!(coalitions(3, 0, 0).len(), 6);
        let cs = sample_coalitions(20, 500, 7);
        let total: f64 = cs.iter().map(|(_, w)| w).sum();
        assert!((total - 500.0).abs() < 1e-9);
        assert!(cs.iter().all(|(z, _)| {
            let s = z.iter().filter(|b| **b).count();
            s > 0 && s < 20
        }));
        assert_eq!(
            sample_coalitions(20, 200, 42),
            sample_coalitions(20, 200, 42)
        );
    }

    #[test]
    fn kernel_is_symmetric() {
        assert!((shap_kernel(4, 1) - shap_kernel(4, 3)).abs() < 1e-12);
    }
}
