use std::collections::HashMap;

use chapkit::{
    Checkpoint, Column, DataFrame, Error, Model, ModelInfo, ModelMetadata, Param, PredictInput,
    Result, SearchSpace, Tensor, TrainInput, cols, prediction_frame,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::linalg::{mean_std, solve};

/// Ridge regression of the target on covariates, around a per-location mean,
/// with normal residual noise. Small, fast and covariate-driven, which makes it
/// the reference model for explainer and HPO tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct RidgeModel;

/// Options for [`RidgeModel`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RidgeConfig {
    /// L2 penalty on standardized coefficients.
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// Covariate columns to use. Default: every numeric column other than
    /// `time_period`, `location` and the target.
    #[serde(default)]
    pub features: Option<Vec<String>>,
    /// Number of samples per forecast.
    #[serde(default = "default_samples")]
    pub n_samples: usize,
    /// RNG seed for sampling.
    #[serde(default)]
    pub seed: u64,
}

fn default_alpha() -> f64 {
    1.0
}
fn default_samples() -> usize {
    100
}

impl Default for RidgeConfig {
    fn default() -> Self {
        Self {
            alpha: default_alpha(),
            features: None,
            n_samples: default_samples(),
            seed: 0,
        }
    }
}

/// Fitted [`RidgeModel`] state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeTrained {
    /// Covariates, in coefficient order.
    pub features: Vec<String>,
    /// Training mean of each covariate.
    pub means: Vec<f64>,
    /// Training standard deviation of each covariate.
    pub stds: Vec<f64>,
    /// Coefficients on standardized covariates.
    pub weights: Vec<f64>,
    /// Per-location target mean.
    pub location_mean: HashMap<String, f64>,
    /// Target mean over all locations.
    pub global_mean: f64,
    /// Residual standard deviation.
    pub resid_std: f64,
}

impl RidgeTrained {
    /// Point prediction (no noise) for one row of covariates, in `features` order.
    pub fn point(&self, location: Option<&str>, x: &[f64]) -> f64 {
        let base = location
            .and_then(|l| self.location_mean.get(l))
            .copied()
            .unwrap_or(self.global_mean);
        let lin: f64 = x
            .iter()
            .zip(&self.means)
            .zip(&self.stds)
            .zip(&self.weights)
            .map(|(((&v, &m), &s), &w)| if v.is_finite() { (v - m) / s * w } else { 0.0 })
            .sum();
        base + lin
    }
}

fn default_features(df: &DataFrame, target: &str) -> Vec<String> {
    df.iter()
        .filter(|(n, c)| {
            *n != cols::TIME_PERIOD
                && *n != cols::LOCATION
                && *n != target
                && !matches!(c, Column::Str(_))
        })
        .map(|(n, _)| n.to_owned())
        .collect()
}

impl Model for RidgeModel {
    type Config = RidgeConfig;
    type Trained = RidgeTrained;

    fn info(&self) -> ModelInfo {
        ModelInfo {
            allow_free_additional_continuous_covariates: true,
            model_metadata: ModelMetadata {
                author: Some("Morten Hansen".into()),
                contact_email: Some("morten@winterop.com".into()),
                author_note: Some(
                    "Ridge regression on covariates around a per-location mean".into(),
                ),
                ..ModelMetadata::default()
            },
            ..ModelInfo::default()
        }
    }

    /// Weights, means and standard deviations go in as tensors; the rest as JSON state.
    fn save(&self, t: &RidgeTrained) -> Result<Checkpoint> {
        let state = serde_json::json!({
            "features": t.features,
            "location_mean": t.location_mean,
            "global_mean": t.global_mean,
            "resid_std": t.resid_std,
        });
        Ok(Checkpoint::from_state(&state)?
            .with_tensor("weights", Tensor::vector(t.weights.clone()))
            .with_tensor("means", Tensor::vector(t.means.clone()))
            .with_tensor("stds", Tensor::vector(t.stds.clone())))
    }

    fn load(&self, ck: &Checkpoint) -> Result<RidgeTrained> {
        #[derive(Deserialize)]
        struct State {
            features: Vec<String>,
            location_mean: HashMap<String, f64>,
            global_mean: f64,
            resid_std: f64,
        }
        let s: State = ck.state()?;
        let t = RidgeTrained {
            weights: ck.tensor("weights")?.to_f64(),
            means: ck.tensor("means")?.to_f64(),
            stds: ck.tensor("stds")?.to_f64(),
            features: s.features,
            location_mean: s.location_mean,
            global_mean: s.global_mean,
            resid_std: s.resid_std,
        };
        let d = t.features.len();
        if t.weights.len() != d || t.means.len() != d || t.stds.len() != d {
            return Err(Error::Serde(
                "checkpoint tensors do not match the feature count".into(),
            ));
        }
        Ok(t)
    }

    fn search_space(&self) -> Option<SearchSpace> {
        Some(SearchSpace::new().with(
            "alpha",
            Param::Float {
                low: 1e-3,
                high: 1e3,
                step: None,
                log: true,
            },
        ))
    }

    fn train(&self, config: &RidgeConfig, input: TrainInput<'_>) -> Result<RidgeTrained> {
        let target = self.info().target;
        let data = input.data;
        let features = config
            .features
            .clone()
            .unwrap_or_else(|| default_features(data, &target));
        let y = data.f64_column(&target)?;
        let xs: Vec<Vec<f64>> = features
            .iter()
            .map(|f| data.f64_column(f))
            .collect::<Result<_, _>>()?;

        let mut location_mean = HashMap::new();
        let mut centered = vec![f64::NAN; y.len()];
        for (loc, idx) in data.group_indices(cols::LOCATION)? {
            if let Some((m, _)) = mean_std(idx.iter().map(|&i| y[i])) {
                for &i in &idx {
                    centered[i] = y[i] - m;
                }
                location_mean.insert(loc, m);
            }
        }
        let (global_mean, _) = mean_std(y.iter().copied())
            .ok_or_else(|| Error::invalid("no observed target values"))?;

        let (means, stds): (Vec<f64>, Vec<f64>) = xs
            .iter()
            .map(|x| {
                mean_std(x.iter().copied())
                    .map_or((0.0, 1.0), |(m, s)| (m, if s > 1e-12 { s } else { 1.0 }))
            })
            .unzip();

        let rows: Vec<usize> = (0..y.len())
            .filter(|&i| centered[i].is_finite() && xs.iter().all(|x| x[i].is_finite()))
            .collect();
        if rows.is_empty() {
            return Err(Error::invalid("no complete training rows"));
        }

        let d = features.len();
        let mut xtx = vec![0.0; d * d];
        let mut xty = vec![0.0; d];
        let mut z = vec![0.0; d];
        for &i in &rows {
            for (k, x) in xs.iter().enumerate() {
                z[k] = (x[i] - means[k]) / stds[k];
            }
            for a in 0..d {
                xty[a] += z[a] * centered[i];
                for b in 0..d {
                    xtx[a * d + b] += z[a] * z[b];
                }
            }
        }
        for a in 0..d {
            xtx[a * d + a] += config.alpha.max(0.0);
        }
        let weights = if d == 0 { Vec::new() } else { solve(xtx, xty)? };

        let mut trained = RidgeTrained {
            features,
            means,
            stds,
            weights,
            location_mean,
            global_mean,
            resid_std: 0.0,
        };
        let locs = data.require(cols::LOCATION)?;
        let mut row_x = vec![0.0; d];
        let resid = rows.iter().map(|&i| {
            for (k, x) in xs.iter().enumerate() {
                row_x[k] = x[i];
            }
            y[i] - trained.point(locs.str_at(i), &row_x)
        });
        trained.resid_std = mean_std(resid).map_or(0.0, |(_, s)| s);
        Ok(trained)
    }

    fn predict(
        &self,
        trained: &RidgeTrained,
        config: &RidgeConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let future = input.future;
        let xs: Vec<Vec<f64>> = trained
            .features
            .iter()
            .map(|f| future.f64_column(f))
            .collect::<Result<_, _>>()?;
        let locs = future.require(cols::LOCATION)?;
        let noise = Normal::new(0.0, trained.resid_std.max(1e-9))
            .map_err(|e| Error::model(e.to_string()))?;
        let mut rng = SmallRng::seed_from_u64(config.seed);
        let mut row_x = vec![0.0; xs.len()];
        prediction_frame(future, config.n_samples, |row, out| {
            for (k, x) in xs.iter().enumerate() {
                row_x[k] = x[row];
            }
            let p = trained.point(locs.str_at(row), &row_x);
            for v in out.iter_mut() {
                *v = (p + noise.sample(&mut rng)).max(0.0);
            }
            Ok(())
        })
    }
}
