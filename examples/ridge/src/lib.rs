//! Ridge regression of the target on covariates around a per-location mean, with
//! normal residual noise. Small, fast and covariate-driven: the reference model for
//! explainer (LIME/SHAP) and HPO work. Stores its weights as safetensors tensors.

mod linalg;
mod model;

pub use model::{RidgeConfig, RidgeModel, RidgeTrained};
