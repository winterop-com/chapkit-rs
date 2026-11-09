//! A small covariate-driven model (ridge regression around a per-location mean),
//! with a search space. The reference model for explainer and HPO work.
//!
//! cargo run --release --example ridge

#[path = "../common/demo.rs"]
mod demo;
mod linalg;
mod model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    demo::run("ridge", &model::RidgeModel, &model::RidgeConfig::default())
}
