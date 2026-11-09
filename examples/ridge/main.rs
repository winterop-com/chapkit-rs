//! A small covariate-driven model (ridge regression around a per-location mean),
//! with a search space and tensor checkpoints. The reference model for explainer
//! and HPO work.
//!
//!     cargo run --release --example ridge            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release --example ridge -- demo    # run once in-process and time it

#[path = "../common/demo.rs"]
mod demo;
mod linalg;
mod model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("demo") {
        return demo::run("ridge", &model::RidgeModel, &model::RidgeConfig::default());
    }
    chapkit::Service::new("ridge", "Ridge regression", model::RidgeModel)
        .description("Ridge regression on covariates around a per-location mean")
        .serve()
        .await?;
    Ok(())
}
