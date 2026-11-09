//! The ridge example as a chapkit-compatible service.
//!
//!     cargo run --release -p ridge            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release -p ridge -- demo    # run once in-process and time it

use ridge::{RidgeConfig, RidgeModel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("demo") {
        return baselines::demo::run("ridge", &RidgeModel, &RidgeConfig::default());
    }
    chapkit::Service::new("ridge", "Ridge regression", RidgeModel)
        .description("Ridge regression on covariates around a per-location mean")
        .serve()
        .await?;
    Ok(())
}
