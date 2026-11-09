//! chap-core's `seasonal_median` baseline, ported to Rust and served as a chapkit-compatible service.
//!
//!     cargo run --release --example seasonal_median            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release --example seasonal_median -- demo    # run once in-process and time it

#[path = "../common/demo.rs"]
mod demo;
#[path = "../common/empirical.rs"]
mod empirical;
mod model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("demo") {
        return demo::run(
            "seasonal_median",
            &model::SeasonalMedian,
            &chapkit::NoConfig {},
        );
    }
    chapkit::Service::new("seasonal-median", "Seasonal median", model::SeasonalMedian)
        .description("Rust port of chap-core's seasonal_median baseline")
        .serve()
        .await?;
    Ok(())
}
