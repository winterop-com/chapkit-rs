//! chap-core's `seasonal_median` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baselines --bin seasonal-median            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release -p baselines --bin seasonal-median -- demo    # run once in-process and time it

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    baselines::main_for(
        "seasonal-median",
        "Seasonal median",
        baselines::SeasonalMedian,
        chapkit::NoConfig {},
    )
    .await
}
