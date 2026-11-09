//! chap-core's `seasonal_shift_persistence` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baselines --bin seasonal-shift-persistence            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release -p baselines --bin seasonal-shift-persistence -- demo    # run once in-process and time it

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    baselines::main_for(
        "seasonal-shift-persistence",
        "Seasonal shift persistence",
        baselines::SeasonalShiftPersistence,
        chapkit::NoConfig {},
    )
    .await
}
