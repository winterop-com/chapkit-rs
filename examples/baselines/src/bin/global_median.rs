//! chap-core's `global_median` baseline as a chapkit-compatible service.
//!
//!     cargo run --release -p baselines --bin global-median            # serve on HOST:PORT (127.0.0.1:8000)
//!     cargo run --release -p baselines --bin global-median -- demo    # run once in-process and time it

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    baselines::main_for(
        "global-median",
        "Global median",
        baselines::GlobalMedian,
        chapkit::NoConfig {},
    )
    .await
}
