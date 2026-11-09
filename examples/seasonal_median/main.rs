//! chap-core's `seasonal_median` baseline, ported to Rust.
//!
//! cargo run --release --example seasonal_median

#[path = "../common/demo.rs"]
mod demo;
#[path = "../common/empirical.rs"]
mod empirical;
mod model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    demo::run(
        "seasonal_median",
        &model::SeasonalMedian,
        &chapkit::NoConfig {},
    )
}
