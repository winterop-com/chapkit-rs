//! chap-core's `seasonal_shift_persistence` baseline, ported to Rust.
//!
//! cargo run --release --example seasonal_shift_persistence

#[path = "../common/demo.rs"]
mod demo;
#[path = "../common/empirical.rs"]
mod empirical;
mod model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    demo::run(
        "seasonal_shift_persistence",
        &model::SeasonalShiftPersistence,
        &chapkit::NoConfig {},
    )
}
