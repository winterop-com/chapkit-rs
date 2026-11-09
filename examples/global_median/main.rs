//! chap-core's `global_median` baseline, ported to Rust.
//!
//! cargo run --release --example global_median

#[path = "../common/demo.rs"]
mod demo;
#[path = "../common/empirical.rs"]
mod empirical;
mod model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    demo::run("global_median", &model::GlobalMedian, &chapkit::NoConfig {})
}
