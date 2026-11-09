//! chap-core's built-in baselines (`chap_core/models/builtin`), ported to chapkit-rs.
//!
//! Same rules as chap-core: predictions cover exactly the future rows, every sample
//! is finite, no randomness (samples are quantiles of observed values), and data a
//! baseline cannot handle is rejected with an invalid-input error. The outputs match
//! chap-core number for number (`tests/parity.rs`).
//!
//! Binaries: `global-median`, `seasonal-median`, `seasonal-shift-persistence`. Each
//! serves its model as a chapkit-compatible service, or runs once in-process with `demo`.

pub mod demo;
pub mod empirical;
mod global_median;
mod seasonal_median;
mod seasonal_shift_persistence;

pub use global_median::GlobalMedian;
pub use seasonal_median::{SeasonalMedian, SeasonalMedianTrained};
pub use seasonal_shift_persistence::{SeasonalShiftPersistence, SeasonalShiftTrained};

/// Serve `model` as a chapkit service, or run the in-process demo when the first
/// argument is `demo`.
pub async fn main_for<M: chapkit::Model>(
    id: &str,
    title: &str,
    model: M,
    config: M::Config,
) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("demo") {
        return demo::run(id, &model, &config);
    }
    chapkit::Service::new(id, title, model)
        .description(format!(
            "Rust port of chap-core's {} baseline",
            id.replace('-', "_")
        ))
        .serve()
        .await?;
    Ok(())
}
