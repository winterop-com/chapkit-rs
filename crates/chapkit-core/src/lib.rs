//! The model contract for chapkit-rs.
//!
//! Pure and IO-free: no async, no HTTP, no database. A model implemented
//! against [`Model`] can be served over the chapkit HTTP API by the `chapkit`
//! crate, and called directly, in-process, by explainers and HPO.

pub mod checkpoint;
mod error;
mod info;
pub mod locations;
mod model;
pub mod period;
mod prediction;
pub mod rng;
mod run;
pub mod search;
pub mod synth;
pub mod testing;
mod validate;

pub use chapkit_dataframe::{self as frame, Column, DType, DataFrame, FrameError, cols};
pub use checkpoint::{Checkpoint, Provenance, Tensor, TensorData};
pub use error::{Error, Result};
pub use info::{AssessedStatus, ModelInfo, ModelMetadata, PeriodType};
pub use model::{Model, NoConfig, PredictInput, TrainInput};
pub use period::Period;
pub use prediction::prediction_frame;
pub use run::{Geo, RunInfo};
pub use search::{Param, SearchSpace};
pub use validate::{Diagnostic, Severity};
