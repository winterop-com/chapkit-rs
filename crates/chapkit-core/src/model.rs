use chapkit_dataframe::DataFrame;
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{Geo, ModelInfo, Result, RunInfo, SearchSpace};

/// Inputs to [`Model::train`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TrainInput<'a> {
    /// Training data: `time_period`, `location`, target and covariates.
    pub data: &'a DataFrame,
    /// Resolved run settings.
    pub run: &'a RunInfo,
    /// Optional GeoJSON for the locations.
    pub geo: Option<&'a Geo>,
}

impl<'a> TrainInput<'a> {
    /// Build a train input.
    pub fn new(data: &'a DataFrame, run: &'a RunInfo, geo: Option<&'a Geo>) -> Self {
        Self { data, run, geo }
    }
}

/// Inputs to [`Model::predict`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PredictInput<'a> {
    /// Observed history up to the forecast origin.
    pub historic: &'a DataFrame,
    /// One row per (`location`, `time_period`) to forecast, with future covariates.
    pub future: &'a DataFrame,
    /// Resolved run settings.
    pub run: &'a RunInfo,
    /// Optional GeoJSON for the locations.
    pub geo: Option<&'a Geo>,
}

impl<'a> PredictInput<'a> {
    /// Build a predict input.
    pub fn new(
        historic: &'a DataFrame,
        future: &'a DataFrame,
        run: &'a RunInfo,
        geo: Option<&'a Geo>,
    ) -> Self {
        Self {
            historic,
            future,
            run,
            geo,
        }
    }
}

/// A forecasting model, called in-process.
///
/// `train` and `predict` are plain synchronous CPU functions with no IO. The
/// HTTP service runs them on a blocking thread; explainers and HPO call them
/// directly in tight loops, so they should be cheap to call repeatedly.
pub trait Model: Send + Sync + 'static {
    /// User-facing options. Its JSON schema becomes `GET /api/v1/configs/$schema`.
    /// Unknown fields (such as the reserved `prediction_periods`) must be ignored,
    /// so do not use `deny_unknown_fields`.
    type Config: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static;

    /// The trained state. Serialized when the service stores a training artifact.
    type Trained: Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Static description of the model.
    fn info(&self) -> ModelInfo;

    /// Hyperparameters worth searching, in chap-core's format.
    fn search_space(&self) -> Option<SearchSpace> {
        None
    }

    /// True when each location is forecast independently of the others.
    /// Explainers use this to batch many perturbations into one `predict` call
    /// by giving each perturbation its own pseudo-location.
    fn locations_independent(&self) -> bool {
        true
    }

    /// Fit the model.
    fn train(&self, config: &Self::Config, input: TrainInput<'_>) -> Result<Self::Trained>;

    /// Forecast. Returns `time_period`, `location` and `sample_0..N` columns,
    /// one row per row of `input.future` (see [`crate::prediction_frame`]).
    fn predict(
        &self,
        trained: &Self::Trained,
        config: &Self::Config,
        input: PredictInput<'_>,
    ) -> Result<DataFrame>;
}

/// Config for models without user options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, serde::Deserialize, JsonSchema)]
pub struct NoConfig {}
