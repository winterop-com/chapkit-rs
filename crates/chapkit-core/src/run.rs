use serde::{Deserialize, Serialize};

/// Per-run settings, resolved by the service before a model is called.
///
/// `prediction_periods` is already final here: the service takes it from the
/// request's `run_info`, falling back to the config, and checks it against the
/// model's min/max bounds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunInfo {
    /// Forecast horizon in periods.
    pub prediction_periods: u32,
    /// Extra covariate columns the caller wants the model to use.
    #[serde(default)]
    pub additional_continuous_covariates: Vec<String>,
    /// Where future covariate values came from (passed through from Chap).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub future_covariate_origin: Option<String>,
}

impl RunInfo {
    /// A run with the given horizon and no extra covariates.
    pub fn with_periods(prediction_periods: u32) -> Self {
        Self {
            prediction_periods,
            ..Self::default()
        }
    }
}

/// A GeoJSON `FeatureCollection`, passed through untouched.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Geo(pub serde_json::Value);
