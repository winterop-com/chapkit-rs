use serde::{Deserialize, Serialize};

/// Temporal resolution a model supports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PeriodType {
    /// Weekly periods (`2024W03`).
    Weekly,
    /// Monthly periods (`2024-01`).
    Monthly,
    /// Either.
    #[default]
    Any,
}

/// Author's self-assessment of model maturity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssessedStatus {
    /// Not assessed.
    Gray,
    /// Do not use.
    Red,
    /// Experimental.
    Orange,
    /// Usable with care.
    Yellow,
    /// Recommended.
    Green,
}

/// Descriptive model metadata (chapkit `ModelMetadata`). Every field is always
/// serialized, nulls included, matching chapkit's `/api/v1/info` output.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub author: Option<String>,
    pub author_note: Option<String>,
    pub author_assessed_status: Option<AssessedStatus>,
    pub contact_email: Option<String>,
    pub organization: Option<String>,
    pub organization_logo_url: Option<String>,
    pub citation_info: Option<String>,
    pub repository_url: Option<String>,
    pub documentation_url: Option<String>,
}

/// What a model declares about itself: the model-specific part of chapkit's `MLServiceInfo`.
/// Service identity (id, display name, version) is supplied by the service builder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub period_type: PeriodType,
    pub min_prediction_periods: u32,
    pub max_prediction_periods: u32,
    pub allow_free_additional_continuous_covariates: bool,
    pub required_covariates: Vec<String>,
    pub requires_geo: bool,
    pub target: String,
    pub model_metadata: ModelMetadata,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            period_type: PeriodType::Any,
            min_prediction_periods: 0,
            max_prediction_periods: 100,
            allow_free_additional_continuous_covariates: false,
            required_covariates: Vec::new(),
            requires_geo: false,
            target: chapkit_dataframe::cols::DISEASE_CASES.to_owned(),
            model_metadata: ModelMetadata::default(),
        }
    }
}
