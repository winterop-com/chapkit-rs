//! Request and response bodies of the chapkit HTTP API.

use chapkit_core::{DataFrame, Geo};
use chapkit_store::{ArtifactRow, ConfigRow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

/// A config to create or replace.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ConfigIn {
    /// Optional client-chosen id (ULID).
    pub id: Option<String>,
    /// Config name.
    pub name: String,
    /// Model options, plus chapkit's `prediction_periods` and `additional_continuous_covariates`.
    #[schema(value_type = Object)]
    pub data: Value,
    /// Tags.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// A stored config.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ConfigOut {
    pub id: String,
    pub created_at: String,
    pub updated_at: String,
    pub tags: Vec<String>,
    pub name: String,
    #[schema(value_type = Object)]
    pub data: Value,
}

impl From<ConfigRow> for ConfigOut {
    fn from(r: ConfigRow) -> Self {
        Self {
            id: r.id,
            created_at: r.created_at,
            updated_at: r.updated_at,
            tags: r.tags,
            name: r.name,
            data: r.data,
        }
    }
}

/// An artifact to create or replace.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ArtifactIn {
    pub id: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[schema(value_type = Object)]
    pub data: Value,
    pub parent_id: Option<String>,
    /// Ignored: the level is computed from the parent.
    pub level: Option<i64>,
}

/// A stored artifact.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ArtifactOut {
    pub id: String,
    pub created_at: String,
    pub updated_at: String,
    pub tags: Vec<String>,
    #[schema(value_type = Object)]
    pub data: Value,
    pub parent_id: Option<String>,
    pub level: i64,
}

impl From<ArtifactRow> for ArtifactOut {
    fn from(r: ArtifactRow) -> Self {
        Self {
            id: r.id,
            created_at: r.created_at,
            updated_at: r.updated_at,
            tags: r.tags,
            data: r.data,
            parent_id: r.parent_id,
            level: r.level,
        }
    }
}

/// An artifact in a tree view.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ArtifactTreeNode {
    #[serde(flatten)]
    pub artifact: ArtifactOut,
    pub level_label: Option<String>,
    pub hierarchy: Option<String>,
    #[schema(no_recursion)]
    pub children: Option<Vec<ArtifactTreeNode>>,
}

/// `{artifact_id}` for config link operations.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct LinkArtifact {
    pub artifact_id: String,
}

/// Runtime context chap-core attaches to train and predict requests.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
pub struct RunInfoIn {
    /// Forecast horizon (`prediction_length` is accepted too).
    #[serde(alias = "prediction_length")]
    pub prediction_periods: Option<u32>,
    pub additional_continuous_covariates: Option<Vec<String>>,
    pub future_covariate_origin: Option<String>,
}

/// `POST /api/v1/ml/$train`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct TrainRequest {
    pub config_id: String,
    /// Training data, chapkit DataFrame JSON.
    #[schema(value_type = Object)]
    pub data: DataFrame,
    #[schema(value_type = Option<Object>)]
    pub geo: Option<Geo>,
    pub run_info: Option<RunInfoIn>,
}

/// `POST /api/v1/ml/$predict` and `$predict-sync`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct PredictRequest {
    pub artifact_id: String,
    #[schema(value_type = Object)]
    pub historic: DataFrame,
    #[schema(value_type = Object)]
    pub future: DataFrame,
    #[schema(value_type = Option<Object>)]
    pub geo: Option<Geo>,
    pub run_info: Option<RunInfoIn>,
}

/// `POST /api/v1/ml/$explain`: explain one forecast cell of a trained model.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ExplainRequest {
    pub artifact_id: String,
    #[schema(value_type = Object)]
    pub historic: DataFrame,
    #[schema(value_type = Object)]
    pub future: DataFrame,
    /// Passed to every model call, as in `$predict`.
    #[schema(value_type = Option<Object>)]
    pub geo: Option<Geo>,
    pub run_info: Option<RunInfoIn>,
    /// Location of the cell to explain.
    pub location: String,
    /// Period of the cell to explain.
    pub time_period: String,
    /// Covariates to explain (default: every numeric covariate in `future`).
    pub features: Option<Vec<String>>,
    /// `temporal` (default): the cell's covariates, their earlier values and the target's
    /// history; `row`: the cell's covariates only.
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub scope: Option<chapkit_xai::Scope>,
    /// Historic periods perturbed under `temporal`. Default 3, at most 52.
    pub lags: Option<usize>,
    /// `lime` (default) or `shap`.
    #[serde(default)]
    #[schema(value_type = String)]
    pub method: chapkit_xai::Method,
    /// Perturbations (LIME) or sampled coalitions (SHAP above 12 features). Default 500.
    pub n_samples: Option<usize>,
    pub seed: Option<u64>,
    pub kernel_width: Option<f64>,
}

/// `$explain` result.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ExplainResponse {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub explanation: chapkit_xai::ForecastExplanation,
    /// Wall time of the explanation, all model calls included.
    pub elapsed_ms: f64,
}

/// 202 response of `$train` and `$predict`.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct JobSubmitted {
    pub job_id: String,
    pub artifact_id: String,
    pub message: String,
}

/// `POST /api/v1/ml/$validate`: a train or predict body plus `type`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ValidateRequest {
    /// `train` or `predict`.
    #[serde(rename = "type")]
    pub kind: String,
    pub config_id: Option<String>,
    pub artifact_id: Option<String>,
    #[schema(value_type = Option<Object>)]
    pub data: Option<DataFrame>,
    #[schema(value_type = Option<Object>)]
    pub historic: Option<DataFrame>,
    #[schema(value_type = Option<Object>)]
    pub future: Option<DataFrame>,
    /// Passed to the model's own checks.
    #[schema(value_type = Option<Object>)]
    pub geo: Option<Geo>,
    pub run_info: Option<RunInfoIn>,
}

/// One validation finding.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Diagnostic {
    /// `error`, `warning` or `info`.
    pub severity: String,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
}

/// `$validate` result.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ValidateResponse {
    pub valid: bool,
    pub diagnostics: Vec<Diagnostic>,
}

/// Backtest settings of `$backtest` and `$hpo`: chap-core's `BacktestParams` and its dataset
/// preparation, each optional.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
pub struct BacktestSettings {
    /// Periods per forecast (default: the config's `prediction_periods`, else 3).
    pub periods: Option<usize>,
    /// Splits (default 7).
    pub splits: Option<usize>,
    /// Periods between splits (default 1).
    pub stride: Option<usize>,
    /// Trainings (default 1).
    pub retrain: Option<usize>,
    /// Future weather: `climatology` (default), `damped_persistence` or `observed`.
    pub weather: Option<String>,
    /// Fill gaps in a location's periods (default true).
    pub fill_gaps: Option<bool>,
    /// Pad locations to the full period range (default true).
    pub pad: Option<bool>,
    /// Leave out locations with no target before the test windows (default true).
    pub drop_untrained: Option<bool>,
}

/// `POST /api/v1/ml/$backtest`: a backtest of the model on `data`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct BacktestRequest {
    /// A stored config; or give `config`, or neither for the defaults.
    pub config_id: Option<String>,
    /// Model options, as a config's `data`.
    #[schema(value_type = Option<Object>)]
    pub config: Option<Value>,
    /// The dataset, chapkit DataFrame JSON.
    #[schema(value_type = Object)]
    pub data: DataFrame,
    #[serde(flatten)]
    pub settings: BacktestSettings,
}

/// `POST /api/v1/ml/$hpo`: a hyperparameter search over the model's search space, each trial a
/// backtest on `data`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct HpoRequest {
    /// A stored config whose options every candidate is merged into; or give `config`.
    pub config_id: Option<String>,
    /// Model options, as a config's `data`.
    #[schema(value_type = Option<Object>)]
    pub config: Option<Value>,
    /// The dataset, chapkit DataFrame JSON.
    #[schema(value_type = Object)]
    pub data: DataFrame,
    #[serde(flatten)]
    pub settings: BacktestSettings,
    /// Trials (default 20).
    pub trials: Option<usize>,
    /// `random` (default) or `grid`.
    pub method: Option<String>,
    /// Metric to minimize, a chap-core `metric_id` (default `crps`).
    pub metric: Option<String>,
    /// Seed of the random searcher (default 0).
    pub seed: Option<u64>,
    /// Let `grid` step through ranges; chap-core's grid takes choices only (default false).
    #[serde(default)]
    pub grid_ranges: bool,
    /// Points a continuous float range is split into by `grid` with `grid_ranges` (default 5).
    pub grid_points: Option<usize>,
    /// Store the best config as a new config (default false).
    #[serde(default)]
    pub create_config: bool,
    /// Name of that config (default `hpo <artifact id>`).
    pub config_name: Option<String>,
}
