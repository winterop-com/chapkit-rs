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
