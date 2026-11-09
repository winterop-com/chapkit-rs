//! Persistence for chapkit-rs.
//!
//! The [`Store`] trait is the interface the service uses; backends implement it.
//! [`SqliteStore`] (rusqlite, bundled SQLite) is the default. Other engines that
//! speak SQLite's file format, such as turso, can be added behind the same trait.
//!
//! Stored things mirror chapkit's public resources:
//! - configs: `{id, name, data, tags, created_at, updated_at}`
//! - artifacts: a tree (`parent_id`, `level`) of JSON `data`, plus an optional binary
//!   `content` blob (for example a safetensors checkpoint) kept out of the JSON
//! - links from a config to the root artifacts trained with it

mod sqlite;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use sqlite::SqliteStore;

/// Store errors. Driver details are kept for logs, never sent to clients.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// A row with this id already exists, or a constraint was violated.
    #[error("conflict: {0}")]
    Conflict(String),
    /// Referenced row does not exist.
    #[error("not found: {0}")]
    NotFound(String),
    /// Invalid request (for example linking a non-root artifact).
    #[error("invalid: {0}")]
    Invalid(String),
    /// Database failure.
    #[error("database: {0}")]
    Database(String),
}

/// Result alias.
pub type StoreResult<T> = Result<T, StoreError>;

/// A stored config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigRow {
    pub id: String,
    pub name: String,
    pub data: Value,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A stored artifact. `content` is only loaded by [`Store::artifact_content`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRow {
    pub id: String,
    pub parent_id: Option<String>,
    pub level: i64,
    pub data: Value,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A new artifact, with an optional binary blob.
#[derive(Debug, Clone)]
pub struct NewArtifact {
    pub id: String,
    pub parent_id: Option<String>,
    pub data: Value,
    pub tags: Vec<String>,
    pub content: Option<Vec<u8>>,
    pub created_at: String,
}

/// Offset/limit for list queries.
#[derive(Debug, Clone, Copy)]
pub struct Page {
    pub offset: u64,
    pub limit: u64,
}

/// Maximum artifact tree depth, as chapkit.
pub const MAX_DEPTH: i64 = 64;

/// The persistence interface.
#[async_trait]
pub trait Store: Send + Sync + 'static {
    /// Check the database answers (`SELECT 1`).
    async fn ping(&self) -> StoreResult<()>;

    /// Insert a config; [`StoreError::Conflict`] if the id exists.
    async fn insert_config(&self, row: ConfigRow) -> StoreResult<()>;
    /// One config.
    async fn get_config(&self, id: &str) -> StoreResult<Option<ConfigRow>>;
    /// Configs ordered by id (creation order), optionally paged, with the total count.
    async fn list_configs(&self, page: Option<Page>) -> StoreResult<(Vec<ConfigRow>, u64)>;
    /// Replace name, data and tags; returns the updated row, `None` if missing.
    async fn update_config(&self, row: ConfigRow) -> StoreResult<Option<ConfigRow>>;
    /// Delete a config and every artifact tree linked to it. `false` if missing.
    async fn delete_config(&self, id: &str) -> StoreResult<bool>;

    /// Insert an artifact. `level` is the parent's level + 1 (0 for roots).
    async fn insert_artifact(&self, a: NewArtifact) -> StoreResult<ArtifactRow>;
    /// One artifact, without its blob.
    async fn get_artifact(&self, id: &str) -> StoreResult<Option<ArtifactRow>>;
    /// An artifact's blob, if it has one.
    async fn artifact_content(&self, id: &str) -> StoreResult<Option<Vec<u8>>>;
    /// Artifacts ordered by id, optionally paged, with the total count.
    async fn list_artifacts(&self, page: Option<Page>) -> StoreResult<(Vec<ArtifactRow>, u64)>;
    /// Replace data and tags; `None` if missing.
    async fn update_artifact(
        &self,
        id: &str,
        data: Value,
        tags: Vec<String>,
        updated_at: String,
    ) -> StoreResult<Option<ArtifactRow>>;
    /// Delete one artifact (children keep existing with `parent_id` cleared). `false` if missing.
    async fn delete_artifact(&self, id: &str) -> StoreResult<bool>;
    /// Direct children, ordered by id.
    async fn children(&self, id: &str) -> StoreResult<Vec<ArtifactRow>>;

    /// Link a root artifact to a config.
    async fn link(&self, config_id: &str, artifact_id: &str) -> StoreResult<()>;
    /// Remove a link. `false` if there was none.
    async fn unlink(&self, config_id: &str, artifact_id: &str) -> StoreResult<bool>;
    /// Root artifacts linked to a config.
    async fn linked_artifacts(&self, config_id: &str) -> StoreResult<Vec<ArtifactRow>>;
    /// The config linked to the root of an artifact's tree.
    async fn config_for_artifact(&self, artifact_id: &str) -> StoreResult<Option<ConfigRow>>;
}

/// Open a store from a database URL. Accepts `:memory:`, a file path, `sqlite:///path`,
/// `sqlite://path` and chapkit's `sqlite+aiosqlite:///path`. `None` or empty means in-memory.
pub fn open(url: Option<&str>) -> StoreResult<SqliteStore> {
    match url.map(str::trim).filter(|u| !u.is_empty()) {
        None | Some(":memory:") => SqliteStore::in_memory(),
        Some(u) => {
            let path = u
                .strip_prefix("sqlite+aiosqlite:///")
                .or_else(|| u.strip_prefix("sqlite:///"))
                .or_else(|| u.strip_prefix("sqlite://"))
                .unwrap_or(u);
            if path == ":memory:" {
                SqliteStore::in_memory()
            } else {
                SqliteStore::open(path)
            }
        }
    }
}
