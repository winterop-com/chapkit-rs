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
//!   and a listing (`data` without `content`, plus a [`summary`] for backtests and forecasts) with
//!   the filterable `type` and `content.dataset_id`, written beside each artifact so listing and
//!   filtering never read or parse `data`. Databases from before listings get them on open.
//! - links from a config to the root artifacts trained with it
//! - datasets (a chapkit-rs extension): a summary as JSON `meta`, and the data as a blob

mod sqlite;
#[cfg(feature = "turso")]
mod turso;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use sqlite::SqliteStore;
#[cfg(feature = "turso")]
pub use turso::TursoStore;

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

/// A stored dataset, without its data ([`Store::dataset_data`]). `meta` is the summary the
/// service computed when the dataset was uploaded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetRow {
    pub id: String,
    pub name: String,
    pub meta: Value,
    pub created_at: String,
    pub updated_at: String,
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
    /// Configs ordered by id (creation order), optionally only those whose name holds `name`
    /// (ignoring the case of ASCII letters), optionally paged, with the count of those kept.
    async fn list_configs(
        &self,
        name: Option<&str>,
        page: Option<Page>,
    ) -> StoreResult<(Vec<ConfigRow>, u64)>;
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
    /// Artifact listings ordered by id: `data` is the artifact's without `content`, with
    /// `summary` ([`summary`]) when it has one. Optionally only those whose `data.type` is `kind`
    /// and, with it, whose `data.content.dataset_id` is one of `datasets`; optionally paged; with
    /// the count of those kept. Reads neither `data` nor `content`.
    async fn list_artifact_summaries(
        &self,
        kind: Option<&str>,
        datasets: Option<&[String]>,
        page: Option<Page>,
    ) -> StoreResult<(Vec<ArtifactRow>, u64)>;
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

    /// Insert a dataset and its data; [`StoreError::Conflict`] if the id exists.
    async fn insert_dataset(&self, row: DatasetRow, data: Vec<u8>) -> StoreResult<()>;
    /// One dataset, without its data.
    async fn get_dataset(&self, id: &str) -> StoreResult<Option<DatasetRow>>;
    /// A dataset's data.
    async fn dataset_data(&self, id: &str) -> StoreResult<Option<Vec<u8>>>;
    /// Datasets ordered by id, optionally only those whose name holds `name` (ignoring
    /// the case of ASCII letters), optionally paged, with the count of those kept.
    async fn list_datasets(
        &self,
        name: Option<&str>,
        page: Option<Page>,
    ) -> StoreResult<(Vec<DatasetRow>, u64)>;
    /// Delete a dataset. `false` if missing.
    async fn delete_dataset(&self, id: &str) -> StoreResult<bool>;
}

/// The `WHERE` clause (with a leading space, or empty) and its parameters, numbered from `?1`,
/// that keep the rows whose `name` holds `name`, ignoring the case of ASCII letters as SQLite's
/// `lower` does. Shared by the backends so they filter alike.
pub(crate) fn name_filter(name: Option<&str>) -> (String, Vec<String>) {
    match name {
        Some(name) => (
            " WHERE instr(lower(name), ?1) > 0".into(),
            vec![name.to_ascii_lowercase()],
        ),
        None => (String::new(), Vec::new()),
    }
}

/// The `WHERE` clause (with a leading space, or empty) and its parameters for listings of a type,
/// and with `datasets` only those made from one of them. `datasets` must not be empty.
pub(crate) fn type_filter(
    kind: Option<&str>,
    datasets: Option<&[String]>,
) -> (String, Vec<String>) {
    let Some(kind) = kind else {
        return (String::new(), Vec::new());
    };
    let mut sql = " WHERE l.kind = ?1".to_owned();
    let mut params = vec![kind.to_owned()];
    if let Some(ids) = datasets {
        let marks: Vec<String> = (2..ids.len() + 2).map(|i| format!("?{i}")).collect();
        sql.push_str(" AND l.dataset_id IN (");
        sql.push_str(&marks.join(", "));
        sql.push(')');
        params.extend(ids.iter().cloned());
    }
    (sql, params)
}

/// The listing of a backtest or forecast carries this summary of its content, enough to list it
/// without it: `dataset_id`, and for a backtest (`ml_backtest`) its `data`, `options` and
/// `aggregate`, for a forecast (`ml_forecast`) its `periods`, `weather`, first period and number
/// of locations. Other artifacts have none.
pub fn summary(data: &Value) -> Option<Value> {
    let content = data.get("content")?;
    let pick = |keys: &[&str]| -> serde_json::Map<String, Value> {
        keys.iter()
            .filter_map(|k| content.get(*k).map(|v| ((*k).to_owned(), v.clone())))
            .collect()
    };
    match data.get("type").and_then(Value::as_str)? {
        "ml_backtest" => Some(Value::Object(pick(&[
            "dataset_id",
            "data",
            "options",
            "aggregate",
        ]))),
        "ml_forecast" => {
            let mut out = pick(&["dataset_id", "periods", "weather"]);
            let rows = content.get("forecasts").and_then(Value::as_array);
            let first = rows
                .and_then(|r| r.first())
                .and_then(|r| r.get("time_period"))
                .cloned();
            let locations = rows.map(|r| {
                r.iter()
                    .filter_map(|row| row.get("location").and_then(Value::as_str))
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            });
            out.insert("first_period".into(), first.unwrap_or(Value::Null));
            out.insert("locations".into(), locations.into());
            Some(Value::Object(out))
        }
        _ => None,
    }
}

/// Whether a listing should carry a [`summary`] but does not (a backtest or forecast written
/// without one): the listing is then made again from the artifact's data, once.
pub(crate) fn lacks_summary(listing: &Value) -> bool {
    matches!(
        listing.get("type").and_then(Value::as_str),
        Some("ml_backtest" | "ml_forecast")
    ) && listing.get("summary").is_none()
}

/// What a listing row stores for `data`: its `type`, its `content.dataset_id`, and the listing
/// itself (`data` without `content`, with its [`summary`]) as JSON text.
pub(crate) struct Listing {
    pub kind: Option<String>,
    pub dataset_id: Option<String>,
    pub text: String,
}

impl Listing {
    pub(crate) fn of(data: &Value) -> StoreResult<Self> {
        let text = |v: Option<&Value>| v.and_then(Value::as_str).map(str::to_owned);
        let mut listing = match data {
            Value::Object(m) => Value::Object(
                m.iter()
                    .filter(|(k, _)| *k != "content")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ),
            other => other.clone(),
        };
        if let (Some(summary), Value::Object(m)) = (summary(data), &mut listing) {
            m.insert("summary".into(), summary);
        }
        Ok(Self {
            kind: text(data.get("type")),
            dataset_id: text(data.pointer("/content/dataset_id")),
            text: serde_json::to_string(&listing)
                .map_err(|e| StoreError::Database(e.to_string()))?,
        })
    }
}

/// Open a store from a database URL.
///
/// - SQLite (rusqlite): `:memory:`, a file path, `sqlite:///path`, `sqlite://path`, chapkit's
///   `sqlite+aiosqlite:///path`. `None`, empty, `sqlite://` and sqlx's `sqlite::memory:` mean
///   in-memory SQLite; any other `scheme://` URL is an error.
/// - turso (feature `turso`): `turso::memory:`, `turso:///path`, `turso://path`.
///
/// As in SQLAlchemy (and so Python chapkit), `sqlite:///data/x.db` is relative to the working
/// directory and an absolute path takes four slashes: `sqlite:////var/lib/x.db`.
pub async fn open(url: Option<&str>) -> StoreResult<std::sync::Arc<dyn Store>> {
    let url = url.map(str::trim).filter(|u| !u.is_empty());
    if let Some(rest) = url.and_then(|u| u.strip_prefix("turso:")) {
        let path = rest
            .strip_prefix("///")
            .or_else(|| rest.strip_prefix("//"))
            .unwrap_or(rest);
        return open_turso(if path.is_empty() { ":memory:" } else { path }).await;
    }
    Ok(std::sync::Arc::new(open_sqlite(url)?))
}

#[cfg(feature = "turso")]
async fn open_turso(path: &str) -> StoreResult<std::sync::Arc<dyn Store>> {
    Ok(std::sync::Arc::new(TursoStore::open(path).await?))
}

#[cfg(not(feature = "turso"))]
#[allow(clippy::unused_async)]
async fn open_turso(_: &str) -> StoreResult<std::sync::Arc<dyn Store>> {
    Err(StoreError::Invalid(
        "turso: URLs need chapkit-store built with the `turso` feature".into(),
    ))
}

/// Open a SQLite store from a database URL (see [`open`]).
pub fn open_sqlite(url: Option<&str>) -> StoreResult<SqliteStore> {
    match sqlite_path(url)? {
        None => SqliteStore::in_memory(),
        Some(path) => SqliteStore::open(path),
    }
}

/// The file a SQLite URL names, or `None` for an in-memory database. Besides `:memory:` and an
/// empty URL, the in-memory spellings of SQLAlchemy (`sqlite://`, `sqlite:///:memory:`) and sqlx
/// (`sqlite::memory:`, `sqlite:`) are understood, so none of them creates a file named after the
/// URL. Any other `scheme://` URL is an error.
fn sqlite_path(url: Option<&str>) -> StoreResult<Option<&str>> {
    let Some(u) = url.map(str::trim).filter(|u| !u.is_empty()) else {
        return Ok(None);
    };
    let path = u
        .strip_prefix("sqlite+aiosqlite:///")
        .or_else(|| u.strip_prefix("sqlite+aiosqlite://"))
        .or_else(|| u.strip_prefix("sqlite:///"))
        .or_else(|| u.strip_prefix("sqlite://"))
        .or_else(|| u.strip_prefix("sqlite:"));
    let path = match path {
        Some(p) => p,
        None if u.contains("://") => {
            return Err(StoreError::Invalid(format!(
                "{u}: not a SQLite URL (sqlite:///path, :memory:, or a file path)"
            )));
        }
        None => u,
    };
    Ok(match path {
        "" | ":memory:" => None,
        p => Some(p),
    })
}

#[cfg(test)]
mod tests {
    use super::sqlite_path;

    #[test]
    fn in_memory_spellings_never_name_a_file() {
        for url in [
            "",
            "  ",
            ":memory:",
            "sqlite::memory:",
            "sqlite:",
            "sqlite://",
            "sqlite:///:memory:",
            "sqlite+aiosqlite:///:memory:",
        ] {
            assert_eq!(sqlite_path(Some(url)).ok().flatten(), None, "{url:?}");
        }
        assert_eq!(sqlite_path(None).ok().flatten(), None);
    }

    #[test]
    fn file_urls_name_their_file() {
        for (url, path) in [
            ("data/chapkit.db", "data/chapkit.db"),
            ("sqlite:///data/chapkit.db", "data/chapkit.db"),
            ("sqlite:////var/lib/chapkit.db", "/var/lib/chapkit.db"),
            ("sqlite+aiosqlite:///data/chapkit.db", "data/chapkit.db"),
            ("sqlite:data/chapkit.db", "data/chapkit.db"),
        ] {
            assert_eq!(sqlite_path(Some(url)).ok().flatten(), Some(path), "{url:?}");
        }
    }

    #[test]
    fn other_schemes_are_refused() {
        assert!(sqlite_path(Some("postgres://localhost/chap")).is_err());
    }
}
