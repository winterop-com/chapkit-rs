//! [`Store`] on turso, a Rust reimplementation of SQLite that reads and writes
//! SQLite's file format. Enabled with the `turso` feature; selected with a
//! `turso:` database URL (`turso::memory:`, `turso:///path/to.db`).
//!
//! Uses plain SQL only: tree walks and cascading deletes are done in code rather
//! than with recursive CTEs or foreign-key actions, so the backend does not depend
//! on which SQLite features turso implements.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value as Json;
use tokio::sync::Mutex;
use turso::{Builder, Connection, Value};

use crate::{
    ArtifactRow, ConfigRow, DatasetRow, MAX_DEPTH, NewArtifact, Page, Store, StoreError,
    StoreResult,
};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS configs (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    data       TEXT NOT NULL,
    tags       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    id         TEXT PRIMARY KEY,
    parent_id  TEXT,
    level      INTEGER NOT NULL,
    data       TEXT NOT NULL,
    content    BLOB,
    tags       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_parent ON artifacts(parent_id);
CREATE TABLE IF NOT EXISTS config_artifacts (
    config_id   TEXT NOT NULL,
    artifact_id TEXT NOT NULL PRIMARY KEY
);
CREATE INDEX IF NOT EXISTS config_artifacts_config ON config_artifacts(config_id);
CREATE TABLE IF NOT EXISTS datasets (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    meta       TEXT NOT NULL,
    data       BLOB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
";

const ARTIFACT_COLS: &str = "id, parent_id, level, data, tags, created_at, updated_at";
const CONFIG_COLS: &str = "id, name, data, tags, created_at, updated_at";
const DATASET_COLS: &str = "id, name, meta, created_at, updated_at";

/// turso store. One connection behind an async mutex, so multi-statement
/// operations are atomic with respect to other requests.
#[derive(Clone)]
pub struct TursoStore {
    conn: Arc<Mutex<Connection>>,
}

fn db(e: &turso::Error) -> StoreError {
    StoreError::Database(e.to_string())
}

fn text(v: &impl serde::Serialize) -> StoreResult<Value> {
    serde_json::to_string(v)
        .map(Value::Text)
        .map_err(|e| StoreError::Database(e.to_string()))
}

fn opt_text(v: Option<&str>) -> Value {
    v.map_or(Value::Null, |s| Value::Text(s.to_owned()))
}

fn s(v: &str) -> Value {
    Value::Text(v.to_owned())
}

fn as_str(v: Value) -> StoreResult<String> {
    match v {
        Value::Text(t) => Ok(t),
        other => Err(StoreError::Database(format!(
            "expected text, got {other:?}"
        ))),
    }
}

fn as_opt_str(v: Value) -> StoreResult<Option<String>> {
    match v {
        Value::Null => Ok(None),
        other => as_str(other).map(Some),
    }
}

fn as_json<T: serde::de::DeserializeOwned>(v: Value) -> StoreResult<T> {
    serde_json::from_str(&as_str(v)?)
        .map_err(|e| StoreError::Database(format!("stored JSON is invalid: {e}")))
}

async fn rows(c: &Connection, sql: &str, params: Vec<Value>) -> StoreResult<Vec<Vec<Value>>> {
    let mut rs = c.query(sql, params).await.map_err(|e| db(&e))?;
    let n = rs.column_count();
    let mut out = Vec::new();
    while let Some(row) = rs.next().await.map_err(|e| db(&e))? {
        out.push(
            (0..n)
                .map(|i| row.get_value(i).unwrap_or(Value::Null))
                .collect(),
        );
    }
    Ok(out)
}

fn config_from(mut r: Vec<Value>) -> StoreResult<ConfigRow> {
    if r.len() < 6 {
        return Err(StoreError::Database("short config row".into()));
    }
    let mut it = r.drain(..);
    let mut next = || it.next().unwrap_or(Value::Null);
    Ok(ConfigRow {
        id: as_str(next())?,
        name: as_str(next())?,
        data: as_json(next())?,
        tags: as_json(next())?,
        created_at: as_str(next())?,
        updated_at: as_str(next())?,
    })
}

fn artifact_from(mut r: Vec<Value>) -> StoreResult<ArtifactRow> {
    if r.len() < 7 {
        return Err(StoreError::Database("short artifact row".into()));
    }
    let mut it = r.drain(..);
    let mut next = || it.next().unwrap_or(Value::Null);
    Ok(ArtifactRow {
        id: as_str(next())?,
        parent_id: as_opt_str(next())?,
        level: match next() {
            Value::Integer(i) => i,
            other => {
                return Err(StoreError::Database(format!(
                    "expected integer level, got {other:?}"
                )));
            }
        },
        data: as_json(next())?,
        tags: as_json(next())?,
        created_at: as_str(next())?,
        updated_at: as_str(next())?,
    })
}

fn dataset_from(mut r: Vec<Value>) -> StoreResult<DatasetRow> {
    if r.len() < 5 {
        return Err(StoreError::Database("short dataset row".into()));
    }
    let mut it = r.drain(..);
    let mut next = || it.next().unwrap_or(Value::Null);
    Ok(DatasetRow {
        id: as_str(next())?,
        name: as_str(next())?,
        meta: as_json(next())?,
        created_at: as_str(next())?,
        updated_at: as_str(next())?,
    })
}

async fn count(c: &Connection, sql: &str, params: Vec<Value>) -> StoreResult<u64> {
    Ok(
        match rows(c, sql, params)
            .await?
            .into_iter()
            .next()
            .and_then(|r| r.into_iter().next())
        {
            Some(Value::Integer(n)) => n.max(0) as u64,
            _ => 0,
        },
    )
}

async fn get_config(c: &Connection, id: &str) -> StoreResult<Option<ConfigRow>> {
    let r = rows(
        c,
        &format!("SELECT {CONFIG_COLS} FROM configs WHERE id = ?1"),
        vec![s(id)],
    )
    .await?;
    r.into_iter().next().map(config_from).transpose()
}

async fn get_artifact(c: &Connection, id: &str) -> StoreResult<Option<ArtifactRow>> {
    let r = rows(
        c,
        &format!("SELECT {ARTIFACT_COLS} FROM artifacts WHERE id = ?1"),
        vec![s(id)],
    )
    .await?;
    r.into_iter().next().map(artifact_from).transpose()
}

async fn exec(c: &Connection, sql: &str, params: Vec<Value>) -> StoreResult<u64> {
    c.execute(sql, params).await.map_err(|e| db(&e))
}

/// Ids of `root` and all its descendants.
async fn subtree_ids(c: &Connection, root: &str) -> StoreResult<Vec<String>> {
    let mut all = vec![root.to_owned()];
    let mut frontier = vec![root.to_owned()];
    for _ in 0..=MAX_DEPTH {
        let mut next = Vec::new();
        for id in &frontier {
            for r in rows(
                c,
                "SELECT id FROM artifacts WHERE parent_id = ?1",
                vec![s(id)],
            )
            .await?
            {
                if let Some(v) = r.into_iter().next() {
                    next.push(as_str(v)?);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        all.extend(next.iter().cloned());
        frontier = next;
    }
    Ok(all)
}

async fn delete_one_artifact(c: &Connection, id: &str) -> StoreResult<u64> {
    exec(
        c,
        "UPDATE artifacts SET parent_id = NULL WHERE parent_id = ?1",
        vec![s(id)],
    )
    .await?;
    exec(
        c,
        "DELETE FROM config_artifacts WHERE artifact_id = ?1",
        vec![s(id)],
    )
    .await?;
    exec(c, "DELETE FROM artifacts WHERE id = ?1", vec![s(id)]).await
}

fn page_sql(page: Option<Page>) -> String {
    page.map_or(String::new(), |p| {
        format!(" LIMIT {} OFFSET {}", p.limit, p.offset)
    })
}

impl TursoStore {
    /// Open (creating if needed) a database: `:memory:` or a file path.
    pub async fn open(path: &str) -> StoreResult<Self> {
        if path != ":memory:"
            && let Some(dir) = std::path::Path::new(path)
                .parent()
                .filter(|d| !d.as_os_str().is_empty())
        {
            std::fs::create_dir_all(dir).map_err(|e| StoreError::Database(e.to_string()))?;
        }
        let database = Builder::new_local(path).build().await.map_err(|e| db(&e))?;
        let conn = database.connect().map_err(|e| db(&e))?;
        conn.execute_batch(SCHEMA).await.map_err(|e| db(&e))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

#[async_trait]
impl Store for TursoStore {
    async fn ping(&self) -> StoreResult<()> {
        let c = self.conn.lock().await;
        rows(&c, "SELECT 1", vec![]).await.map(|_| ())
    }

    async fn insert_config(&self, row: ConfigRow) -> StoreResult<()> {
        let c = self.conn.lock().await;
        if get_config(&c, &row.id).await?.is_some() {
            return Err(StoreError::Conflict(format!(
                "Config with id {} already exists",
                row.id
            )));
        }
        exec(
            &c,
            "INSERT INTO configs (id, name, data, tags, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            vec![s(&row.id), s(&row.name), text(&row.data)?, text(&row.tags)?, s(&row.created_at), s(&row.updated_at)],
        )
        .await
        .map(|_| ())
    }

    async fn get_config(&self, id: &str) -> StoreResult<Option<ConfigRow>> {
        get_config(&*self.conn.lock().await, id).await
    }

    async fn list_configs(&self, page: Option<Page>) -> StoreResult<(Vec<ConfigRow>, u64)> {
        let c = self.conn.lock().await;
        let total = match rows(&c, "SELECT COUNT(*) FROM configs", vec![])
            .await?
            .into_iter()
            .next()
            .and_then(|r| r.into_iter().next())
        {
            Some(Value::Integer(n)) => n.max(0) as u64,
            _ => 0,
        };
        let sql = format!(
            "SELECT {CONFIG_COLS} FROM configs ORDER BY id{}",
            page_sql(page)
        );
        let list = rows(&c, &sql, vec![])
            .await?
            .into_iter()
            .map(config_from)
            .collect::<StoreResult<_>>()?;
        Ok((list, total))
    }

    async fn update_config(&self, row: ConfigRow) -> StoreResult<Option<ConfigRow>> {
        let c = self.conn.lock().await;
        let n = exec(
            &c,
            "UPDATE configs SET name = ?2, data = ?3, tags = ?4, updated_at = ?5 WHERE id = ?1",
            vec![
                s(&row.id),
                s(&row.name),
                text(&row.data)?,
                text(&row.tags)?,
                s(&row.updated_at),
            ],
        )
        .await?;
        if n == 0 {
            Ok(None)
        } else {
            get_config(&c, &row.id).await
        }
    }

    async fn delete_config(&self, id: &str) -> StoreResult<bool> {
        let c = self.conn.lock().await;
        if get_config(&c, id).await?.is_none() {
            return Ok(false);
        }
        exec(&c, "BEGIN", vec![]).await?;
        let result = async {
            let roots = rows(
                &c,
                "SELECT artifact_id FROM config_artifacts WHERE config_id = ?1",
                vec![s(id)],
            )
            .await?;
            for r in roots {
                if let Some(v) = r.into_iter().next() {
                    for aid in subtree_ids(&c, &as_str(v)?).await? {
                        delete_one_artifact(&c, &aid).await?;
                    }
                }
            }
            exec(
                &c,
                "DELETE FROM config_artifacts WHERE config_id = ?1",
                vec![s(id)],
            )
            .await?;
            exec(&c, "DELETE FROM configs WHERE id = ?1", vec![s(id)]).await
        }
        .await;
        exec(
            &c,
            if result.is_ok() { "COMMIT" } else { "ROLLBACK" },
            vec![],
        )
        .await?;
        result.map(|n| n > 0)
    }

    async fn insert_artifact(&self, a: NewArtifact) -> StoreResult<ArtifactRow> {
        let c = self.conn.lock().await;
        let level = match &a.parent_id {
            None => 0,
            Some(p) => {
                let parent = get_artifact(&c, p).await?.ok_or_else(|| {
                    StoreError::NotFound(format!("Parent artifact {p} not found"))
                })?;
                if parent.level + 1 > MAX_DEPTH {
                    return Err(StoreError::Invalid(format!(
                        "Artifact tree deeper than {MAX_DEPTH} levels"
                    )));
                }
                parent.level + 1
            }
        };
        if get_artifact(&c, &a.id).await?.is_some() {
            return Err(StoreError::Conflict(format!(
                "Artifact with id {} already exists",
                a.id
            )));
        }
        exec(
            &c,
            "INSERT INTO artifacts (id, parent_id, level, data, content, tags, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
            vec![
                s(&a.id),
                opt_text(a.parent_id.as_deref()),
                Value::Integer(level),
                text(&a.data)?,
                a.content.map_or(Value::Null, Value::Blob),
                text(&a.tags)?,
                s(&a.created_at),
            ],
        )
        .await?;
        get_artifact(&c, &a.id)
            .await?
            .ok_or_else(|| StoreError::Database("artifact vanished after insert".into()))
    }

    async fn get_artifact(&self, id: &str) -> StoreResult<Option<ArtifactRow>> {
        get_artifact(&*self.conn.lock().await, id).await
    }

    async fn artifact_content(&self, id: &str) -> StoreResult<Option<Vec<u8>>> {
        let c = self.conn.lock().await;
        Ok(
            match rows(
                &c,
                "SELECT content FROM artifacts WHERE id = ?1",
                vec![s(id)],
            )
            .await?
            .into_iter()
            .next()
            .and_then(|r| r.into_iter().next())
            {
                Some(Value::Blob(b)) => Some(b),
                _ => None,
            },
        )
    }

    async fn list_artifacts(&self, page: Option<Page>) -> StoreResult<(Vec<ArtifactRow>, u64)> {
        let c = self.conn.lock().await;
        let total = match rows(&c, "SELECT COUNT(*) FROM artifacts", vec![])
            .await?
            .into_iter()
            .next()
            .and_then(|r| r.into_iter().next())
        {
            Some(Value::Integer(n)) => n.max(0) as u64,
            _ => 0,
        };
        let sql = format!(
            "SELECT {ARTIFACT_COLS} FROM artifacts ORDER BY id{}",
            page_sql(page)
        );
        let list = rows(&c, &sql, vec![])
            .await?
            .into_iter()
            .map(artifact_from)
            .collect::<StoreResult<_>>()?;
        Ok((list, total))
    }

    async fn list_artifacts_of_type(
        &self,
        kind: &str,
        page: Option<Page>,
    ) -> StoreResult<(Vec<ArtifactRow>, u64)> {
        const WHERE: &str = "WHERE json_extract(data, '$.type') = ?1";
        let c = self.conn.lock().await;
        let total = count(
            &c,
            &format!("SELECT COUNT(*) FROM artifacts {WHERE}"),
            vec![s(kind)],
        )
        .await?;
        let sql = format!(
            "SELECT {ARTIFACT_COLS} FROM artifacts {WHERE} ORDER BY id{}",
            page_sql(page)
        );
        let list = rows(&c, &sql, vec![s(kind)])
            .await?
            .into_iter()
            .map(artifact_from)
            .collect::<StoreResult<_>>()?;
        Ok((list, total))
    }

    async fn update_artifact(
        &self,
        id: &str,
        data: Json,
        tags: Vec<String>,
        updated_at: String,
    ) -> StoreResult<Option<ArtifactRow>> {
        let c = self.conn.lock().await;
        let n = exec(
            &c,
            "UPDATE artifacts SET data = ?2, tags = ?3, updated_at = ?4 WHERE id = ?1",
            vec![s(id), text(&data)?, text(&tags)?, s(&updated_at)],
        )
        .await?;
        if n == 0 {
            Ok(None)
        } else {
            get_artifact(&c, id).await
        }
    }

    async fn delete_artifact(&self, id: &str) -> StoreResult<bool> {
        let c = self.conn.lock().await;
        Ok(delete_one_artifact(&c, id).await? > 0)
    }

    async fn children(&self, id: &str) -> StoreResult<Vec<ArtifactRow>> {
        let c = self.conn.lock().await;
        rows(
            &c,
            &format!("SELECT {ARTIFACT_COLS} FROM artifacts WHERE parent_id = ?1 ORDER BY id"),
            vec![s(id)],
        )
        .await?
        .into_iter()
        .map(artifact_from)
        .collect()
    }

    async fn link(&self, config_id: &str, artifact_id: &str) -> StoreResult<()> {
        let c = self.conn.lock().await;
        let a = get_artifact(&c, artifact_id)
            .await?
            .ok_or_else(|| StoreError::NotFound(format!("Artifact {artifact_id} not found")))?;
        if a.level != 0 {
            return Err(StoreError::Invalid(format!(
                "Artifact {artifact_id} is not a root artifact (level {})",
                a.level
            )));
        }
        if get_config(&c, config_id).await?.is_none() {
            return Err(StoreError::NotFound(format!(
                "Config {config_id} not found"
            )));
        }
        exec(
            &c,
            "DELETE FROM config_artifacts WHERE artifact_id = ?1",
            vec![s(artifact_id)],
        )
        .await?;
        exec(
            &c,
            "INSERT INTO config_artifacts (config_id, artifact_id) VALUES (?1, ?2)",
            vec![s(config_id), s(artifact_id)],
        )
        .await
        .map(|_| ())
    }

    async fn unlink(&self, config_id: &str, artifact_id: &str) -> StoreResult<bool> {
        let c = self.conn.lock().await;
        Ok(exec(
            &c,
            "DELETE FROM config_artifacts WHERE config_id = ?1 AND artifact_id = ?2",
            vec![s(config_id), s(artifact_id)],
        )
        .await?
            > 0)
    }

    async fn linked_artifacts(&self, config_id: &str) -> StoreResult<Vec<ArtifactRow>> {
        let c = self.conn.lock().await;
        let ids = rows(
            &c,
            "SELECT artifact_id FROM config_artifacts WHERE config_id = ?1 ORDER BY artifact_id",
            vec![s(config_id)],
        )
        .await?;
        let mut out = Vec::new();
        for r in ids {
            if let Some(v) = r.into_iter().next()
                && let Some(a) = get_artifact(&c, &as_str(v)?).await?
            {
                out.push(a);
            }
        }
        Ok(out)
    }

    async fn config_for_artifact(&self, artifact_id: &str) -> StoreResult<Option<ConfigRow>> {
        let c = self.conn.lock().await;
        let mut current = get_artifact(&c, artifact_id).await?;
        for _ in 0..=MAX_DEPTH {
            let Some(a) = current else { return Ok(None) };
            if let Some(p) = a.parent_id {
                current = get_artifact(&c, &p).await?;
                continue;
            }
            let link = rows(
                &c,
                "SELECT config_id FROM config_artifacts WHERE artifact_id = ?1",
                vec![s(&a.id)],
            )
            .await?;
            return match link.into_iter().next().and_then(|r| r.into_iter().next()) {
                Some(v) => get_config(&c, &as_str(v)?).await,
                None => Ok(None),
            };
        }
        Ok(None)
    }

    async fn insert_dataset(&self, row: DatasetRow, data: Vec<u8>) -> StoreResult<()> {
        let c = self.conn.lock().await;
        if count(
            &c,
            "SELECT COUNT(*) FROM datasets WHERE id = ?1",
            vec![s(&row.id)],
        )
        .await?
            > 0
        {
            return Err(StoreError::Conflict(format!(
                "Dataset with id {} already exists",
                row.id
            )));
        }
        exec(
            &c,
            "INSERT INTO datasets (id, name, meta, data, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            vec![s(&row.id), s(&row.name), text(&row.meta)?, Value::Blob(data), s(&row.created_at), s(&row.updated_at)],
        )
        .await
        .map(|_| ())
    }

    async fn get_dataset(&self, id: &str) -> StoreResult<Option<DatasetRow>> {
        let c = self.conn.lock().await;
        rows(
            &c,
            &format!("SELECT {DATASET_COLS} FROM datasets WHERE id = ?1"),
            vec![s(id)],
        )
        .await?
        .into_iter()
        .next()
        .map(dataset_from)
        .transpose()
    }

    async fn dataset_data(&self, id: &str) -> StoreResult<Option<Vec<u8>>> {
        let c = self.conn.lock().await;
        Ok(
            match rows(&c, "SELECT data FROM datasets WHERE id = ?1", vec![s(id)])
                .await?
                .into_iter()
                .next()
                .and_then(|r| r.into_iter().next())
            {
                Some(Value::Blob(b)) => Some(b),
                _ => None,
            },
        )
    }

    async fn list_datasets(&self, page: Option<Page>) -> StoreResult<(Vec<DatasetRow>, u64)> {
        let c = self.conn.lock().await;
        let total = count(&c, "SELECT COUNT(*) FROM datasets", vec![]).await?;
        let sql = format!(
            "SELECT {DATASET_COLS} FROM datasets ORDER BY id{}",
            page_sql(page)
        );
        let list = rows(&c, &sql, vec![])
            .await?
            .into_iter()
            .map(dataset_from)
            .collect::<StoreResult<_>>()?;
        Ok((list, total))
    }

    async fn delete_dataset(&self, id: &str) -> StoreResult<bool> {
        let c = self.conn.lock().await;
        Ok(exec(&c, "DELETE FROM datasets WHERE id = ?1", vec![s(id)]).await? > 0)
    }
}
