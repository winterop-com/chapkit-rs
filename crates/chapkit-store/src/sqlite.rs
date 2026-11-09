use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde_json::Value;

use crate::{ArtifactRow, ConfigRow, MAX_DEPTH, NewArtifact, Page, Store, StoreError, StoreResult};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS configs (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    data       TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS configs_name ON configs(name);
CREATE TABLE IF NOT EXISTS artifacts (
    id         TEXT PRIMARY KEY,
    parent_id  TEXT REFERENCES artifacts(id) ON DELETE SET NULL,
    level      INTEGER NOT NULL,
    data       TEXT NOT NULL,
    content    BLOB,
    tags       TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_parent ON artifacts(parent_id);
CREATE INDEX IF NOT EXISTS artifacts_level ON artifacts(level);
CREATE TABLE IF NOT EXISTS config_artifacts (
    config_id   TEXT NOT NULL REFERENCES configs(id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL UNIQUE REFERENCES artifacts(id) ON DELETE CASCADE,
    PRIMARY KEY (config_id, artifact_id)
);
PRAGMA user_version = 1;
";

const ARTIFACT_COLS: &str = "id, parent_id, level, data, tags, created_at, updated_at";

/// SQLite store (rusqlite, bundled). One connection behind a mutex; calls run on
/// tokio's blocking pool. The workload is small key lookups, so this is plenty.
#[derive(Clone)]
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
}

fn db(e: &rusqlite::Error) -> StoreError {
    StoreError::Database(e.to_string())
}

fn json_err(e: &serde_json::Error) -> StoreError {
    StoreError::Database(format!("stored JSON is invalid: {e}"))
}

fn to_text(v: &impl serde::Serialize) -> StoreResult<String> {
    serde_json::to_string(v).map_err(|e| json_err(&e))
}

fn is_unique(e: &rusqlite::Error) -> bool {
    matches!(e, rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::ConstraintViolation)
}

fn config_row(r: &Row<'_>) -> rusqlite::Result<(String, String, String, String, String, String)> {
    Ok((
        r.get(0)?,
        r.get(1)?,
        r.get(2)?,
        r.get(3)?,
        r.get(4)?,
        r.get(5)?,
    ))
}

fn parse_config(t: (String, String, String, String, String, String)) -> StoreResult<ConfigRow> {
    Ok(ConfigRow {
        id: t.0,
        name: t.1,
        data: serde_json::from_str(&t.2).map_err(|e| json_err(&e))?,
        tags: serde_json::from_str(&t.3).map_err(|e| json_err(&e))?,
        created_at: t.4,
        updated_at: t.5,
    })
}

type RawArtifact = (String, Option<String>, i64, String, String, String, String);

fn artifact_row(r: &Row<'_>) -> rusqlite::Result<RawArtifact> {
    Ok((
        r.get(0)?,
        r.get(1)?,
        r.get(2)?,
        r.get(3)?,
        r.get(4)?,
        r.get(5)?,
        r.get(6)?,
    ))
}

fn parse_artifact(t: RawArtifact) -> StoreResult<ArtifactRow> {
    Ok(ArtifactRow {
        id: t.0,
        parent_id: t.1,
        level: t.2,
        data: serde_json::from_str(&t.3).map_err(|e| json_err(&e))?,
        tags: serde_json::from_str(&t.4).map_err(|e| json_err(&e))?,
        created_at: t.5,
        updated_at: t.6,
    })
}

impl SqliteStore {
    /// An in-memory database (lost on exit).
    pub fn in_memory() -> StoreResult<Self> {
        Self::init(Connection::open_in_memory().map_err(|e| db(&e))?, false)
    }

    /// A file database, created if missing (parent directories too), in WAL mode.
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        if let Some(dir) = path.as_ref().parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| StoreError::Database(e.to_string()))?;
        }
        Self::init(Connection::open(path).map_err(|e| db(&e))?, true)
    }

    fn init(conn: Connection, file: bool) -> StoreResult<Self> {
        conn.execute_batch(
            "PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 30000; PRAGMA synchronous = NORMAL;",
        )
        .map_err(|e| db(&e))?;
        if file {
            conn.query_row("PRAGMA journal_mode = WAL", [], |_| Ok(()))
                .map_err(|e| db(&e))?;
        }
        conn.execute_batch(SCHEMA).map_err(|e| db(&e))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Run `f` with the connection on the blocking pool.
    async fn with<T: Send + 'static>(
        &self,
        f: impl FnOnce(&mut Connection) -> StoreResult<T> + Send + 'static,
    ) -> StoreResult<T> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard: MutexGuard<'_, Connection> = conn
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            f(&mut guard)
        })
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?
    }
}

fn get_config_sync(c: &Connection, id: &str) -> StoreResult<Option<ConfigRow>> {
    c.query_row(
        "SELECT id, name, data, tags, created_at, updated_at FROM configs WHERE id = ?1",
        [id],
        config_row,
    )
    .optional()
    .map_err(|e| db(&e))?
    .map(parse_config)
    .transpose()
}

fn get_artifact_sync(c: &Connection, id: &str) -> StoreResult<Option<ArtifactRow>> {
    c.query_row(
        &format!("SELECT {ARTIFACT_COLS} FROM artifacts WHERE id = ?1"),
        [id],
        artifact_row,
    )
    .optional()
    .map_err(|e| db(&e))?
    .map(parse_artifact)
    .transpose()
}

fn page_sql(page: Option<Page>) -> String {
    page.map_or(String::new(), |p| {
        format!(" LIMIT {} OFFSET {}", p.limit, p.offset)
    })
}

#[async_trait]
impl Store for SqliteStore {
    async fn ping(&self) -> StoreResult<()> {
        self.with(|c| c.query_row("SELECT 1", [], |_| Ok(())).map_err(|e| db(&e)))
            .await
    }

    async fn insert_config(&self, row: ConfigRow) -> StoreResult<()> {
        self.with(move |c| {
            let (data, tags) = (to_text(&row.data)?, to_text(&row.tags)?);
            c.execute(
                "INSERT INTO configs (id, name, data, tags, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![row.id, row.name, data, tags, row.created_at, row.updated_at],
            )
            .map_err(|e| if is_unique(&e) { StoreError::Conflict(format!("Config with id {} already exists", row.id)) } else { db(&e) })?;
            Ok(())
        })
        .await
    }

    async fn get_config(&self, id: &str) -> StoreResult<Option<ConfigRow>> {
        let id = id.to_owned();
        self.with(move |c| get_config_sync(c, &id)).await
    }

    async fn list_configs(&self, page: Option<Page>) -> StoreResult<(Vec<ConfigRow>, u64)> {
        self.with(move |c| {
            let total: i64 = c
                .query_row("SELECT COUNT(*) FROM configs", [], |r| r.get(0))
                .map_err(|e| db(&e))?;
            let sql = format!(
                "SELECT id, name, data, tags, created_at, updated_at FROM configs ORDER BY id{}",
                page_sql(page)
            );
            let mut stmt = c.prepare(&sql).map_err(|e| db(&e))?;
            let rows = stmt.query_map([], config_row).map_err(|e| db(&e))?;
            let rows = rows
                .map(|r| r.map_err(|e| db(&e)).and_then(parse_config))
                .collect::<StoreResult<Vec<_>>>()?;
            Ok((rows, total.max(0) as u64))
        })
        .await
    }

    async fn update_config(&self, row: ConfigRow) -> StoreResult<Option<ConfigRow>> {
        self.with(move |c| {
            let (data, tags) = (to_text(&row.data)?, to_text(&row.tags)?);
            let n = c
                .execute(
                    "UPDATE configs SET name = ?2, data = ?3, tags = ?4, updated_at = ?5 WHERE id = ?1",
                    params![row.id, row.name, data, tags, row.updated_at],
                )
                .map_err(|e| db(&e))?;
            if n == 0 { Ok(None) } else { get_config_sync(c, &row.id) }
        })
        .await
    }

    async fn delete_config(&self, id: &str) -> StoreResult<bool> {
        let id = id.to_owned();
        self.with(move |c| {
            let tx = c.transaction().map_err(|e| db(&e))?;
            // Delete every artifact tree linked to the config, then the config.
            tx.execute(
                "WITH RECURSIVE tree(id) AS (
                     SELECT artifact_id FROM config_artifacts WHERE config_id = ?1
                     UNION ALL SELECT a.id FROM artifacts a JOIN tree t ON a.parent_id = t.id
                 ) DELETE FROM artifacts WHERE id IN (SELECT id FROM tree)",
                [&id],
            )
            .map_err(|e| db(&e))?;
            let n = tx
                .execute("DELETE FROM configs WHERE id = ?1", [&id])
                .map_err(|e| db(&e))?;
            tx.commit().map_err(|e| db(&e))?;
            Ok(n > 0)
        })
        .await
    }

    async fn insert_artifact(&self, a: NewArtifact) -> StoreResult<ArtifactRow> {
        self.with(move |c| {
            let level = match &a.parent_id {
                None => 0,
                Some(p) => {
                    let parent_level: Option<i64> = c
                        .query_row("SELECT level FROM artifacts WHERE id = ?1", [p], |r| r.get(0))
                        .optional()
                        .map_err(|e| db(&e))?;
                    let pl = parent_level.ok_or_else(|| StoreError::NotFound(format!("Parent artifact {p} not found")))?;
                    if pl + 1 > MAX_DEPTH {
                        return Err(StoreError::Invalid(format!("Artifact tree deeper than {MAX_DEPTH} levels")));
                    }
                    pl + 1
                }
            };
            let (data, tags) = (to_text(&a.data)?, to_text(&a.tags)?);
            c.execute(
                "INSERT INTO artifacts (id, parent_id, level, data, content, tags, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
                params![a.id, a.parent_id, level, data, a.content, tags, a.created_at],
            )
            .map_err(|e| if is_unique(&e) { StoreError::Conflict(format!("Artifact with id {} already exists", a.id)) } else { db(&e) })?;
            get_artifact_sync(c, &a.id)?.ok_or_else(|| StoreError::Database("artifact vanished after insert".into()))
        })
        .await
    }

    async fn get_artifact(&self, id: &str) -> StoreResult<Option<ArtifactRow>> {
        let id = id.to_owned();
        self.with(move |c| get_artifact_sync(c, &id)).await
    }

    async fn artifact_content(&self, id: &str) -> StoreResult<Option<Vec<u8>>> {
        let id = id.to_owned();
        self.with(move |c| {
            c.query_row("SELECT content FROM artifacts WHERE id = ?1", [&id], |r| {
                r.get::<_, Option<Vec<u8>>>(0)
            })
            .optional()
            .map(Option::flatten)
            .map_err(|e| db(&e))
        })
        .await
    }

    async fn list_artifacts(&self, page: Option<Page>) -> StoreResult<(Vec<ArtifactRow>, u64)> {
        self.with(move |c| {
            let total: i64 = c
                .query_row("SELECT COUNT(*) FROM artifacts", [], |r| r.get(0))
                .map_err(|e| db(&e))?;
            let sql = format!(
                "SELECT {ARTIFACT_COLS} FROM artifacts ORDER BY id{}",
                page_sql(page)
            );
            let mut stmt = c.prepare(&sql).map_err(|e| db(&e))?;
            let rows = stmt.query_map([], artifact_row).map_err(|e| db(&e))?;
            let rows = rows
                .map(|r| r.map_err(|e| db(&e)).and_then(parse_artifact))
                .collect::<StoreResult<Vec<_>>>()?;
            Ok((rows, total.max(0) as u64))
        })
        .await
    }

    async fn update_artifact(
        &self,
        id: &str,
        data: Value,
        tags: Vec<String>,
        updated_at: String,
    ) -> StoreResult<Option<ArtifactRow>> {
        let id = id.to_owned();
        self.with(move |c| {
            let (data, tags) = (to_text(&data)?, to_text(&tags)?);
            let n = c
                .execute(
                    "UPDATE artifacts SET data = ?2, tags = ?3, updated_at = ?4 WHERE id = ?1",
                    params![id, data, tags, updated_at],
                )
                .map_err(|e| db(&e))?;
            if n == 0 {
                Ok(None)
            } else {
                get_artifact_sync(c, &id)
            }
        })
        .await
    }

    async fn delete_artifact(&self, id: &str) -> StoreResult<bool> {
        let id = id.to_owned();
        self.with(move |c| {
            Ok(c.execute("DELETE FROM artifacts WHERE id = ?1", [&id])
                .map_err(|e| db(&e))?
                > 0)
        })
        .await
    }

    async fn children(&self, id: &str) -> StoreResult<Vec<ArtifactRow>> {
        let id = id.to_owned();
        self.with(move |c| {
            let mut stmt = c
                .prepare(&format!(
                    "SELECT {ARTIFACT_COLS} FROM artifacts WHERE parent_id = ?1 ORDER BY id"
                ))
                .map_err(|e| db(&e))?;
            let rows = stmt.query_map([&id], artifact_row).map_err(|e| db(&e))?;
            rows.map(|r| r.map_err(|e| db(&e)).and_then(parse_artifact))
                .collect()
        })
        .await
    }

    async fn link(&self, config_id: &str, artifact_id: &str) -> StoreResult<()> {
        let (cid, aid) = (config_id.to_owned(), artifact_id.to_owned());
        self.with(move |c| {
            let a = get_artifact_sync(c, &aid)?
                .ok_or_else(|| StoreError::NotFound(format!("Artifact {aid} not found")))?;
            if a.level != 0 {
                return Err(StoreError::Invalid(format!(
                    "Artifact {aid} is not a root artifact (level {})",
                    a.level
                )));
            }
            if get_config_sync(c, &cid)?.is_none() {
                return Err(StoreError::NotFound(format!("Config {cid} not found")));
            }
            c.execute(
                "INSERT OR REPLACE INTO config_artifacts (config_id, artifact_id) VALUES (?1, ?2)",
                params![cid, aid],
            )
            .map_err(|e| db(&e))?;
            Ok(())
        })
        .await
    }

    async fn unlink(&self, config_id: &str, artifact_id: &str) -> StoreResult<bool> {
        let (cid, aid) = (config_id.to_owned(), artifact_id.to_owned());
        self.with(move |c| {
            Ok(c.execute(
                "DELETE FROM config_artifacts WHERE config_id = ?1 AND artifact_id = ?2",
                params![cid, aid],
            )
            .map_err(|e| db(&e))?
                > 0)
        })
        .await
    }

    async fn linked_artifacts(&self, config_id: &str) -> StoreResult<Vec<ArtifactRow>> {
        let cid = config_id.to_owned();
        self.with(move |c| {
            let sql = format!(
                "SELECT {} FROM artifacts a JOIN config_artifacts l ON l.artifact_id = a.id WHERE l.config_id = ?1 ORDER BY a.id",
                ARTIFACT_COLS.split(", ").map(|col| format!("a.{col}")).collect::<Vec<_>>().join(", ")
            );
            let mut stmt = c.prepare(&sql).map_err(|e| db(&e))?;
            let rows = stmt.query_map([&cid], artifact_row).map_err(|e| db(&e))?;
            rows.map(|r| r.map_err(|e| db(&e)).and_then(parse_artifact)).collect()
        })
        .await
    }

    async fn config_for_artifact(&self, artifact_id: &str) -> StoreResult<Option<ConfigRow>> {
        let aid = artifact_id.to_owned();
        self.with(move |c| {
            let root: Option<String> = c
                .query_row(
                    "WITH RECURSIVE up(id, parent_id) AS (
                         SELECT id, parent_id FROM artifacts WHERE id = ?1
                         UNION ALL SELECT a.id, a.parent_id FROM artifacts a JOIN up ON a.id = up.parent_id
                     ) SELECT id FROM up WHERE parent_id IS NULL LIMIT 1",
                    [&aid],
                    |r| r.get(0),
                )
                .optional()
                .map_err(|e| db(&e))?;
            let Some(root) = root else { return Ok(None) };
            let cid: Option<String> = c
                .query_row("SELECT config_id FROM config_artifacts WHERE artifact_id = ?1", [&root], |r| r.get(0))
                .optional()
                .map_err(|e| db(&e))?;
            match cid {
                Some(cid) => get_config_sync(c, &cid),
                None => Ok(None),
            }
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg(id: &str) -> ConfigRow {
        ConfigRow {
            id: id.into(),
            name: "c".into(),
            data: json!({"prediction_periods": 3, "alpha": 0.5}),
            tags: vec![],
            created_at: "2026-01-01T00:00:00".into(),
            updated_at: "2026-01-01T00:00:00".into(),
        }
    }

    fn art(id: &str, parent: Option<&str>) -> NewArtifact {
        NewArtifact {
            id: id.into(),
            parent_id: parent.map(Into::into),
            data: json!({"type": "x"}),
            tags: vec![],
            content: Some(vec![1, 2, 3]),
            created_at: "2026-01-01T00:00:00".into(),
        }
    }

    #[tokio::test]
    async fn configs_crud_and_conflict() {
        let s = SqliteStore::in_memory().unwrap();
        s.insert_config(cfg("01A")).await.unwrap();
        assert!(matches!(
            s.insert_config(cfg("01A")).await,
            Err(StoreError::Conflict(_))
        ));
        s.insert_config(cfg("01B")).await.unwrap();
        let (rows, total) = s
            .list_configs(Some(Page {
                offset: 1,
                limit: 10,
            }))
            .await
            .unwrap();
        assert_eq!((rows.len(), total), (1, 2));
        let mut c = cfg("01A");
        c.name = "renamed".into();
        assert_eq!(s.update_config(c).await.unwrap().unwrap().name, "renamed");
        assert!(s.update_config(cfg("01Z")).await.unwrap().is_none());
        assert_eq!(
            s.get_config("01A").await.unwrap().unwrap().data["alpha"],
            json!(0.5)
        );
    }

    #[tokio::test]
    async fn artifact_tree_links_and_cascade() {
        let s = SqliteStore::in_memory().unwrap();
        s.insert_config(cfg("C1")).await.unwrap();
        assert_eq!(s.insert_artifact(art("A0", None)).await.unwrap().level, 0);
        assert_eq!(
            s.insert_artifact(art("A1", Some("A0")))
                .await
                .unwrap()
                .level,
            1
        );
        assert_eq!(
            s.insert_artifact(art("A2", Some("A1")))
                .await
                .unwrap()
                .level,
            2
        );
        assert!(matches!(
            s.insert_artifact(art("X", Some("nope"))).await,
            Err(StoreError::NotFound(_))
        ));
        assert_eq!(s.artifact_content("A0").await.unwrap(), Some(vec![1, 2, 3]));

        assert!(matches!(
            s.link("C1", "A1").await,
            Err(StoreError::Invalid(_))
        ));
        s.link("C1", "A0").await.unwrap();
        assert_eq!(s.linked_artifacts("C1").await.unwrap().len(), 1);
        assert_eq!(s.config_for_artifact("A2").await.unwrap().unwrap().id, "C1");
        assert_eq!(s.children("A0").await.unwrap().len(), 1);

        assert!(s.delete_config("C1").await.unwrap());
        assert!(s.get_artifact("A0").await.unwrap().is_none());
        assert!(s.get_artifact("A2").await.unwrap().is_none());
        assert!(!s.delete_config("C1").await.unwrap());
    }

    #[test]
    fn open_parses_database_urls() {
        assert!(crate::open(None).is_ok());
        assert!(crate::open(Some("sqlite+aiosqlite:///:memory:")).is_ok());
        let dir = std::env::temp_dir().join(format!("chapkit-store-test-{}", std::process::id()));
        let path = dir.join("db.sqlite");
        assert!(crate::open(Some(&format!("sqlite:///{}", path.display()))).is_ok());
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
