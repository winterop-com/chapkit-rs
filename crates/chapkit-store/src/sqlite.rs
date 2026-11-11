use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, Row, params, params_from_iter};
use serde_json::Value;

use crate::{
    ArtifactRow, ConfigRow, DatasetRow, Listing, MAX_DEPTH, NewArtifact, Page, Store, StoreError,
    StoreResult,
};

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
CREATE TABLE IF NOT EXISTS artifact_listings (
    id         TEXT PRIMARY KEY REFERENCES artifacts(id) ON DELETE CASCADE,
    kind       TEXT,
    dataset_id TEXT,
    listing    TEXT NOT NULL,
    tags       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifact_listings_kind ON artifact_listings(kind, dataset_id);
CREATE TABLE IF NOT EXISTS config_artifacts (
    config_id   TEXT NOT NULL REFERENCES configs(id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL UNIQUE REFERENCES artifacts(id) ON DELETE CASCADE,
    PRIMARY KEY (config_id, artifact_id)
);
CREATE TABLE IF NOT EXISTS datasets (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    meta       TEXT NOT NULL,
    data       BLOB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
PRAGMA user_version = 2;
";

const DATASET_COLS: &str = "id, name, meta, created_at, updated_at";

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

type RawDataset = (String, String, String, String, String);

fn dataset_row(r: &Row<'_>) -> rusqlite::Result<RawDataset> {
    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
}

fn parse_dataset(t: RawDataset) -> StoreResult<DatasetRow> {
    Ok(DatasetRow {
        id: t.0,
        name: t.1,
        meta: serde_json::from_str(&t.2).map_err(|e| json_err(&e))?,
        created_at: t.3,
        updated_at: t.4,
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
        backfill_listings(&conn)?;
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

/// Write `id`'s listing row (replacing one there was).
fn put_listing(
    c: &Connection,
    id: &str,
    data: &Value,
    tags: &str,
    created_at: &str,
    updated_at: &str,
) -> StoreResult<()> {
    let l = Listing::of(data)?;
    c.execute(
        "INSERT OR REPLACE INTO artifact_listings (id, kind, dataset_id, listing, tags, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![id, l.kind, l.dataset_id, l.text, tags, created_at, updated_at],
    )
    .map_err(|e| db(&e))?;
    Ok(())
}

/// Give every artifact without a listing row one: those of a database from before listings
/// (one at a time, so only one artifact's data is in memory). A no-op afterwards.
fn backfill_listings(c: &Connection) -> StoreResult<()> {
    let ids: Vec<String> = {
        let mut stmt = c
            .prepare("SELECT id FROM artifacts WHERE id NOT IN (SELECT id FROM artifact_listings)")
            .map_err(|e| db(&e))?;
        stmt.query_map([], |r| r.get(0))
            .map_err(|e| db(&e))?
            .collect::<rusqlite::Result<_>>()
            .map_err(|e| db(&e))?
    };
    for id in ids {
        let (data, tags, created, updated): (String, String, String, String) = c
            .query_row(
                "SELECT data, tags, created_at, updated_at FROM artifacts WHERE id = ?1",
                [&id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .map_err(|e| db(&e))?;
        let data: Value = serde_json::from_str(&data).map_err(|e| json_err(&e))?;
        put_listing(c, &id, &data, &tags, &created, &updated)?;
    }
    Ok(())
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

    async fn list_configs(
        &self,
        name: Option<&str>,
        page: Option<Page>,
    ) -> StoreResult<(Vec<ConfigRow>, u64)> {
        let (filter, args) = crate::name_filter(name);
        self.with(move |c| {
            let total: i64 = c
                .query_row(
                    &format!("SELECT COUNT(*) FROM configs{filter}"),
                    params_from_iter(&args),
                    |r| r.get(0),
                )
                .map_err(|e| db(&e))?;
            let sql = format!(
                "SELECT id, name, data, tags, created_at, updated_at FROM configs{filter} ORDER BY id{}",
                page_sql(page)
            );
            let mut stmt = c.prepare(&sql).map_err(|e| db(&e))?;
            let rows = stmt
                .query_map(params_from_iter(&args), config_row)
                .map_err(|e| db(&e))?;
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
            let tx = c.transaction().map_err(|e| db(&e))?;
            tx.execute(
                "INSERT INTO artifacts (id, parent_id, level, data, content, tags, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
                params![a.id, a.parent_id, level, data, a.content, tags, a.created_at],
            )
            .map_err(|e| if is_unique(&e) { StoreError::Conflict(format!("Artifact with id {} already exists", a.id)) } else { db(&e) })?;
            put_listing(&tx, &a.id, &a.data, &tags, &a.created_at, &a.created_at)?;
            tx.commit().map_err(|e| db(&e))?;
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

    async fn list_artifact_summaries(
        &self,
        kind: Option<&str>,
        datasets: Option<&[String]>,
        page: Option<Page>,
    ) -> StoreResult<(Vec<ArtifactRow>, u64)> {
        if kind.is_some() && datasets.is_some_and(<[String]>::is_empty) {
            return Ok((Vec::new(), 0));
        }
        let (filter, args) = crate::type_filter(kind, datasets);
        self.with(move |c| {
            let total: i64 = c
                .query_row(
                    &format!("SELECT COUNT(*) FROM artifact_listings l{filter}"),
                    params_from_iter(&args),
                    |r| r.get(0),
                )
                .map_err(|e| db(&e))?;
            // parent_id and level come before data in an artifact's record: reading them does
            // not read data.
            let sql = format!(
                "SELECT l.id, a.parent_id, a.level, l.listing, l.tags, l.created_at, l.updated_at
                 FROM artifact_listings l JOIN artifacts a ON a.id = l.id{filter} ORDER BY l.id{}",
                page_sql(page)
            );
            let mut stmt = c.prepare(&sql).map_err(|e| db(&e))?;
            let rows = stmt
                .query_map(params_from_iter(&args), artifact_row)
                .map_err(|e| db(&e))?;
            let mut rows = rows
                .map(|r| r.map_err(|e| db(&e)).and_then(parse_artifact))
                .collect::<StoreResult<Vec<_>>>()?;
            drop(stmt);
            for row in rows.iter_mut().filter(|r| crate::lacks_summary(&r.data)) {
                if let Some(full) = get_artifact_sync(c, &row.id)? {
                    put_listing(
                        c,
                        &row.id,
                        &full.data,
                        &to_text(&full.tags)?,
                        &full.created_at,
                        &full.updated_at,
                    )?;
                    row.data = serde_json::from_str(&Listing::of(&full.data)?.text)
                        .map_err(|e| json_err(&e))?;
                }
            }
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
            let (text, tags) = (to_text(&data)?, to_text(&tags)?);
            let tx = c.transaction().map_err(|e| db(&e))?;
            let n = tx
                .execute(
                    "UPDATE artifacts SET data = ?2, tags = ?3, updated_at = ?4 WHERE id = ?1",
                    params![id, text, tags, updated_at],
                )
                .map_err(|e| db(&e))?;
            if n == 0 {
                return Ok(None);
            }
            let created: String = tx
                .query_row(
                    "SELECT created_at FROM artifacts WHERE id = ?1",
                    [&id],
                    |r| r.get(0),
                )
                .map_err(|e| db(&e))?;
            put_listing(&tx, &id, &data, &tags, &created, &updated_at)?;
            tx.commit().map_err(|e| db(&e))?;
            get_artifact_sync(c, &id)
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

    async fn insert_dataset(&self, row: DatasetRow, data: Vec<u8>) -> StoreResult<()> {
        self.with(move |c| {
            let meta = to_text(&row.meta)?;
            c.execute(
                "INSERT INTO datasets (id, name, meta, data, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![row.id, row.name, meta, data, row.created_at, row.updated_at],
            )
            .map_err(|e| if is_unique(&e) { StoreError::Conflict(format!("Dataset with id {} already exists", row.id)) } else { db(&e) })?;
            Ok(())
        })
        .await
    }

    async fn get_dataset(&self, id: &str) -> StoreResult<Option<DatasetRow>> {
        let id = id.to_owned();
        self.with(move |c| {
            c.query_row(
                &format!("SELECT {DATASET_COLS} FROM datasets WHERE id = ?1"),
                [&id],
                dataset_row,
            )
            .optional()
            .map_err(|e| db(&e))?
            .map(parse_dataset)
            .transpose()
        })
        .await
    }

    async fn dataset_data(&self, id: &str) -> StoreResult<Option<Vec<u8>>> {
        let id = id.to_owned();
        self.with(move |c| {
            c.query_row("SELECT data FROM datasets WHERE id = ?1", [&id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(|e| db(&e))
        })
        .await
    }

    async fn list_datasets(
        &self,
        name: Option<&str>,
        page: Option<Page>,
    ) -> StoreResult<(Vec<DatasetRow>, u64)> {
        let (filter, args) = crate::name_filter(name);
        self.with(move |c| {
            let total: i64 = c
                .query_row(
                    &format!("SELECT COUNT(*) FROM datasets{filter}"),
                    params_from_iter(&args),
                    |r| r.get(0),
                )
                .map_err(|e| db(&e))?;
            let sql = format!(
                "SELECT {DATASET_COLS} FROM datasets{filter} ORDER BY id{}",
                page_sql(page)
            );
            let mut stmt = c.prepare(&sql).map_err(|e| db(&e))?;
            let rows = stmt
                .query_map(params_from_iter(&args), dataset_row)
                .map_err(|e| db(&e))?;
            let rows = rows
                .map(|r| r.map_err(|e| db(&e)).and_then(parse_dataset))
                .collect::<StoreResult<Vec<_>>>()?;
            Ok((rows, total.max(0) as u64))
        })
        .await
    }

    async fn delete_dataset(&self, id: &str) -> StoreResult<bool> {
        let id = id.to_owned();
        self.with(move |c| {
            Ok(c.execute("DELETE FROM datasets WHERE id = ?1", [&id])
                .map_err(|e| db(&e))?
                > 0)
        })
        .await
    }
}
