#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! One behavioural suite, run against every backend built in: the in-memory and
//! file SQLite backends always, turso with `--features turso`.

use std::sync::Arc;

use chapkit_store::{ConfigRow, DatasetRow, NewArtifact, Page, Store, StoreError};
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

async fn configs_crud_and_conflict(s: Arc<dyn Store>) {
    s.ping().await.unwrap();
    s.insert_config(cfg("01A")).await.unwrap();
    assert!(matches!(
        s.insert_config(cfg("01A")).await,
        Err(StoreError::Conflict(_))
    ));
    s.insert_config(cfg("01B")).await.unwrap();
    let (rows, total) = s
        .list_configs(
            None,
            Some(Page {
                offset: 1,
                limit: 10,
            }),
        )
        .await
        .unwrap();
    assert_eq!((rows.len(), total), (1, 2));
    assert_eq!(rows[0].id, "01B");
    let mut c = cfg("01A");
    c.name = "renamed".into();
    assert_eq!(s.update_config(c).await.unwrap().unwrap().name, "renamed");
    assert!(s.update_config(cfg("01Z")).await.unwrap().is_none());
    assert_eq!(
        s.get_config("01A").await.unwrap().unwrap().data["alpha"],
        json!(0.5)
    );
}

async fn artifact_tree_links_and_cascade(s: Arc<dyn Store>) {
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
    assert!(matches!(
        s.insert_artifact(art("A0", None)).await,
        Err(StoreError::Conflict(_))
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

    // Deleting an inner artifact keeps its child, detached.
    s.insert_artifact(art("B0", None)).await.unwrap();
    s.insert_artifact(art("B1", Some("B0"))).await.unwrap();
    assert!(s.delete_artifact("B0").await.unwrap());
    assert_eq!(s.get_artifact("B1").await.unwrap().unwrap().parent_id, None);

    // Deleting a config deletes the trees linked to it.
    assert!(s.delete_config("C1").await.unwrap());
    for id in ["A0", "A1", "A2"] {
        assert!(s.get_artifact(id).await.unwrap().is_none(), "{id} survived");
    }
    assert!(s.get_artifact("B1").await.unwrap().is_some());
    assert!(!s.delete_config("C1").await.unwrap());
}

fn dataset(id: &str) -> DatasetRow {
    DatasetRow {
        id: id.into(),
        name: "d".into(),
        meta: json!({"rows": 2, "target": "disease_cases"}),
        created_at: "2026-01-01T00:00:00".into(),
        updated_at: "2026-01-01T00:00:00".into(),
    }
}

async fn datasets_crud_and_artifact_types(s: Arc<dyn Store>) {
    s.insert_dataset(dataset("D1"), b"{\"columns\": []}".to_vec())
        .await
        .unwrap();
    assert!(matches!(
        s.insert_dataset(dataset("D1"), vec![]).await,
        Err(StoreError::Conflict(_))
    ));
    s.insert_dataset(dataset("D2"), vec![7; 3]).await.unwrap();
    assert_eq!(s.get_dataset("D1").await.unwrap().unwrap(), dataset("D1"));
    assert!(s.get_dataset("D9").await.unwrap().is_none());
    assert_eq!(s.dataset_data("D2").await.unwrap(), Some(vec![7; 3]));
    assert_eq!(s.dataset_data("D9").await.unwrap(), None);
    let (rows, total) = s
        .list_datasets(
            None,
            Some(Page {
                offset: 1,
                limit: 5,
            }),
        )
        .await
        .unwrap();
    assert_eq!((rows.len(), total, rows[0].id.as_str()), (1, 2, "D2"));
    assert!(s.delete_dataset("D1").await.unwrap());
    assert!(!s.delete_dataset("D1").await.unwrap());
    assert_eq!(s.list_datasets(None, None).await.unwrap().1, 1);

    for (id, kind) in [
        ("T1", "ml_backtest"),
        ("T2", "ml_training"),
        ("T3", "ml_backtest"),
    ] {
        let mut a = art(id, None);
        a.data = json!({"type": kind});
        s.insert_artifact(a).await.unwrap();
    }
    let (rows, total) = s
        .list_artifact_summaries(Some("ml_backtest"), None, None)
        .await
        .unwrap();
    assert_eq!(total, 2);
    assert_eq!(
        rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["T1", "T3"]
    );
    let (rows, total) = s
        .list_artifact_summaries(
            Some("ml_backtest"),
            None,
            Some(Page {
                offset: 1,
                limit: 1,
            }),
        )
        .await
        .unwrap();
    assert_eq!((rows.len(), total, rows[0].id.as_str()), (1, 2, "T3"));
    assert_eq!(
        s.list_artifact_summaries(Some("nope"), None, None)
            .await
            .unwrap()
            .1,
        0
    );
}

async fn lists_filter_by_name_and_dataset(s: Arc<dyn Store>) {
    for (id, name) in [
        ("01A", "Laos monthly"),
        ("01B", "Vietnam"),
        ("01C", "laos weekly"),
    ] {
        let mut c = cfg(id);
        c.name = name.into();
        s.insert_config(c).await.unwrap();
        let mut d = dataset(id);
        d.name = name.into();
        s.insert_dataset(d, vec![]).await.unwrap();
    }
    let names = |rows: Vec<String>| rows.join(",");
    let (rows, total) = s.list_configs(Some("LAOS"), None).await.unwrap();
    assert_eq!(total, 2);
    assert_eq!(names(rows.into_iter().map(|r| r.id).collect()), "01A,01C");
    let (rows, total) = s
        .list_datasets(
            Some("laos"),
            Some(Page {
                offset: 1,
                limit: 5,
            }),
        )
        .await
        .unwrap();
    assert_eq!((rows.len(), total, rows[0].id.as_str()), (1, 2, "01C"));
    assert_eq!(s.list_datasets(Some("%"), None).await.unwrap().1, 0);
    assert_eq!(s.list_datasets(Some(""), None).await.unwrap().1, 3);

    for (id, kind, dataset) in [
        ("T1", "ml_backtest", "01A"),
        ("T2", "ml_backtest", "01B"),
        ("T3", "ml_forecast", "01A"),
        ("T4", "ml_backtest", "01C"),
    ] {
        let mut a = art(id, None);
        a.data = json!({"type": kind, "content": {"dataset_id": dataset}});
        s.insert_artifact(a).await.unwrap();
    }
    let ids = ["01A".to_owned(), "01C".to_owned()];
    let (rows, total) = s
        .list_artifact_summaries(Some("ml_backtest"), Some(&ids), None)
        .await
        .unwrap();
    assert_eq!(total, 2);
    assert_eq!(names(rows.into_iter().map(|r| r.id).collect()), "T1,T4");
    let none: [String; 0] = [];
    assert_eq!(
        s.list_artifact_summaries(Some("ml_backtest"), Some(&none), None)
            .await
            .unwrap()
            .1,
        0
    );
}

/// A backtest and a forecast as the service stores them, with content to leave out of listings.
fn results() -> [NewArtifact; 2] {
    let mut backtest = art("R1", None);
    backtest.data = json!({
        "type": "ml_backtest", "name": "b",
        "content": {"dataset_id": "D1", "data": {"rows": 3}, "options": {"splits": 2},
                    "aggregate": {"crps": 1.5}, "forecasts": [[1, 2, 3]]}
    });
    let mut forecast = art("R2", Some("R1"));
    forecast.data = json!({
        "type": "ml_forecast",
        "content": {"dataset_id": "D2", "periods": 3, "weather": "climatology",
                    "forecasts": [{"time_period": "2024-01", "location": "A"},
                                  {"time_period": "2024-01", "location": "B"},
                                  {"time_period": "2024-02", "location": "A"}]}
    });
    [backtest, forecast]
}

/// Listings carry `data` without `content`, plus summaries; they follow updates and deletes.
async fn listings_carry_summaries_not_content(s: Arc<dyn Store>) {
    for a in results() {
        s.insert_artifact(a).await.unwrap();
    }
    let (rows, total) = s.list_artifact_summaries(None, None, None).await.unwrap();
    assert_eq!(total, 2);
    assert!(rows.iter().all(|r| r.data.get("content").is_none()));
    assert_eq!(
        rows[0].data["summary"],
        json!({"dataset_id": "D1", "data": {"rows": 3}, "options": {"splits": 2}, "aggregate": {"crps": 1.5}})
    );
    assert_eq!(rows[0].data["name"], "b");
    assert_eq!(
        rows[1].data["summary"],
        json!({"dataset_id": "D2", "periods": 3, "weather": "climatology",
               "first_period": "2024-01", "locations": 2})
    );
    assert_eq!(
        (rows[1].parent_id.as_deref(), rows[1].level),
        (Some("R1"), 1)
    );
    let d2 = ["D2".to_owned()];
    assert_eq!(
        s.list_artifact_summaries(Some("ml_forecast"), Some(&d2), None)
            .await
            .unwrap()
            .1,
        1
    );
    s.update_artifact(
        "R2",
        json!({"type": "ml_backtest", "content": {"dataset_id": "D2"}}),
        vec!["t".into()],
        "2026-02-01T00:00:00".into(),
    )
    .await
    .unwrap();
    let (rows, total) = s
        .list_artifact_summaries(Some("ml_backtest"), Some(&d2), None)
        .await
        .unwrap();
    assert_eq!((total, rows[0].id.as_str()), (1, "R2"));
    assert_eq!(rows[0].tags, ["t"]);
    assert_eq!(rows[0].updated_at, "2026-02-01T00:00:00");
    assert_eq!(rows[0].data["summary"], json!({"dataset_id": "D2"}));
    assert!(s.delete_artifact("R1").await.unwrap());
    let (rows, total) = s.list_artifact_summaries(None, None, None).await.unwrap();
    assert_eq!((total, rows[0].parent_id.as_deref()), (1, None));
}

/// The artifacts table as chapkit-rs 0.6.0 wrote it, before listings.
const OLD_SCHEMA: &str = "CREATE TABLE artifacts (
    id TEXT PRIMARY KEY, parent_id TEXT, level INTEGER NOT NULL, data TEXT NOT NULL, content BLOB,
    tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)";

/// A database written the old way, with `exec` (the backend's own driver), lists its backtests and
/// forecasts with summaries and filters them by type and dataset once opened; a listing that lost
/// its summary gets it back from the artifact's data.
async fn old_databases_list_summaries(
    open: impl AsyncFn() -> Arc<dyn Store>,
    exec: impl AsyncFn(String),
) {
    exec(OLD_SCHEMA.into()).await;
    for (a, level) in results().into_iter().zip([0, 1]) {
        exec(format!(
            "INSERT INTO artifacts (id, parent_id, level, data, tags, created_at, updated_at) \
             VALUES ('{}', {}, {level}, '{}', '[]', '{}', '{}')",
            a.id,
            a.parent_id.map_or("NULL".into(), |p| format!("'{p}'")),
            a.data,
            a.created_at,
            a.created_at
        ))
        .await;
    }
    let check = async |s: Arc<dyn Store>| {
        let (rows, total) = s.list_artifact_summaries(None, None, None).await.unwrap();
        assert_eq!(total, 2);
        assert_eq!(rows[0].data["summary"]["aggregate"], json!({"crps": 1.5}));
        assert_eq!(rows[1].data["summary"]["locations"], 2);
        assert!(rows.iter().all(|r| r.data.get("content").is_none()));
        let d1 = ["D1".to_owned()];
        let (rows, total) = s
            .list_artifact_summaries(Some("ml_backtest"), Some(&d1), None)
            .await
            .unwrap();
        assert_eq!((total, rows[0].id.as_str()), (1, "R1"));
    };
    check(open().await).await;
    exec(
        "UPDATE artifact_listings SET listing = '{\"type\": \"ml_backtest\"}' WHERE id = 'R1'"
            .into(),
    )
    .await;
    check(open().await).await;
}

/// The whole contract, each part on a fresh store from `s`; every backend runs all of it.
async fn suite<F: std::future::Future<Output = Arc<dyn Store>>>(s: impl Fn() -> F) {
    configs_crud_and_conflict(s().await).await;
    lists_filter_by_name_and_dataset(s().await).await;
    artifact_tree_links_and_cascade(s().await).await;
    datasets_crud_and_artifact_types(s().await).await;
    listings_carry_summaries_not_content(s().await).await;
}

#[tokio::test]
async fn sqlite_in_memory() {
    suite(|| async {
        Arc::new(chapkit_store::SqliteStore::in_memory().unwrap()) as Arc<dyn Store>
    })
    .await;
}

#[tokio::test]
async fn sqlite_file_and_urls() {
    let dir = std::env::temp_dir().join(format!("chapkit-store-test-{}", std::process::id()));
    let n = std::sync::atomic::AtomicUsize::new(0);
    let dir2 = dir.clone();
    suite(move || {
        let i = n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = dir2.join(format!("db{i}.sqlite"));
        let s: Arc<dyn Store> = Arc::new(
            chapkit_store::open_sqlite(Some(&format!("sqlite:///{}", path.display()))).unwrap(),
        );
        async move { s }
    })
    .await;
    assert!(
        chapkit_store::open(Some("sqlite+aiosqlite:///:memory:"))
            .await
            .is_ok()
    );
    assert!(chapkit_store::open(None).await.is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn sqlite_old_databases_list_summaries() {
    let path = std::env::temp_dir().join(format!("chapkit-old-{}.sqlite", std::process::id()));
    let _ = std::fs::remove_file(&path);
    old_databases_list_summaries(
        async || Arc::new(chapkit_store::SqliteStore::open(&path).unwrap()) as Arc<dyn Store>,
        async |sql: String| {
            rusqlite::Connection::open(&path)
                .unwrap()
                .execute_batch(&sql)
                .unwrap();
        },
    )
    .await;
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "turso")]
#[tokio::test]
async fn turso_old_databases_list_summaries() {
    let path = std::env::temp_dir().join(format!("chapkit-old-{}.turso", std::process::id()));
    let file = path.display().to_string();
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{file}{suffix}"));
    }
    let url = format!("turso:///{file}");
    old_databases_list_summaries(
        async || chapkit_store::open(Some(&url)).await.unwrap(),
        async |sql: String| {
            let db = turso::Builder::new_local(&file).build().await.unwrap();
            db.connect().unwrap().execute(&sql, ()).await.unwrap();
        },
    )
    .await;
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{file}{suffix}"));
    }
}

#[cfg(feature = "turso")]
#[tokio::test]
async fn turso_in_memory() {
    suite(|| async { chapkit_store::open(Some("turso::memory:")).await.unwrap() }).await;
}

#[cfg(not(feature = "turso"))]
#[tokio::test]
async fn turso_url_without_feature_is_an_error() {
    assert!(matches!(
        chapkit_store::open(Some("turso::memory:")).await,
        Err(StoreError::Invalid(_))
    ));
}
