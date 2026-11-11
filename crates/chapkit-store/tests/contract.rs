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
        .list_artifacts_of_type("ml_backtest", None, None)
        .await
        .unwrap();
    assert_eq!(total, 2);
    assert_eq!(
        rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["T1", "T3"]
    );
    let (rows, total) = s
        .list_artifacts_of_type(
            "ml_backtest",
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
        s.list_artifacts_of_type("nope", None, None)
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
        .list_artifacts_of_type("ml_backtest", Some(&ids), None)
        .await
        .unwrap();
    assert_eq!(total, 2);
    assert_eq!(names(rows.into_iter().map(|r| r.id).collect()), "T1,T4");
    let none: [String; 0] = [];
    assert_eq!(
        s.list_artifacts_of_type("ml_backtest", Some(&none), None)
            .await
            .unwrap()
            .1,
        0
    );
}

async fn suite(s: impl Fn() -> Arc<dyn Store>) {
    configs_crud_and_conflict(s()).await;
    lists_filter_by_name_and_dataset(s()).await;
    artifact_tree_links_and_cascade(s()).await;
    datasets_crud_and_artifact_types(s()).await;
}

#[tokio::test]
async fn sqlite_in_memory() {
    suite(|| Arc::new(chapkit_store::SqliteStore::in_memory().unwrap())).await;
}

#[tokio::test]
async fn sqlite_file_and_urls() {
    let dir = std::env::temp_dir().join(format!("chapkit-store-test-{}", std::process::id()));
    let n = std::sync::atomic::AtomicUsize::new(0);
    let dir2 = dir.clone();
    suite(move || {
        let i = n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = dir2.join(format!("db{i}.sqlite"));
        Arc::new(
            chapkit_store::open_sqlite(Some(&format!("sqlite:///{}", path.display()))).unwrap(),
        )
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

#[cfg(feature = "turso")]
#[tokio::test]
async fn turso_in_memory() {
    configs_crud_and_conflict(chapkit_store::open(Some("turso::memory:")).await.unwrap()).await;
    artifact_tree_links_and_cascade(chapkit_store::open(Some("turso::memory:")).await.unwrap())
        .await;
    datasets_crud_and_artifact_types(chapkit_store::open(Some("turso::memory:")).await.unwrap())
        .await;
}

#[cfg(not(feature = "turso"))]
#[tokio::test]
async fn turso_url_without_feature_is_an_error() {
    assert!(matches!(
        chapkit_store::open(Some("turso::memory:")).await,
        Err(StoreError::Invalid(_))
    ));
}
