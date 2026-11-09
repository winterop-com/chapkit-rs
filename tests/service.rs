#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]
//! The chapkit HTTP contract, end to end and in-process: the exact flow chap-core
//! runs (config, $train, poll job, $predict, poll job, read artifact), plus error shapes.

#[path = "../examples/common/empirical.rs"]
mod empirical;
#[path = "../examples/global_median/model.rs"]
mod global_median;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

fn app() -> Router {
    chapkit::Service::new(
        "global-median",
        "Global median",
        global_median::GlobalMedian,
    )
    .database(":memory:")
    .app()
    .unwrap()
    .build()
    .0
}

async fn call(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value, axum::http::HeaderMap) {
    let mut req = Request::builder()
        .method(method)
        .uri(uri)
        .header("host", "testserver");
    let body = match body {
        Some(v) => {
            req = req.header("content-type", "application/json");
            Body::from(serde_json::to_vec(&v).unwrap())
        }
        None => Body::empty(),
    };
    let res = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
    let (status, headers) = (res.status(), res.headers().clone());
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, value, headers)
}

fn fixture() -> Value {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/baselines_monthly.json"
    ))
    .unwrap();
    serde_json::from_str(&raw).unwrap()
}

#[tokio::test]
async fn chap_core_flow() {
    let app = app();
    let fx = fixture();

    let (s, health, _) = call(&app, "GET", "/health", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        health,
        json!({"status": "healthy", "checks": {"database": {"state": "healthy"}}})
    );

    let (s, info, _) = call(&app, "GET", "/api/v1/info", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(info["id"], "global-median");
    assert_eq!(info["target"], "disease_cases");
    assert!(
        info["model_metadata"]
            .as_object()
            .unwrap()
            .contains_key("citation_info")
    );

    let (_, schema, _) = call(&app, "GET", "/api/v1/configs/$schema", None).await;
    assert!(schema["properties"]["prediction_periods"].is_object());

    // chap-core sends flat options plus covariates and checks they come back unchanged.
    let sent = json!({"name": "cfg", "data": {"additional_continuous_covariates": ["rainfall"], "prediction_periods": 6}});
    let (s, cfg, headers) = call(&app, "POST", "/api/v1/configs", Some(sent)).await;
    assert_eq!(s, StatusCode::CREATED, "{cfg}");
    assert_eq!(
        cfg["data"]["additional_continuous_covariates"],
        json!(["rainfall"])
    );
    assert_eq!(cfg["data"]["prediction_periods"], 6);
    for k in ["id", "created_at", "updated_at", "tags", "name", "data"] {
        assert!(cfg.get(k).is_some(), "ConfigOut missing {k}");
    }
    let config_id = cfg["id"].as_str().unwrap().to_owned();
    assert_eq!(
        headers["location"],
        format!("http://testserver/api/v1/configs/{config_id}").as_str()
    );

    let (_, list, _) = call(&app, "GET", "/api/v1/configs", None).await;
    assert_eq!(list.as_array().unwrap().len(), 1);

    let train_body =
        json!({"config_id": config_id, "data": fx["train"], "run_info": {"prediction_length": 6}});
    let (s, sub, _) = call(&app, "POST", "/api/v1/ml/$train", Some(train_body)).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{sub}");
    let (job_id, model_id) = (
        sub["job_id"].as_str().unwrap().to_owned(),
        sub["artifact_id"].as_str().unwrap().to_owned(),
    );

    // Eager completion: finished by the first poll, so chap-core never sleeps.
    let (_, job, _) = call(&app, "GET", &format!("/api/v1/jobs/{job_id}"), None).await;
    assert_eq!(job["status"], "completed", "{job}");

    let predict_body = json!({"artifact_id": model_id, "historic": fx["historic"], "future": fx["future"], "run_info": {"prediction_length": 6}});
    let (s, sub, _) = call(
        &app,
        "POST",
        "/api/v1/ml/$predict",
        Some(predict_body.clone()),
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{sub}");
    let (_, job, _) = call(
        &app,
        "GET",
        &format!("/api/v1/jobs/{}", sub["job_id"].as_str().unwrap()),
        None,
    )
    .await;
    assert_eq!(job["status"], "completed", "{job}");

    let (s, art, _) = call(
        &app,
        "GET",
        &format!("/api/v1/artifacts/{}", sub["artifact_id"].as_str().unwrap()),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    // chap-core validates exactly these keys.
    let mut keys: Vec<&str> = art["data"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "content",
            "content_size",
            "content_type",
            "metadata",
            "type"
        ]
    );
    assert_eq!(art["data"]["type"], "ml_prediction");
    assert_eq!(art["parent_id"], model_id.as_str());
    for k in [
        "status",
        "config_id",
        "started_at",
        "completed_at",
        "duration_seconds",
    ] {
        assert!(
            art["data"]["metadata"].get(k).is_some(),
            "metadata missing {k}"
        );
    }
    let content = &art["data"]["content"];
    assert_eq!(content["columns"][0], "time_period");
    assert_eq!(content["columns"][1], "location");
    assert_eq!(
        content["data"].as_array().unwrap().len(),
        fx["future"]["data"].as_array().unwrap().len()
    );

    // The served forecast equals chap-core's own global_median, sample for sample.
    let expected = &fx["predictions"]["global_median"];
    for row in content["data"].as_array().unwrap() {
        let (tp, loc) = (row[0].as_str().unwrap(), row[1].as_str().unwrap());
        let key = format!("{loc}|{}", tp.replace('-', ""));
        let want = expected[&key].as_array().unwrap();
        for (g, w) in row.as_array().unwrap()[2..].iter().zip(want) {
            assert!((g.as_f64().unwrap() - w.as_f64().unwrap()).abs() < 1e-9);
        }
    }

    // Fast path: same forecast, returned directly.
    let (s, frame, _) = call(&app, "POST", "/api/v1/ml/$predict-sync", Some(predict_body)).await;
    assert_eq!(s, StatusCode::OK, "{frame}");
    assert_eq!(frame["data"], content["data"]);

    let (_, tree, _) = call(
        &app,
        "GET",
        &format!("/api/v1/artifacts/{model_id}/$tree"),
        None,
    )
    .await;
    assert_eq!(tree["level_label"], "ml_training");
    assert_eq!(tree["children"].as_array().unwrap().len(), 1);
    assert!(tree["children"][0]["data"].get("content").is_none());

    let (_, linked, _) = call(
        &app,
        "GET",
        &format!("/api/v1/configs/{config_id}/$artifacts"),
        None,
    )
    .await;
    assert_eq!(linked[0]["id"], model_id.as_str());

    let (s, _, _) = call(
        &app,
        "DELETE",
        &format!("/api/v1/configs/{config_id}"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    let (s, _, _) = call(&app, "GET", &format!("/api/v1/artifacts/{model_id}"), None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn failed_jobs_and_problem_responses() {
    let app = app();
    let fx = fixture();

    let missing = ulid::Ulid::generate().to_string();
    let (s, sub, _) = call(
        &app,
        "POST",
        "/api/v1/ml/$train",
        Some(json!({"config_id": missing, "data": fx["train"]})),
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED);
    let (_, job, _) = call(
        &app,
        "GET",
        &format!("/api/v1/jobs/{}", sub["job_id"].as_str().unwrap()),
        None,
    )
    .await;
    assert_eq!(job["status"], "failed");
    assert_eq!(
        job["error"],
        format!("ValueError: Config {missing} not found").as_str()
    );

    let (s, p, h) = call(&app, "GET", &format!("/api/v1/configs/{missing}"), None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_eq!(h["content-type"], "application/problem+json");
    assert_eq!(p["type"], "urn:servicekit:error:not-found");
    assert_eq!(p["instance"], format!("/api/v1/configs/{missing}").as_str());

    let (s, p, _) = call(&app, "GET", "/api/v1/configs/not-a-ulid", None).await;
    assert_eq!(
        (s, p["type"].as_str()),
        (
            StatusCode::BAD_REQUEST,
            Some("urn:servicekit:error:invalid-ulid")
        )
    );

    let (s, p, _) = call(&app, "POST", "/api/v1/configs", Some(json!({"data": {}}))).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(p["type"], "urn:servicekit:error:validation-failed");
    assert!(p["errors"].is_array());

    let (s, _, _) = call(&app, "GET", "/api/v1/configs?page=0&size=10", None).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    let (s, page, _) = call(&app, "GET", "/api/v1/configs?page=1&size=10", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        page,
        json!({"items": [], "total": 0, "page": 1, "size": 10, "pages": 0})
    );

    let id = ulid::Ulid::generate().to_string();
    let body = json!({"id": id, "name": "a", "data": {}});
    assert_eq!(
        call(&app, "POST", "/api/v1/configs", Some(body.clone()))
            .await
            .0,
        StatusCode::CREATED
    );
    assert_eq!(
        call(&app, "POST", "/api/v1/configs", Some(body)).await.0,
        StatusCode::CONFLICT
    );

    let (_, h, headers) = call(&app, "GET", "/health", None).await;
    assert_eq!(h["status"], "healthy");
    assert!(headers.contains_key("x-request-id"));

    let (s, doc, _) = call(&app, "GET", "/openapi.json", None).await;
    assert_eq!(s, StatusCode::OK);
    for path in [
        "/api/v1/ml/$train",
        "/api/v1/ml/$predict",
        "/api/v1/jobs/{job_id}",
        "/api/v1/configs/{id}",
        "/health",
    ] {
        assert!(doc["paths"].get(path).is_some(), "openapi missing {path}");
    }
}
