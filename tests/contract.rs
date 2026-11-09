#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]
//! The chapkit HTTP contract, end to end and in-process, with a tiny model defined
//! here: the exact flow chap-core runs (config, $train, poll job, $predict, poll job,
//! read artifact), plus error shapes. Model-specific checks (parity with chap-core's
//! baselines) live in the chapkit-rs-examples repository.

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use chapkit::{
    DataFrame, Model, ModelInfo, PredictInput, Result as ModelResult, TrainInput, cols,
    prediction_frame,
};
use http_body_util::BodyExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use tower::ServiceExt;

/// Per-location mean times `scale`, every sample the same.
struct MeanModel;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct MeanConfig {
    #[serde(default = "one")]
    scale: f64,
}

fn one() -> f64 {
    1.0
}

impl Model for MeanModel {
    type Config = MeanConfig;
    type Trained = HashMap<String, f64>;

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn train(&self, _: &MeanConfig, input: TrainInput<'_>) -> ModelResult<HashMap<String, f64>> {
        let y = input.data.f64_column(cols::DISEASE_CASES)?;
        Ok(input
            .data
            .group_indices(cols::LOCATION)?
            .into_iter()
            .map(|(loc, idx)| {
                (
                    loc,
                    idx.iter().map(|&i| y[i]).sum::<f64>() / idx.len() as f64,
                )
            })
            .collect())
    }

    fn predict(
        &self,
        t: &HashMap<String, f64>,
        cfg: &MeanConfig,
        input: PredictInput<'_>,
    ) -> ModelResult<DataFrame> {
        let loc = input.future.require(cols::LOCATION)?;
        prediction_frame(input.future, 10, |row, out| {
            let m = loc
                .str_at(row)
                .and_then(|l| t.get(l))
                .copied()
                .unwrap_or(0.0);
            out.fill(m * cfg.scale);
            Ok(())
        })
    }
}

fn app() -> Router {
    chapkit::Service::new("mean", "Mean", MeanModel)
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

/// Two locations x 12 months of training data, then 3 future months.
fn fixture() -> Value {
    let (mut train, mut future) = (vec![], vec![]);
    for loc in ["A", "B"] {
        for m in 1..=12 {
            let y = if loc == "A" { 10.0 } else { 20.0 };
            train.push(json!([format!("2024-{m:02}"), loc, y, 100.0]));
        }
        for m in 1..=3 {
            future.push(json!([format!("2025-{m:02}"), loc, 100.0]));
        }
    }
    json!({
        "train": {"columns": ["time_period", "location", "disease_cases", "rainfall"], "data": train},
        "historic": {"columns": ["time_period", "location", "disease_cases", "rainfall"], "data": train},
        "future": {"columns": ["time_period", "location", "rainfall"], "data": future},
    })
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
    assert_eq!(info["id"], "mean");
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
    let sent = json!({"name": "cfg", "data": {"additional_continuous_covariates": ["rainfall"], "prediction_periods": 3}});
    let (s, cfg, headers) = call(&app, "POST", "/api/v1/configs", Some(sent)).await;
    assert_eq!(s, StatusCode::CREATED, "{cfg}");
    assert_eq!(
        cfg["data"]["additional_continuous_covariates"],
        json!(["rainfall"])
    );
    assert_eq!(cfg["data"]["prediction_periods"], 3);
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
        json!({"config_id": config_id, "data": fx["train"], "run_info": {"prediction_length": 3}});
    let (s, sub, _) = call(&app, "POST", "/api/v1/ml/$train", Some(train_body)).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{sub}");
    let (job_id, model_id) = (
        sub["job_id"].as_str().unwrap().to_owned(),
        sub["artifact_id"].as_str().unwrap().to_owned(),
    );

    // Eager completion: finished by the first poll, so chap-core never sleeps.
    let (_, job, _) = call(&app, "GET", &format!("/api/v1/jobs/{job_id}"), None).await;
    assert_eq!(job["status"], "completed", "{job}");

    let predict_body = json!({"artifact_id": model_id, "historic": fx["historic"], "future": fx["future"], "run_info": {"prediction_length": 3}});
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

    // Model output: per-location mean times the config's default scale.
    let first = &content["data"][0];
    assert_eq!(first[1], "A");
    assert_eq!(first[2], 10.0);
    assert_eq!(
        cfg["data"]["scale"], 1.0,
        "model defaults are filled into the stored config"
    );

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
