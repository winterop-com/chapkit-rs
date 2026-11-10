#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]
//! The chapkit HTTP contract, end to end and in-process, with a tiny model defined
//! here: the exact flow chap-core runs (config, $train, poll job, $predict, poll job,
//! read artifact), plus error shapes. Model-specific checks (parity with chap-core's
//! baselines) live in the chapkit-rs-models repository.

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use chapkit::{
    DataFrame, Diagnostic, Model, ModelInfo, PredictInput, Result as ModelResult, TrainInput, cols,
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

    fn search_space(&self) -> Option<chapkit::SearchSpace> {
        Some(chapkit::SearchSpace::new().with(
            "scale",
            chapkit::Param::Float {
                low: 0.5,
                high: 2.0,
                step: Some(0.25),
                log: false,
            },
        ))
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

    fn validate_train(&self, cfg: &MeanConfig, _: TrainInput<'_>) -> Vec<Diagnostic> {
        if cfg.scale < 0.0 {
            vec![
                Diagnostic::error("negative_scale", "scale must not be negative")
                    .at("config.scale"),
            ]
        } else {
            Vec::new()
        }
    }

    fn validate_predict(&self, _: &MeanConfig, input: PredictInput<'_>) -> Vec<Diagnostic> {
        if input.historic.n_rows() < 2 {
            vec![
                Diagnostic::warning("short_history", "less than two rows of history")
                    .at("historic"),
            ]
        } else {
            Vec::new()
        }
    }
}

async fn app() -> Router {
    chapkit::Service::new("mean", "Mean", MeanModel)
        // CHAPKIT_TEST_DATABASE_URL picks the backend (`make test-turso` runs this on turso).
        .database(std::env::var("CHAPKIT_TEST_DATABASE_URL").unwrap_or_else(|_| ":memory:".into()))
        .app()
        .await
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
    let app = app().await;
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
    let (s, frame, _) = call(
        &app,
        "POST",
        "/api/v1/ml/$predict-sync",
        Some(predict_body.clone()),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{frame}");
    assert_eq!(frame["data"], content["data"]);

    // Explain one forecast cell. The mean model ignores rainfall, so its importance is zero.
    let mut explain_body = predict_body.clone();
    explain_body["location"] = json!("A");
    explain_body["time_period"] = json!("2025-01");
    explain_body["method"] = json!("shap");
    let (s, ex, _) = call(&app, "POST", "/api/v1/ml/$explain", Some(explain_body)).await;
    assert_eq!(s, StatusCode::OK, "{ex}");
    assert_eq!(ex["method"], "shap");
    assert_eq!(ex["prediction"], 10.0);
    assert_eq!(
        ex["features"],
        json!([
            "rainfall",
            "rainfall@t-1",
            "disease_cases@t-1",
            "rainfall@t-2",
            "disease_cases@t-2",
            "rainfall@t-3",
            "disease_cases@t-3"
        ])
    );
    assert!(
        ex["importances"]
            .as_array()
            .unwrap()
            .iter()
            .all(|i| i["weight"] == 0.0),
        "{ex}"
    );
    assert!(
        ex.get("r2").is_none(),
        "a constant forecast has no R2: {ex}"
    );
    assert!(ex["elapsed_ms"].as_f64().unwrap() >= 0.0);

    // Sample data generator, as cdx and the console use it.
    let (s, sample, _) = call(
        &app,
        "GET",
        "/api/v1/ml/$generate-sample-data?num_locations=2&num_periods=6&seed=1",
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{sample}");
    assert_eq!(sample["data"]["columns"][2], "disease_cases");
    assert_eq!(sample["data"]["data"].as_array().unwrap().len(), 12);
    let (_, sample2, _) = call(
        &app,
        "GET",
        "/api/v1/ml/$generate-sample-data?num_locations=2&num_periods=6&seed=1",
        None,
    )
    .await;
    assert_eq!(sample, sample2, "same seed, same data");
    let (_, pred, _) = call(
        &app,
        "GET",
        "/api/v1/ml/$generate-sample-data?kind=predict&num_periods=4&include_geo=true",
        None,
    )
    .await;
    assert_eq!(pred["future"]["data"].as_array().unwrap().len(), 4 * 5);
    assert_eq!(pred["geo"]["type"], "FeatureCollection");
    assert!(
        info["chapkit_version"]
            .as_str()
            .unwrap()
            .starts_with("2.6.0+chapkit-rs.")
    );

    // A finished job's stream sends its record once and ends.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/jobs/{job_id}/$stream?poll_interval=0.01"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.headers()["content-type"], "text/event-stream");
    assert_eq!(res.headers()["x-accel-buffering"], "no");
    let text =
        String::from_utf8(res.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
    assert!(
        text.starts_with("data: {") && text.contains("\"completed\"") && text.ends_with("\n\n"),
        "{text}"
    );
    let (s, _, _) = call(
        &app,
        "GET",
        &format!("/api/v1/jobs/{job_id}/$stream?poll_interval=0"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        res.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/plain; version=0.0.4")
    );
    let text =
        String::from_utf8(res.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
    assert!(text.contains("ml_train_jobs_total"), "{text}");
    assert!(
        text.contains(
            "http_server_requests_total{method=\"POST\",route=\"/api/v1/ml/$train\",status=\"202\"}"
        ),
        "{text}"
    );

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
    let app = app().await;
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

/// Forecasts the location's last observed case count plus the step's rainfall, so it reads
/// `historic` as well as `future`.
struct LastCasesModel;

impl Model for LastCasesModel {
    type Config = MeanConfig;
    type Trained = f64;

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn train(&self, _: &MeanConfig, _: TrainInput<'_>) -> ModelResult<f64> {
        Ok(1.0)
    }

    fn predict(&self, _: &f64, _: &MeanConfig, input: PredictInput<'_>) -> ModelResult<DataFrame> {
        let cases = input.historic.f64_column(cols::DISEASE_CASES)?;
        let hist_loc = input.historic.require(cols::LOCATION)?;
        let last: HashMap<String, f64> = (0..input.historic.n_rows())
            .filter_map(|r| Some((hist_loc.str_at(r)?.to_owned(), *cases.get(r)?)))
            .collect();
        let rain = input.future.f64_column("rainfall")?;
        let loc = input.future.require(cols::LOCATION)?;
        prediction_frame(input.future, 4, |row, out| {
            let base = loc.str_at(row).and_then(|l| last.get(l)).copied();
            out.fill(base.unwrap_or(0.0) + rain.get(row).copied().unwrap_or(0.0));
            Ok(())
        })
    }
}

/// `$explain` perturbs the cell's covariates, their earlier values and the target's history
/// (`scope`, `lags`), and reports the surrogate's fit (`r2`, `effective_n`).
#[tokio::test]
async fn explain_temporal_features() {
    let app = chapkit::Service::new("lag", "Lag", LastCasesModel)
        .database(":memory:")
        .app()
        .await
        .unwrap()
        .build()
        .0;
    let fx = fixture();
    let (_, sub) = train_model(&app, &fx["train"]).await;
    let body = |extra: Value| {
        let mut b = json!({"artifact_id": sub["artifact_id"], "historic": fx["historic"],
            "future": fx["future"], "location": "A"});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b
    };
    let explain = |b: Value| {
        let app = app.clone();
        async move { call(&app, "POST", "/api/v1/ml/$explain", Some(b)).await }
    };

    // Default: temporal scope, three historic lags. The forecast follows the last observed
    // count one for one; historic counts are 10 (A) and 20 (B), so one SD is 5.
    let (s, ex, _) = explain(body(json!({"time_period": "2025-01"}))).await;
    assert_eq!(s, StatusCode::OK, "{ex}");
    assert_eq!(ex["features"].as_array().unwrap().len(), 7, "{ex}");
    assert_eq!(ex["importances"][0]["feature"], "disease_cases@t-1");
    assert!((ex["importances"][0]["weight"].as_f64().unwrap() - 5.0).abs() < 1e-6);
    assert!((ex["r2"].as_f64().unwrap() - 1.0).abs() < 1e-9, "{ex}");
    let n_eff = ex["effective_n"].as_f64().unwrap();
    assert!(n_eff > 1.0 && n_eff <= 500.0, "{ex}");

    // Earlier forecast steps are features too; lags 0 leaves history out.
    let (_, ex, _) = explain(body(json!({"time_period": "2025-03", "lags": 0}))).await;
    assert_eq!(
        ex["features"],
        json!(["rainfall", "rainfall@t-1", "rainfall@t-2"])
    );

    // Row scope: only the cell's own covariates, as before.
    let (_, ex, _) = explain(body(json!({"time_period": "2025-03", "scope": "row"}))).await;
    assert_eq!(ex["features"], json!(["rainfall"]));

    // SHAP over the same features reports its additive fit, no effective N.
    let (_, ex, _) = explain(body(json!({"time_period": "2025-01", "method": "shap"}))).await;
    assert!((ex["r2"].as_f64().unwrap() - 1.0).abs() < 1e-9, "{ex}");
    assert!(ex.get("effective_n").is_none());

    let (s, _, _) = explain(body(json!({"time_period": "2025-01", "scope": "lagged"}))).await;
    assert!(s.is_client_error(), "{s}");
}

/// Needs geo to predict: fails without it, like a model with spatial smoothing.
struct GeoModel;

impl Model for GeoModel {
    type Config = MeanConfig;
    type Trained = f64;

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn search_space(&self) -> Option<chapkit::SearchSpace> {
        Some(chapkit::SearchSpace::new().with(
            "scale",
            chapkit::Param::Float {
                low: 0.5,
                high: 2.0,
                step: Some(0.25),
                log: false,
            },
        ))
    }

    fn train(&self, _: &MeanConfig, _: TrainInput<'_>) -> ModelResult<f64> {
        Ok(1.0)
    }

    fn predict(&self, _: &f64, _: &MeanConfig, input: PredictInput<'_>) -> ModelResult<DataFrame> {
        if input.geo.is_none() {
            return Err(chapkit::Error::InvalidInput("geo is required".into()));
        }
        let rain = input.future.f64_column("rainfall")?;
        prediction_frame(input.future, 4, |row, out| {
            out.fill(rain.get(row).copied().unwrap_or(0.0));
            Ok(())
        })
    }
}

async fn train_model(app: &Router, data: &Value) -> (StatusCode, Value) {
    let (_, cfg, _) = call(
        app,
        "POST",
        "/api/v1/configs",
        Some(json!({"name": format!("c{}", ulid::Ulid::generate()), "data": {}})),
    )
    .await;
    let (s, sub, _) = call(
        app,
        "POST",
        "/api/v1/ml/$train",
        Some(json!({"config_id": cfg["id"], "data": data, "run_info": {"prediction_length": 3}})),
    )
    .await;
    (s, sub)
}

async fn job_status(app: &Router, sub: &Value) -> Value {
    let (_, job, _) = call(
        app,
        "GET",
        &format!("/api/v1/jobs/{}", sub["job_id"].as_str().unwrap()),
        None,
    )
    .await;
    job["status"].clone()
}

#[tokio::test]
async fn review_regressions() {
    let app = app().await;
    let fx = fixture();

    // A non-ASCII period id used to panic in provenance and leave the job "running".
    let mut bad = fx["train"].clone();
    bad["data"][0][0] = json!("123é4");
    let (s, sub) = train_model(&app, &bad).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{sub}");
    assert_eq!(job_status(&app, &sub).await, "completed");

    // Bodies over axum's old 2 MiB default are accepted: 50,000 training rows.
    let rows: Vec<Value> = (0..50_000)
        .map(|i| {
            json!([
                format!("{}-{:02}", 1900 + i / 12 % 120, i % 12 + 1),
                format!("loc{}", i / 1440),
                12.5,
                100.0,
                25.25,
                123_456.0
            ])
        })
        .collect();
    let big = json!({"columns": ["time_period", "location", "disease_cases", "rainfall", "mean_temperature", "population"], "data": rows});
    assert!(serde_json::to_vec(&big).unwrap().len() > 2 << 20);
    let (s, sub) = train_model(&app, &big).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{sub}");
    assert_eq!(job_status(&app, &sub).await, "completed");

    // A configured limit answers 413, not a validation error.
    let small = chapkit::Service::new("mean", "Mean", MeanModel)
        .database(":memory:")
        .max_body_bytes(1024)
        .app()
        .await
        .unwrap()
        .build()
        .0;
    let (s, body) = train_model(&small, &big).await;
    assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["type"], "urn:servicekit:error:payload-too-large");

    // $validate: a prediction without artifact_id is not valid.
    let (s, v, _) = call(
        &app,
        "POST",
        "/api/v1/ml/$validate",
        Some(json!({"type": "predict", "historic": fx["historic"], "future": fx["future"]})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["valid"], false, "{v}");
    assert!(
        v["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["code"] == "artifact_id_missing"),
        "{v}"
    );

    // Pagination far past the end is an empty page, not an overflow.
    for path in ["/api/v1/configs", "/api/v1/artifacts"] {
        let (s, page, _) = call(&app, "GET", &format!("{path}?page=42949674&size=100"), None).await;
        assert_eq!(s, StatusCode::OK, "{path} {page}");
        assert_eq!(page["items"], json!([]));
        assert_eq!(page["page"], 42_949_674);
        let (s, _, _) = call(
            &app,
            "GET",
            &format!("{path}?page=9223372036854775807&size=100"),
            None,
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{path} at i64::MAX");
    }

    // $explain passes geo to the model, as $predict-sync does.
    let geo_app = chapkit::Service::new("geo", "Geo", GeoModel)
        .database(":memory:")
        .app()
        .await
        .unwrap()
        .build()
        .0;
    let (_, sub) = train_model(&geo_app, &fx["train"]).await;
    let geo = json!({"type": "FeatureCollection", "features": []});
    let body = json!({"artifact_id": sub["artifact_id"], "historic": fx["historic"], "future": fx["future"], "geo": geo});
    let (s, out, _) = call(
        &geo_app,
        "POST",
        "/api/v1/ml/$predict-sync",
        Some(body.clone()),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{out}");
    let mut explain = body;
    explain["location"] = json!("A");
    explain["time_period"] = json!("2025-01");
    explain["method"] = json!("shap");
    let (s, ex, _) = call(&geo_app, "POST", "/api/v1/ml/$explain", Some(explain)).await;
    assert_eq!(s, StatusCode::OK, "{ex}");
    assert_eq!(ex["prediction"], 100.0, "{ex}");
}

#[tokio::test]
async fn model_validation_hooks() {
    let app = app().await;
    let fx = fixture();
    let codes = |v: &Value| -> Vec<String> {
        v["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["code"].as_str().unwrap().to_owned())
            .collect()
    };
    let config = |scale: f64| json!({"name": format!("c{}", ulid::Ulid::generate()), "data": {"scale": scale}});
    let (_, bad, _) = call(&app, "POST", "/api/v1/configs", Some(config(-1.0))).await;
    let (_, good, _) = call(&app, "POST", "/api/v1/configs", Some(config(1.0))).await;

    // The model's error makes the request invalid, naming the field.
    let body = json!({"type": "train", "config_id": bad["id"], "data": fx["train"]});
    let (_, v, _) = call(&app, "POST", "/api/v1/ml/$validate", Some(body)).await;
    assert_eq!(v["valid"], false, "{v}");
    assert_eq!(codes(&v), ["negative_scale"]);
    assert_eq!(v["diagnostics"][0]["field"], "config.scale");

    // When chapkit's own checks fail, the model's do not run.
    let empty = json!({"columns": ["time_period", "location", "disease_cases"], "data": []});
    let body = json!({"type": "train", "config_id": bad["id"], "data": empty});
    let (_, v, _) = call(&app, "POST", "/api/v1/ml/$validate", Some(body)).await;
    assert_eq!(codes(&v), ["data_empty"], "{v}");

    // A clean request has no findings.
    let body = json!({"type": "train", "config_id": good["id"], "data": fx["train"]});
    let (_, v, _) = call(&app, "POST", "/api/v1/ml/$validate", Some(body)).await;
    assert_eq!(v, json!({"valid": true, "diagnostics": []}));

    // A warning is reported but the request stays valid.
    let (_, sub, _) = call(
        &app,
        "POST",
        "/api/v1/ml/$train",
        Some(json!({"config_id": good["id"], "data": fx["train"], "run_info": {"prediction_length": 3}})),
    )
    .await;
    let mut one_row = fx["historic"].clone();
    one_row["data"] = json!([fx["historic"]["data"][0]]);
    let body = json!({"type": "predict", "artifact_id": sub["artifact_id"], "historic": one_row, "future": fx["future"]});
    let (_, v, _) = call(&app, "POST", "/api/v1/ml/$validate", Some(body)).await;
    assert_eq!(v["valid"], true, "{v}");
    assert_eq!(codes(&v), ["short_history"]);
    assert_eq!(v["diagnostics"][0]["severity"], "warning");
}

/// The flow behind `chapkit-rs test` and a model binary's `test`, over HTTP against a served app;
/// and the check that tells the binary's `test` nothing is running.
#[tokio::test]
async fn test_command_flow_over_http() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let router = app().await;
    tokio::spawn(async move { axum::serve(listener, router).await });

    let cli = <chapkit::cli::Cli as clap::Parser>::try_parse_from([
        "mean",
        "test",
        "--url",
        &url,
        "--locations",
        "3",
        "--configs",
        "2",
        "--seed",
        "7",
    ])
    .unwrap();
    let chapkit::cli::Command::Test(t) = cli.command else {
        panic!("expected test")
    };
    assert!(chapkit::flow::reachable(&url).await);
    chapkit::flow::run(&url, &t.flow).await.unwrap();

    let free = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unused = format!("http://{}", free.local_addr().unwrap());
    drop(free);
    assert!(!chapkit::flow::reachable(&unused).await);
}

/// Trains slowly, so jobs overlap.
struct SlowModel;

impl Model for SlowModel {
    type Config = MeanConfig;
    type Trained = f64;

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn search_space(&self) -> Option<chapkit::SearchSpace> {
        Some(chapkit::SearchSpace::new().with(
            "scale",
            chapkit::Param::Float {
                low: 0.5,
                high: 2.0,
                step: Some(0.25),
                log: false,
            },
        ))
    }

    fn train(&self, _: &MeanConfig, _: TrainInput<'_>) -> ModelResult<f64> {
        std::thread::sleep(std::time::Duration::from_millis(300));
        Ok(1.0)
    }

    fn predict(&self, _: &f64, _: &MeanConfig, input: PredictInput<'_>) -> ModelResult<DataFrame> {
        prediction_frame(input.future, 1, |_, out| {
            out.fill(1.0);
            Ok(())
        })
    }
}

/// Three `$train` jobs on a service allowed two at once: two run, one waits, then all complete.
async fn check_two_jobs_at_once(service: chapkit::Service) {
    let app = service
        .database(":memory:")
        .eager_wait(std::time::Duration::ZERO)
        .app()
        .await
        .unwrap()
        .build()
        .0;
    let data = fixture()["train"].clone();
    for _ in 0..3 {
        assert_eq!(train_model(&app, &data).await.0, StatusCode::ACCEPTED);
    }
    let count = |jobs: &Value, status: &str| {
        jobs.as_array()
            .unwrap()
            .iter()
            .filter(|j| j["status"] == status)
            .count()
    };
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let (_, jobs, _) = call(&app, "GET", "/api/v1/jobs", None).await;
    assert_eq!(
        (count(&jobs, "running"), count(&jobs, "pending")),
        (2, 1),
        "{jobs}"
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let (_, jobs, _) = call(&app, "GET", "/api/v1/jobs", None).await;
        if count(&jobs, "completed") == 3 {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "{jobs}");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// `max_concurrency` and `serve --max-concurrency` reach the job scheduler.
#[tokio::test]
async fn max_concurrency_limits_running_jobs() {
    check_two_jobs_at_once(chapkit::Service::new("slow", "Slow", SlowModel).max_concurrency(2))
        .await;
    let cli = <chapkit::cli::Cli as clap::Parser>::try_parse_from([
        "slow",
        "serve",
        "--max-concurrency",
        "2",
    ])
    .unwrap();
    let chapkit::cli::Command::Serve(args) = cli.command else {
        panic!("expected serve")
    };
    check_two_jobs_at_once(chapkit::Service::new("slow", "Slow", SlowModel).with_serve_args(&args))
        .await;
}

/// Two locations x 24 months with a seasonal target, for backtests.
fn backtest_data() -> Value {
    let mut rows = vec![];
    for (loc, base) in [("A", 10.0), ("B", 30.0)] {
        for i in 0..24 {
            let y = base + f64::from(i % 12) * 2.0 + f64::from(i % 5);
            rows.push(json!([
                format!("{}-{:02}", 2022 + i / 12, i % 12 + 1),
                loc,
                y,
                100.0 + f64::from(i)
            ]));
        }
    }
    json!({"columns": ["time_period", "location", "disease_cases", "rainfall"], "data": rows})
}

/// Poll a job until it is done; its final record.
async fn finished(app: &Router, sub: &Value) -> Value {
    let uri = format!("/api/v1/jobs/{}", sub["job_id"].as_str().unwrap());
    for _ in 0..500 {
        let (_, job, _) = call(app, "GET", &uri, None).await;
        if job["status"] != "pending" && job["status"] != "running" {
            return job;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("job {uri} did not finish");
}

#[tokio::test]
async fn backtest_job_stores_the_eval_result() {
    let app = app().await;
    let (_, info, _) = call(&app, "GET", "/api/v1/info", None).await;
    assert!(
        info["extensions"]
            .as_array()
            .unwrap()
            .contains(&json!("$backtest"))
    );
    let data = backtest_data();

    // Inline model options, chap-core's backtest settings.
    let body =
        json!({"config": {"scale": 2.0}, "data": data, "periods": 2, "splits": 3, "stride": 2});
    let (s, sub, _) = call(&app, "POST", "/api/v1/ml/$backtest", Some(body)).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{sub}");
    assert!(
        sub["message"]
            .as_str()
            .unwrap()
            .starts_with("Backtest job submitted")
    );
    let job = finished(&app, &sub).await;
    assert_eq!(job["status"], "completed", "{job}");
    let uri = format!("/api/v1/artifacts/{}", sub["artifact_id"].as_str().unwrap());
    let (s, artifact, _) = call(&app, "GET", &uri, None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(artifact["level"], 0);
    assert_eq!(artifact["data"]["type"], "ml_backtest");
    assert_eq!(artifact["data"]["content_type"], "application/json");
    let result = &artifact["data"]["content"];
    assert_eq!(result["model"], "mean");
    assert_eq!(result["config"]["scale"], 2.0);
    assert_eq!(result["config"]["prediction_periods"], 2);
    assert_eq!(
        result["options"],
        json!({"periods": 2, "splits": 3, "stride": 2, "retrain": 1, "weather": "climatology",
               "fill_gaps": true, "pad": true, "drop_untrained": true})
    );
    assert_eq!(result["data"]["source"], "request");
    assert_eq!(result["data"]["locations"], 2);
    assert_eq!(result["rows"], 12);
    let doubled = result["aggregate"]["crps"].as_f64().unwrap();
    assert!(doubled > 0.0);
    assert_eq!(result["by_horizon"].as_array().unwrap().len(), 2);
    assert_eq!(result["by_location"][1]["location"], "B");
    let (_, tree, _) = call(&app, "GET", &format!("{uri}/$tree"), None).await;
    assert_eq!(tree["level_label"], "ml_backtest");
    assert!(tree["data"].get("content").is_none());

    // A stored config; the scores follow its options.
    let (_, cfg, _) = call(
        &app,
        "POST",
        "/api/v1/configs",
        Some(json!({"name": "unscaled", "data": {"prediction_periods": 2}})),
    )
    .await;
    let body = json!({"config_id": cfg["id"], "data": data, "splits": 3, "stride": 2, "weather": "damped_persistence"});
    let (s, sub, _) = call(&app, "POST", "/api/v1/ml/$backtest", Some(body)).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{sub}");
    assert_eq!(finished(&app, &sub).await["status"], "completed");
    let (_, artifact, _) = call(
        &app,
        "GET",
        &format!("/api/v1/artifacts/{}", sub["artifact_id"].as_str().unwrap()),
        None,
    )
    .await;
    let result = &artifact["data"]["content"];
    assert_eq!(artifact["data"]["metadata"]["config_id"], cfg["id"]);
    assert_eq!(result["options"]["periods"], 2, "from the config");
    assert_eq!(result["options"]["weather"], "damped_persistence");
    assert!(result["aggregate"]["crps"].as_f64().unwrap() < doubled);

    // Bad requests are refused before a job starts; data that cannot be split fails the job.
    for (body, status) in [
        (
            json!({"config": {}, "config_id": cfg["id"], "data": data}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"config_id": ulid::Ulid::generate().to_string(), "data": data}),
            StatusCode::NOT_FOUND,
        ),
        (
            json!({"data": data, "weather": "mstl_arima"}),
            StatusCode::BAD_REQUEST,
        ),
        (json!({"data": data, "retrain": 9}), StatusCode::BAD_REQUEST),
        (
            json!({"config": {"scale": "big"}, "data": data}),
            StatusCode::BAD_REQUEST,
        ),
    ] {
        let (s, p, _) = call(&app, "POST", "/api/v1/ml/$backtest", Some(body)).await;
        assert_eq!(s, status, "{p}");
    }
    let (s, sub, _) = call(
        &app,
        "POST",
        "/api/v1/ml/$backtest",
        Some(json!({"data": data, "splits": 30})),
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED);
    let job = finished(&app, &sub).await;
    assert_eq!(job["status"], "failed");
    assert!(
        job["error"]
            .as_str()
            .unwrap_or_default()
            .contains("too few"),
        "{job}"
    );
}

#[tokio::test]
async fn hpo_job_stores_the_search_and_can_create_the_best_config() {
    let app = app().await;
    let (_, info, _) = call(&app, "GET", "/api/v1/info", None).await;
    assert!(
        info["extensions"]
            .as_array()
            .unwrap()
            .contains(&json!("$hpo"))
    );
    assert_eq!(info["hpo_search_space"]["scale"]["step"], 0.25);
    let data = backtest_data();

    // The grid of `scale` (0.5 to 2 by 0.25: 7 values), the best config created.
    // chap-core's grid takes choices only, so a range needs grid_ranges.
    let refused = json!({"data": data, "periods": 2, "splits": 3, "method": "grid", "trials": 50});
    let (s, err, _) = call(&app, "POST", "/api/v1/ml/$hpo", Some(refused)).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{err}");
    assert!(
        err["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("is a range"),
        "{err}"
    );
    let body = json!({"data": data, "periods": 2, "splits": 3, "method": "grid", "trials": 50,
                      "grid_ranges": true, "metric": "mae", "create_config": true,
                      "config_name": "best scale"});
    let (s, sub, _) = call(&app, "POST", "/api/v1/ml/$hpo", Some(body)).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{sub}");
    assert!(
        sub["message"]
            .as_str()
            .unwrap()
            .starts_with("Hyperparameter search job submitted")
    );
    let job = finished(&app, &sub).await;
    assert_eq!(job["status"], "completed", "{job}");
    let uri = format!("/api/v1/artifacts/{}", sub["artifact_id"].as_str().unwrap());
    let (_, artifact, _) = call(&app, "GET", &uri, None).await;
    assert_eq!(artifact["data"]["type"], "ml_hpo");
    let result = &artifact["data"]["content"];
    assert_eq!(result["metric"], "mae");
    let search = &result["search"];
    assert_eq!(search["method"], "grid");
    assert_eq!(search["n_trials"], 7);
    assert_eq!(search["stop_reason"], "search_exhausted");
    assert_eq!(search["leaderboard"].as_array().unwrap().len(), 7);
    let best = search["best_params"]["scale"].as_f64().unwrap();
    assert_eq!(result["best_config"]["scale"], best);
    assert_eq!(result["best"]["aggregate"]["mae"], search["best_score"]);
    let scores: Vec<f64> = search["leaderboard"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["score"].as_f64().unwrap())
        .collect();
    assert!(scores.windows(2).all(|w| w[0] <= w[1]));
    let (_, tree, _) = call(&app, "GET", &format!("{uri}/$tree"), None).await;
    assert_eq!(tree["level_label"], "ml_hpo");

    // The created config holds the best options; training with it works as with any config.
    let best_id = result["best_config_id"].as_str().unwrap();
    let (s, cfg, _) = call(&app, "GET", &format!("/api/v1/configs/{best_id}"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(cfg["name"], "best scale");
    assert_eq!(cfg["data"]["scale"], best);
    let (s, sub, _) = call(
        &app,
        "POST",
        "/api/v1/ml/$train",
        Some(json!({"config_id": best_id, "data": data})),
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED);
    assert_eq!(finished(&app, &sub).await["status"], "completed");

    // Random search from a stored config, without creating one; reproducible by seed.
    let (_, base, _) = call(
        &app,
        "POST",
        "/api/v1/configs",
        Some(json!({"name": "base", "data": {"prediction_periods": 2}})),
    )
    .await;
    let mut boards = vec![];
    for _ in 0..2 {
        let body =
            json!({"config_id": base["id"], "data": data, "splits": 3, "trials": 6, "seed": 4});
        let (s, sub, _) = call(&app, "POST", "/api/v1/ml/$hpo", Some(body)).await;
        assert_eq!(s, StatusCode::ACCEPTED, "{sub}");
        assert_eq!(finished(&app, &sub).await["status"], "completed");
        let (_, a, _) = call(
            &app,
            "GET",
            &format!("/api/v1/artifacts/{}", sub["artifact_id"].as_str().unwrap()),
            None,
        )
        .await;
        let content = &a["data"]["content"];
        assert!(content.get("best_config_id").is_none());
        assert_eq!(content["search"]["method"], "random");
        assert_eq!(content["metric"], "crps");
        assert_eq!(a["data"]["metadata"]["config_id"], base["id"]);
        let board: Vec<(Value, Value)> = content["search"]["leaderboard"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t["params"].clone(), t["score"].clone()))
            .collect();
        boards.push(board);
    }
    assert_eq!(boards[0], boards[1]);
    assert_eq!(boards[0].len(), 6);

    for body in [
        json!({"data": data, "metric": "mape"}),
        json!({"data": data, "metric": "nope"}),
        json!({"data": data, "method": "tpe"}),
        json!({"data": data, "trials": 0}),
        json!({"data": data, "weather": "mstl_arima"}),
    ] {
        let (s, p, _) = call(&app, "POST", "/api/v1/ml/$hpo", Some(body)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{p}");
    }
}
