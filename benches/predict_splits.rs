#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]
//! A backtest's forecasts through the HTTP API, in-process: 7 splits of 12 weekly periods over
//! 136 locations x 300 weeks, as chap-core sends them one `$predict` at a time (submit, job status,
//! artifact read) against one `$predict-splits` call, with full histories and with one shared one.

use std::collections::HashMap;
use std::hint::black_box;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use chapkit::synth::{Generator, Resolution, Spec};
use chapkit::{
    DataFrame, Model, ModelInfo, NoConfig, Period, PredictInput, Result, TrainInput, cols,
    prediction_frame,
};
use criterion::{Criterion, criterion_group, criterion_main};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

fn key(location: &str, p: Period) -> String {
    format!("{location}|{}", p.period_of_year())
}

/// Mean of each location and week of year; 100 samples spread +-20% around it.
struct SeasonalMean;

impl Model for SeasonalMean {
    type Config = NoConfig;
    type Trained = HashMap<String, f64>;

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<Self::Trained> {
        let (tp, loc) = (
            input.data.require(cols::TIME_PERIOD)?,
            input.data.require(cols::LOCATION)?,
        );
        let y = input.data.f64_column(cols::DISEASE_CASES)?;
        let mut sums: HashMap<String, (f64, f64)> = HashMap::new();
        for (row, v) in y.iter().enumerate() {
            let p = Period::parse(tp.str_at(row).unwrap()).unwrap();
            let e = sums.entry(key(loc.str_at(row).unwrap(), p)).or_default();
            e.0 += v;
            e.1 += 1.0;
        }
        Ok(sums.into_iter().map(|(k, (s, n))| (k, s / n)).collect())
    }

    fn predict(
        &self,
        trained: &Self::Trained,
        _: &NoConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let (tp, loc) = (
            input.future.require(cols::TIME_PERIOD)?,
            input.future.require(cols::LOCATION)?,
        );
        prediction_frame(input.future, 100, |row, out| {
            let p = Period::parse(tp.str_at(row).unwrap()).unwrap();
            let mean = trained
                .get(&key(loc.str_at(row).unwrap(), p))
                .copied()
                .unwrap_or(0.0);
            for (i, s) in out.iter_mut().enumerate() {
                *s = mean * (0.8 + 0.4 * i as f64 / 99.0);
            }
            Ok(())
        })
    }
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Vec<u8>>) -> (StatusCode, Value) {
    let req = Request::builder().method(method).uri(uri);
    let req = match body {
        Some(b) => req
            .header("content-type", "application/json")
            .body(Body::from(b)),
        None => req.body(Body::empty()),
    };
    let res = app.clone().oneshot(req.unwrap()).await.unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// chap-core's per-split flow: submit, read the job, read the prediction artifact.
async fn predict_one(app: &Router, body: Vec<u8>) -> Value {
    let (s, sub) = call(app, "POST", "/api/v1/ml/$predict", Some(body)).await;
    assert_eq!(s, StatusCode::ACCEPTED);
    let job = format!("/api/v1/jobs/{}", sub["job_id"].as_str().unwrap());
    let (_, job) = call(app, "GET", &job, None).await;
    assert_eq!(job["status"], "completed", "{job}");
    let art = format!("/api/v1/artifacts/{}", sub["artifact_id"].as_str().unwrap());
    call(app, "GET", &art, None).await.1
}

fn bench(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let spec = Spec {
        num_locations: 136,
        num_periods: 300,
        resolution: Resolution::Weekly,
        ..Spec::default()
    };
    let data = serde_json::to_value(Generator::new(Some(1)).training_data(&spec).unwrap()).unwrap();
    let app = rt.block_on(async {
        let app = chapkit::Service::new("seasonal", "Seasonal", SeasonalMean)
            .database(":memory:")
            .app()
            .await
            .unwrap()
            .build()
            .0;
        let cfg = serde_json::to_vec(&json!({"name": "c", "data": {}})).unwrap();
        let (_, cfg) = call(&app, "POST", "/api/v1/configs", Some(cfg)).await;
        let train = serde_json::to_vec(&json!({"config_id": cfg["id"], "data": data})).unwrap();
        let (_, sub) = call(&app, "POST", "/api/v1/ml/$train", Some(train)).await;
        let job = format!("/api/v1/jobs/{}", sub["job_id"].as_str().unwrap());
        let (_, job) = call(&app, "GET", &job, None).await;
        assert_eq!(job["status"], "completed", "{job}");
        (app, sub["artifact_id"].as_str().unwrap().to_owned())
    });
    let (app, model_id) = app;

    // Splits as chap-core makes them: 12 periods each, stride 4, the last ending at the data's end.
    let columns = data["columns"].as_array().unwrap();
    let col = |name: &str| columns.iter().position(|c| c == name).unwrap();
    let (tp, target) = (col(cols::TIME_PERIOD), col(cols::DISEASE_CASES));
    let rows = data["data"].as_array().unwrap();
    let mut periods: Vec<&str> = rows.iter().map(|r| r[tp].as_str().unwrap()).collect();
    periods.sort_by_key(|p| Period::parse(p).unwrap());
    periods.dedup();
    let rank: HashMap<&str, usize> = periods.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let future_columns: Vec<&Value> = columns
        .iter()
        .filter(|c| *c != cols::DISEASE_CASES)
        .collect();
    let (mut explicit, mut shared) = (vec![], vec![]);
    for k in 0..7 {
        let end = periods.len() - 12 - 4 * (6 - k) - 1;
        let historic: Vec<&Value> = rows
            .iter()
            .filter(|r| rank[r[tp].as_str().unwrap()] <= end)
            .collect();
        let future: Vec<Value> = rows
            .iter()
            .filter(|r| (end + 1..=end + 12).contains(&rank[r[tp].as_str().unwrap()]))
            .map(|r| {
                let mut r = r.as_array().unwrap().clone();
                r.remove(target);
                Value::Array(r)
            })
            .collect();
        let future = json!({"columns": future_columns, "data": future});
        explicit
            .push(json!({"historic": {"columns": columns, "data": historic}, "future": future}));
        shared.push(json!({"historic_end": periods[end], "future": future}));
    }
    let per_split: Vec<Vec<u8>> = explicit
        .iter()
        .map(|s| {
            let mut body = s.clone();
            body["artifact_id"] = json!(model_id);
            serde_json::to_vec(&body).unwrap()
        })
        .collect();
    let one_call =
        serde_json::to_vec(&json!({"artifact_id": model_id, "splits": explicit})).unwrap();
    let one_shared =
        serde_json::to_vec(&json!({"artifact_id": model_id, "historic": data, "splits": shared}))
            .unwrap();
    eprintln!(
        "request bytes: 7 x $predict {}, $predict-splits {}, shared history {}",
        per_split.iter().map(Vec::len).sum::<usize>(),
        one_call.len(),
        one_shared.len()
    );

    let mut g = c.benchmark_group("backtest_forecasts_136x300_weekly_7x12");
    g.sample_size(20);
    g.bench_function("7x_predict_job", |b| {
        b.iter(|| {
            rt.block_on(async {
                for body in &per_split {
                    black_box(predict_one(&app, body.clone()).await);
                }
            });
        });
    });
    for (name, body) in [
        ("predict_splits", &one_call),
        ("predict_splits_shared", &one_shared),
    ] {
        g.bench_function(name, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let (s, v) = call(
                        &app,
                        "POST",
                        "/api/v1/ml/$predict-splits",
                        Some(body.clone()),
                    )
                    .await;
                    assert_eq!(s, StatusCode::OK);
                    black_box(v)
                })
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
