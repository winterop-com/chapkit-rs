//! chap-core's flow as a test: configurations from the service's schema, `$validate`, training
//! and forecasting jobs, and the shape of every stored forecast, with test data generated here and
//! shaped by what the service declares, as Python chapkit's `chapkit test` does.
//! Used by `chapkit-rs test` and by every model binary's `test` subcommand.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chapkit_core::synth::{Generator, Resolution, Spec};
use serde_json::{Map, Value, json};

/// Period format of the generated data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PeriodType {
    Monthly,
    Weekly,
}

/// Test options shared by `chapkit-rs test` and a model binary's `test`.
#[derive(Debug, Clone, PartialEq, Eq, clap::Args)]
pub struct Options {
    /// Configurations to create
    #[arg(long, default_value_t = 1)]
    pub configs: usize,
    /// Trainings per configuration
    #[arg(long, default_value_t = 1)]
    pub trainings: usize,
    /// Forecasts per trained model
    #[arg(long, default_value_t = 1)]
    pub predictions: usize,
    /// Locations in the test data
    #[arg(long, default_value_t = 5)]
    pub locations: usize,
    /// Training periods
    #[arg(long, default_value_t = 50)]
    pub periods: usize,
    /// History periods sent with a forecast
    #[arg(long, default_value_t = 30)]
    pub history: usize,
    /// Period format [default: the service's]
    #[arg(long, value_enum)]
    pub period_type: Option<PeriodType>,
    /// Seed for the test data [default: random]
    #[arg(long)]
    pub seed: Option<u64>,
    /// Seconds to wait per job
    #[arg(long, default_value_t = 300)]
    pub timeout: u64,
    /// Save the generated data in DIR
    #[arg(long, value_name = "DIR")]
    pub save_data: Option<PathBuf>,
    /// Show service details and diagnostics
    #[arg(long, short)]
    pub verbose: bool,
    /// Keep what the test creates
    #[arg(long)]
    pub keep: bool,
}

/// What the service declares in `/api/v1/info`.
struct ServiceInfo {
    id: String,
    chapkit_version: String,
    required_covariates: Vec<String>,
    requires_geo: bool,
    allow_free_covariates: bool,
    min_periods: u32,
    max_periods: u32,
    weekly: bool,
}

impl ServiceInfo {
    fn from_json(v: &Value) -> Self {
        let text = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("?").to_owned();
        let flag = |k: &str| v.get(k).and_then(Value::as_bool).unwrap_or(false);
        let number = |k: &str, d: u32| {
            v.get(k)
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .unwrap_or(d)
        };
        Self {
            id: text("id"),
            chapkit_version: text("chapkit_version"),
            required_covariates: v
                .get("required_covariates")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            requires_geo: flag("requires_geo"),
            allow_free_covariates: flag("allow_free_additional_continuous_covariates"),
            min_periods: number("min_prediction_periods", 0),
            max_periods: number("max_prediction_periods", 100),
            weekly: v.get("period_type").and_then(Value::as_str) == Some("weekly"),
        }
    }
}

/// A configuration created for the test.
struct TestConfig {
    id: String,
    horizon: u32,
    covariates: Vec<String>,
}

#[derive(Default)]
struct Stats {
    configs: usize,
    trained: usize,
    training_failed: usize,
    predicted: usize,
    prediction_failed: usize,
    validations: usize,
    validations_skipped: usize,
}

struct Client {
    http: reqwest::Client,
    base: String,
    timeout: Duration,
}

impl Client {
    async fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<(u16, Value), String> {
        let mut req = self.http.request(method, format!("{}{path}", self.base));
        if let Some(b) = body {
            req = req.json(b);
        }
        let res = req.send().await.map_err(|e| {
            if e.is_connect() {
                format!("no service at {}", self.base)
            } else {
                format!("{path}: {e}")
            }
        })?;
        let status = res.status().as_u16();
        let text = res.text().await.map_err(|e| format!("{path}: {e}"))?;
        let value = if text.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&text).unwrap_or(Value::String(text))
        };
        Ok((status, value))
    }

    async fn get(&self, path: &str) -> Result<Value, String> {
        expect(
            self.call(reqwest::Method::GET, path, None).await?,
            200,
            path,
        )
    }

    async fn post(&self, path: &str, body: &Value, want: u16) -> Result<Value, String> {
        expect(
            self.call(reqwest::Method::POST, path, Some(body)).await?,
            want,
            path,
        )
    }

    /// Poll a job until it is finished, backing off from 50 ms to 1 s; fail unless it completed.
    async fn wait(&self, job_id: &str) -> Result<(), String> {
        let start = Instant::now();
        let mut pause = Duration::from_millis(50);
        loop {
            let job = self.get(&format!("/api/v1/jobs/{job_id}")).await?;
            match job.get("status").and_then(Value::as_str) {
                Some("completed") => return Ok(()),
                Some(s @ ("failed" | "canceled")) => {
                    let error = job
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("no error message");
                    return Err(format!("job {s}: {error}"));
                }
                _ if start.elapsed() > self.timeout => {
                    return Err(format!(
                        "job did not finish in {} s",
                        self.timeout.as_secs()
                    ));
                }
                _ => {
                    tokio::time::sleep(pause).await;
                    pause = (pause * 2).min(Duration::from_secs(1));
                }
            }
        }
    }

    /// `$validate` the payload. `Ok(None)` when the service has no `$validate` (older services);
    /// `Err` with the error diagnostics when it is not valid.
    async fn validate(&self, kind: &str, payload: &Value) -> Result<Option<Vec<Value>>, String> {
        let mut body = payload.clone();
        if let Some(map) = body.as_object_mut() {
            map.insert("type".into(), json!(kind));
        }
        let (status, res) = self
            .call(reqwest::Method::POST, "/api/v1/ml/$validate", Some(&body))
            .await?;
        if status == 404 {
            return Ok(None);
        }
        let res = expect((status, res), 200, "$validate")?;
        let diagnostics = res
            .get("diagnostics")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if res.get("valid").and_then(Value::as_bool) == Some(false) {
            let errors: Vec<String> = diagnostics
                .iter()
                .filter(|d| d.get("severity").and_then(Value::as_str) == Some("error"))
                .map(|d| {
                    let code = d.get("code").and_then(Value::as_str).unwrap_or("?");
                    let message = d.get("message").and_then(Value::as_str).unwrap_or("");
                    format!("[{code}] {message}")
                })
                .collect();
            return Err(format!("$validate: {}", errors.join("; ")));
        }
        Ok(Some(diagnostics))
    }
}

fn expect((status, body): (u16, Value), want: u16, path: &str) -> Result<Value, String> {
    if status == want {
        Ok(body)
    } else {
        Err(format!("{path}: HTTP {status}, expected {want}: {body}"))
    }
}

fn field<'a>(v: &'a Value, key: &str, context: &str) -> Result<&'a Value, String> {
    v.get(key)
        .ok_or_else(|| format!("{context}: missing '{key}'"))
}

fn str_field(v: &Value, key: &str, context: &str) -> Result<String, String> {
    field(v, key, context)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{context}: '{key}' is not a string"))
}

/// Configuration data from the service's config schema, as Python chapkit's test generates it:
/// each field's default (numbers shifted by `variation`), else a value of the field's type.
pub fn config_from_schema(schema: &Value, variation: usize) -> Map<String, Value> {
    let properties = schema.get("properties").and_then(Value::as_object);
    properties
        .into_iter()
        .flatten()
        .filter(|(name, _)| name.as_str() != "id")
        .map(|(name, field)| (name.clone(), value_for(field, variation)))
        .collect()
}

fn value_for(field: &Value, variation: usize) -> Value {
    let v = u32::try_from(variation).unwrap_or(u32::MAX);
    if let Some(default) = field.get("default") {
        return match default {
            Value::Number(n) if n.is_i64() => json!(n.as_i64().unwrap_or(0) + i64::from(v)),
            Value::Number(n) => json!(n.as_f64().unwrap_or(0.0) + f64::from(v) * 0.1),
            other => other.clone(),
        };
    }
    match field
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("string")
    {
        "integer" => json!(v.max(1)),
        "number" => json!(f64::from(v) * 0.1),
        "boolean" => json!(variation.is_multiple_of(2)),
        "array" => json!([]),
        "object" => json!({}),
        "string" => match field.get("enum").and_then(Value::as_array) {
            Some(values) if !values.is_empty() => values
                .get(variation % values.len())
                .cloned()
                .unwrap_or(Value::Null),
            _ => json!(format!("test_value_{variation}")),
        },
        _ => Value::Null,
    }
}

/// Print a step's outcome; `Some` on success.
fn report<T>(label: &str, start: Instant, result: Result<T, String>) -> Option<T> {
    match result {
        Ok(v) => {
            println!(
                "  ok    {label:<44} {:>8.1} ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
            Some(v)
        }
        Err(e) => {
            println!("  FAIL  {label}\n        {e}");
            None
        }
    }
}

/// Like [`report`], but a failure ends the test: used for the steps everything else needs.
fn required<T>(label: &str, start: Instant, result: Result<T, String>) -> Result<T, String> {
    report(label, start, result).ok_or_else(|| format!("{label} failed"))
}

fn save(dir: Option<&Path>, name: &str, value: &Value) -> Result<(), String> {
    let Some(dir) = dir else { return Ok(()) };
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{name}.json"));
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))
}

fn frame_json(df: &chapkit_core::DataFrame) -> Result<Value, String> {
    serde_json::to_value(df).map_err(|e| e.to_string())
}

/// Whether a service answers on `url` (any HTTP response counts).
pub async fn reachable(url: &str) -> bool {
    reqwest::Client::new()
        .get(format!("{}/health", url.trim_end_matches('/')))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .is_ok()
}

/// Run the test against the service at `url`; `Err` when it is not reachable, a required step
/// failed or any job failed. Unless `keep`, the configurations it created (and with them its
/// models and forecasts) are deleted.
pub async fn run(url: &str, o: &Options) -> Result<(), String> {
    println!("Testing {url}\n");
    let client = Client {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(o.timeout))
            .build()
            .map_err(|e| e.to_string())?,
        base: url.trim_end_matches('/').to_owned(),
        timeout: Duration::from_secs(o.timeout),
    };
    let seed = o.seed.unwrap_or_else(random_seed);
    if o.seed.is_none() {
        println!("Random seed {seed} (repeat this run with --seed {seed})\n");
    }
    let started = Instant::now();
    let mut stats = Stats::default();
    let mut created: Vec<String> = Vec::new();
    let outcome = test(&client, o, seed, &mut stats, &mut created).await;
    if !o.keep {
        for id in &created {
            let _ = client
                .call(
                    reqwest::Method::DELETE,
                    &format!("/api/v1/configs/{id}"),
                    None,
                )
                .await;
        }
    }
    outcome?;
    print_summary(&stats, started);
    let failures = stats.training_failed + stats.prediction_failed;
    if failures > 0 {
        Err(format!("{failures} job(s) failed"))
    } else if stats.trained == 0 {
        Err("nothing was trained".into())
    } else {
        Ok(())
    }
}

/// Health, service info and configuration schema: the steps everything else needs.
async fn discover(c: &Client, o: &Options) -> Result<(ServiceInfo, Value), String> {
    let t = Instant::now();
    required(
        "health",
        t,
        async {
            let h = c.get("/health").await?;
            match h.get("status").and_then(Value::as_str) {
                Some("healthy") => Ok(()),
                _ => Err(format!("not healthy: {h}")),
            }
        }
        .await,
    )?;
    let t = Instant::now();
    let info = ServiceInfo::from_json(&required("service info", t, c.get("/api/v1/info").await)?);
    if o.verbose {
        println!(
            "        {} (chapkit {}): required covariates {:?}, geo {}, free covariates {}, horizon {}-{}",
            info.id,
            info.chapkit_version,
            info.required_covariates,
            info.requires_geo,
            info.allow_free_covariates,
            info.min_periods,
            info.max_periods
        );
    }
    let t = Instant::now();
    let schema = required(
        "configuration schema",
        t,
        c.get("/api/v1/configs/$schema").await,
    )?;
    Ok((info, schema))
}

async fn test(
    c: &Client,
    o: &Options,
    seed: u64,
    stats: &mut Stats,
    created: &mut Vec<String>,
) -> Result<(), String> {
    let (info, schema) = discover(c, o).await?;
    let weekly = o
        .period_type
        .map_or(info.weekly, |p| matches!(p, PeriodType::Weekly));
    let resolution = if weekly {
        Resolution::Weekly
    } else {
        Resolution::Monthly
    };
    let geo = info
        .requires_geo
        .then(|| Generator::geo(o.locations, false));
    if let Some(g) = &geo {
        save(o.save_data.as_deref(), "geo", g)?;
    }

    let mut configs = Vec::new();
    for i in 0..o.configs {
        let t = Instant::now();
        let label = format!("configuration {}", i + 1);
        let config = required(&label, t, create_config(c, o, &schema, &info, i).await)?;
        created.push(config.id.clone());
        stats.configs += 1;
        configs.push(config);
    }

    let mut job = 0u64;
    for (ci, config) in configs.iter().enumerate() {
        for ti in 0..o.trainings {
            job += 1;
            let spec = spec(o, &info, config, resolution, o.periods);
            let label = format!("configuration {}: train {}", ci + 1, ti + 1);
            let t = Instant::now();
            let trained = train(
                c,
                o,
                seed + job,
                &spec,
                config,
                geo.as_ref(),
                &format!("training_{ci}_{ti}"),
                stats,
            )
            .await;
            let Some(model) = report(&label, t, trained) else {
                stats.training_failed += 1;
                continue;
            };
            stats.trained += 1;
            for pi in 0..o.predictions {
                job += 1;
                let spec = spec_for_forecast(o, &info, config, resolution);
                let label = format!(
                    "configuration {}: train {}: forecast {}",
                    ci + 1,
                    ti + 1,
                    pi + 1
                );
                let t = Instant::now();
                let name = format!("prediction_{ci}_{ti}_{pi}");
                let forecast = predict(
                    c,
                    o,
                    seed + job,
                    &spec,
                    config,
                    &model,
                    geo.as_ref(),
                    &name,
                    stats,
                )
                .await;
                if report(&label, t, forecast).is_some() {
                    stats.predicted += 1;
                } else {
                    stats.prediction_failed += 1;
                }
            }
        }
    }
    Ok(())
}

async fn create_config(
    c: &Client,
    o: &Options,
    schema: &Value,
    info: &ServiceInfo,
    i: usize,
) -> Result<TestConfig, String> {
    let data = config_from_schema(schema, i);
    save(
        o.save_data.as_deref(),
        &format!("config_{i}"),
        &Value::Object(data.clone()),
    )?;
    // The horizon the configuration asks for, kept within what the service declares.
    let requested = data
        .get("prediction_periods")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(3);
    let lower = info.min_periods.max(1);
    let horizon = requested.clamp(lower, info.max_periods.max(lower));
    let covariates = data
        .get("additional_continuous_covariates")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let name = format!("chapkit-rs-test-{}-{i}", std::process::id());
    let res = c
        .post("/api/v1/configs", &json!({"name": name, "data": data}), 201)
        .await?;
    Ok(TestConfig {
        id: str_field(&res, "id", "/api/v1/configs")?,
        horizon,
        covariates,
    })
}

fn spec(
    o: &Options,
    info: &ServiceInfo,
    config: &TestConfig,
    resolution: Resolution,
    periods: usize,
) -> Spec {
    Spec {
        num_locations: o.locations,
        num_periods: periods,
        // Two extra covariates when the service accepts free ones, as Python chapkit's test does.
        num_features: if info.allow_free_covariates { 2 } else { 0 },
        required_covariates: info.required_covariates.clone(),
        additional_covariates: config.covariates.clone(),
        resolution,
        ..Spec::default()
    }
}

fn spec_for_forecast(
    o: &Options,
    info: &ServiceInfo,
    config: &TestConfig,
    resolution: Resolution,
) -> Spec {
    spec(o, info, config, resolution, config.horizon as usize)
}

fn run_info(config: &TestConfig) -> Value {
    json!({"prediction_periods": config.horizon, "additional_continuous_covariates": config.covariates})
}

/// Validate, then train; the trained model's artifact id.
#[allow(clippy::too_many_arguments)]
async fn train(
    c: &Client,
    o: &Options,
    seed: u64,
    spec: &Spec,
    config: &TestConfig,
    geo: Option<&Value>,
    name: &str,
    stats: &mut Stats,
) -> Result<String, String> {
    let data = frame_json(
        &Generator::new(Some(seed))
            .training_data(spec)
            .map_err(|e| e.to_string())?,
    )?;
    save(o.save_data.as_deref(), name, &data)?;
    let mut payload = json!({"config_id": config.id, "data": data, "run_info": run_info(config)});
    if let (Some(g), Some(map)) = (geo, payload.as_object_mut()) {
        map.insert("geo".into(), g.clone());
    }
    validate(c, o, "train", &payload, stats).await?;
    let sub = c.post("/api/v1/ml/$train", &payload, 202).await?;
    c.wait(&str_field(&sub, "job_id", "$train")?).await?;
    let model = str_field(&sub, "artifact_id", "$train")?;
    c.get(&format!("/api/v1/artifacts/{model}")).await?;
    Ok(model)
}

/// Validate, forecast, and check the stored forecast.
#[allow(clippy::too_many_arguments)]
async fn predict(
    c: &Client,
    o: &Options,
    seed: u64,
    spec: &Spec,
    config: &TestConfig,
    model: &str,
    geo: Option<&Value>,
    name: &str,
    stats: &mut Stats,
) -> Result<(), String> {
    let (historic, future) = Generator::new(Some(seed))
        .prediction_data(spec, o.history)
        .map_err(|e| e.to_string())?;
    let (historic, future) = (frame_json(&historic)?, frame_json(&future)?);
    save(
        o.save_data.as_deref(),
        &format!("{name}_historic"),
        &historic,
    )?;
    save(o.save_data.as_deref(), &format!("{name}_future"), &future)?;
    let future_rows = future
        .get("data")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let mut payload = json!({"artifact_id": model, "historic": historic, "future": future, "run_info": run_info(config)});
    if let (Some(g), Some(map)) = (geo, payload.as_object_mut()) {
        map.insert("geo".into(), g.clone());
    }
    validate(c, o, "predict", &payload, stats).await?;
    let sub = c.post("/api/v1/ml/$predict", &payload, 202).await?;
    c.wait(&str_field(&sub, "job_id", "$predict")?).await?;
    check_forecast(c, &str_field(&sub, "artifact_id", "$predict")?, future_rows).await
}

async fn validate(
    c: &Client,
    o: &Options,
    kind: &str,
    payload: &Value,
    stats: &mut Stats,
) -> Result<(), String> {
    match c.validate(kind, payload).await? {
        None => stats.validations_skipped += 1,
        Some(diagnostics) => {
            stats.validations += 1;
            if o.verbose {
                for d in &diagnostics {
                    let get = |k: &str| d.get(k).and_then(Value::as_str).unwrap_or("?").to_owned();
                    println!(
                        "        {kind} {}: [{}] {}",
                        get("severity"),
                        get("code"),
                        get("message")
                    );
                }
            }
        }
    }
    Ok(())
}

/// The stored forecast has the keys chap-core reads, starts with `time_period` and `location`,
/// has sample columns, and one row per future row.
async fn check_forecast(c: &Client, prediction_id: &str, future_rows: usize) -> Result<(), String> {
    let art = c.get(&format!("/api/v1/artifacts/{prediction_id}")).await?;
    let data = field(&art, "data", "forecast artifact")?;
    for key in [
        "type",
        "metadata",
        "content",
        "content_type",
        "content_size",
    ] {
        field(data, key, "forecast artifact data")?;
    }
    let content = field(data, "content", "forecast artifact")?;
    let columns: Vec<&str> = field(content, "columns", "forecast")?
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if columns.first() != Some(&"time_period") || columns.get(1) != Some(&"location") {
        return Err(format!(
            "forecast columns start with {columns:?}, expected time_period, location"
        ));
    }
    let samples = columns.iter().filter(|c| c.starts_with("sample_")).count();
    let rows = field(content, "data", "forecast")?
        .as_array()
        .map_or(0, Vec::len);
    if samples == 0 || rows != future_rows {
        return Err(format!(
            "forecast has {rows} rows and {samples} samples; expected {future_rows} rows"
        ));
    }
    Ok(())
}

fn print_summary(s: &Stats, started: Instant) {
    println!();
    println!("  configurations   {}", s.configs);
    println!(
        "  trainings        {} ok, {} failed",
        s.trained, s.training_failed
    );
    println!(
        "  forecasts        {} ok, {} failed",
        s.predicted, s.prediction_failed
    );
    if s.validations_skipped > 0 {
        println!(
            "  validations      {} ok, {} skipped (no $validate)",
            s.validations, s.validations_skipped
        );
    } else {
        println!("  validations      {} ok", s.validations);
    }
    println!(
        "  time             {:.2} s",
        started.elapsed().as_secs_f64()
    );
}

fn random_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    u64::from(nanos % 2_147_483_647).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_follows_the_schema_like_python_chapkit() {
        let schema = json!({"properties": {
            "id": {"type": "string"},
            "prediction_periods": {"type": "integer", "default": 3},
            "scale": {"type": "number", "default": 1.5},
            "flag": {"type": "boolean", "default": true},
            "additional_continuous_covariates": {"type": "array", "default": ["rainfall"]},
            "method": {"type": "string", "enum": ["a", "b"]},
            "lags": {"type": "integer"},
            "label": {"type": "string"}
        }});
        let c0 = config_from_schema(&schema, 0);
        assert!(!c0.contains_key("id"));
        assert_eq!(c0["prediction_periods"], 3);
        assert_eq!(c0["flag"], true);
        assert_eq!(c0["additional_continuous_covariates"], json!(["rainfall"]));
        assert_eq!(c0["method"], "a");
        assert_eq!(c0["lags"], 1);
        assert_eq!(c0["label"], "test_value_0");
        let c1 = config_from_schema(&schema, 1);
        assert_eq!(c1["prediction_periods"], 4);
        assert_eq!(c1["scale"], 1.6);
        assert_eq!(c1["method"], "b");
    }
}
