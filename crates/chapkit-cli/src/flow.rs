//! `chapkit-rs test`: run chap-core's flow against a running chapkit service (Rust or Python) and
//! check every response has the shape chap-core relies on.

use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// What to test against.
pub struct Options {
    pub url: String,
    pub locations: usize,
    pub periods: usize,
    pub timeout: u64,
    pub keep: bool,
}

struct Client {
    http: reqwest::Client,
    base: String,
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
        let res = req.send().await.map_err(|e| format!("{path}: {e}"))?;
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

    /// Poll a job until it is finished; fail unless it completed.
    async fn wait(&self, job_id: &str, timeout: Duration) -> Result<(), String> {
        let start = Instant::now();
        loop {
            let job = self.get(&format!("/api/v1/jobs/{job_id}")).await?;
            match job.get("status").and_then(Value::as_str) {
                Some("completed") => return Ok(()),
                Some(s @ ("failed" | "canceled")) => {
                    return Err(format!(
                        "job {job_id} {s}: {}",
                        job.get("error").unwrap_or(&Value::Null)
                    ));
                }
                _ if start.elapsed() > timeout => {
                    return Err(format!("job {job_id} did not finish in {timeout:?}"));
                }
                _ => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
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
        .ok_or_else(|| format!("{context}: missing '{key}' in {v}"))
}

fn str_field(v: &Value, key: &str, context: &str) -> Result<String, String> {
    field(v, key, context)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{context}: '{key}' is not a string"))
}

/// Run every step, printing one line each; stop at the first failure.
pub async fn run(o: Options) -> Result<(), String> {
    let client = Client {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(o.timeout))
            .build()
            .map_err(|e| e.to_string())?,
        base: o.url.trim_end_matches('/').to_owned(),
    };
    let timeout = Duration::from_secs(o.timeout);
    println!("Testing {}\n", client.base);
    let total = Instant::now();
    let mut config_id: Option<String> = None;
    let result = steps(&client, &o, timeout, &mut config_id).await;
    if !o.keep
        && let Some(id) = &config_id
    {
        let _ = client
            .call(
                reqwest::Method::DELETE,
                &format!("/api/v1/configs/{id}"),
                None,
            )
            .await;
    }
    match result {
        Ok(()) => {
            println!(
                "\nAll steps passed in {:.2} s.",
                total.elapsed().as_secs_f64()
            );
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Evaluate a step, print `ok` with its time or `FAIL`, and return early on failure.
macro_rules! step {
    ($label:expr, $body:expr) => {{
        let t = Instant::now();
        match $body {
            Ok(v) => {
                println!(
                    "  ok    {:<44} {:>8.1} ms",
                    $label,
                    t.elapsed().as_secs_f64() * 1000.0
                );
                v
            }
            Err(e) => {
                println!("  FAIL  {}", $label);
                return Err(e);
            }
        }
    }};
}

async fn steps(
    c: &Client,
    o: &Options,
    timeout: Duration,
    config_id: &mut Option<String>,
) -> Result<(), String> {
    step!(
        "health",
        async {
            let h = c.get("/health").await?;
            match h.get("status").and_then(Value::as_str) {
                Some("healthy") => Ok(()),
                _ => Err(format!("/health: not healthy: {h}")),
            }
        }
        .await
    );

    let info = step!("service info", c.get("/api/v1/info").await);
    let id = str_field(&info, "id", "/api/v1/info")?;
    let version = info
        .get("chapkit_version")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_owned();

    step!(
        "configuration schema",
        async {
            let s = c.get("/api/v1/configs/$schema").await?;
            field(&s, "properties", "/api/v1/configs/$schema").map(|_| ())
        }
        .await
    );

    let config = step!(
        "create a configuration",
        async {
            let name = format!("chapkit-rs-test-{}", std::process::id());
            c.post("/api/v1/configs", &json!({"name": name, "data": {}}), 201)
                .await
        }
        .await
    );
    *config_id = Some(str_field(&config, "id", "/api/v1/configs")?);
    let cid = config_id.clone().unwrap_or_default();

    let q = format!(
        "num_locations={}&num_periods={}&seed=1",
        o.locations, o.periods
    );
    let train = step!(
        "generate training data",
        c.get(&format!("/api/v1/ml/$generate-sample-data?{q}"))
            .await
    );
    let predict = step!(
        "generate prediction data",
        c.get(&format!(
            "/api/v1/ml/$generate-sample-data?kind=predict&{q}"
        ))
        .await
    );
    let train_data = field(&train, "data", "sample data")?;
    let future = field(&predict, "future", "prediction sample data")?;
    let future_rows = field(future, "data", "future")?
        .as_array()
        .map_or(0, Vec::len);

    let model_id = step!(
        "train (job)",
        async {
            let body =
                json!({"config_id": cid, "data": train_data, "run_info": {"prediction_length": 3}});
            let sub = c.post("/api/v1/ml/$train", &body, 202).await?;
            c.wait(&str_field(&sub, "job_id", "$train")?, timeout)
                .await?;
            str_field(&sub, "artifact_id", "$train")
        }
        .await
    );

    let prediction_id = step!(
        "predict (job)",
        async {
            let body = json!({
                "artifact_id": model_id,
                "historic": field(&predict, "historic", "prediction sample data")?,
                "future": future,
                "run_info": {"prediction_length": 3},
            });
            let sub = c.post("/api/v1/ml/$predict", &body, 202).await?;
            c.wait(&str_field(&sub, "job_id", "$predict")?, timeout)
                .await?;
            str_field(&sub, "artifact_id", "$predict")
        }
        .await
    );

    step!(
        "forecast has the shape chap-core expects",
        check_forecast(c, &prediction_id, future_rows).await
    );

    println!(
        "\n{id} (chapkit {version}): {} locations, {} periods.",
        o.locations, o.periods
    );
    Ok(())
}

/// The stored forecast has the keys chap-core reads, starts with `time_period` and `location`,
/// has sample columns, and one row per future row.
async fn check_forecast(c: &Client, prediction_id: &str, future_rows: usize) -> Result<(), String> {
    let art = c.get(&format!("/api/v1/artifacts/{prediction_id}")).await?;
    let data = field(&art, "data", "prediction artifact")?;
    for key in [
        "type",
        "metadata",
        "content",
        "content_type",
        "content_size",
    ] {
        field(data, key, "prediction artifact data")?;
    }
    let content = field(data, "content", "prediction artifact")?;
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
