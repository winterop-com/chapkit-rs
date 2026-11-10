//! A small Prometheus registry and `GET /metrics` (text format 0.0.4).
//!
//! Counters are process-global so any layer can record without plumbing state:
//! HTTP requests per route and status with durations, job outcomes, and counters
//! the ML layer adds (`ml_train_jobs_total`, `ml_predict_jobs_total`, as chapkit).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

type Key = (String, Vec<(String, String)>);

#[derive(Default)]
struct Registry {
    counters: BTreeMap<Key, f64>,
    help: BTreeMap<String, &'static str>,
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(Mutex::default)
}

fn start_time() -> f64 {
    static T: OnceLock<f64> = OnceLock::new();
    *T.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64())
    })
}

/// Add `value` to a counter.
pub fn add(name: &str, help: &'static str, labels: &[(&str, &str)], value: f64) {
    let key = (
        name.to_owned(),
        labels
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
    );
    let mut r = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *r.counters.entry(key).or_insert(0.0) += value;
    r.help.entry(name.to_owned()).or_insert(help);
}

/// Increment a counter by one.
pub fn inc(name: &str, help: &'static str, labels: &[(&str, &str)]) {
    add(name, help, labels, 1.0);
}

/// Record one HTTP request.
pub fn observe_http(method: &str, route: &str, status: u16, seconds: f64) {
    let status = status.to_string();
    let labels = [
        ("method", method),
        ("route", route),
        ("status", status.as_str()),
    ];
    inc(
        "http_server_requests_total",
        "HTTP requests handled.",
        &labels,
    );
    add(
        "http_server_request_duration_seconds_sum",
        "Total time spent handling HTTP requests.",
        &labels,
        seconds,
    );
}

fn escape(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// The registry in Prometheus text format.
pub fn render() -> String {
    let _ = start_time();
    let r = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut out = String::new();
    let mut last = "";
    for ((name, labels), value) in &r.counters {
        if name != last {
            let help = r.help.get(name).copied().unwrap_or("");
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} counter");
            last = name;
        }
        let labels: Vec<String> = labels
            .iter()
            .map(|(k, v)| format!("{k}=\"{}\"", escape(v)))
            .collect();
        let _ = writeln!(out, "{name}{{{}}} {value}", labels.join(","));
    }
    let _ = writeln!(
        out,
        "# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.\n\
         # TYPE process_start_time_seconds gauge\nprocess_start_time_seconds {}",
        start_time()
    );
    let _ = writeln!(
        out,
        "# HELP chapkit_rs_info chapkit-rs build information.\n# TYPE chapkit_rs_info gauge\nchapkit_rs_info{{version=\"{}\"}} 1",
        env!("CARGO_PKG_VERSION")
    );
    out
}

/// Prometheus metrics.
#[utoipa::path(get, path = "/metrics", tag = "Observability",
    responses((status = OK, content_type = "text/plain; version=0.0.4; charset=utf-8", body = String)))]
pub async fn handler() -> impl axum::response::IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        render(),
    )
}

/// The metrics router (mounted at the root).
pub fn router() -> utoipa_axum::router::OpenApiRouter {
    utoipa_axum::router::OpenApiRouter::new().routes(utoipa_axum::routes!(handler))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_counters() {
        inc("test_total", "A test counter.", &[("a", "x\"y")]);
        inc("test_total", "A test counter.", &[("a", "x\"y")]);
        let text = render();
        assert!(text.contains("# TYPE test_total counter"));
        assert!(text.contains("test_total{a=\"x\\\"y\"} 2"));
        assert!(text.contains("process_start_time_seconds"));
    }
}
