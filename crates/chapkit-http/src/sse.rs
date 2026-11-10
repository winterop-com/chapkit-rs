//! Server-sent event streams, as servicekit's: `data: <json>\n\n` events with
//! `Cache-Control: no-cache`, `Connection: keep-alive` and `X-Accel-Buffering: no`.

use std::convert::Infallible;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use futures_util::Stream;
use serde::Deserialize;
use serde_json::json;
use utoipa::IntoParams;

use crate::problem::Problem;

/// `poll_interval` query parameter, in seconds (`0 < x <= 60`).
#[derive(Debug, Deserialize, IntoParams)]
pub struct PollParams {
    /// Seconds between events.
    pub poll_interval: Option<f64>,
}

impl PollParams {
    /// The interval, or `default`; a 422 problem when out of range.
    pub fn interval(&self, default: f64) -> Result<Duration, Problem> {
        let v = self.poll_interval.unwrap_or(default);
        if v > 0.0 && v <= 60.0 {
            Ok(Duration::from_secs_f64(v))
        } else {
            Err(Problem::validation(vec![json!({
                "loc": ["query", "poll_interval"],
                "msg": "Input should be greater than 0 and less than or equal to 60",
                "type": "value_error"
            })]))
        }
    }
}

/// One SSE event carrying `data`.
pub fn event(data: &impl serde::Serialize) -> Bytes {
    let json = serde_json::to_string(data).unwrap_or_else(|_| "null".into());
    Bytes::from(format!("data: {json}\n\n"))
}

/// An SSE response from a stream of encoded events.
pub fn response(
    stream: impl Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
) -> Response {
    let mut res = Body::from_stream(stream).into_response();
    let h = res.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    h.insert("x-accel-buffering", HeaderValue::from_static("no"));
    res
}
