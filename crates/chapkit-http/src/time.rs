//! Timestamp formats that match Python chapkit's JSON output.

use chrono::Utc;

/// Entity timestamp: naive UTC, second resolution (`2026-11-09T20:24:16`),
/// as SQLite `CURRENT_TIMESTAMP` gives through SQLAlchemy and pydantic.
pub fn entity_now() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}

/// Job timestamp: UTC with microseconds and `Z` (`2026-11-09T20:24:16.490430Z`).
pub fn job_now() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string()
}

/// RFC 3339 UTC timestamp with seconds precision.
pub fn rfc3339_now() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}
