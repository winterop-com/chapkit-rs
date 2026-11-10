//! Findings a model reports on a request it has not run yet, for `$validate`.

use serde::{Deserialize, Serialize};

/// How serious a finding is. Any `Error` makes the request invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// The request would fail or give a wrong forecast.
    Error,
    /// The request runs, but the forecast may be poor.
    Warning,
    /// Worth knowing.
    Info,
}

impl Severity {
    /// `error`, `warning` or `info`.
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        }
    }
}

/// One finding: a stable machine-readable `code` (for example `insufficient_history`), a message for
/// people, and optionally the input it is about (for example `config.lags` or `historic`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub field: Option<String>,
}

impl Diagnostic {
    fn new(severity: Severity, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity,
            code: code.into(),
            message: message.into(),
            field: None,
        }
    }

    /// An error: the request is invalid.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Severity::Error, code, message)
    }

    /// A warning.
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, code, message)
    }

    /// Information.
    pub fn info(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Severity::Info, code, message)
    }

    /// Name the input this finding is about, for example `config.lags` or `historic`.
    #[must_use]
    pub fn at(mut self, field: impl Into<String>) -> Self {
        self.field = Some(field.into());
        self
    }
}
