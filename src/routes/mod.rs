//! HTTP routes of the chapkit API.

pub mod artifacts;
pub mod configs;
pub mod datasets;
pub mod ml;

/// `page` and `size`, plus the chapkit-rs filter `q`.
#[derive(Debug, Clone, Default, serde::Deserialize, utoipa::IntoParams)]
pub(crate) struct NamedListParams {
    /// Page number, starting at 1.
    page: Option<i64>,
    /// Page size, 1-100.
    size: Option<i64>,
    /// Only those whose name holds this text, ignoring the case of ASCII letters. chapkit-rs
    /// extension.
    q: Option<String>,
}

impl NamedListParams {
    /// `(page, size)` as [`chapkit_http::PageParams::resolve`] answers it.
    pub(crate) fn paging(&self) -> Result<Option<(u64, u64)>, chapkit_http::Problem> {
        chapkit_http::PageParams {
            page: self.page,
            size: self.size,
        }
        .resolve()
    }

    /// The name filter: `q` trimmed, none when blank.
    pub(crate) fn name(&self) -> Option<&str> {
        search_text(self.q.as_deref())
    }
}

/// A search text trimmed, none when blank.
pub(crate) fn search_text(q: Option<&str>) -> Option<&str> {
    q.map(str::trim).filter(|q| !q.is_empty())
}
