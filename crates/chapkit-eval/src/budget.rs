//! Memory estimates for the frames preparation and splitting build, checked before they are built.
//!
//! Sizes are of the in-memory column storage, by type: a float or integer cell is an
//! `Option<f64>`/`Option<i64>` (16 bytes), a text cell a `u32` dictionary code (4), a boolean 2 and
//! a JSON value 32 (its content more). Arithmetic is in `u64` and saturates.

use chapkit_core::{Column, DataFrame, Error, Result};

/// Bytes one row of `col` takes.
pub(crate) fn cell_bytes(col: &Column) -> u64 {
    match col {
        Column::Float(_) | Column::Int(_) => 16,
        Column::Str(_) => 4,
        Column::Bool(_) => 2,
        Column::Json(_) => 32,
    }
}

/// Bytes one row of `data` takes, every column.
pub(crate) fn row_bytes(data: &DataFrame) -> u64 {
    data.iter()
        .map(|(_, c)| cell_bytes(c))
        .fold(0, u64::saturating_add)
}

/// `rows` times `bytes`, saturating.
pub(crate) fn times(rows: usize, bytes: u64) -> u64 {
    u64::try_from(rows)
        .unwrap_or(u64::MAX)
        .saturating_mul(bytes)
}

/// The size of a type, as `u64`.
pub(crate) fn size<T>() -> u64 {
    u64::try_from(std::mem::size_of::<T>()).unwrap_or(u64::MAX)
}

/// Refuse `bytes` over `max`, saying what would have been built.
pub(crate) fn check(what: &str, bytes: u64, max: u64, hint: &str) -> Result<()> {
    if bytes > max {
        let gb = |b: u64| b as f64 / 1e9;
        return Err(Error::invalid(format!(
            "{what} would take about {:.1} GB, more than the limit of {:.1} GB; {hint}",
            gb(bytes),
            gb(max)
        )));
    }
    Ok(())
}
