//! Target masking, as chap-core's `target_mask`: inside a masked range `disease_cases` is hidden
//! (null) and the `target_masked` indicator is 1, elsewhere 0, so a model can tell a masked period
//! from a missing one. A window's future always has the indicator at 0, and no masked period may
//! fall in the evaluation window.

use chapkit_core::{Column, DataFrame, Error, Period, Result, cols};

use crate::split::Panel;

/// The mask indicator column.
pub const MASK_INDICATOR: &str = "target_masked";

/// An inclusive range of masked periods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskedRange {
    /// First masked period.
    pub start: Period,
    /// Last masked period.
    pub end: Period,
}

/// `data` with `disease_cases` nulled inside the ranges and the indicator added (as a float column,
/// after the others). Without ranges, `data` is returned unchanged, with no indicator.
pub fn mask_target(data: &DataFrame, ranges: &[MaskedRange]) -> Result<DataFrame> {
    if ranges.is_empty() {
        return Ok(data.clone());
    }
    if data.has(MASK_INDICATOR) {
        return Err(Error::invalid(format!(
            "{MASK_INDICATOR} is reserved: it is added by masking"
        )));
    }
    let tp = data.require(cols::TIME_PERIOD)?;
    let target = data.f64_column(cols::DISEASE_CASES)?;
    let mut masked = Vec::with_capacity(data.n_rows());
    let mut indicator = Vec::with_capacity(data.n_rows());
    for (row, value) in target.iter().enumerate() {
        let period = tp.str_at(row).and_then(Period::parse);
        let inside = period.is_some_and(|p| ranges.iter().any(|r| r.start <= p && p <= r.end));
        masked.push((!inside && !value.is_nan()).then_some(*value));
        indicator.push(if inside { 1.0 } else { 0.0 });
    }
    let mut out = data.clone();
    out.set_column(cols::DISEASE_CASES, Column::Float(masked))?;
    out.push_column(MASK_INDICATOR, Column::from_f64(indicator))?;
    Ok(out)
}

/// chap-core's `check_mask_outside_evaluation_window`: no masked period in the last `window`
/// periods, which hold every split's truth.
pub(crate) fn check_mask_outside_window(
    data: &DataFrame,
    panel: &Panel,
    window: usize,
) -> Result<()> {
    let indicator = data.f64_column(MASK_INDICATOR)?;
    let start = panel.periods.len().saturating_sub(window);
    let first = indicator
        .iter()
        .zip(&panel.row_period)
        .filter(|(v, p)| **v > 0.5 && **p >= start)
        .map(|(_, p)| *p)
        .min();
    match first {
        Some(p) => Err(Error::invalid(format!(
            "masked period {} falls in the evaluation window, which starts at {}; backtests cannot \
             evaluate on masked data",
            panel.labels.get(p).map_or("", String::as_str),
            panel.labels.get(start).map_or("", String::as_str),
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_inside_the_ranges_only() {
        let data = DataFrame::from_columns(vec![
            (
                "time_period",
                Column::from_strs(["2020-01", "2020-02", "2020-03"]),
            ),
            ("location", Column::from_strs(["A", "A", "A"])),
            (
                "disease_cases",
                Column::Int(vec![Some(1), Some(2), Some(3)]),
            ),
        ])
        .unwrap();
        let p = |s| Period::parse(s).unwrap();
        let out = mask_target(
            &data,
            &[MaskedRange {
                start: p("2020-02"),
                end: p("2020-02"),
            }],
        )
        .unwrap();
        assert_eq!(
            out.column("disease_cases"),
            Some(&Column::Float(vec![Some(1.0), None, Some(3.0)]))
        );
        assert_eq!(out.f64_column(MASK_INDICATOR).unwrap(), [0.0, 1.0, 0.0]);
        assert_eq!(mask_target(&data, &[]).unwrap(), data);
        assert!(mask_target(&out, &[]).is_ok());
        assert!(
            mask_target(
                &out,
                &[MaskedRange {
                    start: p("2020-01"),
                    end: p("2020-01")
                }]
            )
            .is_err()
        );
    }
}
