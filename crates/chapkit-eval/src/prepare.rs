//! Dataset preparation, as chap-core loads every dataset for a backtest
//! (`DataSet.from_pandas(df, fill_missing=True)`).
//!
//! chap-core groups the rows by location (in sorted order) and sorts each location by period;
//! a period twice in one location is an error. It then fills the gaps inside each location's series
//! with missing values in every column, and pads every location with missing rows to the dataset's
//! first and last period, so all locations share one contiguous period range. Filled numeric
//! columns become floats; a text column that needs filling is an error, as in chap-core.

use std::collections::BTreeMap;

use chapkit_core::{Column, DataFrame, Error, Period, Result, cols};

use crate::budget;

/// What [`prepare`] does beyond grouping, sorting and rejecting duplicates. The default is
/// chap-core's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prepare {
    /// Fill missing periods inside each location's series with missing rows. With this on, the
    /// dataset's periods are the contiguous range from its first to its last period.
    pub fill_gaps: bool,
    /// Pad every location with missing rows to the dataset's first and last period.
    pub pad: bool,
}

impl Default for Prepare {
    fn default() -> Self {
        Self {
            fill_gaps: true,
            pad: true,
        }
    }
}

/// The most rows [`prepare`] makes, after filling and padding: chap-core's preparation of a
/// larger dataset is refused before anything is allocated. 100,000 locations of ten years of
/// weekly data fit. [`MAX_PREPARED_BYTES`] bounds their width.
pub const MAX_PREPARED_ROWS: usize = 60_000_000;

/// The most memory [`prepare`]'s result and its bookkeeping take, estimated from the rows it makes
/// times the bytes per row (every column by type, filled numeric columns as floats, plus the row's
/// period, source row and location while building): a larger dataset is refused before anything is
/// allocated. 100,000 locations x 120 months x 6 numeric columns take about 1.9 GB, and ten years of
/// weekly data about 8 GB, so 16 GiB leaves room while refusing a small request that would expand
/// into tens of gigabytes.
pub const MAX_PREPARED_BYTES: u64 = 16 << 30;

/// The number of periods from `first` to `last` (inclusive, 0 when `last` is earlier). Fails for
/// a monthly and a weekly period, and for Sunday-start weeks.
pub(crate) fn span(first: Period, last: Period) -> Result<usize> {
    if std::mem::discriminant(&first) != std::mem::discriminant(&last) {
        return Err(mixed(first, last));
    }
    if last <= first {
        return Ok(usize::from(first == last));
    }
    let (Some(a), Some(b)) = (first.ordinal(), last.ordinal()) else {
        return Err(Error::invalid(format!(
            "cannot fill or pad Sunday-start weeks ({first}); keep the gaps and do not pad"
        )));
    };
    usize::try_from(b - a + 1).map_err(|_| Error::invalid("period range out of bounds"))
}

fn mixed(a: Period, b: Period) -> Error {
    Error::invalid(format!(
        "data mixes monthly and weekly periods ({a} and {b})"
    ))
}

/// Refuse `rows` prepared rows over [`MAX_PREPARED_ROWS`].
fn check_rows(rows: usize) -> Result<()> {
    if rows > MAX_PREPARED_ROWS {
        return Err(Error::invalid(format!(
            "the prepared dataset would have {rows} rows (periods filled and padded per location), \
             more than the limit of {MAX_PREPARED_ROWS}"
        )));
    }
    Ok(())
}

/// Bytes one prepared row of `data` takes: its columns (numeric ones as floats, as filling makes
/// them) and the period, source row and location kept per row while building.
fn prepared_row_bytes(data: &DataFrame) -> u64 {
    let columns = data
        .iter()
        .map(|(_, c)| match c {
            Column::Bool(_) => 16,
            c => budget::cell_bytes(c),
        })
        .fold(0, u64::saturating_add);
    columns
        .saturating_add(budget::size::<Period>())
        .saturating_add(budget::size::<Option<usize>>())
        .saturating_add(budget::size::<&str>())
}

/// The periods from `first` to `last`, contiguous. Fails for Sunday-start weeks, for a monthly
/// and a weekly period, and for more than [`MAX_PREPARED_ROWS`] periods.
pub(crate) fn contiguous(first: Period, last: Period) -> Result<Vec<Period>> {
    let n = span(first, last)?;
    check_rows(n)?;
    let mut out = Vec::with_capacity(n);
    out.push(first);
    let mut p = first;
    while p < last {
        p = p.succ().ok_or_else(|| {
            Error::invalid(format!(
                "cannot fill or pad Sunday-start weeks ({p}); keep the gaps and do not pad"
            ))
        })?;
        out.push(p);
    }
    Ok(out)
}

/// A row's location as text: non-text locations (numbers) become text, as in chap-core.
pub(crate) fn location_key(loc: &Column, row: usize) -> String {
    loc.str_at(row)
        .map_or_else(|| loc.json_at(row).to_string(), str::to_owned)
}

/// Row index per (location, period): locations sorted, each location's rows by period.
type Index = BTreeMap<String, BTreeMap<Period, usize>>;

fn index(data: &DataFrame) -> Result<Index> {
    let (tp, loc) = (
        data.require(cols::TIME_PERIOD)?,
        data.require(cols::LOCATION)?,
    );
    let mut out: Index = BTreeMap::new();
    let mut seen: Option<Period> = None;
    for row in 0..data.n_rows() {
        let label = tp.str_at(row).unwrap_or_default();
        let period = Period::parse(label).ok_or_else(|| {
            Error::invalid(format!(
                "time_period {label:?} in row {row} is not a month or week"
            ))
        })?;
        match seen {
            Some(s) if std::mem::discriminant(&s) != std::mem::discriminant(&period) => {
                return Err(mixed(s, period));
            }
            _ => seen = Some(period),
        }
        let location = location_key(loc, row);
        let rows = out.entry(location.clone()).or_default();
        if rows.insert(period, row).is_some() {
            return Err(Error::invalid(format!(
                "location {location} has more than one row for {period}"
            )));
        }
    }
    Ok(out)
}

/// A column over `rows` (`None` = a filled row): taken as is when nothing is filled, else as floats.
fn fill(name: &str, col: &Column, rows: &[Option<usize>]) -> Result<Column> {
    if let Some(idx) = rows.iter().copied().collect::<Option<Vec<usize>>>() {
        return col
            .take(&idx)
            .ok_or_else(|| Error::invalid("row out of range"));
    }
    if !matches!(col, Column::Float(_) | Column::Int(_) | Column::Bool(_)) {
        return Err(Error::invalid(format!(
            "column '{name}' is not numeric and cannot be filled for missing periods (as in chap-core); \
             keep the gaps and do not pad, or drop the column"
        )));
    }
    Ok(Column::Float(
        rows.iter().map(|r| r.and_then(|r| col.f64_at(r))).collect(),
    ))
}

/// Prepare `data` for a backtest as chap-core does, with `options` (see [`Prepare`]). Rows come out
/// by location (sorted), then period; `time_period` is written as [`Period`] displays it.
pub fn prepare(data: &DataFrame, options: &Prepare) -> Result<DataFrame> {
    prepare_within(data, *options, MAX_PREPARED_BYTES)
}

/// [`prepare`] with a memory budget of `max_bytes`.
fn prepare_within(data: &DataFrame, options: Prepare, max_bytes: u64) -> Result<DataFrame> {
    let index = index(data)?;
    let first = index
        .values()
        .filter_map(|r| r.keys().next())
        .min()
        .copied();
    let last = index
        .values()
        .filter_map(|r| r.keys().next_back())
        .max()
        .copied();
    let (Some(first), Some(last)) = (first, last) else {
        return Ok(data.clone());
    };
    // The rows this makes and their memory, counted before any is allocated.
    let mut total = 0usize;
    for by_period in index.values() {
        let (Some(&own_first), Some(&own_last)) =
            (by_period.keys().next(), by_period.keys().next_back())
        else {
            continue;
        };
        let n = match (options.pad, options.fill_gaps) {
            (true, true) => span(first, last)?,
            (true, false) => span(first, own_first)? + by_period.len() + span(own_last, last)? - 2,
            (false, true) => span(own_first, own_last)?,
            (false, false) => by_period.len(),
        };
        total = total.saturating_add(n);
    }
    check_rows(total)?;
    budget::check(
        &format!("the prepared dataset ({total} rows, periods filled and padded per location)"),
        budget::times(total, prepared_row_bytes(data)),
        max_bytes,
        "keep the gaps, do not pad, or drop columns",
    )?;
    let mut periods: Vec<Period> = Vec::new();
    let mut rows: Vec<Option<usize>> = Vec::new();
    let mut locations: Vec<&str> = Vec::new();
    for (location, by_period) in &index {
        let (Some(&own_first), Some(&own_last)) =
            (by_period.keys().next(), by_period.keys().next_back())
        else {
            continue;
        };
        let inner: Vec<Period> = if options.fill_gaps {
            contiguous(own_first, own_last)?
        } else {
            by_period.keys().copied().collect()
        };
        let series: Vec<Period> = if options.pad {
            let mut before = contiguous(first, own_first)?;
            before.pop();
            let after = contiguous(own_last, last)?;
            before
                .into_iter()
                .chain(inner)
                .chain(after.into_iter().skip(1))
                .collect()
        } else {
            inner
        };
        for p in series {
            periods.push(p);
            rows.push(by_period.get(&p).copied());
            locations.push(location);
        }
    }
    let mut out = DataFrame::new();
    out.push_column(
        cols::TIME_PERIOD,
        Column::from_strs(periods.iter().map(ToString::to_string)),
    )?;
    out.push_column(cols::LOCATION, Column::from_strs(locations))?;
    for (name, col) in data.iter() {
        if name == cols::TIME_PERIOD || name == cols::LOCATION {
            continue;
        }
        out.push_column(name, fill(name, col, &rows)?)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rows: &[(&str, &str, Option<f64>)]) -> DataFrame {
        DataFrame::from_columns(vec![
            ("time_period", Column::from_strs(rows.iter().map(|r| r.0))),
            ("location", Column::from_strs(rows.iter().map(|r| r.1))),
            (
                "disease_cases",
                Column::Float(rows.iter().map(|r| r.2).collect()),
            ),
        ])
        .unwrap()
    }

    fn cells(df: &DataFrame) -> Vec<(String, String, Option<f64>)> {
        (0..df.n_rows())
            .map(|r| {
                let row = df.row_json(r);
                (
                    row[0].as_str().unwrap().to_owned(),
                    row[1].as_str().unwrap().to_owned(),
                    row[2].as_f64(),
                )
            })
            .collect()
    }

    #[test]
    fn fills_pads_and_sorts_as_chap_core() {
        // The example probed in chap-core 2.7.0: B has a gap and ends early, A misses 2020-03.
        let data = frame(&[
            ("2020-03", "B", Some(1.0)),
            ("2020-01", "B", Some(2.0)),
            ("2020-01", "A", Some(3.0)),
            ("2020-02", "A", Some(4.0)),
            ("2020-04", "A", Some(5.0)),
        ]);
        let got = cells(&prepare(&data, &Prepare::default()).unwrap());
        let want = [
            ("2020-01", "A", Some(3.0)),
            ("2020-02", "A", Some(4.0)),
            ("2020-03", "A", None),
            ("2020-04", "A", Some(5.0)),
            ("2020-01", "B", Some(2.0)),
            ("2020-02", "B", None),
            ("2020-03", "B", Some(1.0)),
            ("2020-04", "B", None),
        ];
        let want: Vec<_> = want
            .iter()
            .map(|(p, l, v)| ((*p).to_owned(), (*l).to_owned(), *v))
            .collect();
        assert_eq!(got, want);

        let keep = Prepare {
            fill_gaps: false,
            pad: false,
        };
        assert_eq!(prepare(&data, &keep).unwrap().n_rows(), 5);
        let pad_only = Prepare {
            fill_gaps: false,
            pad: true,
        };
        // B is padded after 2020-03, its gap at 2020-02 stays.
        assert_eq!(prepare(&data, &pad_only).unwrap().n_rows(), 6);
    }

    /// Months order before weeks, so filling from a month towards a week never ended.
    #[test]
    fn mixed_months_and_weeks_are_an_error() {
        let data = frame(&[("2024-01", "A", Some(1.0)), ("2024W02", "A", Some(2.0))]);
        for (fill_gaps, pad) in [(true, true), (false, false)] {
            let e = prepare(&data, &Prepare { fill_gaps, pad })
                .unwrap_err()
                .to_string();
            assert!(e.contains("mixes monthly and weekly"), "{e}");
        }
    }

    /// The rows filling and padding would make are counted before any is allocated.
    #[test]
    fn prepared_rows_are_bounded() {
        // 108,000 months each for 600 locations: 64.8 million rows.
        let names: Vec<String> = (0..600).map(|i| format!("L{i}")).collect();
        let mut rows = vec![("1000-01", "L0", Some(1.0))];
        rows.extend(names.iter().map(|l| ("9999-12", l.as_str(), Some(1.0))));
        let data = frame(&rows);
        let e = prepare(&data, &Prepare::default()).unwrap_err().to_string();
        assert!(e.contains("more than the limit"), "{e}");
        let keep = Prepare {
            fill_gaps: false,
            pad: false,
        };
        assert_eq!(prepare(&data, &keep).unwrap().n_rows(), 601);
        // 100,000 locations of ten years, monthly or weekly, fit.
        assert!(check_rows(100_000 * 522).is_ok());
        assert_eq!(
            span(
                Period::parse("2020W01").unwrap(),
                Period::parse("2021W01").unwrap()
            )
            .unwrap(),
            54
        );
    }

    /// The budget counts bytes, so column width matters: 500 locations over 1000-01..9999-12
    /// (54 million rows) with 100 numeric columns are about 87 GB and refused; checked with a small
    /// budget first, so nothing large is allocated.
    #[test]
    fn prepared_bytes_count_columns() {
        let wide = |rows: &[(&str, &str)], width: usize| {
            let mut df = frame(
                &rows
                    .iter()
                    .map(|r| (r.0, r.1, Some(1.0)))
                    .collect::<Vec<_>>(),
            );
            for c in 0..width {
                df.push_column(format!("x{c}"), Column::Float(vec![Some(1.0); rows.len()]))
                    .unwrap();
            }
            df
        };
        // Two locations, one filled to three months: 6 rows; 10 columns of 16 bytes do not fit 1 KB.
        let small = wide(&[("2020-01", "A"), ("2020-03", "A"), ("2020-01", "B")], 10);
        let e = prepare_within(&small, Prepare::default(), 1_000)
            .unwrap_err()
            .to_string();
        assert!(e.contains("more than the limit"), "{e}");
        assert!(prepare_within(&small, Prepare::default(), 10_000).is_ok());
        let row = prepared_row_bytes(&wide(&[("2020-01", "A")], 100));
        let n = span(
            Period::parse("1000-01").unwrap(),
            Period::parse("9999-12").unwrap(),
        )
        .unwrap();
        assert!(budget::times(n * 500, row) > MAX_PREPARED_BYTES);
        // 100,000 locations x 120 months x 6 numeric columns fit, as do ten years of weekly data.
        let realistic = prepared_row_bytes(&wide(&[("2020-01", "A")], 5));
        assert!(budget::times(100_000 * 120, realistic) < MAX_PREPARED_BYTES / 4);
        assert!(budget::times(100_000 * 522, realistic) < MAX_PREPARED_BYTES);
    }

    #[test]
    fn duplicates_and_unfillable_text_are_errors() {
        let dup = frame(&[("2020-01", "A", Some(1.0)), ("2020-01", "A", Some(2.0))]);
        let e = prepare(&dup, &Prepare::default()).unwrap_err().to_string();
        assert!(
            e.contains("location A has more than one row for 2020-01"),
            "{e}"
        );

        let mut text = frame(&[("2020-01", "A", Some(1.0)), ("2020-03", "A", Some(2.0))]);
        text.push_column("region", Column::from_strs(["x", "x"]))
            .unwrap();
        assert!(prepare(&text, &Prepare::default()).is_err());
        let keep = Prepare {
            fill_gaps: false,
            pad: false,
        };
        assert!(prepare(&text, &keep).is_ok());
    }
}
