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

/// The periods from `first` to `last`, contiguous. Fails for Sunday-start weeks.
pub(crate) fn contiguous(first: Period, last: Period) -> Result<Vec<Period>> {
    let mut out = vec![first];
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

/// Row index per (location, period): locations sorted, each location's rows by period.
type Index = BTreeMap<String, BTreeMap<Period, usize>>;

fn index(data: &DataFrame) -> Result<Index> {
    let (tp, loc) = (
        data.require(cols::TIME_PERIOD)?,
        data.require(cols::LOCATION)?,
    );
    let mut out: Index = BTreeMap::new();
    for row in 0..data.n_rows() {
        let label = tp.str_at(row).unwrap_or_default();
        let period = Period::parse(label).ok_or_else(|| {
            Error::invalid(format!(
                "time_period {label:?} in row {row} is not a month or week"
            ))
        })?;
        // Non-text locations (numbers) become text, as in chap-core.
        let location = loc
            .str_at(row)
            .map_or_else(|| loc.json_at(row).to_string(), str::to_owned);
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
