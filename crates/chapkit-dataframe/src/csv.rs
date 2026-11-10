//! CSV input, as chap-core datasets are written: a header row, comma-separated, optional quotes.

use crate::column::Column;
use crate::frame::{DataFrame, FrameError};

/// Cells read as missing in numeric columns (pandas' defaults that occur in chap datasets).
const MISSING: [&str; 5] = ["", "NA", "NaN", "nan", "null"];

/// Columns that are always text, whatever they look like (`202401` is a period, not a number).
const TEXT: [&str; 2] = ["time_period", "location"];

/// Split CSV text into records of fields: comma-separated, `"` quotes with `""` as a literal quote,
/// LF or CRLF line ends. Blank lines are skipped.
fn records(text: &str) -> Result<Vec<Vec<String>>, FrameError> {
    let mut out = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => quoted = false,
                c => field.push(c),
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            ',' => record.push(std::mem::take(&mut field)),
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' => {
                record.push(std::mem::take(&mut field));
                if !(record.len() == 1 && record.first().is_some_and(String::is_empty)) {
                    out.push(std::mem::take(&mut record));
                }
                record.clear();
            }
            c => field.push(c),
        }
    }
    if quoted {
        return Err(FrameError::Convert("CSV ends inside a quoted field".into()));
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        out.push(record);
    }
    Ok(out)
}

/// A column from its text cells: integers if every present cell is one, else numbers if every
/// present cell is one, else text (with empty cells missing).
fn column(name: &str, cells: Vec<String>) -> Column {
    let present = || cells.iter().filter(|c| !MISSING.contains(&c.trim()));
    if !TEXT.contains(&name) {
        if present().all(|c| c.trim().parse::<i64>().is_ok()) && present().next().is_some() {
            return Column::Int(cells.iter().map(|c| c.trim().parse().ok()).collect());
        }
        if present().all(|c| c.trim().parse::<f64>().is_ok()) {
            return Column::Float(
                cells
                    .iter()
                    .map(|c| c.trim().parse::<f64>().ok().filter(|v| !v.is_nan()))
                    .collect(),
            );
        }
    }
    Column::Str(
        cells
            .into_iter()
            .map(|c| (!c.is_empty()).then_some(c))
            .collect(),
    )
}

impl DataFrame {
    /// Read CSV text with a header row. Column types are inferred: integer, number, else text;
    /// `time_period` and `location` are always text.
    pub fn from_csv(text: &str) -> Result<DataFrame, FrameError> {
        let mut rows = records(text.strip_prefix('\u{feff}').unwrap_or(text))?.into_iter();
        let header = rows.next().unwrap_or_default();
        let mut cells: Vec<Vec<String>> = header.iter().map(|_| Vec::new()).collect();
        for (i, row) in rows.enumerate() {
            if row.len() != header.len() {
                return Err(FrameError::RowWidth {
                    row: i,
                    got: row.len(),
                    expected: header.len(),
                });
            }
            for (col, cell) in cells.iter_mut().zip(row) {
                col.push(cell);
            }
        }
        let mut df = DataFrame::new();
        for (name, col) in header.into_iter().zip(cells) {
            let col = column(&name, col);
            df.push_column(name, col)?;
        }
        Ok(df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_chap_csv() {
        let df = DataFrame::from_csv(
            "time_period,location,disease_cases,rainfall,name\r\n\
             202401,A,3,1.5,\"x, y\"\r\n\
             202402,A,,NaN,\"say \"\"hi\"\"\"\n\n",
        )
        .unwrap();
        assert_eq!(df.n_rows(), 2);
        assert_eq!(df.column("time_period").unwrap().str_at(0), Some("202401"));
        assert_eq!(
            df.column("disease_cases"),
            Some(&Column::Int(vec![Some(3), None]))
        );
        assert_eq!(
            df.column("rainfall"),
            Some(&Column::Float(vec![Some(1.5), None]))
        );
        assert_eq!(df.column("name").unwrap().str_at(0), Some("x, y"));
        assert_eq!(df.column("name").unwrap().str_at(1), Some("say \"hi\""));
    }

    #[test]
    fn rejects_ragged_rows_and_open_quotes() {
        assert!(DataFrame::from_csv("a,b\n1\n").is_err());
        assert!(DataFrame::from_csv("a\n\"1\n").is_err());
    }
}
