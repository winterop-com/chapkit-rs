//! Dictionary-encoded string columns.
//!
//! Chap tables repeat a few strings many times: every row has a `location` and a `time_period`,
//! but there are only as many distinct values as locations and periods. Storing one `u32` code
//! per row plus each distinct string once makes such columns far smaller, grouping by them a
//! matter of integer bucketing, and decoding them free of a heap allocation per cell.

use std::collections::HashMap;

/// A string column: a code per row into a dictionary of distinct values.
#[derive(Debug, Clone, Default)]
pub struct StrColumn {
    codes: Vec<u32>,
    dict: Vec<String>,
}

/// The code of a missing value.
pub const NULL: u32 = u32::MAX;

impl StrColumn {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// True when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// The value at `row`; `None` when it is missing or `row` is out of range.
    pub fn get(&self, row: usize) -> Option<&str> {
        let code = *self.codes.get(row)?;
        self.dict.get(code as usize).map(String::as_str)
    }

    /// Values in row order (`None` for missing ones).
    pub fn iter(&self) -> impl Iterator<Item = Option<&str>> {
        self.codes
            .iter()
            .map(|&c| self.dict.get(c as usize).map(String::as_str))
    }

    /// The code of each row ([`NULL`] for missing values), indexing [`StrColumn::dict`].
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }

    /// The distinct values, in order of first appearance.
    pub fn dict(&self) -> &[String] {
        &self.dict
    }

    /// Rows `idx` (indices may repeat). `None` if any index is out of range.
    pub fn take(&self, idx: &[usize]) -> Option<StrColumn> {
        let codes = idx
            .iter()
            .map(|&i| self.codes.get(i).copied())
            .collect::<Option<Vec<u32>>>()?;
        Some(StrColumn {
            codes,
            dict: self.dict.clone(),
        })
    }

    /// Append another column's rows.
    pub fn append(&mut self, other: &StrColumn) {
        let mut builder = StrBuilder::resume(std::mem::take(self));
        for v in other.iter() {
            builder.push(v);
        }
        *self = builder.finish();
    }
}

impl PartialEq for StrColumn {
    /// Equal when the values are equal row by row, however they are encoded.
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<'a> FromIterator<Option<&'a str>> for StrColumn {
    fn from_iter<I: IntoIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        let mut b = StrBuilder::default();
        for v in iter {
            b.push(v);
        }
        b.finish()
    }
}

impl FromIterator<Option<String>> for StrColumn {
    fn from_iter<I: IntoIterator<Item = Option<String>>>(iter: I) -> Self {
        let mut b = StrBuilder::default();
        for v in iter {
            b.push(v.as_deref());
        }
        b.finish()
    }
}

/// Builds a [`StrColumn`] value by value, interning each distinct string once.
#[derive(Debug, Default)]
pub struct StrBuilder {
    col: StrColumn,
    index: HashMap<String, u32>,
    /// The previous value and its code: sorted data repeats values in runs.
    last: Option<u32>,
}

impl StrBuilder {
    /// Continue building on an existing column.
    fn resume(col: StrColumn) -> Self {
        let index = col
            .dict
            .iter()
            .enumerate()
            .filter_map(|(i, s)| Some((s.clone(), u32::try_from(i).ok()?)))
            .collect();
        Self {
            col,
            index,
            last: None,
        }
    }

    /// A builder that starts with `n` missing values.
    pub fn with_nulls(n: usize) -> Self {
        let mut b = Self::default();
        b.col.codes.resize(n, NULL);
        b
    }

    /// Append a value (`None` for missing).
    pub fn push(&mut self, value: Option<&str>) {
        let code = match value {
            None => NULL,
            Some(s) => self.code_of(s),
        };
        self.col.codes.push(code);
    }

    fn code_of(&mut self, s: &str) -> u32 {
        if let Some(code) = self.last
            && self.col.dict.get(code as usize).is_some_and(|d| d == s)
        {
            return code;
        }
        let code = if let Some(&code) = self.index.get(s) {
            code
        } else {
            // More than u32::MAX - 1 distinct strings cannot occur in a frame that fits in memory.
            let code = u32::try_from(self.col.dict.len()).unwrap_or(NULL - 1);
            self.col.dict.push(s.to_owned());
            self.index.insert(s.to_owned(), code);
            code
        };
        self.last = Some(code);
        code
    }

    /// Number of values pushed.
    pub fn len(&self) -> usize {
        self.col.len()
    }

    /// True when nothing was pushed.
    pub fn is_empty(&self) -> bool {
        self.col.is_empty()
    }

    /// Values pushed so far, in order.
    pub fn iter(&self) -> impl Iterator<Item = Option<&str>> {
        self.col.iter()
    }

    /// The finished column.
    pub fn finish(self) -> StrColumn {
        self.col
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interns_and_round_trips() {
        let values = [Some("A"), Some("A"), None, Some("B"), Some("A"), Some("")];
        let col: StrColumn = values.into_iter().collect();
        assert_eq!(col.len(), 6);
        assert_eq!(col.dict(), ["A", "B", ""]);
        assert_eq!(col.codes(), [0, 0, NULL, 1, 0, 2]);
        assert!(col.iter().eq(values));
        assert_eq!(col.get(2), None);
        assert_eq!(col.get(9), None);

        let picked = col.take(&[3, 0, 3]).unwrap_or_default();
        assert!(picked.iter().eq([Some("B"), Some("A"), Some("B")]));
        assert!(col.take(&[6]).is_none());

        // Same values, different encoding: still equal.
        let other: StrColumn = [Some("B"), Some("A")].into_iter().collect();
        let reordered: StrColumn = [Some("B"), Some("A")]
            .into_iter()
            .collect::<Vec<_>>()
            .into_iter()
            .collect();
        assert_eq!(other, reordered);
        let mut a: StrColumn = [Some("X")].into_iter().collect();
        a.append(&other);
        assert!(a.iter().eq([Some("X"), Some("B"), Some("A")]));
        assert_eq!(a.dict().len(), 3);
    }
}
