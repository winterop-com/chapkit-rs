use std::collections::HashMap;

use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::column::{Column, DType};

/// Errors raised when building or reshaping a frame.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum FrameError {
    /// A column's length does not match the frame's row count.
    #[error("column '{name}' has {got} rows, expected {expected}")]
    Length {
        name: String,
        got: usize,
        expected: usize,
    },
    /// A row in the wire format has the wrong number of cells.
    #[error("row {row} has {got} cells, expected {expected}")]
    RowWidth {
        row: usize,
        got: usize,
        expected: usize,
    },
    /// A column name appears twice.
    #[error("duplicate column '{0}'")]
    Duplicate(String),
    /// A required column is missing.
    #[error("missing column '{0}'")]
    Missing(String),
    /// A row index is past the end of the frame.
    #[error("row index out of range (frame has {rows} rows)")]
    RowOutOfRange { rows: usize },
    /// Frames being stacked have different column names.
    #[error("cannot stack frames with different columns")]
    ColumnMismatch,
    /// Converting to or from another frame library failed.
    #[error("conversion failed: {0}")]
    Convert(String),
}

/// A small columnar table.
///
/// It serializes to and from the chapkit wire format
/// `{"columns": [...], "data": [[row], ...], "schema": {"fields": [...]}}`;
/// `schema` is optional on input and always written on output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DataFrame {
    names: Vec<String>,
    columns: Vec<Column>,
    n_rows: usize,
}

impl DataFrame {
    /// An empty frame with no columns.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a frame from `(name, column)` pairs.
    pub fn from_columns<S: Into<String>>(
        cols: impl IntoIterator<Item = (S, Column)>,
    ) -> Result<Self, FrameError> {
        let mut df = DataFrame::new();
        for (name, col) in cols {
            df.push_column(name, col)?;
        }
        Ok(df)
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// True when the frame has no rows.
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    /// Column names in order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Iterate `(name, column)` pairs in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.names
            .iter()
            .map(String::as_str)
            .zip(self.columns.iter())
    }

    /// Position of a column.
    pub fn position(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// True if the column exists.
    pub fn has(&self, name: &str) -> bool {
        self.position(name).is_some()
    }

    /// Borrow a column by name.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.position(name).map(|i| &self.columns[i])
    }

    /// Borrow a column by name, or fail with [`FrameError::Missing`].
    pub fn require(&self, name: &str) -> Result<&Column, FrameError> {
        self.column(name)
            .ok_or_else(|| FrameError::Missing(name.to_owned()))
    }

    /// Set one cell of a numeric column. The column becomes a float column if it was an integer
    /// one. Fails if the column is missing or not numeric, or the row is out of range. There is no
    /// `&mut Column` accessor: resizing a column would break the frame's row count.
    pub fn set_f64(
        &mut self,
        name: &str,
        row: usize,
        value: Option<f64>,
    ) -> Result<(), FrameError> {
        let rows = self.n_rows;
        let i = self
            .position(name)
            .ok_or_else(|| FrameError::Missing(name.to_owned()))?;
        let col = self
            .columns
            .get_mut(i)
            .ok_or_else(|| FrameError::Missing(name.to_owned()))?;
        if let Column::Int(v) = col {
            #[allow(clippy::cast_precision_loss)]
            let widened = v.iter().map(|x| x.map(|x| x as f64)).collect();
            *col = Column::Float(widened);
        }
        match col {
            Column::Float(v) => {
                let cell = v.get_mut(row).ok_or(FrameError::RowOutOfRange { rows })?;
                *cell = value;
                Ok(())
            }
            _ => Err(FrameError::Missing(format!("{name} (numeric)"))),
        }
    }

    /// Numeric values of a column (nulls as `NaN`).
    pub fn f64_column(&self, name: &str) -> Result<Vec<f64>, FrameError> {
        self.require(name)?
            .to_f64_vec()
            .ok_or_else(|| FrameError::Missing(format!("{name} (numeric)")))
    }

    /// Append a column. Fails on duplicate names or a row-count mismatch.
    pub fn push_column(&mut self, name: impl Into<String>, col: Column) -> Result<(), FrameError> {
        let name = name.into();
        if self.has(&name) {
            return Err(FrameError::Duplicate(name));
        }
        if self.columns.is_empty() {
            self.n_rows = col.len();
        } else if col.len() != self.n_rows {
            return Err(FrameError::Length {
                name,
                got: col.len(),
                expected: self.n_rows,
            });
        }
        self.names.push(name);
        self.columns.push(col);
        Ok(())
    }

    /// Insert or replace a column.
    pub fn set_column(&mut self, name: impl Into<String>, col: Column) -> Result<(), FrameError> {
        let name = name.into();
        match self.position(&name) {
            Some(i) => {
                if self.columns.len() > 1 && col.len() != self.n_rows {
                    return Err(FrameError::Length {
                        name,
                        got: col.len(),
                        expected: self.n_rows,
                    });
                }
                self.n_rows = col.len();
                self.columns[i] = col;
                Ok(())
            }
            None => self.push_column(name, col),
        }
    }

    /// Keep only the named columns, in the given order.
    pub fn select(&self, names: &[&str]) -> Result<DataFrame, FrameError> {
        let mut out = DataFrame::new();
        for n in names {
            out.push_column(*n, self.require(n)?.clone())?;
        }
        out.n_rows = self.n_rows;
        Ok(out)
    }

    /// Select rows by index (indices may repeat).
    pub fn take(&self, idx: &[usize]) -> Result<DataFrame, FrameError> {
        let columns = self
            .columns
            .iter()
            .map(|c| c.take(idx))
            .collect::<Option<Vec<_>>>()
            .ok_or(FrameError::RowOutOfRange { rows: self.n_rows })?;
        Ok(DataFrame {
            names: self.names.clone(),
            columns,
            n_rows: idx.len(),
        })
    }

    /// Keep rows where `mask` is true. `mask` must have one entry per row.
    pub fn filter(&self, mask: &[bool]) -> Result<DataFrame, FrameError> {
        if mask.len() != self.n_rows {
            return Err(FrameError::Length {
                name: "mask".into(),
                got: mask.len(),
                expected: self.n_rows,
            });
        }
        let idx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| m.then_some(i))
            .collect();
        self.take(&idx)
    }

    /// Row indices grouped by the string value of `key`, in order of first appearance.
    /// Null keys are grouped under `""`.
    pub fn group_indices(&self, key: &str) -> Result<Vec<(String, Vec<usize>)>, FrameError> {
        let col = self.require(key)?;
        let mut order: Vec<(String, Vec<usize>)> = Vec::new();
        let mut pos: HashMap<String, usize> = HashMap::new();
        for row in 0..self.n_rows {
            let k = match col.str_at(row) {
                Some(s) => s.to_owned(),
                None => match col.json_at(row) {
                    Value::Null => String::new(),
                    v => v.to_string(),
                },
            };
            if let Some(&i) = pos.get(&k) {
                order[i].1.push(row);
            } else {
                pos.insert(k.clone(), order.len());
                order.push((k, vec![row]));
            }
        }
        Ok(order)
    }

    /// Stack frames vertically. All frames must have the same column names in the same order.
    pub fn vstack(frames: &[&DataFrame]) -> Result<DataFrame, FrameError> {
        let Some((first, rest)) = frames.split_first() else {
            return Ok(DataFrame::new());
        };
        let mut out = (*first).clone();
        for f in rest {
            if f.names != out.names {
                return Err(FrameError::ColumnMismatch);
            }
            for (a, b) in out.columns.iter_mut().zip(&f.columns) {
                a.append(b);
            }
            out.n_rows += f.n_rows;
        }
        Ok(out)
    }

    /// Row-major JSON cells of one row.
    pub fn row_json(&self, row: usize) -> Vec<Value> {
        self.columns.iter().map(|c| c.json_at(row)).collect()
    }

    fn from_wire(w: Wire) -> Result<Self, FrameError> {
        let width = w.columns.len();
        let types: HashMap<&str, DType> = w
            .schema
            .as_ref()
            .map(|s| {
                s.fields
                    .iter()
                    .map(|f| (f.name.as_str(), DType::parse(&f.r#type)))
                    .collect()
            })
            .unwrap_or_default();
        let n_rows = w.data.len();
        let mut cells: Vec<Vec<Value>> = (0..width).map(|_| Vec::with_capacity(n_rows)).collect();
        for (r, row) in w.data.into_iter().enumerate() {
            if row.len() != width {
                return Err(FrameError::RowWidth {
                    row: r,
                    got: row.len(),
                    expected: width,
                });
            }
            for (c, v) in row.into_iter().enumerate() {
                cells[c].push(v);
            }
        }
        let mut df = DataFrame::new();
        for (name, col_cells) in w.columns.iter().zip(cells) {
            let dtype = types.get(name.as_str()).copied();
            df.push_column(name.clone(), Column::from_json_cells(col_cells, dtype))?;
        }
        df.n_rows = n_rows;
        Ok(df)
    }
}

#[derive(Deserialize)]
struct Wire {
    columns: Vec<String>,
    data: Vec<Vec<Value>>,
    #[serde(default)]
    schema: Option<WireSchema>,
}

#[derive(Deserialize, Serialize)]
struct WireSchema {
    fields: Vec<WireField>,
}

#[derive(Deserialize, Serialize)]
struct WireField {
    name: String,
    #[serde(rename = "type")]
    r#type: String,
}

impl<'de> Deserialize<'de> for DataFrame {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        DataFrame::from_wire(w).map_err(serde::de::Error::custom)
    }
}

struct Rows<'a>(&'a DataFrame);
struct Row<'a>(&'a DataFrame, usize);
struct Cell<'a>(&'a Column, usize);

impl Serialize for DataFrame {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let schema = WireSchema {
            fields: self
                .iter()
                .map(|(n, c)| WireField {
                    name: n.to_owned(),
                    r#type: c.dtype().as_str().to_owned(),
                })
                .collect(),
        };
        let mut m = s.serialize_map(Some(3))?;
        m.serialize_entry("columns", &self.names)?;
        m.serialize_entry("data", &Rows(self))?;
        m.serialize_entry("schema", &schema)?;
        m.end()
    }
}

impl Serialize for Rows<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.n_rows))?;
        for r in 0..self.0.n_rows {
            seq.serialize_element(&Row(self.0, r))?;
        }
        seq.end()
    }
}

impl Serialize for Row<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.columns.len()))?;
        for c in &self.0.columns {
            seq.serialize_element(&Cell(c, self.1))?;
        }
        seq.end()
    }
}

impl Serialize for Cell<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let row = self.1;
        let missing = || serde::ser::Error::custom(format!("row {row} out of range"));
        match self.0 {
            Column::Float(v) => match v.get(row).ok_or_else(missing)? {
                Some(x) if x.is_finite() => s.serialize_f64(*x),
                _ => s.serialize_none(),
            },
            Column::Int(v) => match v.get(row).ok_or_else(missing)? {
                Some(x) => s.serialize_i64(*x),
                None => s.serialize_none(),
            },
            Column::Str(v) => match v.get(row).ok_or_else(missing)? {
                Some(x) => s.serialize_str(x),
                None => s.serialize_none(),
            },
            Column::Bool(v) => match v.get(row).ok_or_else(missing)? {
                Some(x) => s.serialize_bool(*x),
                None => s.serialize_none(),
            },
            Column::Json(v) => v.get(row).ok_or_else(missing)?.serialize(s),
        }
    }
}
