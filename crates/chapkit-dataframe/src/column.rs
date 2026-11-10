use serde_json::Value;

use crate::strings::StrColumn;

/// Logical column type, mirroring the chapkit DataFrame schema field types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// `"string"`
    String,
    /// `"integer"`
    Integer,
    /// `"number"`
    Number,
    /// `"boolean"`
    Boolean,
    /// `"any"`: arbitrary JSON values.
    Any,
}

impl DType {
    /// The chapkit schema name of this type.
    pub fn as_str(self) -> &'static str {
        match self {
            DType::String => "string",
            DType::Integer => "integer",
            DType::Number => "number",
            DType::Boolean => "boolean",
            DType::Any => "any",
        }
    }

    /// Parse a chapkit schema type name. Unknown names map to [`DType::Any`].
    pub fn parse(s: &str) -> DType {
        match s {
            "string" => DType::String,
            "integer" => DType::Integer,
            "number" => DType::Number,
            "boolean" => DType::Boolean,
            _ => DType::Any,
        }
    }
}

/// A typed, null-aware column.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    /// Floating point values. `None` and `NaN` are both written as JSON `null`.
    Float(Vec<Option<f64>>),
    /// Integer values.
    Int(Vec<Option<i64>>),
    /// String values, dictionary-encoded (see [`StrColumn`]).
    Str(StrColumn),
    /// Boolean values.
    Bool(Vec<Option<bool>>),
    /// Arbitrary JSON values (mixed or nested content).
    Json(Vec<Value>),
}

impl Column {
    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Column::Float(v) => v.len(),
            Column::Int(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Json(v) => v.len(),
        }
    }

    /// True when the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Logical type.
    pub fn dtype(&self) -> DType {
        match self {
            Column::Float(_) => DType::Number,
            Column::Int(_) => DType::Integer,
            Column::Str(_) => DType::String,
            Column::Bool(_) => DType::Boolean,
            Column::Json(_) => DType::Any,
        }
    }

    /// Build a float column from plain values (`NaN` stays `NaN`, written as null).
    pub fn from_f64(values: impl IntoIterator<Item = f64>) -> Column {
        Column::Float(values.into_iter().map(Some).collect())
    }

    /// Build a string column from plain values.
    pub fn from_strs<S: Into<String>>(values: impl IntoIterator<Item = S>) -> Column {
        Column::Str(values.into_iter().map(|s| Some(s.into())).collect())
    }

    /// Value at `row` as `f64`, if numeric, present and in range. `NaN` is returned as-is.
    pub fn f64_at(&self, row: usize) -> Option<f64> {
        match self {
            Column::Float(v) => v.get(row).copied().flatten(),
            Column::Int(v) => v.get(row).copied().flatten().map(|x| x as f64),
            Column::Bool(v) => v
                .get(row)
                .copied()
                .flatten()
                .map(|b| if b { 1.0 } else { 0.0 }),
            Column::Json(v) => v.get(row).and_then(Value::as_f64),
            Column::Str(_) => None,
        }
    }

    /// Value at `row` as `&str`, if this is a string column and the value is present and in range.
    pub fn str_at(&self, row: usize) -> Option<&str> {
        match self {
            Column::Str(v) => v.get(row),
            Column::Json(v) => v.get(row).and_then(Value::as_str),
            _ => None,
        }
    }

    /// Value at `row` as JSON. Non-finite floats, nulls and out-of-range rows become `null`.
    pub fn json_at(&self, row: usize) -> Value {
        match self {
            Column::Float(v) => v
                .get(row)
                .copied()
                .flatten()
                .and_then(serde_json::Number::from_f64)
                .map_or(Value::Null, Value::Number),
            Column::Int(v) => v
                .get(row)
                .copied()
                .flatten()
                .map_or(Value::Null, Value::from),
            Column::Str(v) => v
                .get(row)
                .map_or(Value::Null, |s| Value::String(s.to_owned())),
            Column::Bool(v) => v
                .get(row)
                .copied()
                .flatten()
                .map_or(Value::Null, Value::Bool),
            Column::Json(v) => v.get(row).cloned().unwrap_or(Value::Null),
        }
    }

    /// Numeric view of the whole column: nulls become `NaN`. `None` for string columns.
    pub fn to_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            Column::Float(v) => Some(v.iter().map(|x| x.unwrap_or(f64::NAN)).collect()),
            Column::Int(v) => Some(v.iter().map(|x| x.map_or(f64::NAN, |x| x as f64)).collect()),
            Column::Bool(v) => Some(
                v.iter()
                    .map(|x| x.map_or(f64::NAN, |b| if b { 1.0 } else { 0.0 }))
                    .collect(),
            ),
            Column::Json(v) => Some(v.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()),
            Column::Str(_) => None,
        }
    }

    /// Select rows by index (indices may repeat). `None` if any index is out of range.
    pub fn take(&self, idx: &[usize]) -> Option<Column> {
        fn pick<T: Clone>(v: &[T], idx: &[usize]) -> Option<Vec<T>> {
            idx.iter().map(|&i| v.get(i).cloned()).collect()
        }
        Some(match self {
            Column::Float(v) => Column::Float(pick(v, idx)?),
            Column::Int(v) => Column::Int(pick(v, idx)?),
            Column::Str(v) => Column::Str(v.take(idx)?),
            Column::Bool(v) => Column::Bool(pick(v, idx)?),
            Column::Json(v) => Column::Json(pick(v, idx)?),
        })
    }

    /// Append another column, widening the type when needed
    /// (int + float gives float; anything else that differs gives JSON).
    pub fn append(&mut self, other: &Column) {
        match (&mut *self, other) {
            (Column::Float(a), Column::Float(b)) => a.extend_from_slice(b),
            (Column::Int(a), Column::Int(b)) => a.extend_from_slice(b),
            (Column::Str(a), Column::Str(b)) => a.append(b),
            (Column::Bool(a), Column::Bool(b)) => a.extend_from_slice(b),
            (Column::Json(a), _) => a.extend((0..other.len()).map(|i| other.json_at(i))),
            (Column::Float(a), Column::Int(b)) => a.extend(b.iter().map(|x| x.map(|x| x as f64))),
            (Column::Int(a), Column::Float(b)) => {
                let mut widened: Vec<Option<f64>> = a.iter().map(|x| x.map(|x| x as f64)).collect();
                widened.extend_from_slice(b);
                *self = Column::Float(widened);
            }
            _ => {
                let mut widened: Vec<Value> = (0..self.len()).map(|i| self.json_at(i)).collect();
                widened.extend((0..other.len()).map(|i| other.json_at(i)));
                *self = Column::Json(widened);
            }
        }
    }

    /// Build a column from JSON cells, using `dtype` when given, otherwise inferring it.
    pub(crate) fn from_json_cells(cells: Vec<Value>, dtype: Option<DType>) -> Column {
        let dtype = dtype.unwrap_or_else(|| infer(&cells));
        match dtype {
            DType::Integer if cells.iter().all(|c| c.is_null() || c.is_i64()) => {
                Column::Int(cells.iter().map(Value::as_i64).collect())
            }
            DType::Number | DType::Integer => {
                Column::Float(cells.iter().map(Value::as_f64).collect())
            }
            DType::Boolean if cells.iter().all(|c| c.is_null() || c.is_boolean()) => {
                Column::Bool(cells.iter().map(Value::as_bool).collect())
            }
            DType::String if cells.iter().all(|c| c.is_null() || c.is_string()) => Column::Str(
                cells
                    .into_iter()
                    .map(|c| match c {
                        Value::String(s) => Some(s),
                        _ => None,
                    })
                    .collect(),
            ),
            _ => Column::Json(cells),
        }
    }
}

/// Infer a column type from JSON cells. All-null columns are numeric (NaN sent as null).
fn infer(cells: &[Value]) -> DType {
    let mut kind: Option<DType> = None;
    for c in cells {
        let k = match c {
            Value::Null => continue,
            Value::Bool(_) => DType::Boolean,
            Value::Number(n) if n.is_i64() => DType::Integer,
            Value::Number(_) => DType::Number,
            Value::String(_) => DType::String,
            _ => return DType::Any,
        };
        kind = Some(match (kind, k) {
            (None, k) => k,
            (Some(a), b) if a == b => a,
            (Some(DType::Integer | DType::Number), DType::Integer | DType::Number) => DType::Number,
            _ => return DType::Any,
        });
    }
    kind.unwrap_or(DType::Number)
}
