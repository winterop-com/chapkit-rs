//! Streaming decoder for the chapkit wire format.
//!
//! Cells go straight into typed column builders while the JSON is parsed, instead of becoming a
//! `serde_json::Value` each, collected row by row and transposed. The resulting columns follow the
//! same rules as [`Column::from_json_cells`]: types are inferred (all-null is numeric, integers only
//! when every number is an `i64`, integers mixed with floats are floats, anything else mixed is
//! JSON), and a schema type, when given, takes precedence. Unusual combinations fall back to that
//! function on the original cell values, so the result is identical.

use std::collections::HashMap;
use std::fmt;

use serde::Deserialize;
use serde::de::{self, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::{Number, Value};

use crate::column::{Column, DType};
use crate::frame::{DataFrame, FrameError, WireSchema};
use crate::strings::StrBuilder;

/// One decoded cell. Scalars avoid `Value`'s allocation; arrays and objects keep it. Strings are
/// normally handed to the column builder straight from the parser (see [`CellSeed`]); `Str` only
/// occurs for strings inside rebuilt values.
enum Cell {
    Null,
    Bool(bool),
    Num(Number),
    Str(String),
    Other(Value),
}

/// A column being built. Starts `Empty` (only nulls so far) and settles on a type.
enum Builder {
    Empty(usize),
    Num(Vec<Option<Number>>),
    Str(StrBuilder),
    Bool(Vec<Option<bool>>),
    Json(Vec<Value>),
}

impl Builder {
    /// Push a string without allocating: interned into the column's dictionary.
    fn push_str(&mut self, s: &str) {
        match self {
            Builder::Str(b) => b.push(Some(s)),
            Builder::Json(v) => v.push(Value::String(s.to_owned())),
            Builder::Empty(n) => {
                let mut b = StrBuilder::with_nulls(*n);
                b.push(Some(s));
                *self = Builder::Str(b);
            }
            _ => {
                let mut v = std::mem::replace(self, Builder::Empty(0)).into_values();
                v.push(Value::String(s.to_owned()));
                *self = Builder::Json(v);
            }
        }
    }

    fn push(&mut self, cell: Cell) {
        match (&mut *self, cell) {
            (Builder::Empty(n), Cell::Null) => *n += 1,
            (Builder::Num(v), Cell::Null) => v.push(None),
            (Builder::Str(b), Cell::Null) => b.push(None),
            (_, Cell::Str(s)) => self.push_str(&s),
            (Builder::Bool(v), Cell::Null) => v.push(None),
            (Builder::Num(v), Cell::Num(x)) => v.push(Some(x)),
            (Builder::Bool(v), Cell::Bool(x)) => v.push(Some(x)),
            (Builder::Json(v), cell) => v.push(cell.into_value()),
            (Builder::Empty(n), cell) => {
                let n = *n;
                *self = match cell {
                    Cell::Num(x) => Builder::Num(nulls_then(n, x)),
                    Cell::Bool(x) => Builder::Bool(nulls_then(n, x)),
                    other => {
                        let mut v = vec![Value::Null; n];
                        v.push(other.into_value());
                        Builder::Json(v)
                    }
                };
            }
            (_, cell) => {
                let mut v = std::mem::replace(self, Builder::Empty(0)).into_values();
                v.push(cell.into_value());
                *self = Builder::Json(v);
            }
        }
    }

    fn into_values(self) -> Vec<Value> {
        match self {
            Builder::Empty(n) => vec![Value::Null; n],
            Builder::Num(v) => v
                .into_iter()
                .map(|x| x.map_or(Value::Null, Value::Number))
                .collect(),
            Builder::Str(b) => b
                .iter()
                .map(|x| x.map_or(Value::Null, |s| Value::String(s.to_owned())))
                .collect(),
            Builder::Bool(v) => v
                .into_iter()
                .map(|x| x.map_or(Value::Null, Value::Bool))
                .collect(),
            Builder::Json(v) => v,
        }
    }

    /// The finished column, by the same rules as `Column::from_json_cells`.
    fn finish(self, dtype: Option<DType>) -> Column {
        match (dtype, self) {
            (None | Some(DType::Number | DType::Integer), Builder::Empty(n))
                if dtype != Some(DType::Integer) =>
            {
                Column::Float(vec![None; n])
            }
            (Some(DType::Integer), Builder::Empty(n)) => Column::Int(vec![None; n]),
            (None | Some(DType::Integer), Builder::Num(v))
                if v.iter().flatten().all(Number::is_i64) =>
            {
                Column::Int(
                    v.iter()
                        .map(|x| x.as_ref().and_then(Number::as_i64))
                        .collect(),
                )
            }
            (None | Some(DType::Number | DType::Integer), Builder::Num(v)) => Column::Float(
                v.iter()
                    .map(|x| x.as_ref().and_then(Number::as_f64))
                    .collect(),
            ),
            (None | Some(DType::String), Builder::Str(b)) => Column::Str(b.finish()),
            (Some(DType::String), Builder::Empty(n)) => {
                Column::Str(StrBuilder::with_nulls(n).finish())
            }
            (None | Some(DType::Boolean), Builder::Bool(v)) => Column::Bool(v),
            (Some(DType::Boolean), Builder::Empty(n)) => Column::Bool(vec![None; n]),
            (None, Builder::Json(v)) => Column::Json(v),
            // Rare: a schema type that does not match the cells.
            (dtype, b) => Column::from_json_cells(b.into_values(), dtype),
        }
    }
}

fn nulls_then<T>(n: usize, x: T) -> Vec<Option<T>> {
    let mut v: Vec<Option<T>> = Vec::with_capacity(n + 1);
    v.resize_with(n, || None);
    v.push(Some(x));
    v
}

impl Cell {
    fn from_value(v: Value) -> Cell {
        match v {
            Value::Null => Cell::Null,
            Value::Bool(b) => Cell::Bool(b),
            Value::Number(n) => Cell::Num(n),
            Value::String(s) => Cell::Str(s),
            other => Cell::Other(other),
        }
    }

    fn into_value(self) -> Value {
        match self {
            Cell::Null => Value::Null,
            Cell::Bool(b) => Value::Bool(b),
            Cell::Num(n) => Value::Number(n),
            Cell::Str(s) => Value::String(s),
            Cell::Other(v) => v,
        }
    }
}

/// Seed for one cell: decodes it straight into its column's builder (`None` for cells beyond
/// the frame's width, which are only counted).
struct CellSeed<'a>(Option<&'a mut Builder>);

impl<'de> DeserializeSeed<'de> for CellSeed<'_> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for CellSeed<'_> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_str<E>(self, v: &str) -> Result<(), E> {
        if let Some(b) = self.0 {
            b.push_str(v);
        }
        Ok(())
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        self.store(Cell::Null);
        Ok(())
    }
    fn visit_none<E>(self) -> Result<(), E> {
        self.store(Cell::Null);
        Ok(())
    }
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
    fn visit_bool<E>(self, v: bool) -> Result<(), E> {
        self.store(Cell::Bool(v));
        Ok(())
    }
    fn visit_i64<E>(self, v: i64) -> Result<(), E> {
        self.store(Cell::Num(v.into()));
        Ok(())
    }
    fn visit_u64<E>(self, v: u64) -> Result<(), E> {
        self.store(Cell::Num(v.into()));
        Ok(())
    }
    fn visit_f64<E>(self, v: f64) -> Result<(), E> {
        self.store(Number::from_f64(v).map_or(Cell::Null, Cell::Num));
        Ok(())
    }
    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<(), A::Error> {
        let v = Value::deserialize(de::value::SeqAccessDeserializer::new(seq))?;
        self.store(Cell::Other(v));
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<(), A::Error> {
        // With serde_json's `arbitrary_precision` feature, numbers arrive as a special map;
        // rebuilding the `Value` turns them back into numbers.
        let v = Value::deserialize(de::value::MapAccessDeserializer::new(map))?;
        self.store(Cell::from_value(v));
        Ok(())
    }
}

impl CellSeed<'_> {
    fn store(self, cell: Cell) {
        if let Some(b) = self.0 {
            b.push(cell);
        }
    }
}

/// Decoded `data`: one builder per column, plus the row count.
struct Data {
    builders: Vec<Builder>,
    n_rows: usize,
}

/// Seed for `data`: rows must all have `width` cells; the first row sets it when unknown.
struct DataSeed(Option<usize>);

impl<'de> DeserializeSeed<'de> for DataSeed {
    type Value = Data;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Data, D::Error> {
        d.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for DataSeed {
    type Value = Data;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a list of rows")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Data, A::Error> {
        let mut data = Data {
            builders: (0..self.0.unwrap_or(0))
                .map(|_| Builder::Empty(0))
                .collect(),
            n_rows: 0,
        };
        let mut width = self.0;
        while let Some(got) = seq.next_element_seed(RowSeed {
            builders: &mut data.builders,
            first: width.is_none(),
        })? {
            let expected = *width.get_or_insert(got);
            if got != expected {
                return Err(de::Error::custom(FrameError::RowWidth {
                    row: data.n_rows,
                    got,
                    expected,
                }));
            }
            data.n_rows += 1;
        }
        Ok(data)
    }
}

/// Seed for one row: pushes its cells into the builders and returns how many there were.
struct RowSeed<'a> {
    builders: &'a mut Vec<Builder>,
    /// First row of a frame whose width is not known yet: grow the builders as needed.
    first: bool,
}

impl<'de> DeserializeSeed<'de> for RowSeed<'_> {
    type Value = usize;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<usize, D::Error> {
        d.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for RowSeed<'_> {
    type Value = usize;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a row (list of cells)")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<usize, A::Error> {
        let mut i = 0;
        loop {
            if self.first && i >= self.builders.len() {
                // Only grow if there really is another cell.
                let mut probe = Builder::Empty(0);
                if seq.next_element_seed(CellSeed(Some(&mut probe)))?.is_none() {
                    break;
                }
                self.builders.push(probe);
            } else if seq
                .next_element_seed(CellSeed(self.builders.get_mut(i)))?
                .is_none()
            {
                // Extra cells are counted (and reported as a width error) but not stored.
                break;
            }
            i += 1;
        }
        Ok(i)
    }
}

/// The frame visitor: `columns`, `data` and optional `schema` in any order; other keys ignored.
pub(crate) struct FrameVisitor;

impl<'de> Visitor<'de> for FrameVisitor {
    type Value = DataFrame;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(r#"a chapkit DataFrame {"columns": [...], "data": [[...], ...]}"#)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<DataFrame, A::Error> {
        let (mut columns, mut data, mut schema): (
            Option<Vec<String>>,
            Option<Data>,
            Option<WireSchema>,
        ) = (None, None, None);
        let mut seen_schema = false;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "columns" if columns.is_some() => {
                    return Err(de::Error::duplicate_field("columns"));
                }
                "data" if data.is_some() => return Err(de::Error::duplicate_field("data")),
                "schema" if seen_schema => return Err(de::Error::duplicate_field("schema")),
                "columns" => columns = Some(map.next_value()?),
                "data" => {
                    data = Some(map.next_value_seed(DataSeed(columns.as_ref().map(Vec::len)))?);
                }
                "schema" => {
                    seen_schema = true;
                    schema = map.next_value()?;
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        let columns = columns.ok_or_else(|| de::Error::missing_field("columns"))?;
        let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
        assemble(columns, data, schema.as_ref()).map_err(de::Error::custom)
    }

    /// The positional form `[columns, data, schema?]`, which serde's derived decoders accept too.
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<DataFrame, A::Error> {
        let columns: Vec<String> = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let data = seq
            .next_element_seed(DataSeed(Some(columns.len())))?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let schema: Option<WireSchema> = seq.next_element()?.flatten();
        if seq.next_element::<IgnoredAny>()?.is_some() {
            return Err(de::Error::invalid_length(4, &self));
        }
        assemble(columns, data, schema.as_ref()).map_err(de::Error::custom)
    }
}

fn assemble(
    columns: Vec<String>,
    data: Data,
    schema: Option<&WireSchema>,
) -> Result<DataFrame, FrameError> {
    let Data {
        mut builders,
        n_rows,
    } = data;
    // `data` came before `columns` (or had no rows): reconcile the width now.
    if builders.len() != columns.len() {
        if n_rows > 0 {
            return Err(FrameError::RowWidth {
                row: 0,
                got: builders.len(),
                expected: columns.len(),
            });
        }
        builders = columns.iter().map(|_| Builder::Empty(0)).collect();
    }
    let types: HashMap<&str, DType> = schema
        .map(|s| {
            s.fields
                .iter()
                .map(|f| (f.name.as_str(), DType::parse(&f.r#type)))
                .collect()
        })
        .unwrap_or_default();
    let mut df = DataFrame::new();
    for (name, b) in columns.into_iter().zip(builders) {
        let dtype = types.get(name.as_str()).copied();
        df.push_column(name, b.finish(dtype))?;
    }
    df.set_rows(n_rows);
    Ok(df)
}

#[cfg(test)]
mod tests {
    use crate::DataFrame;
    use crate::frame::Wire;
    use serde_json::json;

    /// The streaming decoder must give exactly what the previous decoder gave.
    #[test]
    fn matches_the_reference_decoder() {
        let big = u64::MAX;
        let cases = [
            json!({"columns": ["a", "b"], "data": [[1, "x"], [2, "y"]]}),
            json!({"columns": ["a"], "data": [[1], [2.5], [null]]}),
            json!({"columns": ["a"], "data": [[null], [null], [3]]}),
            json!({"columns": ["a"], "data": [[null], [null]]}),
            json!({"columns": ["a"], "data": [[1], ["x"], [true]]}),
            json!({"columns": ["a"], "data": [["x"], [null], [2]]}),
            json!({"columns": ["a"], "data": [[true], [false], [null]]}),
            json!({"columns": ["a"], "data": [[[1, 2]], [{"k": 1}], [null]]}),
            json!({"columns": ["a"], "data": [[null], [[1]], [3]]}),
            json!({"columns": ["a"], "data": [[1], [big], [-3]]}),
            json!({"columns": ["a"], "data": [[big], [big]]}),
            json!({"columns": ["a", "b"], "data": []}),
            json!({"columns": [], "data": [[], []]}),
            json!({"data": [[1, "x"]], "columns": ["a", "b"]}),
            json!({"columns": ["a"], "data": [[1]], "extra": {"ignored": true}}),
            // Schema types, matching and not matching the cells.
            json!({"columns": ["a", "b"], "data": [[1, "x"]],
                   "schema": {"fields": [{"name": "a", "type": "number"}, {"name": "b", "type": "string"}]}}),
            json!({"columns": ["a"], "data": [[1.5], [2]], "schema": {"fields": [{"name": "a", "type": "integer"}]}}),
            json!({"columns": ["a"], "data": [[null]], "schema": {"fields": [{"name": "a", "type": "integer"}]}}),
            json!({"columns": ["a"], "data": [[null]], "schema": {"fields": [{"name": "a", "type": "string"}]}}),
            json!({"columns": ["a"], "data": [[null]], "schema": {"fields": [{"name": "a", "type": "boolean"}]}}),
            json!({"columns": ["a"], "data": [["x"], [1]], "schema": {"fields": [{"name": "a", "type": "number"}]}}),
            json!({"columns": ["a"], "data": [[1], [2]], "schema": {"fields": [{"name": "a", "type": "string"}]}}),
            json!({"columns": ["a"], "data": [[1], [true]], "schema": {"fields": [{"name": "a", "type": "boolean"}]}}),
            json!({"columns": ["a"], "data": [[1], ["x"]], "schema": {"fields": [{"name": "a", "type": "any"}]}}),
            json!({"columns": ["a"], "data": [[1]], "schema": {"fields": [{"name": "zz", "type": "string"}]}}),
            // Errors: ragged rows, duplicate names, missing keys.
            json!({"columns": ["a", "b"], "data": [[1, 2], [3]]}),
            json!({"columns": ["a"], "data": [[1, 2]]}),
            json!({"data": [[1, 2]], "columns": ["a"]}),
            json!({"columns": ["a", "a"], "data": [[1, 2]]}),
            json!({"columns": ["a"]}),
            json!({"data": [[1]]}),
        ];
        for case in cases {
            let new = serde_json::from_value::<DataFrame>(case.clone());
            // Requests arrive as bytes: that path must agree with the in-memory one.
            let bytes = serde_json::to_vec(&case).unwrap();
            match (&new, serde_json::from_slice::<DataFrame>(&bytes)) {
                (Ok(a), Ok(b)) => assert_eq!(a, &b, "{case} from bytes"),
                (Err(_), Err(_)) => {}
                (a, b) => panic!("{case}: from_value {a:?} vs from_slice {b:?}"),
            }
            let old = serde_json::from_value::<Wire>(case.clone())
                .map_err(|e| e.to_string())
                .and_then(|w| DataFrame::from_wire_reference(w).map_err(|e| e.to_string()));
            match (new, old) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a, b, "{case}");
                    assert_eq!(a.n_rows(), b.n_rows(), "{case}");
                    for ((_, ca), (_, cb)) in a.iter().zip(b.iter()) {
                        assert_eq!(ca.dtype(), cb.dtype(), "{case}");
                    }
                }
                (Err(_), Err(_)) => {}
                (a, b) => panic!("{case}: new {a:?} vs reference {b:?}"),
            }
        }
    }

    /// Inputs that only exist as raw text: duplicate keys, the positional form.
    #[test]
    fn matches_the_reference_on_raw_json() {
        let cases = [
            r#"{"columns":["a"],"data":[[1]],"data":[[2]]}"#,
            r#"{"columns":["a"],"columns":["a"],"data":[[1]]}"#,
            r#"{"columns":["a"],"data":[[1]],"schema":null,"schema":null}"#,
            r#"[["a"],[[1]],null]"#,
            r#"[["a"],[[1]]]"#,
            r#"[["a"]]"#,
            r#"[["a"],[[1]],null,1]"#,
        ];
        for raw in cases {
            let new = serde_json::from_str::<DataFrame>(raw);
            let old = serde_json::from_str::<Wire>(raw)
                .map_err(|e| e.to_string())
                .and_then(|w| DataFrame::from_wire_reference(w).map_err(|e| e.to_string()));
            match (new, old) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "{raw}"),
                (Err(_), Err(_)) => {}
                (a, b) => panic!("{raw}: new {a:?} vs reference {b:?}"),
            }
        }
    }

    #[test]
    fn decodes_from_a_byte_stream_too() {
        let bytes = br#"{"columns":["time_period","location","y"],"data":[["2024-01","A",1],["2024-02","A",null]]}"#;
        let df: DataFrame = serde_json::from_slice(bytes).unwrap();
        assert_eq!(df.n_rows(), 2);
        assert_eq!(df.require("location").unwrap().str_at(1), Some("A"));
        assert_eq!(df.f64_column("y").unwrap()[0], 1.0);
    }
}
