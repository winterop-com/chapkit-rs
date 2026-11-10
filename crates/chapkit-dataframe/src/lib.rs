//! A lightweight columnar `DataFrame` that speaks the chapkit wire format.
//!
//! chapkit exchanges tables as `{"columns": [...], "data": [[row], ...], "schema"?: {...}}`.
//! This crate decodes that into typed columns that are cheap to slice, clone and stack,
//! which matters when explainers build thousands of perturbed inputs.

mod column;
mod decode;
mod frame;
#[cfg(feature = "polars")]
mod polars_conv;

/// The polars crate this conversion is built against (feature `polars`).
#[cfg(feature = "polars")]
pub use polars;

pub use column::{Column, DType};
pub use frame::{DataFrame, FrameError};

/// Well-known chapkit column names.
pub mod cols {
    /// Period column, e.g. `2024-01` (monthly) or `2024W03` (weekly).
    pub const TIME_PERIOD: &str = "time_period";
    /// Location (org unit) id column.
    pub const LOCATION: &str = "location";
    /// Default prediction target.
    pub const DISEASE_CASES: &str = "disease_cases";

    /// Name of the `i`-th sample column in prediction output.
    pub fn sample(i: usize) -> String {
        format!("sample_{i}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn decode(v: serde_json::Value) -> DataFrame {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn infers_types_and_round_trips() {
        let df = decode(json!({
            "columns": ["time_period", "location", "rainfall", "disease_cases", "flag"],
            "data": [
                ["2024-01", "A", 1.5, 3, true],
                ["2024-02", "A", null, 4, false],
                ["2024-01", "B", 2, null, null]
            ]
        }));
        assert_eq!(df.n_rows(), 3);
        assert_eq!(df.column("time_period").unwrap().dtype(), DType::String);
        assert_eq!(df.column("rainfall").unwrap().dtype(), DType::Number);
        assert_eq!(df.column("disease_cases").unwrap().dtype(), DType::Integer);
        assert_eq!(df.column("flag").unwrap().dtype(), DType::Boolean);

        let out = serde_json::to_value(&df).unwrap();
        assert_eq!(out["data"][1], json!(["2024-02", "A", null, 4, false]));
        assert_eq!(out["data"][2][2], json!(2.0));
        assert_eq!(
            out["schema"]["fields"][2],
            json!({"name": "rainfall", "type": "number"})
        );

        let again: DataFrame = serde_json::from_value(out).unwrap();
        assert_eq!(again, df);
    }

    #[test]
    fn schema_overrides_inference() {
        let df = decode(json!({
            "columns": ["x"],
            "data": [[1], [2]],
            "schema": {"fields": [{"name": "x", "type": "number", "title": "X"}]}
        }));
        assert_eq!(df.column("x").unwrap().dtype(), DType::Number);
    }

    #[test]
    fn mixed_cells_become_json() {
        let df = decode(json!({"columns": ["x"], "data": [[1], ["a"]]}));
        assert_eq!(df.column("x").unwrap().dtype(), DType::Any);
    }

    #[test]
    fn rejects_ragged_rows() {
        let r: Result<DataFrame, _> =
            serde_json::from_value(json!({"columns": ["a", "b"], "data": [[1]]}));
        assert!(r.is_err());
    }

    #[test]
    fn nan_is_written_as_null() {
        let df = DataFrame::from_columns([("x", Column::from_f64([f64::NAN, 1.0]))]).unwrap();
        let out = serde_json::to_value(&df).unwrap();
        assert_eq!(out["data"], json!([[null], [1.0]]));
    }

    #[test]
    fn groups_and_stacks() {
        let df = decode(json!({
            "columns": ["location", "v"],
            "data": [["B", 1], ["A", 2], ["B", 3]]
        }));
        let g = df.group_indices("location").unwrap();
        assert_eq!(g, vec![("B".into(), vec![0, 2]), ("A".into(), vec![1])]);
        // Runs of the same key and keys coming back after a gap, plus a null key.
        let runs = decode(json!({
            "columns": ["location"],
            "data": [["A"], ["A"], ["B"], ["B"], ["A"], [null], ["B"]]
        }));
        assert_eq!(
            runs.group_indices("location").unwrap(),
            vec![
                ("A".into(), vec![0, 1, 4]),
                ("B".into(), vec![2, 3, 6]),
                (String::new(), vec![5])
            ]
        );

        let other = decode(json!({"columns": ["location", "v"], "data": [["C", 0.5]]}));
        let s = DataFrame::vstack(&[&df, &other]).unwrap();
        assert_eq!(s.n_rows(), 4);
        assert_eq!(s.column("v").unwrap().dtype(), DType::Number);
        assert_eq!(s.f64_column("v").unwrap(), vec![1.0, 2.0, 3.0, 0.5]);
    }

    #[test]
    fn push_column_checks_length() {
        let mut df = DataFrame::from_columns([("a", Column::from_f64([1.0, 2.0]))]).unwrap();
        assert!(df.push_column("b", Column::from_f64([1.0])).is_err());
        assert!(df.push_column("a", Column::from_f64([1.0, 2.0])).is_err());
    }

    #[test]
    fn set_f64_keeps_the_frame_consistent() {
        let mut df = decode(json!({
            "columns": ["location", "rainfall", "cases"],
            "data": [["A", 1.5, 3], ["B", 2.5, 4]]
        }));
        df.set_f64("rainfall", 1, Some(9.0)).unwrap();
        // An integer column is widened to float.
        df.set_f64("cases", 0, None).unwrap();
        assert!(matches!(
            df.set_f64("rainfall", 2, Some(0.0)),
            Err(FrameError::RowOutOfRange { rows: 2 })
        ));
        assert!(df.set_f64("location", 0, Some(0.0)).is_err());
        assert!(df.set_f64("nope", 0, Some(0.0)).is_err());
        assert_eq!(
            serde_json::to_value(&df).unwrap()["data"],
            json!([["A", 1.5, null], ["B", 9.0, 4.0]])
        );
    }

    #[test]
    fn out_of_range_access_is_an_error_not_a_panic() {
        let df = DataFrame::from_columns([("a", Column::from_f64([1.0, 2.0]))]).unwrap();
        let a = df.column("a").unwrap();
        assert_eq!(a.f64_at(5), None);
        assert_eq!(a.json_at(5), serde_json::Value::Null);
        assert!(df.take(&[0, 2]).is_err());
        assert!(df.filter(&[true]).is_err());
        assert_eq!(
            df.take(&[1, 1]).unwrap().f64_column("a").unwrap(),
            vec![2.0, 2.0]
        );
    }
}
