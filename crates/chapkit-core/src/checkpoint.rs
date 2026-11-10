//! Model checkpoints in the [safetensors](https://github.com/huggingface/safetensors) format.
//!
//! A checkpoint holds a model's trained state: numeric weights as tensors, and any
//! other state as JSON in the safetensors `__metadata__` header. It also records
//! provenance (which model, which data, trained up to which period), so a
//! pre-trained model can be checked for overlap with a backtest's test periods.
//!
//! Safetensors files cannot execute code on load (unlike pickle) and are read by
//! the Python, Rust and JS safetensors libraries alike.

use std::collections::{BTreeMap, HashMap};

use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DataFrame, Error, Period, Result, cols};

/// Format tag written to every checkpoint header.
pub const FORMAT: &str = "chapkit-checkpoint/1";

const KEY_FORMAT: &str = "chapkit.format";
const KEY_STATE: &str = "chapkit.state";
const KEY_PROVENANCE: &str = "chapkit.provenance";

/// Tensor element storage.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    /// 32-bit floats.
    F32(Vec<f32>),
    /// 64-bit floats.
    F64(Vec<f64>),
    /// 64-bit integers.
    I64(Vec<i64>),
}

impl TensorData {
    fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::F64(v) => v.len(),
            TensorData::I64(v) => v.len(),
        }
    }

    fn dtype(&self) -> Dtype {
        match self {
            TensorData::F32(_) => Dtype::F32,
            TensorData::F64(_) => Dtype::F64,
            TensorData::I64(_) => Dtype::I64,
        }
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    fn from_le_bytes(dtype: Dtype, bytes: &[u8]) -> Result<Self> {
        fn chunks<const N: usize, T>(bytes: &[u8], f: impl Fn([u8; N]) -> T) -> Vec<T> {
            bytes.as_chunks::<N>().0.iter().map(|c| f(*c)).collect()
        }
        Ok(match dtype {
            Dtype::F32 => TensorData::F32(chunks(bytes, f32::from_le_bytes)),
            Dtype::F64 => TensorData::F64(chunks(bytes, f64::from_le_bytes)),
            Dtype::I64 => TensorData::I64(chunks(bytes, i64::from_le_bytes)),
            other => return Err(Error::Serde(format!("unsupported tensor dtype {other:?}"))),
        })
    }
}

/// A named n-dimensional array.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Dimensions; their product is the number of elements.
    pub shape: Vec<usize>,
    /// Row-major elements.
    pub data: TensorData,
}

impl Tensor {
    /// A tensor, checking that `shape` matches the number of elements.
    pub fn new(shape: Vec<usize>, data: TensorData) -> Result<Self> {
        if shape.iter().product::<usize>() != data.len() {
            return Err(Error::invalid(format!(
                "tensor shape {shape:?} does not match {} elements",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// A 1-D `f64` tensor.
    pub fn vector(values: Vec<f64>) -> Self {
        Self {
            shape: vec![values.len()],
            data: TensorData::F64(values),
        }
    }

    /// The elements as `f64`, whatever the stored type.
    pub fn to_f64(&self) -> Vec<f64> {
        match &self.data {
            TensorData::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            TensorData::F64(v) => v.clone(),
            TensorData::I64(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }
}

/// Where a checkpoint came from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Model (service) id.
    pub model: String,
    /// Model version.
    pub model_version: String,
    /// chapkit-rs version that wrote the checkpoint.
    pub chapkit_version: String,
    /// When the checkpoint was written (RFC 3339, UTC).
    pub created_at: Option<String>,
    /// Last period in the training data (chap period id). A backtest whose test
    /// periods start at or before this period overlaps the training data.
    pub trained_until: Option<String>,
    /// Locations in the training data.
    #[serde(default)]
    pub locations: Vec<String>,
    /// Free-form description of the training data sources.
    #[serde(default)]
    pub data_sources: Vec<String>,
    /// The config the model was trained with.
    #[serde(default)]
    pub config: Value,
}

impl Provenance {
    /// Fill `trained_until` and `locations` from training data.
    pub fn with_training_data(mut self, data: &DataFrame) -> Self {
        self.trained_until = last_period(data).map(|p| p.to_string());
        self.locations = data
            .group_indices(cols::LOCATION)
            .map(|g| g.into_iter().map(|(loc, _)| loc).collect())
            .unwrap_or_default();
        self
    }

    /// True when a test period starting at `first_test_period` is covered by the
    /// training data. Unknown or unparseable periods are treated as overlapping,
    /// since the safe answer for an evaluation is "cannot rule it out".
    pub fn overlaps(&self, first_test_period: &str) -> bool {
        match (
            self.trained_until.as_deref().and_then(Period::parse),
            Period::parse(first_test_period),
        ) {
            (Some(until), Some(test)) => test <= until,
            _ => true,
        }
    }
}

/// The latest parseable period in `data`'s `time_period` column.
pub fn last_period(data: &DataFrame) -> Option<Period> {
    let col = data.column(cols::TIME_PERIOD)?;
    (0..data.n_rows())
        .filter_map(|r| col.str_at(r).and_then(Period::parse))
        .max()
}

/// A model checkpoint: tensors, JSON state and provenance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Checkpoint {
    /// Named weights.
    pub tensors: BTreeMap<String, Tensor>,
    /// Non-tensor state as JSON.
    pub state: Value,
    /// Where it came from.
    pub provenance: Provenance,
}

impl Checkpoint {
    /// A checkpoint holding `state` as JSON and no tensors. The default for models
    /// whose trained state is small or not numeric.
    pub fn from_state<T: Serialize>(state: &T) -> Result<Self> {
        let state = serde_json::to_value(state).map_err(|e| Error::Serde(e.to_string()))?;
        Ok(Self {
            state,
            ..Self::default()
        })
    }

    /// Read the JSON state back as `T`.
    pub fn state<T: DeserializeOwned>(&self) -> Result<T> {
        T::deserialize(&self.state).map_err(|e| Error::Serde(e.to_string()))
    }

    /// Add a tensor (builder style).
    #[must_use]
    pub fn with_tensor(mut self, name: impl Into<String>, t: Tensor) -> Self {
        self.tensors.insert(name.into(), t);
        self
    }

    /// Borrow a tensor, or fail if it is missing.
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Serde(format!("checkpoint has no tensor '{name}'")))
    }

    /// Serialize to safetensors bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let ser = |e: &dyn std::fmt::Display| Error::Serde(e.to_string());
        let mut meta = HashMap::new();
        meta.insert(KEY_FORMAT.to_owned(), FORMAT.to_owned());
        meta.insert(
            KEY_STATE.to_owned(),
            serde_json::to_string(&self.state).map_err(|e| ser(&e))?,
        );
        meta.insert(
            KEY_PROVENANCE.to_owned(),
            serde_json::to_string(&self.provenance).map_err(|e| ser(&e))?,
        );
        let bytes: Vec<(&String, Dtype, &Vec<usize>, Vec<u8>)> = self
            .tensors
            .iter()
            .map(|(n, t)| (n, t.data.dtype(), &t.shape, t.data.to_le_bytes()))
            .collect();
        let views = bytes
            .iter()
            .map(|(n, dtype, shape, b)| {
                Ok((
                    (*n).clone(),
                    TensorView::new(*dtype, (*shape).clone(), b).map_err(|e| ser(&e))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        safetensors::serialize(views, Some(meta)).map_err(|e| ser(&e))
    }

    /// Parse safetensors bytes written by [`Checkpoint::to_bytes`]. Files from other
    /// tools load too: their tensors are kept, and state and provenance stay empty.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let ser = |e: &dyn std::fmt::Display| Error::Serde(e.to_string());
        let (_, header) = SafeTensors::read_metadata(bytes).map_err(|e| ser(&e))?;
        let st = SafeTensors::deserialize(bytes).map_err(|e| ser(&e))?;
        let meta = header.metadata().clone().unwrap_or_default();
        if let Some(format) = meta.get(KEY_FORMAT)
            && format != FORMAT
        {
            return Err(Error::Serde(format!(
                "unsupported checkpoint format '{format}'"
            )));
        }
        let state = match meta.get(KEY_STATE) {
            Some(s) => serde_json::from_str(s).map_err(|e| ser(&e))?,
            None => Value::Null,
        };
        let provenance = match meta.get(KEY_PROVENANCE) {
            Some(s) => serde_json::from_str(s).map_err(|e| ser(&e))?,
            None => Provenance::default(),
        };
        let mut tensors = BTreeMap::new();
        for (name, view) in st.tensors() {
            let data = TensorData::from_le_bytes(view.dtype(), view.data())?;
            tensors.insert(name, Tensor::new(view.shape().to_vec(), data)?);
        }
        Ok(Self {
            tensors,
            state,
            provenance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Column;
    use serde_json::json;

    #[test]
    fn round_trips_tensors_state_and_provenance() {
        let ck = Checkpoint::from_state(&json!({"alpha": 0.5, "features": ["rain"]}))
            .unwrap()
            .with_tensor("w", Tensor::vector(vec![1.0, -2.5, 3.25]))
            .with_tensor(
                "m",
                Tensor::new(vec![2, 2], TensorData::F32(vec![1.0, 2.0, 3.0, 4.0])).unwrap(),
            )
            .with_tensor(
                "i",
                Tensor::new(vec![2], TensorData::I64(vec![7, -7])).unwrap(),
            );
        let ck = Checkpoint {
            provenance: Provenance {
                model: "demo".into(),
                trained_until: Some("2021-04".into()),
                ..Provenance::default()
            },
            ..ck
        };
        let bytes = ck.to_bytes().unwrap();
        assert_eq!(Checkpoint::from_bytes(&bytes).unwrap(), ck);
    }

    #[test]
    fn rejects_bad_shapes_and_garbage() {
        assert!(Tensor::new(vec![2, 2], TensorData::F64(vec![1.0])).is_err());
        assert!(Checkpoint::from_bytes(b"not a checkpoint").is_err());
    }

    #[test]
    fn provenance_from_training_data_and_overlap() {
        let df = DataFrame::from_columns([
            (
                "time_period",
                Column::from_strs(["2021-03", "2021-04", "2021-02"]),
            ),
            ("location", Column::from_strs(["A", "A", "B"])),
        ])
        .unwrap();
        let p = Provenance::default().with_training_data(&df);
        assert_eq!(p.trained_until.as_deref(), Some("2021-04"));
        assert_eq!(p.locations, vec!["A", "B"]);
        assert!(p.overlaps("2021-04"));
        assert!(p.overlaps("2020-12"));
        assert!(!p.overlaps("2021-05"));
        assert!(p.overlaps("garbage"));
    }
}
