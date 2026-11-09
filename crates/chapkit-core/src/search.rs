//! Hyperparameter search spaces in chap-core's format
//! (`chap_core/hpo/search_space.py`), so `hpo_search_space` in `/api/v1/info`
//! is read by chap-core unchanged.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};

/// One hyperparameter's domain.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    /// Pick one of a fixed list of values (`{"values": [...]}`).
    Categorical(Vec<Value>),
    /// Integers in `[low, high]`, on a `step` grid, optionally log-uniform.
    Int {
        low: i64,
        high: i64,
        step: i64,
        log: bool,
    },
    /// Floats in `[low, high]`, continuous or on a `step` grid, optionally log-uniform.
    Float {
        low: f64,
        high: f64,
        step: Option<f64>,
        log: bool,
    },
}

impl Param {
    /// Check chap-core's constraints on this domain.
    pub fn validate(&self, name: &str) -> Result<(), String> {
        match *self {
            Param::Categorical(ref v) if v.is_empty() => {
                Err(format!("'{name}': 'values' must be a non-empty list"))
            }
            Param::Categorical(_) => Ok(()),
            Param::Int {
                low,
                high,
                step,
                log,
            } => {
                if low >= high {
                    return Err(format!("Int('{name}'): low < high required"));
                }
                if step < 1 {
                    return Err(format!("Int('{name}'): step must be >= 1"));
                }
                if log && (step != 1 || low <= 0) {
                    return Err(format!(
                        "Int('{name}'): log=True requires step == 1 and low > 0"
                    ));
                }
                Ok(())
            }
            Param::Float {
                low,
                high,
                step,
                log,
            } => {
                if low >= high {
                    return Err(format!("Float('{name}'): low < high required"));
                }
                if log && (step.is_some() || low <= 0.0) {
                    return Err(format!(
                        "Float('{name}'): log=True requires no step and low > 0"
                    ));
                }
                if step.is_some_and(|s| s <= 0.0) {
                    return Err(format!("Float('{name}'): step must be > 0"));
                }
                Ok(())
            }
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Param::Categorical(v) => json!({ "values": v }),
            Param::Int {
                low,
                high,
                step,
                log,
            } => {
                json!({ "type": "int", "low": low, "high": high, "step": step, "log": log })
            }
            Param::Float {
                low,
                high,
                step,
                log,
            } => {
                let mut m = Map::new();
                m.insert("type".into(), "float".into());
                m.insert("low".into(), json!(low));
                m.insert("high".into(), json!(high));
                if let Some(s) = step {
                    m.insert("step".into(), json!(s));
                }
                m.insert("log".into(), json!(log));
                Value::Object(m)
            }
        }
    }

    fn from_json(name: &str, v: &Value) -> Result<Param, String> {
        let spec = v
            .as_object()
            .ok_or_else(|| format!("'{name}': each spec must be a mapping"))?;
        if let Some(values) = spec.get("values") {
            let values = values
                .as_array()
                .ok_or_else(|| format!("'{name}': 'values' must be a list"))?;
            return Ok(Param::Categorical(values.clone()));
        }
        let (Some(low), Some(high)) = (spec.get("low"), spec.get("high")) else {
            return Err(format!("'{name}': expected 'low' and 'high'"));
        };
        let ty = spec
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_lowercase);
        let log = spec.get("log").and_then(Value::as_bool).unwrap_or(false);
        let step = spec.get("step").filter(|s| !s.is_null());
        let is_int = match ty.as_deref() {
            Some("int") => true,
            None => low.is_i64() && high.is_i64(),
            Some("float") => false,
            Some(other) => return Err(format!("'{name}': unknown spec type '{other}'")),
        };
        let num = |v: &Value| {
            v.as_f64()
                .ok_or_else(|| format!("'{name}': bounds must be numbers"))
        };
        Ok(if is_int {
            Param::Int {
                low: num(low)? as i64,
                high: num(high)? as i64,
                step: step.map(num).transpose()?.map_or(1, |s| s as i64),
                log,
            }
        } else {
            Param::Float {
                low: num(low)?,
                high: num(high)?,
                step: step.map(num).transpose()?,
                log,
            }
        })
    }
}

/// A named set of hyperparameter domains, keyed by config field name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchSpace(pub BTreeMap<String, Param>);

impl SearchSpace {
    /// An empty space.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter (builder style).
    #[must_use]
    pub fn with(mut self, name: impl Into<String>, p: Param) -> Self {
        self.0.insert(name.into(), p);
        self
    }

    /// Validate every parameter.
    pub fn validate(&self) -> Result<(), String> {
        if self.0.is_empty() {
            return Err("search_space must be a non-empty dict".into());
        }
        self.0.iter().try_for_each(|(k, p)| p.validate(k))
    }
}

impl Serialize for SearchSpace {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let m: Map<String, Value> = self
            .0
            .iter()
            .map(|(k, p)| (k.clone(), p.to_json()))
            .collect();
        m.serialize(s)
    }
}

impl<'de> Deserialize<'de> for SearchSpace {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let m = Map::<String, Value>::deserialize(d)?;
        let mut out = BTreeMap::new();
        for (k, v) in &m {
            let p = Param::from_json(k, v).map_err(serde::de::Error::custom)?;
            p.validate(k).map_err(serde::de::Error::custom)?;
            out.insert(k.clone(), p);
        }
        Ok(SearchSpace(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chap_core_format() {
        let s: SearchSpace = serde_json::from_value(json!({
            "window": {"low": 3, "high": 24},
            "alpha": {"low": 0.001, "high": 1.0, "log": true},
            "kind": {"values": ["a", "b"]},
            "beta": {"low": 1, "high": 2, "type": "float"}
        }))
        .unwrap();
        assert_eq!(
            s.0["window"],
            Param::Int {
                low: 3,
                high: 24,
                step: 1,
                log: false
            }
        );
        assert_eq!(
            s.0["alpha"],
            Param::Float {
                low: 0.001,
                high: 1.0,
                step: None,
                log: true
            }
        );
        assert_eq!(
            s.0["beta"],
            Param::Float {
                low: 1.0,
                high: 2.0,
                step: None,
                log: false
            }
        );
        assert!(matches!(s.0["kind"], Param::Categorical(_)));
        let round: SearchSpace = serde_json::from_value(serde_json::to_value(&s).unwrap()).unwrap();
        assert_eq!(round, s);
    }

    #[test]
    fn rejects_bad_specs() {
        assert!(
            serde_json::from_value::<SearchSpace>(json!({"a": {"low": 5, "high": 1}})).is_err()
        );
        assert!(
            serde_json::from_value::<SearchSpace>(
                json!({"a": {"low": 0.0, "high": 1.0, "log": true}})
            )
            .is_err()
        );
        assert!(serde_json::from_value::<SearchSpace>(json!({"a": {"values": []}})).is_err());
    }
}
