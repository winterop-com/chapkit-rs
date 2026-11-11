//! Serde for floats that may be `NaN` or infinite (residuals of missing observations, undefined
//! criteria). JSON has no such numbers, and serde_json writes them as `null`, which does not read
//! back into `f64`; here they are the strings `"NaN"`, `"inf"` and `"-inf"`, and `null` reads as
//! `NaN` (state written before).

use std::fmt;

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer};

#[derive(Clone, Copy)]
struct Float(f64);

impl Serialize for Float {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            x if x.is_nan() => s.serialize_str("NaN"),
            f64::INFINITY => s.serialize_str("inf"),
            f64::NEG_INFINITY => s.serialize_str("-inf"),
            x => s.serialize_f64(x),
        }
    }
}

struct FloatVisitor;

impl Visitor<'_> for FloatVisitor {
    type Value = Float;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(r#"a number, "NaN", "inf", "-inf" or null"#)
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Float, E> {
        Ok(Float(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Float, E> {
        Ok(Float(v as f64))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Float, E> {
        Ok(Float(v as f64))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Float, E> {
        match v {
            "NaN" => Ok(Float(f64::NAN)),
            "inf" => Ok(Float(f64::INFINITY)),
            "-inf" => Ok(Float(f64::NEG_INFINITY)),
            _ => Err(E::invalid_value(de::Unexpected::Str(v), &self)),
        }
    }

    fn visit_unit<E: de::Error>(self) -> Result<Float, E> {
        Ok(Float(f64::NAN))
    }

    fn visit_none<E: de::Error>(self) -> Result<Float, E> {
        Ok(Float(f64::NAN))
    }
}

impl<'de> Deserialize<'de> for Float {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Float, D::Error> {
        d.deserialize_any(FloatVisitor)
    }
}

/// One float (`#[serde(with = "crate::serde_float::one")]`).
pub(crate) mod one {
    use super::{Deserialize, Deserializer, Float, Serialize, Serializer};

    #[allow(clippy::trivially_copy_pass_by_ref)] // serde's `with` passes a reference
    pub(crate) fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        Float(*v).serialize(s)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Float::deserialize(d).map(|f| f.0)
    }
}

/// A vector of floats (`#[serde(with = "crate::serde_float::many")]`).
pub(crate) mod many {
    use super::{Deserialize, Deserializer, Float, Serializer};

    pub(crate) fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(v.iter().map(|x| Float(*x)))
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        Vec::<Float>::deserialize(d).map(|v| v.into_iter().map(|f| f.0).collect())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn nonfinite_floats_round_trip() {
        let v = [1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0];
        let json = serde_json::to_string(&v.map(Float)).unwrap();
        assert_eq!(json, r#"[1.5,"NaN","inf","-inf",-0.0]"#);
        let back: Vec<Float> = serde_json::from_str(&json).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let back: Vec<f64> = back.into_iter().map(|f| f.0).collect();
        assert_eq!(bits(&back[..1]), bits(&v[..1]));
        assert!(back[1].is_nan());
        assert_eq!(bits(&back[2..]), bits(&v[2..]));
        // `null`, as serde_json wrote NaN before, and integers.
        let old: Vec<Float> = serde_json::from_str("[null, 2]").unwrap();
        assert!(old[0].0.is_nan() && old[1].0 == 2.0);
        assert!(serde_json::from_str::<Vec<Float>>(r#"["x"]"#).is_err());
    }
}
