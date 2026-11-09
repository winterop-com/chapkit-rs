//! Type erasure: lets the HTTP layer drive any [`Model`] through one object-safe trait.
//!
//! Configs travel as JSON (as stored and sent by Chap) and are parsed into the
//! model's typed `Config` per call; trained state is kept as `Arc<dyn Any>` so the
//! warm-model cache can hold any model's `Trained` type.

use std::any::Any;
use std::sync::Arc;

use chapkit_core::{
    Checkpoint, DataFrame, Error, Model, ModelInfo, PredictInput, Result, SearchSpace, TrainInput,
};
use serde_json::{Map, Value, json};

/// Trained state of some model, shared across requests.
pub type Trained = Arc<dyn Any + Send + Sync>;

/// Keys chapkit reserves in every config's `data`.
pub const RESERVED_KEYS: [&str; 2] = ["prediction_periods", "additional_continuous_covariates"];

/// Default forecast horizon when neither the request nor the config sets one.
pub const DEFAULT_PREDICTION_PERIODS: u64 = 3;

/// Object-safe view of a [`Model`].
pub trait DynModel: Send + Sync + 'static {
    /// Model info.
    fn info(&self) -> ModelInfo;
    /// Search space.
    fn search_space(&self) -> Option<SearchSpace>;
    /// JSON schema of the config `data`, with chapkit's reserved keys added.
    fn config_schema(&self) -> Value;
    /// Flatten `user_option_values`, add reserved defaults, validate against the
    /// model's config type and fill its defaults. Keys the caller sent are kept as sent.
    fn normalize_config(&self, data: Value) -> std::result::Result<Value, String>;
    /// Train; returns the trained state and its checkpoint (provenance not yet filled).
    fn train(&self, config: &Value, input: TrainInput<'_>) -> Result<(Trained, Checkpoint)>;
    /// Load trained state from a checkpoint.
    fn load(&self, checkpoint: &Checkpoint) -> Result<Trained>;
    /// Predict with trained state from [`DynModel::train`] or [`DynModel::load`].
    fn predict(
        &self,
        trained: &Trained,
        config: &Value,
        input: PredictInput<'_>,
    ) -> Result<DataFrame>;
    /// Whether locations are forecast independently.
    fn locations_independent(&self) -> bool;
}

/// Wraps a typed model.
pub struct Erased<M>(pub M);

fn parse_config<M: Model>(config: &Value) -> Result<M::Config> {
    M::Config::deserialize(config).map_err(|e| Error::Config(e.to_string()))
}

use serde::Deserialize as _;

impl<M: Model> DynModel for Erased<M> {
    fn info(&self) -> ModelInfo {
        self.0.info()
    }

    fn search_space(&self) -> Option<SearchSpace> {
        self.0.search_space()
    }

    fn config_schema(&self) -> Value {
        let mut schema =
            serde_json::to_value(schemars::schema_for!(M::Config)).unwrap_or_else(|_| json!({}));
        if let Value::Object(obj) = &mut schema {
            obj.remove("$schema");
            obj.insert("type".into(), json!("object"));
            let props = obj.entry("properties").or_insert_with(|| json!({}));
            if let Value::Object(props) = props {
                props.entry("prediction_periods").or_insert_with(|| {
                    json!({"type": "integer", "title": "Prediction Periods", "default": DEFAULT_PREDICTION_PERIODS,
                           "description": "Number of periods to forecast"})
                });
                props.entry("additional_continuous_covariates").or_insert_with(|| {
                    json!({"type": "array", "items": {"type": "string"}, "title": "Additional Continuous Covariates",
                           "default": [], "description": "Extra covariate columns to use"})
                });
            }
        }
        schema
    }

    fn normalize_config(&self, data: Value) -> std::result::Result<Value, String> {
        let Value::Object(mut obj) = data else {
            return Err("config data must be a JSON object".into());
        };
        // chap-core style nesting: user options under `user_option_values`. Flat keys win.
        if let Some(Value::Object(nested)) = obj.remove("user_option_values") {
            for (k, v) in nested {
                obj.entry(k).or_insert(v);
            }
        }
        obj.entry("prediction_periods")
            .or_insert_with(|| json!(DEFAULT_PREDICTION_PERIODS));
        obj.entry("additional_continuous_covariates")
            .or_insert_with(|| json!([]));
        match obj.get("prediction_periods") {
            Some(Value::Number(n)) if n.as_u64().is_some() => {}
            _ => return Err("prediction_periods must be a non-negative integer".into()),
        }
        if !obj
            .get("additional_continuous_covariates")
            .is_some_and(|v| v.as_array().is_some_and(|a| a.iter().all(Value::is_string)))
        {
            return Err("additional_continuous_covariates must be a list of strings".into());
        }
        let data = Value::Object(obj);
        let typed: M::Config = M::Config::deserialize(&data).map_err(|e| e.to_string())?;
        let Value::Object(mut out) = data else {
            return Err("config data must be a JSON object".into());
        };
        // Fill the model's defaults for keys the caller left out; never touch keys they sent.
        if let Ok(Value::Object(defaults)) = serde_json::to_value(&typed) {
            for (k, v) in defaults {
                out.entry(k).or_insert(v);
            }
        }
        Ok(Value::Object(out))
    }

    fn train(&self, config: &Value, input: TrainInput<'_>) -> Result<(Trained, Checkpoint)> {
        let cfg = parse_config::<M>(config)?;
        let trained = self.0.train(&cfg, input)?;
        let checkpoint = self.0.save(&trained)?;
        Ok((Arc::new(trained), checkpoint))
    }

    fn load(&self, checkpoint: &Checkpoint) -> Result<Trained> {
        Ok(Arc::new(self.0.load(checkpoint)?))
    }

    fn predict(
        &self,
        trained: &Trained,
        config: &Value,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let cfg = parse_config::<M>(config)?;
        let t = trained
            .downcast_ref::<M::Trained>()
            .ok_or_else(|| Error::model("trained state belongs to a different model"))?;
        self.0.predict(t, &cfg, input)
    }

    fn locations_independent(&self) -> bool {
        self.0.locations_independent()
    }
}

/// Remove `content` from an artifact's data (list views return summaries).
pub fn without_content(data: &Value) -> Value {
    match data {
        Value::Object(m) => Value::Object(
            m.iter()
                .filter(|(k, _)| *k != "content")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<Map<_, _>>(),
        ),
        other => other.clone(),
    }
}
