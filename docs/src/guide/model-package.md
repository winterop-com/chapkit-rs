# Writing a model package

A model package is a Rust crate that depends on `chapkit`, implements `Model`, and has a `main` that serves
it. The packages in `chapkit-rs-examples` are complete examples: `baselines` (three models, one binary
each) and `ridge`.

## Cargo.toml

```toml
[package]
name = "my-model"
edition = "2024"

[dependencies]
chapkit = { path = "../chapkit-rs" }
serde = { version = "1", features = ["derive"] }
schemars = "1"
tokio = { version = "1", features = ["full"] }
```

## The model

```rust,ignore
use chapkit::{DataFrame, Model, ModelInfo, PredictInput, Result, TrainInput, cols, prediction_frame};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-location mean, scaled by an option.
struct MeanModel;

/// User options. The JSON schema becomes GET /api/v1/configs/$schema.
/// Do not use deny_unknown_fields: configs also carry chapkit's reserved keys.
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct MeanConfig {
    #[serde(default = "one")]
    scale: f64,
}
fn one() -> f64 { 1.0 }

impl Model for MeanModel {
    type Config = MeanConfig;
    type Trained = HashMap<String, f64>; // stored as JSON in the checkpoint by default

    fn info(&self) -> ModelInfo {
        ModelInfo::default() // target disease_cases, any period type, 0..=100 periods
    }

    fn train(&self, _: &MeanConfig, input: TrainInput<'_>) -> Result<Self::Trained> {
        let y = input.data.f64_column(cols::DISEASE_CASES)?;
        Ok(input.data.group_indices(cols::LOCATION)?
            .into_iter()
            .map(|(loc, rows)| (loc, rows.iter().map(|&i| y[i]).sum::<f64>() / rows.len() as f64))
            .collect())
    }

    fn predict(&self, means: &Self::Trained, cfg: &MeanConfig, input: PredictInput<'_>) -> Result<DataFrame> {
        let loc = input.future.require(cols::LOCATION)?;
        // time_period, location, sample_0..99: one row per future row
        prediction_frame(input.future, 100, |row, samples| {
            let m = loc.str_at(row).and_then(|l| means.get(l)).copied().unwrap_or(0.0);
            samples.fill(m * cfg.scale);
            Ok(())
        })
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    chapkit::Service::new("mean", "Mean", MeanModel)
        .description("Per-location mean")
        .serve()
        .await
}
```

## What you get

Running the binary serves the full chapkit API on `127.0.0.1:8000`: configs, artifacts, jobs, `$train`,
`$predict`, `$validate`, `$predict-sync`, `$explain`, `$generate-sample-data`, health, info, system,
metrics, SSE streams, OpenAPI at `/openapi.json` and docs at `/docs`.

## Tips

- **Statistical models** with nothing to store use `type Trained = ()` and do their work in `predict`
  from `input.historic`.
- **Models with weights** override `Model::save` and `Model::load` to store tensors in the safetensors
  checkpoint; see `ridge`.
- **Hyperparameters** are declared with `Model::search_space` in chap-core's format and published in
  `/api/v1/info` as `hpo_search_space`.
- **Errors**: return `Error::InvalidInput` for data the model cannot handle; the job fails with
  `ValueError: ...`, which chap-core shows to the user.
- **Testing** in-process is a plain function call; `chapkit::synth::Generator` makes realistic data.
