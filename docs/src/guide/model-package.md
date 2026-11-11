# Writing a model package

A model package is a Rust crate that depends on `chapkit`, implements `Model`, and has a `main` that serves
it. The packages in `chapkit-rs-models` are complete examples: the three `baseline-*` ports of chap-core's
baselines. Each keeps its calculation in `src/model.rs` (plain functions over numbers), the chapkit
plumbing in `src/lib.rs` and the binary in `src/main.rs`; `chapkit::locations` does the per-location splitting,
parallelism and prediction table.

## Cargo.toml

```toml
[package]
name = "my-model"
edition = "2024"

[dependencies]
chapkit = { git = "https://github.com/winterop-com/chapkit-rs", tag = "v0.2.0" }
serde = { version = "1", features = ["derive"] }
schemars = "1"
tokio = { version = "1", features = ["full"] }
```

`chapkit-rs init` writes the dependency pinned to the tag of its own release. To move to a newer release, change the
tag; with a chapkit-rs checkout beside the package, `chapkit = { path = "../chapkit-rs" }` builds against it instead.

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
async fn main() -> std::process::ExitCode {
    chapkit::Service::new("mean", "Mean", MeanModel)
        .description("Per-location mean")
        .serve()
        .await
}
```

## What you get

Running the binary serves the full chapkit API on `127.0.0.1:8000`: configs, artifacts, jobs, `$train`,
`$predict`, `$validate`, `$predict-sync`, `$predict-splits`, `$explain`, `$backtest`, `$hpo`, `$generate-sample-data`, health, info, system,
metrics, SSE streams, OpenAPI at `/openapi.json` and docs at `/docs`.

## Using polars

For pandas/numpy-style work (group-by, joins, lags, rolling windows) enable the `polars` feature and
convert at the edges of `train` and `predict`:

```toml
chapkit = { path = "../chapkit-rs", features = ["polars"] }
polars = { version = "0.55", features = ["lazy"] }   # the polars features your model uses
```

```rust,ignore
use chapkit::polars::prelude::*;

fn train(&self, _: &Self::Config, input: TrainInput<'_>) -> chapkit::Result<Self::Trained> {
    let df = input.data.to_polars()?;
    let means = df
        .lazy()
        .group_by_stable([col("location")])
        .agg([col("disease_cases").mean()])
        .collect()
        .map_err(|e| chapkit::Error::Model(e.to_string()))?;
    // ... build Self::Trained from `means`; DataFrame::from_polars converts back.
}
```

The chapkit `DataFrame` stays the wire type: the JSON chap-core sends is decoded into it, and predictions
are returned as one. Numbers, integers, strings and booleans map to Float64, Int64, String and Boolean;
from polars, other numeric widths are widened and other types become strings. Each conversion copies the
data once, which is small next to decoding the request. polars is optional so small models, and the
library itself, do not pay its compile time and binary size. Your crate's `polars` dependency must be the
same minor version (`chapkit::polars` re-exports the one in use); cargo merges the features.

## Validation

chap-core and the chap modeling app can check a request with `$validate` before submitting it. chapkit-rs checks
what it knows: the configuration exists, the data has rows and the `time_period` and `location` columns, the
horizon is within the model's bounds. A model adds what only it knows by implementing `validate_train` and
`validate_predict`, which return findings:

```rust,ignore
fn validate_train(&self, config: &Config, input: TrainInput<'_>) -> Vec<Diagnostic> {
    let mut found = Vec::new();
    if config.lags > 24 {
        found.push(
            Diagnostic::error("too_many_lags", "lags above 24 are not supported").at("config.lags"),
        );
    }
    found
}
```

`Diagnostic::error` makes the request invalid; `warning` and `info` are reported but let it through. Each has a
stable `code` for programs and a message for people, and `at` names the input it is about. The checks run only when
chapkit's own checks passed, inside the request (on a blocking thread, so a slow or failing check cannot take the
service down); `input.run` carries the resolved horizon. Both default to no findings. This matches Python chapkit's
`on_validate_train` and `on_validate_predict`, and `chapkit-rs test -v` shows the findings.

## Tips

- **Statistical models** with nothing to store use `type Trained = ()` and do their work in `predict`
  from `input.historic`.
- **Models with weights** override `Model::save` and `Model::load` to store tensors in the safetensors
  checkpoint (`Checkpoint::tensors`, `Tensor`).
- **The default horizon** is 3 periods. A model with another default declares the reserved key in its config,
  `#[serde(default = "twelve")] prediction_periods: u32`, as a Python chapkit config subclass does; a config
  that sets `prediction_periods` still wins.
- **Hyperparameters** are declared with `Model::search_space` in chap-core's format and published in
  `/api/v1/info` as `hpo_search_space`.
- **Errors**: return `Error::InvalidInput` for data the model cannot handle; the job fails with
  `ValueError: ...`, which chap-core shows to the user.
- **Testing** in-process is a plain function call; `chapkit::synth::Generator` makes realistic data.
