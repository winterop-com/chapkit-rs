# chapkit-rs

Experimental Rust take on [chapkit](https://github.com/dhis2-chap/chapkit): serve forecasting models to Chap
from Rust, **in-process**, behind an HTTP API that stays compatible with Python chapkit, so chap-core and
chap-marketplace work unchanged.

## Why

A call to a Python chapkit model costs 2.5-3 s of overhead (job polling, workspace copy, zip, pickle).
That is fine for one backtest, but not for explainers (LIME/SHAP), hyperparameter search or AutoML,
which call a model hundreds to thousands of times. In chapkit-rs a model is a Rust trait called
directly, so a call costs microseconds:

| model (50 locations x 120 months, predict 3 periods) | predict |
|---|---|
| global_median | 55 us |
| seasonal_median | 79 us |
| seasonal_shift_persistence | 0.8 ms |
| ridge (covariates) | 104 us |

Measured with `make bench` on an Apple Silicon laptop.

Shell, R and Python models stay on Python chapkit. chapkit-rs only covers in-process Rust models.

## Status

Early. Done:

- `chapkit-dataframe`: columnar `DataFrame` with the chapkit `{columns, data, schema?}` wire format
- `chapkit-core`: the `Model` trait, run info, model info, chap-core search spaces, chap period ids
- Example models: ports of chap-core's built-in baselines, verified against chap-core number for number

Next:

- HTTP layer: chapkit-compatible `/health`, `/api/v1/info`, configs, artifacts, jobs, `$train` and `$predict`
- Fast paths: synchronous predict, in-process `$explain` (LIME/SHAP) and `$hpo`
- safetensors checkpoints and pre-trained models

## A model

```rust
use chapkit::{DataFrame, Model, ModelInfo, NoConfig, PredictInput, Result, TrainInput};

struct MyModel;

impl Model for MyModel {
    type Config = NoConfig;   // user options; their JSON schema becomes /api/v1/configs/$schema
    type Trained = ();        // fitted state; () for pure-stats models that fit at predict time

    fn info(&self) -> ModelInfo { ModelInfo::default() }

    fn train(&self, _: &NoConfig, _: TrainInput<'_>) -> Result<()> { Ok(()) }

    fn predict(&self, _: &(), _: &NoConfig, input: PredictInput<'_>) -> Result<DataFrame> {
        // time_period, location, sample_0..N for each row of input.future
        chapkit::prediction_frame(input.future, 100, |_row, samples| {
            samples.fill(0.0);
            Ok(())
        })
    }
}
```

## Examples

Ports of chap-core's built-in baselines (`chap_core/models/builtin`), plus a small covariate model:

```sh
make examples-release                       # release binaries into ./bin
bin/global-median                           # serve on 127.0.0.1:8000 (HOST, PORT to change)
bin/seasonal-median demo                    # run once in-process and time it
cargo run --release -p ridge                # or straight from cargo
```

Each example is its own crate under `examples/`, depending on `chapkit` by path, the way a model
project would.

## Development

```sh
make lint      # fmt + clippy, warnings are errors
make test      # all tests, including parity with chap-core's baselines
make bench     # criterion benchmarks
make fixtures  # regenerate baseline golden outputs from a chap-core checkout (CHAP_CORE=../chap-core)
```
