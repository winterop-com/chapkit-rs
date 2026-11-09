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

## Layout

| crate | role |
|---|---|
| `chapkit` (root) | What model authors depend on. `Service` turns a `Model` into the full chapkit HTTP API. |
| `chapkit-core` | The model contract: `Model` trait, run and model info, search spaces, periods, safetensors checkpoints. Pure: no IO. |
| `chapkit-dataframe` | Columnar `DataFrame` with the chapkit `{columns, data}` wire format. |
| `chapkit-http` | servicekit-compatible plumbing: problem+json errors, jobs, health, system, auth, registration, OpenAPI. |
| `chapkit-store` | Persistence behind an async `Store` trait; SQLite backend. |

Example models (ports of chap-core's baselines, checked against chap-core number for number, and a
covariate model for xAI/HPO) live next door in `../chapkit-rs-examples`.

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

## Serving

```rust,ignore
#[tokio::main]
async fn main() -> std::io::Result<()> {
    chapkit::Service::new("my-model", "My model", MyModel).serve().await
}
```

Environment: `HOST`/`PORT` (127.0.0.1:8000), `DATABASE_URL` (in-memory by default),
`SERVICEKIT_ORCHESTRATOR_URL` to register with chap-core, `SERVICEKIT_API_KEYS` for auth,
`CHAPKIT_EAGER_WAIT_MS` (1000), `CHAPKIT_MAX_CONCURRENCY` (1 job at a time),
`CHAPKIT_SYNC_CONCURRENCY` (cores), `LOG_LEVEL`, `LOG_FORMAT=json`.

## Development

```sh
make lint      # fmt + clippy, warnings are errors
make test      # all tests, including the chapkit HTTP contract
make bench     # criterion benchmarks
```
