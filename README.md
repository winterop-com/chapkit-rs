# chapkit-rs

Experimental Rust take on [chapkit](https://github.com/dhis2-chap/chapkit): serve forecasting models to Chap
from Rust, **in-process**, behind an HTTP API that stays compatible with Python chapkit, so chap-core and
chap-marketplace work unchanged.

## Why

A call to a Python chapkit model costs 2.5-3 s of overhead (job polling, workspace copy, zip, pickle).
That is fine for one backtest, but not for explainers (LIME/SHAP), hyperparameter search or AutoML,
which call a model hundreds to thousands of times. Measured on the same model (chap-core's
`global_median`, 20 locations x 120 months):

| | Python chapkit | chapkit-rs |
|---|---|---|
| train + predict through chap-core's job flow | 4.08 s | 0.013 s |
| CPU per prediction | 154 ms | 2.3 ms |
| `$predict-sync` (forecast returned directly) | | 2.7 ms |
| memory at idle | 150 MB | 16 MB |
| startup | 0.9 s | 0.02 s |
| install size | 522 MB (Python environment) | 11.9 MB (binary) |

Most of the chap-core speedup comes from finishing fast jobs before chap-core's first status check, which
Python chapkit could adopt too; the CPU, memory and in-process numbers are Rust. Details, method and caveats:
the [benchmarks chapter](docs/src/results/benchmarks.md) of the book.

Through a real chap-core (via chap-devops), the Rust ports of chap-core's baselines score identically to
chap-core's own built-in baselines, to six decimals.

Every model binary also backtests itself in-process with chap-core's splits, weather and metrics (`eval`, and
`$backtest` as a job), and searches its hyperparameters with a backtest per trial (`hpo`, `$hpo`), about 2000
trials per second for a baseline-sized model.

Every model binary also serves a web UI for its model at `/ui/` (`<bin> serve`, then
<http://127.0.0.1:8000/ui/>): upload a dataset, backtest the model on it, see the forecasts against the
truth, and forecast ahead ([guide](docs/src/guide/ui.md)).

Shell, R and Python models stay on Python chapkit. chapkit-rs only covers in-process Rust models.

## Getting started

```sh
cargo install --git https://github.com/winterop-com/chapkit-rs chapkit-cli
chapkit-rs init my-model && cd my-model     # a model package: the calculation in src/model.rs
cargo run --release -- serve                # serve on http://127.0.0.1:8000
cargo run --release -- test                 # in another terminal: chap-core's flow against it
```

The [tutorials](https://winterop-com.github.io/chapkit-rs/tutorials/install.html) walk through it step by step.

## Documentation

**The book** is online at <https://winterop-com.github.io/chapkit-rs/> (source in `docs/`; `make docs`,
`make docs-serve`). It explains the approach, every design decision, how to
write and serve a model package, testing with chap-core, and the results.

## Layout

| crate | role |
|---|---|
| `chapkit` (root) | What model authors depend on. `Service` turns a `Model` into the full chapkit HTTP API. |
| `chapkit-core` | The model contract: `Model` trait, run and model info, search spaces, periods, safetensors checkpoints, RNG, synthetic data. Pure: no IO. |
| `chapkit-xai` | LIME and KernelSHAP over forecasts, in-process and parallel. |
| `chapkit-ts` | Time series for models and evaluation: STL, KPSS, R's `stats::arima` and fable's ARIMA search, with R's numbers. Pure: no IO. |
| `chapkit-eval` | chap-core's backtest in-process: splits, future weather, metrics, with chap-core's numbers. Pure: no IO. |
| `chapkit-hpo` | Hyperparameter search: random and grid over a model's search space, each trial an in-process backtest, in parallel. Pure: no IO. |
| `chapkit-dataframe` | Columnar `DataFrame` with the chapkit `{columns, data}` wire format. |
| `chapkit-http` | servicekit-compatible plumbing: problem+json errors, jobs, health, system, SSE, metrics, auth, registration, OpenAPI. |
| `chapkit-store` | Persistence behind an async `Store` trait; SQLite backend, turso backend (feature `turso`). |
| `chapkit-ui` | The web UI, embedded in every model binary and served at `/ui/`. |

Model packages (ports of chap-core's baselines, checked against chap-core number for number, and a
covariate model for xAI/HPO) live next door in `../chapkit-rs-models`.

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
async fn main() -> std::process::ExitCode {
    chapkit::Service::new("my-model", "My model", MyModel).serve().await
}
```

The binary has the subcommands `serve` (`--host`, `--port`, `--database`, `--max-concurrency`), `test`
(chap-core's flow against the running service) and `healthcheck`; without arguments it prints its help.

Environment: `HOST`/`PORT` (127.0.0.1:8000), `DATABASE_URL` (in-memory by default),
`SERVICEKIT_ORCHESTRATOR_URL` to register with chap-core, `SERVICEKIT_API_KEYS` for auth,
`CHAPKIT_EAGER_WAIT_MS` (1000), `CHAPKIT_MAX_CONCURRENCY` (1 job at a time),
`CHAPKIT_SYNC_CONCURRENCY` (cores), `CHAPKIT_MAX_BODY_MB` (1024), `LOG_LEVEL`, `LOG_FORMAT=json`.

## Development

```sh
make check            # lint + test; run before every commit
make release          # the chapkit-rs command into bin/ (universal on macOS)
make test-turso       # store suite and HTTP contract on the turso backend
make test-polars      # the polars conversion (feature polars)
make bench            # criterion benchmarks
make openapi          # regenerate the committed openapi.json
make compare-openapi PY=<url> RS=<url>   # diff against a running Python chapkit service (base URLs)
make models-check     # make check in ../chapkit-rs-models
make models-release   # model binaries into ../chapkit-rs-models/bin
make docs             # build the book
```
