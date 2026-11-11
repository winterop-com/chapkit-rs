# Crates

| crate | depends on | contents |
|---|---|---|
| `chapkit-dataframe` | serde, polars (optional) | `DataFrame`, `Column` (float, int, string, bool, JSON; null-aware), the chapkit wire format with optional schema, CSV input, grouping, stacking, checked row access; `to_polars`/`from_polars` (feature `polars`) |
| `chapkit-core` | dataframe, safetensors, schemars | `Model`, `TrainInput`, `PredictInput`, `RunInfo`, `ModelInfo`, `ModelMetadata`, `SearchSpace` (chap-core format), `Period`, `prediction_frame`, `Checkpoint`, `Provenance`, `rng::Rng`, `synth::Generator`, `NoConfig` |
| `chapkit-xai` | core, rayon | `explain` (LIME, KernelSHAP over any value function), `explain_forecast` (a cell, with lagged features) |
| `chapkit-ts` | rayon | time series, pure: `stl::Stl` (R's and statsmodels' STL), `unitroot` (KPSS, seasonal strength, fable's `ndiffs`/`nsdiffs`), `arima` (R's `stats::arima`: `fit_arima`, `fit_arima_fixed`, `Arima::{forecast, filter, reestimate}`; fable's search: `auto_arima`, `AutoArima`), `statsmodels::SmArima` |
| `chapkit-eval` | core, ts, rayon | chap-core's backtest in-process: `prepare` (`Prepare`), `train_test_splits` (with chap-core's untrained-location filter), `retrain_points`, `WeatherProvider` (`Climatology`, `DampedPersistence`, `MstlArima`, `Observed`), `mask_target`, `backtest`, `backtest_splits`, `backtest_model`, `score` (`Metric`, `Scores`), `summarize` (`Quantiles`), `forecast_ahead`. No IO |
| `chapkit-hpo` | core, eval, rayon | hyperparameter search: `candidates` (random, grid over a `SearchSpace`), `search` (any objective, trials in parallel), `search_model` (backtest objective), `direction`, `with_params`, `Search`, `Trial`. No IO |
| `chapkit-http` | axum, utoipa, tokio, reqwest | `App`, `Problem`, `Scheduler`, jobs router, health, system, info, SSE, metrics, API keys, request ids, registration, extractors, pagination |
| `chapkit-store` | rusqlite, turso (optional) | `Store` trait (configs, artifacts, datasets), `SqliteStore`, `TursoStore` (feature `turso`), rows, `open(url)` |
| `chapkit-ui` | axum, rust-embed | the web UI (`ui/`'s build in `dist/`), embedded and served at `/ui/` |
| `chapkit-flow` | core, clap, reqwest | chap-core's flow as a test against a running service: generated data, configurations, `$validate`, jobs, checks; used by `chapkit-rs test` and every model binary's `test` |
| `chapkit` | all of the above, clap | `Service`, the binary's command line (`chapkit::cli`: `serve`, `test`, `eval`, `hpo`, `healthcheck`), the configs, artifacts, datasets and ML routers (with `$backtest`, `$forecast`, `$hpo`), the UI (feature `ui`, default on), the type-erased model, the warm cache; re-exports the crates as `chapkit::{dataframe, eval, flow, hpo, http, store, ts, xai}` and `chapkit-core` at the top level |
| `chapkit-cli` | flow, clap, reqwest | the `chapkit-rs` command: `init` scaffolds a model package, `test` runs chap-core's flow against a running service |
