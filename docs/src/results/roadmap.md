# Roadmap

Done: the model contract, safetensors checkpoints with provenance, the full chapkit HTTP API (38 of 38
routes), eager job completion, `$predict-sync`, `$explain` (LIME, KernelSHAP; over the cell's covariates,
their earlier values and the target's history, with the surrogate's R2), the sample data generator, the
committed OpenAPI document with a Python comparison, and chap-core end-to-end runs with identical scores.

Also done: the turso storage backend (feature `turso`), passing the same store suite and HTTP contract; a
shared container image for model packages (published to GHCR, run as managed models in chap-devops); a
command line in every model binary (`serve`, `test`, `eval`, `hpo`, `healthcheck`).

Next:

- **Backtests in-process** (`chapkit-eval`, [guide](../guide/eval.md)): chap-core's evaluation, run inside
  the model service. In the full-stack run one backtest took 47 s through chap-core, of which the model took
  under a second; in-process it is a matter of milliseconds, which is also what hyperparameter search needs.
  - Done: a pure crate with chap-core 2.7.0's splitter (`train_test_generator`, with retraining), the
    metrics (CRPS and its log and normalised forms, MAE, RMSE, MAPE, coverage, Winkler, ratio above truth;
    aggregate, by horizon and by location), the observed and climatology weather providers, target
    masking, and chap-core's dataset preparation (`fill_missing=True`: gaps filled, locations padded); parity with fixtures chap-core generates (`make eval-fixtures`), to 1e-12; `<bin> eval` on a
    CSV or JSON dataset or generated data.
  - Done since: the damped persistence weather provider; chap-core's untrained-location filter;
    `eval --json` (every score at full precision, to compare with chap-core's stored `aggregateMetrics`)
    and `eval --config` (model options); every period form chap-core reads, including the week ranges of
    its CSV export; the Rust-only `$backtest` job endpoint, storing the result as an `ml_backtest` artifact.
  - Next: target masking on the command line; checkpoint provenance (splits that start before a
    pre-trained model's `trained_until` are flagged); refreshing the fixtures on every chap-core release;
    the workbench UI below, to show the scores and forecasts.
  - Done since: chap-core's MSTL + ARIMA weather provider, a port of the statsmodels code it runs; and
    `chapkit::ts` (crate `chapkit-ts`): STL, KPSS, R's `stats::arima` and fable's automatic ARIMA search,
    with parity against R and fable, for models such as the auto-ARIMA baseline.
  - Done: `$predict-splits`, every split's forecasts in one call, so chap-core's own backtest can make one
    round trip per training instead of one per split and still score the forecasts itself
    ([proposal](../design/predict-splits.md)). Next, with chap-core: using it from chap-core's REST wrapper.
- **A model workbench UI**: Python chapkit's console (configs, train, predict, jobs, artifacts) plus what
  the backtests make possible: a dataset, a backtest, scores by horizon and location, forecasts against the
  truth, skill against the baselines, explanations and search trials. For one model and its developer; the
  Modeling App remains the tool for comparing models across a country.
  Built with [sokkel-ui](https://github.com/winterop-com/sokkel-ui) (React components on Base UI, with charts
  and maps), served by the model binary; a component it lacks is added to sokkel-ui, not built in the app.
  Shipped in 0.5.0 ([guide](../guide/ui.md)), served at `/ui/` by every model binary: stored datasets with
  upload and charts, evaluations (`$backtest` on a stored dataset, scores by horizon and location, forecasts
  against the truth), forecasts (`$forecast` past the end of the data), configs, jobs, artifacts, system and API
  pages. Released without a design review; 0.6.0 reviews it with the user and fixes what that finds.

- **Generated datasets**: make a stored dataset from the synthetic generator instead of uploading one, so a
  model can be tried, evaluated and forecast straight away. The generator exists (`chapkit::synth`, served as
  `$generate-sample-data`; [guide](../guide/sample-data.md)); what is missing is storing its output as a
  dataset: a `POST /api/v1/datasets/$generate` taking the generator's settings (name, locations, periods,
  monthly or weekly, start, covariates, seed), defaulting to the model's period type and required covariates,
  bounded like `$generate-sample-data`. In the UI, a "Generate" choice beside Upload on the Datasets page (and
  in the command palette), with the seed shown so a dataset can be made again.

- **Hyperparameter search** ([guide](../guide/hpo.md)). Done: `chapkit-hpo`, random and grid search over
  the chap-core search space, each trial an in-process backtest, trials in parallel and reproducible by
  seed; `<bin> hpo` and the `$hpo` job endpoint, which can create the best config. About 2000 trials per
  second for a baseline-sized model. Next: an adaptive searcher (chap-core's TPE), early stopping of poor
  trials, and search results in the workbench UI.
- **Pre-trained models end to end**: a model package that ships a checkpoint, and a backtest check of
  `trained_until` against the test periods.
