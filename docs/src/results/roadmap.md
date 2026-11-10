# Roadmap

Done: the model contract, safetensors checkpoints with provenance, the full chapkit HTTP API (38 of 38
routes), eager job completion, `$predict-sync`, `$explain` (LIME, KernelSHAP), the sample data generator,
the committed OpenAPI document with a Python comparison, and chap-core end-to-end runs with identical scores.

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
    the workbench UI below, to show the scores and forecasts. MSTL + ARIMA weather stays out: it is
    statsforecast's AutoARIMA search, a large port of its own.
  - Later, with chap-core: a `$predict-splits` endpoint returning every split's forecasts in one call, so
    chap-core's own backtest makes one round trip per training instead of one per split and still scores
    the forecasts itself.
- **A model workbench UI**: Python chapkit's console (configs, train, predict, jobs, artifacts) plus what
  the backtests make possible: a dataset, a backtest, scores by horizon and location, forecasts against the
  truth, skill against the baselines, explanations and search trials. For one model and its developer; the
  Modeling App remains the tool for comparing models across a country.
  Built with [sokkel-ui](https://github.com/winterop-com/sokkel-ui) (React components on Base UI, with charts
  and maps), served by the model binary; a component it lacks is added to sokkel-ui, not built in the app.

- **Lagged explainer features**: perturb covariates at `t-1`, `t-2` from the historic frame
  ([Explainers](../design/explainers.md)).
- **Hyperparameter search** ([guide](../guide/hpo.md)). Done: `chapkit-hpo`, random and grid search over
  the chap-core search space, each trial an in-process backtest, trials in parallel and reproducible by
  seed; `<bin> hpo` and the `$hpo` job endpoint, which can create the best config. About 2000 trials per
  second for a baseline-sized model. Next: an adaptive searcher (chap-core's TPE), early stopping of poor
  trials, and search results in the workbench UI.
- **Pre-trained models end to end**: a model package that ships a checkpoint, and a backtest check of
  `trained_until` against the test periods.
