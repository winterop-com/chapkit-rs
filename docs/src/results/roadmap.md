# Roadmap

Done: the model contract, safetensors checkpoints with provenance, the full chapkit HTTP API (38 of 38
routes), eager job completion, `$predict-sync`, `$explain` (LIME, KernelSHAP), the sample data generator,
the committed OpenAPI document with a Python comparison, and chap-core end-to-end runs with identical scores.

Also done: the turso storage backend (feature `turso`), passing the same store suite and HTTP contract; a
shared container image for model packages (published to GHCR, run as managed models in chap-devops); a
command line in every model binary (`serve`, `test`, `healthcheck`).

Next:

- **Backtests in-process** (`chapkit-eval`): chap-core's evaluation, run inside the model service. chap-core
  splits the data into expanding train windows with a forecast window after each (`n_splits`, `stride`,
  `n_retrain`), fills the future climate from a weather provider so nothing leaks, and scores the forecasts.
  In the full-stack run one backtest took 47 s through chap-core, of which the model took under a second;
  in-process it is a matter of milliseconds, which is also what hyperparameter search needs.
  - A pure crate, like `chapkit-core`: the splitter (chap-core's `train_test_generator`, with retraining),
    the metrics (CRPS and its log and normalised forms, MAE, RMSE, MAPE, coverage, Winkler, ratio above truth),
    the observed and seasonal climatology weather providers, and target masking.
  - Parity with a pinned chap-core release, checked against fixtures chap-core generates, like the baselines;
    refreshed on every chap-core release. chap-core stays the source of record: this is the developer's preview
    of its evaluation, and must give the same numbers.
  - Exposed as `<bin> eval` (a dataset file or generated data; scores by horizon and location) and as a
    Rust-only `$backtest` job endpoint, like `$explain`.
  - Uses checkpoint provenance: splits that start before a pre-trained model's `trained_until` are flagged.
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
- **Hyperparameter search** (`$hpo`): random and grid search over the chap-core search space, each trial an
  in-process backtest, trials in parallel inside one job. Parked until chap-core's own HPO matures; needs the
  backtests above.
- **Pre-trained models end to end**: a model package that ships a checkpoint, and a backtest check of
  `trained_until` against the test periods.
