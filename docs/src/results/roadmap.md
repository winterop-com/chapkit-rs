# Roadmap

Done: the model contract, safetensors checkpoints with provenance, the full chapkit HTTP API (38 of 38
routes), eager job completion, `$predict-sync`, `$explain` (LIME, KernelSHAP), the sample data generator,
the committed OpenAPI document with a Python comparison, and chap-core end-to-end runs with identical scores.

Also done: the turso storage backend (feature `turso`), passing the same store suite and HTTP contract.

Next:

- **Lagged explainer features**: perturb covariates at `t-1`, `t-2` from the historic frame
  ([Explainers](../design/explainers.md)).
- **Hyperparameter search** (`$hpo`): random and grid search over the chap-core search space with a
  rolling-origin backtest, trials in parallel inside one job. Parked until chap-core's own HPO matures.
- **Pre-trained models end to end**: a model package that ships a checkpoint, and a backtest check of
  `trained_until` against the test periods.
- **A container image** for model packages, to compare image sizes and run them as managed models in
  chap-devops.
