# Why chapkit-rs

## Where the 2.5 to 3 seconds go

A chap-core backtest calls a chapkit model service through jobs: it submits `$train`, polls the job until
it is done, submits `$predict`, polls again, and reads the prediction artifact. Two things make each call
slow in Python chapkit:

1. **Polling.** chap-core asks for the job status, and if the job is not finished it sleeps 2 seconds before
   asking again (`chap_core/models/chapkit_rest_api_wrapper.py`, `wait_for_job`). A job that is not done at
   the first check always costs at least 2 seconds.
2. **The job itself.** Python chapkit copies the model's project into a temporary workspace, writes the data
   to CSV files, pickles the trained model into a zip, and stores the zip as an artifact. Even a model whose
   own computation takes microseconds needs tens to hundreds of milliseconds for this, so it is never done
   at the first check.

Together that is the 2.5 to 3 seconds per call that show up in every backtest and every chap CLI run.

## Why that rules out explainers, search and AutoML

LIME explains one forecast by perturbing its inputs some 500 to 1000 times and refitting a local surrogate.
KernelSHAP needs a similar number of evaluations. A hyperparameter search runs a backtest per trial; AutoML
runs searches over several models. At 2.5 seconds per call, one LIME explanation with 1000 perturbations
spends about 45 minutes on overhead alone.

These methods only work when the model can be called directly, in the same process, as a function.

## What chapkit-rs changes

- Models are Rust trait implementations, called in-process. A train or predict is a function call.
- Explainers and search run inside the service, next to the model, with every perturbation a direct call
  run in parallel across cores (`POST /api/v1/ml/$explain`).
- `$predict-sync` returns a forecast directly from a model kept warm in memory: no job, no artifact.
- The normal job flow stays, because chap-core depends on it, but jobs that finish within a small budget
  are finished before the reply, so chap-core's first status check already sees them complete
  (see [Finishing fast jobs](../design/eager-completion.md)).
