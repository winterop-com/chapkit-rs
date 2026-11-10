# Concurrency

| part | how it runs |
|---|---|
| HTTP | tokio's multi-threaded runtime, one worker thread per core; axum serves requests in parallel |
| jobs (`$train`, `$predict`, `$backtest`, `$hpo`) | queued behind a semaphore, one at a time by default (`CHAPKIT_MAX_CONCURRENCY`), as in chapkit; the model runs on tokio's blocking pool so it never stalls request handling or health checks |
| `$predict-sync`, `$explain` | no job; bounded by a second semaphore, one per core by default (`CHAPKIT_SYNC_CONCURRENCY`), so a burst cannot start more CPU-bound work than there are cores |
| models | `&self` methods on a `Send + Sync` type; trained state is shared behind an `Arc` and read with no lock during a call |
| explainers | perturbations are evaluated in parallel with rayon, across all cores, unless the batch is cheaper on one thread ([Explainers](explainers.md)) |
| backtests, searches | inside one job, trainings, splits and search trials run in parallel with rayon, across all cores |
| store | one SQLite connection behind a mutex on the blocking pool; writes are small metadata rows |
| warm model cache | an LRU of trained models by artifact id, behind a mutex held only for the lookup |

Python chapkit runs one uvicorn worker, and the GIL limits it to one thread of Python at a time; its jobs
also run one at a time by default.

Job listings are ordered by a submission counter rather than by ULID: ULIDs created within the same
millisecond have a random order.

## Deleting a running job

A model call is plain Rust on a blocking thread and cannot be stopped from outside. Deleting a running job
therefore stops the service from waiting for it and discards its result, but the model call runs to the
end. The job's concurrency slot travels with the model call (`chapkit_http::jobs::blocking`) and is freed
only when the call returns, so `CHAPKIT_MAX_CONCURRENCY` is never exceeded.

The trade-off: with the default of one job at a time, a model call that hangs keeps blocking later train
and predict jobs even after its job is deleted; only a restart clears it. For slow or experimental models, raise `CHAPKIT_MAX_CONCURRENCY`. `$predict-sync`
and `$explain` do not use job slots and are not affected.

A job that panics is marked failed with the panic message instead of staying `running`.
