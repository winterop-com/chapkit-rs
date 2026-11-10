# Concurrency

| part | how it runs |
|---|---|
| HTTP | tokio's multi-threaded runtime, one worker thread per core; axum serves requests in parallel |
| jobs (`$train`, `$predict`) | queued behind a semaphore, one at a time by default (`CHAPKIT_MAX_CONCURRENCY`), as in chapkit; the model runs on tokio's blocking pool so it never stalls request handling or health checks |
| `$predict-sync`, `$explain` | no job; bounded by a second semaphore, one per core by default (`CHAPKIT_SYNC_CONCURRENCY`), so a burst cannot start more CPU-bound work than there are cores |
| models | `&self` methods on a `Send + Sync` type; trained state is shared behind an `Arc` and read with no lock during a call |
| explainers | perturbations are evaluated in parallel with rayon, across all cores |
| store | one SQLite connection behind a mutex on the blocking pool; writes are small metadata rows |
| warm model cache | an LRU of trained models by artifact id, behind a mutex held only for the lookup |

Python chapkit runs one uvicorn worker, and the GIL limits it to one thread of Python at a time; its jobs
also run one at a time by default.

Job listings are ordered by a submission counter rather than by ULID: ULIDs created within the same
millisecond have a random order.
