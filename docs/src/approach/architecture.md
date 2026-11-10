# Architecture

chapkit-rs is a Cargo workspace of small crates, each with one job. Lower layers know nothing about the
layers above them.

| crate | layer | knows about |
|---|---|---|
| `chapkit-dataframe` | data | a columnar table and the chapkit `{columns, data, schema}` JSON format |
| `chapkit-core` | model contract | the `Model` trait, run settings, model info, search spaces, periods, checkpoints, RNG, synthetic data. No IO, no async, no HTTP |
| `chapkit-xai` | explainers | LIME and KernelSHAP over any value function; explaining a forecast cell |
| `chapkit-eval` | backtests | chap-core's splits, future weather, metrics and target masking, over any `Model`. No IO |
| `chapkit-http` | HTTP plumbing | problem+json errors, jobs, health, system, info, SSE, metrics, API keys, registration, OpenAPI. No models, no database |
| `chapkit-store` | persistence | the `Store` trait and its SQLite backend. No HTTP |
| `chapkit` (root) | service | wires a `Model` to all of the above: configs, artifacts, ML routes, the warm model cache, `Service` |

Model packages (the chap-core baselines) live in a separate repository, `chapkit-rs-models`, and
depend on `chapkit` like any downstream project would.

## A request through the stack

`POST /api/v1/ml/$train` with a config id and a training frame:

1. `chapkit-http` decodes `%24` to `$` (chap-core's proxy spelling), assigns a request id, checks the API
   key if configured, and records metrics.
2. The JSON body is decoded into a `TrainRequest`; the training frame becomes a typed columnar
   `DataFrame`. Malformed bodies get a 422 problem response.
3. A job is submitted to the scheduler, which queues it behind a semaphore (one job at a time by default,
   as in chapkit). The handler waits up to the eager budget (1 s) for it.
4. The job loads the config from the store, resolves the horizon (request, else config), checks it against
   the model's bounds, and calls `Model::train` on tokio's blocking pool.
5. `Model::save` turns the trained state into a `Checkpoint`; provenance (model, version, training period
   range, locations, config) is added, and the safetensors bytes are stored as the training artifact.
6. The trained state is also kept in the warm cache, so a following predict skips loading it.
7. The handler replies 202 with the job id and artifact id. A fast job is already `completed`.

`$predict` is the same shape; the prediction frame is stored as the child artifact chap-core reads.
`$predict-sync` and `$explain` skip the job and the artifact entirely.

## The type-erased model

Handlers are not generic over the model. `Service::new` wraps the typed `Model` in `Erased<M>`, which
implements the object-safe `DynModel` trait: configs travel as JSON and are parsed into the model's typed
`Config` on each call, and trained state is held as `Arc<dyn Any>` so one cache serves any model. The
parse costs microseconds and keeps every route a plain function.
