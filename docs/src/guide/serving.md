# Serving and configuration

```rust,ignore
chapkit::Service::new("id", "Display name", model)
    .version("1.0.0")              // MLServiceInfo.version
    .description("...")
    .database("data/chapkit.db")   // default: DATABASE_URL, else in memory
    .max_concurrency(1)            // jobs at once
    .eager_wait(Duration::from_millis(1000))
    .cache_size(16)                // trained models kept warm
    .serve()
    .await
```

`Service::app()` returns the app instead of serving it; `app.build()` gives an axum `Router` for in-process
tests and benchmarks.

The binary prints `INFO:     Uvicorn running on http://HOST:PORT`, the line chap-core's MLproject runner
waits for. All environment variables are listed in the [reference](../reference/environment.md).

## Use a database file in deployments

Without `DATABASE_URL` the database is in memory, the same default as Python chapkit. Every config, trained
model and prediction then stays in RAM until the process exits, and a restart loses them. That suits tests
and benchmarks, not a service chap-core uses: memory grows by about 0.1 MB per stored prediction (the
benchmark's load test reached 1.2 GB after 12,000 predictions). Set `DATABASE_URL` to a file
(`data/chapkit.db`, or `turso:///data/chapkit.db` with the `turso` feature) so results live on disk and
survive restarts.

## Registering with chap-core

Set `SERVICEKIT_ORCHESTRATOR_URL` (for example `http://localhost:8700/v2/services/$register`), and
`SERVICEKIT_HOST`/`SERVICEKIT_PORT` to the address chap-core should call back on. The service registers
when it starts, sends a keepalive every 10 seconds, registers again if chap-core forgets it, and deregisters
on shutdown. `GIT_REVISION` (a commit sha) is reported as `git_revision`; chap-core 2.4 and later stores a
model template only for services that report one.
