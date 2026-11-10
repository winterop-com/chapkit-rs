# Environment variables

`HOST`, `PORT`, `DATABASE_URL` and `CHAPKIT_MAX_CONCURRENCY` are also options of a model binary's `serve`
(`--host`, `--port`, `--database`, `--max-concurrency`); a flag wins over the variable, and the
variable over a value set in the model's code ([command line](cli.md#a-model-binary)).

| variable | default | meaning |
|---|---|---|
| `HOST` | `127.0.0.1` | address to bind; `0.0.0.0` to accept calls from containers |
| `PORT` | `8000` | port to bind |
| `DATABASE_URL` | in memory | `:memory:`, a path, `sqlite:///path` or chapkit's `sqlite+aiosqlite:///path`; with the `turso` feature also `turso::memory:` or `turso:///path`. Three slashes mean a path relative to the working directory, four an absolute one (`sqlite:////var/lib/chapkit.db`), as in SQLAlchemy |
| `CHAPKIT_EAGER_WAIT_MS` | `1000` | how long `$train`/`$predict` wait for a job before replying 202 |
| `CHAPKIT_MAX_CONCURRENCY` | `1` | jobs running at once |
| `CHAPKIT_MAX_BODY_MB` | `1024` | largest accepted request body; larger bodies get a 413 problem |
| `CHAPKIT_SYNC_CONCURRENCY` | number of cores | `$predict-sync` and `$explain` calls running at once |
| `SERVICEKIT_ORCHESTRATOR_URL` | unset | register with chap-core at this URL (`.../v2/services/$register`) |
| `SERVICEKIT_HOST` | host name | host chap-core should call back on |
| `SERVICEKIT_PORT` | `PORT`, else `8000` | port chap-core should call back on |
| `SERVICEKIT_REGISTRATION_KEY` | unset | sent as `X-Service-Key` when registering |
| `SERVICEKIT_API_KEYS` | unset | comma-separated keys; when set, requests need `X-API-Key` (docs and health stay open) |
| `GIT_REVISION` | unset | commit sha reported as `git_revision` |
| `LOG_LEVEL` | `info` | log filter |
| `LOG_FORMAT` | console | `json` for structured logs |
