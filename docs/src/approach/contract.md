# The compatibility contract

The **public HTTP API** is the contract. Everything a client can observe over HTTP matches Python chapkit:
paths, methods, status codes, JSON shapes, error format and the job lifecycle. Everything behind the API is
free to differ.

## What must match

The routes chap-core calls, from `chap_core/models/chapkit_rest_api_wrapper.py`:

| call | what chap-core relies on |
|---|---|
| `GET /health` | `status` is the string `"healthy"` |
| `GET /api/v1/info` | chapkit's `MLServiceInfo` (chap-core accepts extra fields) |
| `GET /api/v1/configs/$schema` | top-level `properties` of the config data schema |
| `POST /api/v1/configs` | keys sent in `data` come back unchanged; `ConfigOut` shape |
| `GET /api/v1/configs`, `DELETE /api/v1/configs/{id}` | plain list; best-effort cleanup |
| `POST /api/v1/ml/$train`, `$predict` | 202 with `{job_id, artifact_id, message}` |
| `GET /api/v1/jobs/{id}` | job record with `status`, `error`, `error_traceback` |
| `GET /api/v1/artifacts/{id}` | prediction `data` with exactly `type`, `metadata`, `content`, `content_type`, `content_size` |
| orchestrator registration | `POST $register`, keepalive `PUT ping_url`, deregister on shutdown |

chap-devops additionally uses `GET /api/v1/ml/$generate-sample-data` (through chap-core's service proxy,
which spells `$` as `%24`) and the `chapkit_version` in `/api/v1/info`.

Beyond those, chapkit-rs serves every other route Python chapkit serves (38 of 38): artifact trees,
config links, `$validate`, `$stats`, `$schema`, pagination, SSE `$stream` endpoints, `/metrics` and
`/api/v1/system`.

## What may differ

- **Artifacts.** A training artifact holds a safetensors checkpoint instead of a zipped Python workspace.
  chap-core never reads training artifacts.
- **Storage.** Configs and artifacts are stored as JSON and blobs in SQLite, not pickled.
- **Extensions.** `$predict-sync`, `$explain`, `$backtest` and `$hpo` exist only in chapkit-rs;
  `/api/v1/info` lists them under `extensions`. Python clients ignore them.

## The version string

`/api/v1/info` reports `chapkit_version` as `2.6.0+chapkit-rs.0.1.0`: the Python chapkit release whose API
chapkit-rs matches, with chapkit-rs as semver build metadata. chap-devops gates features on this version,
so it has to parse as a chapkit version. The claim is backed by the route-by-route comparison in
[OpenAPI as a reviewed artifact](../design/openapi.md); `chapkit_rs_version` carries the real version.

## How the contract is kept

- `tests/contract.rs` runs the exact chap-core flow in-process against a tiny model and checks every shape
  chap-core reads.
- `scripts/compare_openapi.py` diffs a running Python chapkit service against a chapkit-rs one and fails on
  any difference in a route chap-core uses.
- The example baselines are backtested through a real chap-core with chap-devops and score identically to
  chap-core's own built-in baselines ([results](../results/end-to-end.md)).
