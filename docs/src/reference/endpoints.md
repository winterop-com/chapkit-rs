# HTTP endpoints

Generated from `openapi.json`. Every service also serves the full document at `/openapi.json` and
interactive docs at `/docs`. Routes marked chapkit-rs only are extensions Python chapkit does not have:
`$predict-sync`, `$explain`, `$backtest` and `$hpo`.

| method | path | summary |
|---|---|---|
| `GET` | `/api/v1/artifacts` | List artifact summaries (`data` without `content`). |
| `POST` | `/api/v1/artifacts` | Create an artifact. The level is the parent's level + 1. |
| `GET` | `/api/v1/artifacts/$schema` | JSON schema of an artifact. |
| `GET` | `/api/v1/artifacts/$stats` | Artifact count. |
| `GET` | `/api/v1/artifacts/{id}` | Get an artifact. |
| `PUT` | `/api/v1/artifacts/{id}` | Replace an artifact's data and tags. |
| `DELETE` | `/api/v1/artifacts/{id}` | Delete an artifact. |
| `GET` | `/api/v1/artifacts/{id}/$config` | The config linked to the artifact's root. |
| `GET` | `/api/v1/artifacts/{id}/$download` | Download the artifact's content (checkpoint bytes, or the prediction frame as JSON records). |
| `GET` | `/api/v1/artifacts/{id}/$expand` | The artifact as a tree node, without children. |
| `GET` | `/api/v1/artifacts/{id}/$metadata` | The artifact's `data.metadata`. |
| `GET` | `/api/v1/artifacts/{id}/$tree` | The artifact's whole subtree (content stripped). |
| `GET` | `/api/v1/configs` | List configs (paged when both `page` and `size` are given). |
| `POST` | `/api/v1/configs` | Create a config. Defaults are filled; keys you send come back unchanged. |
| `GET` | `/api/v1/configs/$schema` | JSON schema of the config `data` (model options plus reserved keys). |
| `GET` | `/api/v1/configs/$stats` | Config count. |
| `GET` | `/api/v1/configs/{id}` | Get a config. |
| `PUT` | `/api/v1/configs/{id}` | Replace a config. |
| `DELETE` | `/api/v1/configs/{id}` | Delete a config and the artifact trees linked to it. |
| `GET` | `/api/v1/configs/{id}/$artifacts` | Root artifacts linked to the config. |
| `POST` | `/api/v1/configs/{id}/$link-artifact` | Link a root artifact to the config. |
| `POST` | `/api/v1/configs/{id}/$unlink-artifact` | Unlink an artifact from the config. |
| `GET` | `/api/v1/info` | Service information (the service's `ServiceInfo` / `MLServiceInfo` JSON). |
| `GET` | `/api/v1/jobs` | List jobs, newest first. |
| `GET` | `/api/v1/jobs/$schema` | JSON schema of the job list. |
| `GET` | `/api/v1/jobs/{job_id}` | Get a job. |
| `DELETE` | `/api/v1/jobs/{job_id}` | Cancel (if running) and delete a job. |
| `GET` | `/api/v1/jobs/{job_id}/$stream` | Stream a job's record as server-sent events until it finishes. Sends |
| `POST` | `/api/v1/ml/$backtest` | Backtest the model on a dataset in-process, scored as chap-core scores it. Returns 202; the |
| `POST` | `/api/v1/ml/$explain` | Explain one forecast cell with LIME or KernelSHAP, computed in-process: every |
| `GET` | `/api/v1/ml/$generate-sample-data` | A ready-to-submit sample `$train` or `$predict` payload from the synthetic data |
| `POST` | `/api/v1/ml/$hpo` | Search the model's hyperparameters (random or grid over its `hpo_search_space`), each trial |
| `POST` | `/api/v1/ml/$predict` | Predict with a trained model. Returns 202; the prediction artifact is a child of the training artifact. |
| `POST` | `/api/v1/ml/$predict-sync` | Predict and return the forecast directly: no job, no stored artifact. The trained |
| `POST` | `/api/v1/ml/$train` | Train a model. Returns 202 with the job id and the artifact id the model will be stored under. |
| `POST` | `/api/v1/ml/$validate` | Check a train or predict request without running it. |
| `GET` | `/api/v1/system` | System information. |
| `GET` | `/api/v1/system/apps` | Bundled static web apps. chapkit-rs serves none; kept for API compatibility. |
| `GET` | `/api/v1/system/apps/$schema` | JSON schema of the app list. |
| `GET` | `/health` | Service health. 503 when any check is unhealthy; degraded still returns 200. |
| `GET` | `/health/$stream` | Stream the health status as server-sent events, forever. |
| `GET` | `/metrics` | Prometheus metrics. |
