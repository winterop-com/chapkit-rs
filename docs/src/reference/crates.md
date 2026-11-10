# Crates

| crate | depends on | contents |
|---|---|---|
| `chapkit-dataframe` | serde, polars (optional) | `DataFrame`, `Column` (float, int, string, bool, JSON; null-aware), the chapkit wire format with optional schema, grouping, stacking, checked row access; `to_polars`/`from_polars` (feature `polars`) |
| `chapkit-core` | dataframe, safetensors, schemars | `Model`, `TrainInput`, `PredictInput`, `RunInfo`, `ModelInfo`, `ModelMetadata`, `SearchSpace` (chap-core format), `Period`, `prediction_frame`, `Checkpoint`, `Provenance`, `rng::Rng`, `synth::Generator`, `NoConfig` |
| `chapkit-xai` | core, rayon | `explain` (LIME, KernelSHAP over any value function), `explain_forecast` |
| `chapkit-http` | axum, utoipa, tokio, reqwest | `App`, `Problem`, `Scheduler`, jobs router, health, system, info, SSE, metrics, API keys, request ids, registration, extractors, pagination |
| `chapkit-store` | rusqlite, turso (optional) | `Store` trait, `SqliteStore`, `TursoStore` (feature `turso`), rows, `open(url)` |
| `chapkit` | all of the above | `Service`, the configs, artifacts and ML routers, the type-erased model, the warm cache; re-exports the crates as `chapkit::{dataframe, http, store, xai}` and `chapkit-core` at the top level |
