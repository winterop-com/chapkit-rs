# Sample data

`chapkit::synth` is the Rust counterpart of Python chapkit's `TestDataGenerator`, and
`GET /api/v1/ml/$generate-sample-data` serves it with the same parameters.

```rust,ignore
use chapkit::synth::{Generator, Resolution, Spec};

let mut generator = Generator::new(Some(42));        // None: fresh data each time
let spec = Spec { num_locations: 20, num_periods: 120, resolution: Resolution::Monthly, ..Spec::default() };
let train = generator.training_data(&spec)?;
let (historic, future) = generator.prediction_data(&Spec { num_periods: 3, ..spec }, 120)?;
let geo = Generator::geo(20, false);                  // GeoJSON grid, properties.id = location_N
```

Columns: `time_period`, `location`, `disease_cases`, `population`, `rainfall`, `mean_temperature`, then
`feature_0..N`, the model's required covariates and a config's additional covariates. Climate covariates
follow an annual cycle with a per-location phase, and `disease_cases` depends on the season and on rainfall,
so models have something to learn. Future periods continue the historic calendar without a gap, with
`disease_cases` null.

Query parameters of the endpoint: `kind` (`train` or `predict`), `config_id` (adds its covariates),
`num_locations` (5), `num_periods` (50), `num_features` (3), `period_type` (`monthly` or `weekly`, default
from the model), `geo_type` (`polygon` or `point`), `include_geo` (default: the model's `requires_geo`),
`seed` (negative for fresh data). A sample holds at most 20 million cells (locations x periods x columns,
history and future for `predict`), about 200 MB of JSON; a larger one is a 400. Weekly periods follow the
ISO calendar, week 53 included.

## As a stored dataset

`POST /api/v1/datasets/$generate` (chapkit-rs only) stores the generator's training data as a dataset, as an
upload is stored, for the UI's evaluations and forecasts. It takes a `name` and the endpoint's parameters as
JSON: `num_locations`, `num_periods`, `num_features`, `period_type`, `config_id` and `seed`. The data is what
`$generate-sample-data` returns as `data` for the same parameters, under the same cell limit; the target is
`disease_cases`. `kind`, `geo_type` and `include_geo` are not taken: a stored dataset is a table, with no
geometry, so a model that requires geo gets it in the request that uses the dataset.

```json
{"name": "synthetic", "num_locations": 4, "num_periods": 48, "period_type": "monthly", "seed": 42}
```

It answers 201 with the dataset's description, as an upload does, plus `generator`: the settings as resolved
and the seed used. Without a seed (or with a negative one) the service draws one below 2^53, so it reads back
exactly in JavaScript, and records it; the same settings and seed give the same data again, from either
endpoint.

```json
"generator": {"num_locations": 4, "num_periods": 48, "num_features": 3, "period_type": "monthly", "seed": 42}
```

`chapkit::rng::Rng` is the seedable generator behind it, also used by the explainers; models can use it for
reproducible sampling.
