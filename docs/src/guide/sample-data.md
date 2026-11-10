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
`seed` (negative for fresh data).

`chapkit::rng::Rng` is the seedable generator behind it, also used by the explainers; models can use it for
reproducible sampling.
