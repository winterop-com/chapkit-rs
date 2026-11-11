# The web UI

Every model binary serves a web UI for its own model: the Modeling App for one model and its developer.
Upload a dataset, backtest the model on it, see the scores by horizon and location and the forecasts against
the truth, and forecast past the end of the data. Comparing models across a country stays with the Modeling
App and chap-core.

## Opening it

```sh
baseline-seasonal-median serve
```

Then open <http://127.0.0.1:8000/ui/>. `/` redirects there (Python chapkit serves its console at `/`;
nothing in the chapkit API uses it). With API keys set (`SERVICEKIT_API_KEYS`), the UI's files are open
like `/docs`, but its API calls need a key.

The UI is React on [sokkel-ui](https://github.com/winterop-com/sokkel-ui), in `ui/`. `bun run build` there
writes `crates/chapkit-ui/dist/`, which is compiled into the binary (rust-embed): no files beside it, no
network. A build without it serves a page saying the UI is not built. Hashed files under `assets/` are
cached forever; `index.html` is revalidated on every load. Any other path under `/ui/` answers
`index.html`, since the UI routes on the client.

The UI is the default-on feature `ui` of the `chapkit` crate; `default-features = false` leaves it out.

## What it calls

Everything is in the service's `/openapi.json`. Besides the chapkit API (configs, artifacts, jobs, info),
it uses these chapkit-rs extensions:

| endpoint | |
|---|---|
| `POST /api/v1/datasets` | Store a dataset: JSON `{name, data, target?}`, or a CSV or JSON `file` as `multipart/form-data` (with `name`, `target`) |
| `GET /api/v1/datasets` | List datasets (paged with `page` and `size`, like configs) |
| `GET`, `DELETE /api/v1/datasets/{id}` | A dataset's description; delete it |
| `GET /api/v1/datasets/{id}/$data` | The data as stored, chapkit DataFrame JSON; `?location=A&location=B` for some locations |
| `POST /api/v1/ml/$backtest` | A backtest of `data` or a stored `dataset_id` |
| `POST /api/v1/ml/$forecast` | A forecast past the end of the dataset |
| `GET /api/v1/artifacts?type=ml_backtest` | Artifacts of one `data.type` |

**Datasets** are stored as given, after a check: `time_period` in any form chap-core reads (`2024-01`,
`2024W01`, chap-core's CSV week ranges `2024-01-01/2024-01-07`), a `location` on every row, no location
twice in a period, one kind of period, and a numeric target (`disease_cases` unless `target` names
another). The description:

```json
{"id": "01M7...", "name": "laos_subset", "created_at": "...", "updated_at": "...",
 "target": "disease_cases", "rows": 108, "locations": 3,
 "location_ids": ["Bokeo", "Savannakhet", "Vientiane[prefecture]"],
 "period_type": "month", "first_period": "2010-01", "last_period": "2012-12", "periods": 36,
 "columns": ["time_period", "rainfall", "mean_temperature", "disease_cases", "population", "location"],
 "covariates": ["rainfall", "mean_temperature", "population"]}
```

`first_period` and `last_period` are written as chapkit writes periods (`2024-01`, `2024W01`), as are the
periods of backtest and forecast results; `$data` returns the periods as uploaded.

**`$backtest`** takes `dataset_id` in place of `data`. Its result (the `<bin> eval --json` document) then
carries `dataset_id`, and always `forecasts`: every forecast of every split with the observed target
(`null` when missing) and the quantiles of the samples (numpy's linear method), enough to draw forecasts
against the truth without the samples:

```json
{"split": 0, "location": "Bokeo", "time_period": "2012-04", "horizon": 1,
 "observed": 2.0, "median": 0.5, "q10": 0.105, "q25": 0.255, "q75": 0.745, "q90": 0.895}
```

**`$forecast`** is chap-core's `forecast_ahead`: train on the whole dataset (prepared as for a backtest),
build each location's next `periods` periods with the weather provider, and predict.

```json
{"config_id": "01M7...", "dataset_id": "01M7...", "periods": 6, "weather": "climatology"}
```

`config_id` or `config` (or neither, for the defaults), `dataset_id` or `data`; `periods` defaults to the
config's `prediction_periods`; `weather` is `climatology` (default) or `damped_persistence`
(`damped-persistence` too). `observed` is refused: there is no observed future to read. Like `$backtest`
it answers 202 with a job and stores an `ml_forecast` artifact whose `data.content` is
`{dataset_id, config, periods, weather, forecasts: [{location, time_period, median, q10, q25, q75, q90}],
frame}`, `frame` being the model's prediction frame.

## Screens

The seasonal median model with the Laos monthly and weekly district datasets uploaded, each evaluated and
forecast, in the light and the dark scheme.

**Overview**: the model, what the service holds, and the latest evaluation's scores.

![Overview, light](ui/overview-light.png)
![Overview, dark](ui/overview-dark.png)

**Datasets**: upload a CSV or JSON file by the picker or by dropping it on the page.

![Datasets, light](ui/datasets-light.png)
![Datasets, dark](ui/datasets-dark.png)

**A dataset**: its facts, the target over time for the locations chosen, and its rows.

![A dataset, light](ui/dataset-light.png)
![A dataset, dark](ui/dataset-dark.png)

**Evaluate**: a backtest's settings, and the evaluations run before.

![Evaluate, light](ui/evaluate-light.png)
![Evaluate, dark](ui/evaluate-dark.png)

**An evaluation**: CRPS, MAE, RMSE and coverage, the scores by horizon and by location, and each split's
forecasts against what happened.

![An evaluation, light](ui/evaluation-light.png)
![An evaluation, dark](ui/evaluation-dark.png)

**Forecast**: a forecast's settings, and the forecasts made before.

![Forecast, light](ui/forecast-light.png)
![Forecast, dark](ui/forecast-dark.png)

**A forecast**: the recent history and the forecast with its intervals by location, its values, and a CSV
download.

![A forecast, light](ui/forecast-result-light.png)
![A forecast, dark](ui/forecast-result-dark.png)

**Configs**: named option sets, made from the model's JSON schema.

![Configs, light](ui/configs-light.png)
![Configs, dark](ui/configs-dark.png)

**Jobs**, refreshed while any runs, **Artifacts**, **System** (host, health and requests from `/metrics`)
and **API** (the routes of `/openapi.json`, with links to it and to `/docs`):

![Jobs, light](ui/jobs-light.png)
![Jobs, dark](ui/jobs-dark.png)
![Artifacts, light](ui/artifacts-light.png)
![Artifacts, dark](ui/artifacts-dark.png)
![System, light](ui/system-light.png)
![System, dark](ui/system-dark.png)
![API, light](ui/api-light.png)
![API, dark](ui/api-dark.png)
