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
like `/docs`, but its API calls need a key: the first refused call asks for it, and the account menu
sets, changes or forgets it. The UI sends it as `X-API-Key` on every call, downloads included, keeps it
for the browser tab (session storage) and never puts it in a link.

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
| `GET /api/v1/datasets` | List datasets (paged with `page` and `size`, like configs); `q` keeps those whose name holds it |
| `GET`, `DELETE /api/v1/datasets/{id}` | A dataset's description; delete it |
| `GET /api/v1/datasets/{id}/$data` | The data as stored, chapkit DataFrame JSON; or a projection: `location` and `column` (repeatable) for some locations and columns, `before` and `last` for the last periods before one, `canonical=true` for the periods as chapkit writes them, in period order |
| `POST /api/v1/ml/$backtest` | A backtest of `data` or a stored `dataset_id` |
| `POST /api/v1/ml/$forecast` | A forecast past the end of the dataset |
| `GET /api/v1/artifacts?type=ml_backtest&order=desc` | Artifacts of one `data.type`, newest first with `order=desc` (paged with `page` and `size`); backtests and forecasts carry `data.summary` in place of their content; with `type`, `q` keeps those made from a dataset whose name holds it |
| `GET /api/v1/configs?q=` | Configs whose name holds `q` (`q` is a chapkit-rs extension) |

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

`first_period` and `last_period` are written as chapkit writes periods (`2024-01`, `2024W01`,
`2024SunW01` for a week starting on Sunday), as are the periods of backtest and forecast results;
`$data` returns the periods as uploaded unless asked for `canonical=true`. The UI asks for that and
never reads a period itself: a chart history of a forecast is one call,
`$data?location=A&location=B&column=cases&before=2024-07&last=12&canonical=true`.

`/api/v1/info` also says `max_body_bytes`, the largest request the service takes
(`CHAPKIT_MAX_BODY_MB`), so the upload dialog refuses a larger file before sending it.

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

## Search and shortcuts

The search in the top bar finds datasets and configs by name, and evaluations and forecasts by the
name of the dataset they were made from: a few of each, newest runs first, with how many match in
all. It asks the service with `q` (case-insensitive for ASCII letters), so a large list is never
downloaded. Its list ends in See all, the results page (`#/search?q=laos`): every match, 25 of each
kind at a time with More, narrowed by kind, the query kept in the address so Back from a result
returns to it. A config found opens on its own page (`#/configs/{id}`), its options with ways to
evaluate or forecast with it. The command palette goes to every page, runs the actions (upload a dataset, new
evaluation, forecast or config, set or change the API key, light, dark or system scheme, the
shortcuts) and lists the same search's results under them.

| keys | where | |
|---|---|---|
| ⌘K (Ctrl K off a Mac) | everywhere, in fields too | Open or close the command palette |
| `/` | everywhere | Search datasets, runs and configs |
| `?` | everywhere | Show the keyboard shortcuts |
| `g` then `o` `d` `e` `f` `c` `j` `a` `s` | everywhere | Go to Overview, Datasets, Evaluate, Forecast, Configs, Jobs, Artifacts, System |
| `u` | Datasets | Upload a dataset |
| `n` | Configs | New config |
| ⌘Enter (Ctrl Enter) | Evaluate, Forecast, in their fields too | Run the evaluation or forecast |

Single keys and `g` sequences are quiet while typing in a field and while a dialog or menu is open;
the second key of a sequence must come within a second, and Escape drops it. None needs Alt, so they
work on a Norwegian keyboard as on an American one. ⌘K over an open search result list closes it and
opens the palette.

## Screens

The seasonal median model with the Laos monthly and weekly district datasets uploaded, each evaluated and
forecast, in the light and the dark scheme.

**Overview**: the model, what the service holds, and the latest evaluation's scores.

![Overview, light](ui/overview-light.png)
![Overview, dark](ui/overview-dark.png)

**Datasets**: upload a CSV or JSON file by the picker or by dropping it on the page, naming its target
column (`disease_cases` by default).

![Datasets, light](ui/datasets-light.png)
![Datasets, dark](ui/datasets-dark.png)

**A dataset**: its facts, the target over time for the locations chosen, and its rows.

![A dataset, light](ui/dataset-light.png)
![A dataset, dark](ui/dataset-dark.png)

**Evaluate**: a backtest's settings, and the evaluations run before.

![Evaluate, light](ui/evaluate-light.png)
![Evaluate, dark](ui/evaluate-dark.png)

**An evaluation**: CRPS, MAE, RMSE and coverage, the scores by horizon and by location, and each split's
forecasts against what happened. A chart's Data button, here and on every page, shows its rows as a table.

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
