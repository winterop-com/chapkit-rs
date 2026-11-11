# Backtests in-process

`chapkit-eval` is chap-core's evaluation run inside the model binary: the same splits, the same future
weather and the same metrics, giving the same numbers, but with `train` and `predict` as plain function
calls. A backtest that takes chap-core most of a minute through the job flow takes milliseconds here, which
is what a developer iterating on a model, and hyperparameter search ([Hyperparameter search](hpo.md)), need.

chap-core stays the source of record: this is the developer's preview of its evaluation. The port follows
chap-core 2.7.0 and is checked against fixtures chap-core generates ([Parity](#parity)).

## From the command line

Every model binary has `eval`. Without `--data` it generates a dataset the way `test` does (the model's
required covariates, monthly or weekly as the model declares):

```sh
baseline-seasonal-median eval --data laos_subset.csv
```

```text
seasonal-median: 7 splits of 3 periods, stride 1, 1 training, climatology weather
data: laos_subset.csv, 3 locations x 36 months
backtest: 1.4 ms, 63 forecasts scored

metric                       all        h1        h2        h3
CRPS                       66.88     52.64     69.64     78.35
CRPS (log1p)              0.4446    0.3575    0.4303    0.5461
CRPS Normalized           0.0518    0.0408    0.0539    0.0607
MAE                        88.50     73.52     91.40    100.57
RMSE                      134.08    123.31    137.17    141.10
MAPE                       46.90     43.61     45.77     51.32
Coverage 10-90            0.3968    0.4762    0.3810    0.3333
Coverage 25-75            0.2540    0.3333    0.2381    0.1905
Winkler Score 10-90       478.74    343.56    504.43    588.23
Winkler Score 25-75       286.89    229.01    298.39    333.27
Winkler 10-90 (log1p)     3.6309    2.8236    3.4918    4.5773
Winkler 25-75 (log1p)     1.8549    1.4931    1.8009    2.2706
Ratio Above Truth         0.2735    0.3252    0.2581    0.2371
```

`all` is the aggregate chap-core stores for a backtest; `h1`, `h2`, ... are the same metrics by horizon.
`--by-location` adds a table by location. chap-core's own backtest of its `seasonal_median` on chap-core's
`example_data/laos_subset.csv` with the same settings gives the same aggregate (CRPS 66.8757872374539,
RMSE 134.08000750558756). The options are in the [command reference](../reference/cli.md#eval).

A dataset is a CSV file or chapkit's JSON (`{columns, data}`) with `time_period`, `location`,
`disease_cases` and the covariates, as chap-core reads them. Periods may be in any monthly or weekly form
chap-core reads: `2024-01`, `202401`, `2024-1`, `2024W03`, `2024-W03`, `2024SunW03`, `2024-S03`, and the
7-day ranges of chap-core's CSV export (`2019-01-07/2019-01-13`). It is prepared as chap-core prepares it
([Preparation](#what-it-does)); `--keep-gaps`, `--no-pad` and `--keep-untrained` turn parts of that off,
and the first line of the output says so.

The model's default configuration is used unless `--config` gives options, as JSON or `@file`, flat or
nested as chap-core sends them (`{"user_option_values": {...}}`). They go through the same checks and
defaults as `POST /api/v1/configs`; `prediction_periods` is always `--periods`:

```sh
configurable-baseline eval --data laos.csv --config '{"spread_factor": 1.5}'
```

`--json` prints the whole result instead of the tables, every number at full precision (each `f64` as
the shortest text that reads back to it), so it can be compared with chap-core's stored
`aggregateMetrics` digit for digit: the settings, the config that ran, the dataset (with any dropped
locations), the splits, and the metrics (`aggregate`, `by_horizon`, `by_location`), keyed by chap-core's
`metric_id`, `null` where undefined:

```sh
configurable-baseline eval --data laos.csv --json | jq .aggregate.crps
```

`forecasts` lists every forecast with its split, horizon, observed target and the median and 10th, 25th,
75th and 90th percentiles of its samples. `$backtest` runs the same backtest as a job of a running service,
on the data it is sent or a stored dataset, and stores this JSON as an `ml_backtest` artifact
([The web UI](ui.md), [HTTP endpoints](../reference/endpoints.md)).

## From Rust

```rust,ignore
use chapkit::eval::{Climatology, Options, backtest_model};

let options = Options::default(); // chap-core's: 3 periods, 7 splits, stride 1, 1 training, its preparation
let backtest = backtest_model(&model, &config, &data, &options, &Climatology)?;
let scores = backtest.score(&data)?;
println!("CRPS {}", scores.get(chapkit::eval::Metric::Crps));
```

`backtest` takes plain `train` and `predict` functions instead of a `Model`; `score` scores any list of
forecasts. Trainings run in parallel, then every split's forecast.

## What it does

**Preparation**, as chap-core's REST API loads every dataset (`DataSet.from_pandas(df, fill_missing=True)`):

1. Rows are grouped by location (locations sorted; a numeric location becomes text) and each location's
   rows are sorted by period.
2. A period twice in one location is an error naming both, always.
3. Gaps inside a location's series are filled with rows whose values are all missing.
4. Every location is padded with such rows to the dataset's first and last period, so all locations share
   one contiguous period range; a period missing for every location is filled too.

Filled numeric columns become floats. A text column that would need filling is an error, as it is in
chap-core. The period range is then contiguous, so a horizon counts periods, and a model sees missing rows
where chap-core's would: a model counting the periods since the last observation (chap-core's persistence)
gives chap-core's forecasts only with filling on.

| option (`Prepare`, `eval` flag) | default | off |
|---|---|---|
| `fill_gaps`, `--keep-gaps` | fill gaps (chap-core) | keep the rows as given; the periods are those that occur in the data |
| `pad`, `--no-pad` | pad to the full range (chap-core) | each location keeps its own first and last period |

`chapkit::eval::prepare` is the step on its own; a backtest runs it before splitting. Sunday-start weeks
cannot be filled or padded yet, and data mixing monthly and weekly periods is an error.

Limits keep a request from exhausting memory; all are checked before anything is allocated, with
memory estimated from the rows times their columns' widths (16 bytes a number, 4 a text code). The
prepared dataset holds at most 60 million rows (`MAX_PREPARED_ROWS`) and 16 GiB (`MAX_PREPARED_BYTES`):
100,000 locations of ten years take about 1.9 GB monthly and 8 GB weekly with six numeric columns. The
splits, built at once (the first training set, and per split its history, truth and generated future,
which is every location with history times the window, however sparse the data), hold at most 250
million rows (`MAX_SPLIT_ROWS`) and 32 GiB (`MAX_SPLIT_BYTES`) together: 100,000 locations of ten years
of monthly data with five covariates and 7 splits hold about 100 million rows and 12 GB.

**Untrained locations**, as chap-core's REST backtest (`validate_and_filter_dataset_for_evaluation`, after
preparation and masking): a location with no observed target before the test windows (in the first
`n - (periods + (splits - 1) * stride)` periods) is left out of the backtest, and when none is left the
backtest fails with chap-core's error. Its observations still count in the range `crps_norm` divides by,
as in chap-core's stored aggregate. `eval` names the count on its `data:` line (`1 dropped (no target before
the test windows)`) and the locations in `--json` (`data.dropped`). `Options::drop_untrained` (on by
default), `--keep-untrained` to keep them.

**Splits** (chap-core's `train_test_generator`). With `n` periods, the first training uses every period
up to index `n - (periods + (splits - 1) * stride + 1)`; split `i` has the history up to `stride * i`
periods later and forecasts the `periods` after it. With `--retrain N` the model is trained `N` times,
at splits `{(j * splits) / N}`: the first on the training set, the later ones on that split's history.

**Future weather**: what the model is given for the periods it forecasts.

| provider | |
|---|---|
| `climatology` (default) | per location and covariate, the mean of each month (or week; week 53 counts as 52) of the history. A season never observed gets the mean of the observed seasons' means, as chap-core's regression gives. `population` and text columns are carried forward from each location's last row. |
| `damped_persistence` (`--weather damped-persistence`) | the climatology plus the last anomaly from it, damped with the horizon: `climatology(T+h) + phi^h * (y(T) - climatology(T))`, with `phi` the lag-1 autocorrelation of the location's anomaly series, clipped to [0, 1]. The reference forecast chap-core recommends over plain climatology. As in chap-core, a covariate missing anywhere in a location's history gives a missing forecast for it. |
| `mstl_arima` (`--weather mstl-arima`) | chap-core's MSTL + ARIMA, ported from the statsmodels code it runs: per location and covariate, the seasonal component from MSTL (one STL pass, seasonal window 11), an ARIMA on the rest with `d` from KPSS (at most 2) and `p, q` in 0..3 by AICc, candidates with a root within 1.001 of the unit circle rejected (ARIMA(0,0,0) if none is left), and the seasonal component continued as the mean of the complete cycles. Needs more than two full cycles of history (25 months, 105 weeks); a covariate missing anywhere in a location's history gives a missing forecast for it. |
| `observed` | the covariates as observed in the window. Looks ahead: for diagnostics only, the scores are not comparable to production. |

**Metrics**, with chap-core's definitions: CRPS (the order-statistic estimator), CRPS on `log1p` values,
CRPS divided by the range of the observations, MAE, RMSE and MAPE of the sample median, coverage of the
10-90 and 25-75 percentile intervals (numpy's linear percentiles), Winkler scores of both intervals (also on
`log1p` values), and the share of samples above the truth. A forecast is scored only where the target was
observed. The aggregate is the mean over all (location, period, split) rows, skipping undefined values (MAPE
of a zero), and the root mean square for RMSE.

The CRPS normalisation follows chap-core's REST API, which stores the aggregate the Modeling App shows: it
uses the range of every observed target in the dataset (`chap evaluate` on the command line uses the test
windows only). RMSE by horizon or location is the root mean square per group. With nothing to score,
chap-core 2.7.0's grouped RMSE raises where its MAE returns no rows.

**Target masking** (`chapkit::eval::mask_target`): inside masked ranges `disease_cases` is hidden and
`target_masked` is 1; the window's future has it at 0, and a masked period inside the evaluation window is
an error, as in chap-core. Mask prepared data, as chap-core masks after padding. Not yet on the command line.

**Checks**, as chap-core's: every observed location and period of a window needs a forecast with finite
samples, there must be more periods than the windows cover, and `retrain` cannot exceed `splits`. With
`--keep-gaps` the periods are those in the data, and a horizon is the position among them.

## Parity

`scripts/gen_eval_fixtures.py` runs chap-core's own `train_test_generator`, `backtest` (with retraining)
and metrics on six generated datasets (monthly, weekly across ISO week 53, monthly with a masked range,
two messy ones, and one with a location that has no target before the test windows, which chap-core's own
filter drops), with a missing and a zero target in the evaluation window and a covariate never
observed in one season. The messy datasets, monthly and weekly, have a gap inside one location (in the
weekly one across 2020W53), a location starting late and one ending early, a period missing for every
location, and their rows shuffled; chap-core loads every dataset with `fill_missing=True`. It writes `crates/chapkit-eval/fixtures/eval_<case>.json`; the data comes from a fixed seed, so a
rerun with the same chap-core release gives the same files:

```sh
make eval-fixtures CHAP_CORE=../../chap-core   # runs $CHAP_CORE/.venv/bin/python
```

The tests in `crates/chapkit-eval/tests/parity.rs` check, for each case:

| check | tolerance | largest difference seen |
|---|---|---|
| the prepared dataset, row for row | exact | 0 |
| every split's windows (train end, history end, forecast periods) | exact | - |
| the locations dropped as untrained | exact | - |
| the future frame from the climatology, damped persistence and observed providers | 1e-9 relative | below 1e-13 |
| the future frame from MSTL + ARIMA | 1e-3 relative | 3.8e-4 |
| every metric (aggregate, by horizon, by location) from chap-core's forecasts of `seasonal_median` and three test models (one with samples below -1, whose log1p is `NaN`: numpy's percentile then makes the log-Winkler scores `NaN`) | 1e-12 relative | 1.4e-15 |
| whole backtests of the two test models (repeated in Rust; one counts periods since the last observation): forecasts and metrics | 1e-12 relative | 5.0e-16 |

A duplicate row is rejected, and with `--keep-gaps --no-pad` the periods-since-last-observation model no
longer matches chap-core (a test pins both).

The remaining differences are summation order (numpy sums pairwise), and for MSTL + ARIMA where
statsmodels' optimiser stops: some fits end at its 50-iteration limit before converging.

`scripts/gen_mstl_arima_fixtures.py` (also run by `make eval-fixtures`) records every stage of chap-core's
MSTL + ARIMA provider on 14 monthly and weekly series. The test in
`crates/chapkit-eval/src/weather/mstl_arima.rs` checks them stage by stage:

| stage | agreement |
|---|---|
| MSTL seasonal component | 6e-14 relative |
| `d` from KPSS | every series |
| the chosen `(p, d, q)` | every series with a noisy deseasonalised series (11 of 14) |
| forecasts | 4e-6 where statsmodels converged; 1.6e-3 where it stopped at 50 iterations |

The three series that differ in order are almost exactly predictable after deseasonalising: a constant, a
straight line and a dry series with one storm. Their likelihood grows without bound as the variance goes
to zero, so where statsmodels stops, and which candidates pass the root check, is down to rounding. The
constant series still forecasts the constant.

## Speed

`make bench BENCH=backtest`: one backtest of a seasonal-mean model (100 samples) on 136 locations x 300
weeks, 7 splits of 12 periods, climatology weather, then the scores (11424 forecasts of 100 samples):
112 ms (criterion's interval 103 to 123 ms) on an Apple Silicon laptop, measured 2026-11-10 while other builds
were running.
