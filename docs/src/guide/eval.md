# Backtests in-process

`chapkit-eval` is chap-core's evaluation run inside the model binary: the same splits, the same future
weather and the same metrics, giving the same numbers, but with `train` and `predict` as plain function
calls. A backtest that takes chap-core most of a minute through the job flow takes milliseconds here, which
is what a developer iterating on a model, and hyperparameter search, need.

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
`disease_cases` and the covariates, as chap-core reads them. The model's default configuration is used.
It is prepared as chap-core prepares it ([Preparation](#preparation)); `--keep-gaps` and `--no-pad` turn
parts of that off, and the first line of the output says so.

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
cannot be filled or padded yet.

**Splits** (chap-core's `train_test_generator`). With `n` periods, the first training uses every period
up to index `n - (periods + (splits - 1) * stride + 1)`; split `i` has the history up to `stride * i`
periods later and forecasts the `periods` after it. With `--retrain N` the model is trained `N` times,
at splits `{(j * splits) / N}`: the first on the training set, the later ones on that split's history.

**Future weather**: what the model is given for the periods it forecasts.

| provider | |
|---|---|
| `climatology` (default) | per location and covariate, the mean of each month (or week; week 53 counts as 52) of the history. A season never observed gets the mean of the observed seasons' means, as chap-core's regression gives. `population` and text columns are carried forward from each location's last row. |
| `observed` | the covariates as observed in the window. Looks ahead: for diagnostics only, the scores are not comparable to production. |

chap-core's damped persistence and MSTL + ARIMA providers are not ported yet.

**Metrics**, with chap-core's definitions: CRPS (the order-statistic estimator), CRPS on `log1p` values,
CRPS divided by the range of the observations, MAE, RMSE and MAPE of the sample median, coverage of the
10-90 and 25-75 percentile intervals (numpy's linear percentiles), Winkler scores of both intervals (also on
`log1p` values), and the share of samples above the truth. A forecast is scored only where the target was
observed. The aggregate is the mean over all (location, period, split) rows, skipping undefined values (MAPE
of a zero), and the root mean square for RMSE.

Two details follow chap-core's REST API, which stores the aggregate the Modeling App shows: the CRPS
normalisation uses the range of every observed target in the dataset (`chap evaluate` on the command line
uses the test windows only), and RMSE by horizon or location is the root mean square per group (chap-core
2.7.0's own grouped RMSE fails).

**Target masking** (`chapkit::eval::mask_target`): inside masked ranges `disease_cases` is hidden and
`target_masked` is 1; the window's future has it at 0, and a masked period inside the evaluation window is
an error, as in chap-core. Mask prepared data, as chap-core masks after padding. Not yet on the command line.

**Checks**, as chap-core's: every observed location and period of a window needs a forecast with finite
samples, there must be more periods than the windows cover, and `retrain` cannot exceed `splits`. With
`--keep-gaps` the periods are those in the data, and a horizon is the position among them.

## Parity

`scripts/gen_eval_fixtures.py` runs chap-core's own `train_test_generator`, `backtest` (with retraining)
and metrics on five generated datasets (monthly, weekly across ISO week 53, monthly with a masked range,
and two messy ones), with a missing and a zero target in the evaluation window and a covariate never
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
| the future frame from both providers | 1e-9 relative | 1.2e-15 |
| every metric (aggregate, by horizon, by location) from chap-core's forecasts of `seasonal_median` and two test models | 1e-12 relative | 1.4e-15 |
| whole backtests of the two test models (repeated in Rust; one counts periods since the last observation): forecasts and metrics | 1e-12 relative | 5.0e-16 |

A duplicate row is rejected, and with `--keep-gaps --no-pad` the periods-since-last-observation model no
longer matches chap-core (a test pins both).

The remaining differences are summation order (numpy sums pairwise).

## Speed

`make bench BENCH=backtest`: one backtest of a seasonal-mean model (100 samples) on 136 locations x 300
weeks, 7 splits of 12 periods, climatology weather, then the scores (11424 forecasts of 100 samples):
112 ms (criterion's interval 103 to 123 ms) on an Apple Silicon laptop, measured 2026-11-10 while other builds
were running.
