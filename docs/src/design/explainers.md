# Explainers

`chapkit-xai` implements LIME and KernelSHAP, ported from `chap-rs`, generic over any value function
`f(features) -> value`. `POST /api/v1/ml/$explain` uses them to explain one forecast cell, a location and a
period, of a trained model.

## What is perturbed

A forecast at period `t` can depend on more than that month's covariates: a recursive model feeds each step
into the next, and climate models react to rain one to three months earlier. With `scope: "temporal"` (the
default) the features of a cell are:

| feature | value | frame |
|---|---|---|
| `rainfall` | the covariate at the cell | `future` |
| `rainfall@t-1`, `rainfall@t-2`, ... | the covariate at the location's earlier forecast steps | `future` |
| `rainfall@t-j` (further back) | the covariate in the location's last `lags` historic periods | `historic` |
| `disease_cases@t-j` | the target's observed values in those historic periods | `historic` |

Features are named by their offset from the cell's period, so the same name means the same input for every
cell: at the third forecast step, `rainfall@t-1` and `rainfall@t-2` are the first two steps and
`disease_cases@t-3` is the last observed count. Covariates come from `features` (default: every numeric
column of `future`); a lagged value that is missing is left out. `scope: "row"` keeps only the cell's own
covariates, which was the only behaviour before.

Each perturbation draws every feature from a normal around its column's background mean with its
background spread (the column over `historic`, all locations; `future` for covariates historic lacks),
writes the values into copies of both frames, and calls the model once with the whole frames; the value is
the mean of the samples at the cell. LIME weights perturbations by a Gaussian kernel on the standardized
distance (width `0.75 * sqrt(k)` for `k` features) and fits a weighted least squares surrogate.
KernelSHAP uses the same features with the background means as the reference; up to 12 features it
enumerates every coalition, above that it samples `n_samples` of them.

LIME importances are local slopes per standard deviation of each feature (the column's background SD),
so a weight of 5 means one SD more of that input moves the forecast by 5 cases. SHAP importances are
additive contributions that sum exactly to `prediction - baseline`.

## Options

| field | default | |
|---|---|---|
| `method` | `lime` | `lime` or `shap` |
| `scope` | `temporal` | `temporal` or `row` |
| `lags` | 3 | historic periods (at most 52) |
| `features` | numeric covariates of `future` | covariates to explain |
| `n_samples` | 500 | perturbations (LIME) or sampled coalitions (SHAP) |
| `seed` | 0 | |
| `kernel_width` | `0.75 * sqrt(k)` | LIME kernel width |

```json
{
  "artifact_id": "01K...",
  "historic": {"columns": ["time_period", "location", "rainfall", "mean_temperature", "disease_cases"], "data": []},
  "future": {"columns": ["time_period", "location", "rainfall", "mean_temperature"], "data": []},
  "location": "L1",
  "time_period": "2023-03",
  "lags": 3
}
```

The response lists `features` in input order and `importances` largest first, with:

- `r2`: weighted R2 of the surrogate on the model's values. For LIME, the fit of the linear surrogate on the
  perturbations; for SHAP, how well sums of contributions reproduce the coalition values (1 for an additive
  model). A low value means the importances describe the model poorly at this cell. Absent when the
  forecast does not vary.
- `effective_n` (LIME): `(sum w)^2 / sum w^2` of the kernel weights, the number of perturbations the fit
  effectively rests on. Far below `n_model_calls` means few perturbations land near the instance; widen
  `kernel_width` or raise `n_samples`.

## Threads

Evaluations run on rayon's pool, unless the batch is so cheap that one thread is faster. The explainer
times the first model call; if that times the number of calls is under 75 µs, it stays on one thread.
`make bench BENCH=xai_threads` measures the crossover: LIME, 100 calls, the model busy-waiting a fixed time
(Apple M2 Max, 12 cores, warm pool; medians):

| per call | one thread | rayon | auto |
|---|---|---|---|
| 0.5 µs | **68 µs** | 88 µs | 66 µs |
| 1 µs | 119 µs | **102 µs** | 101 µs |
| 2 µs | 218 µs | **99 µs** | 103 µs |
| 4 µs | 419 µs | **115 µs** | 118 µs |
| 8 µs | 820 µs | **148 µs** | 176 µs |
| 16 µs | 1.62 ms | **224 µs** | 245 µs |

Rayon's floor is about 90 µs per batch; one thread wins below about 0.75 µs per call (75 µs per batch).
Auto pays one sequential call before it decides, a few percent above rayon on costly models.
The library's `Options::parallel` can force either (`never` for callers that parallelize themselves).

## Measured: a recursive model

The comparison with chap-core 2.7.0's LIME used a model whose answer is known:
`y_t = 2 rainfall_t + 3 mean_temperature_t + 0.5 y_{t-1} + 10`, per location, from the last observed count,
on 4 locations with 60 historic months and 3 forecast months, explaining L1 at the third step (2023-03).
Seven inputs drive that cell; the true LIME weight of each is its slope times the column's background SD.
Defaults (temporal, `lags` 3, 21 features with population), seed 0:

| input | feature | slope | background SD | truth (slope x SD) | LIME weight |
|---|---|---|---|---|---|
| rainfall, step 3 | `rainfall` | 2 | 54.817 | 109.635 | 109.635 |
| rainfall, step 2 | `rainfall@t-1` | 1 | 54.817 | 54.817 | 54.817 |
| rainfall, step 1 | `rainfall@t-2` | 0.5 | 54.817 | 27.409 | 27.409 |
| temperature, step 3 | `mean_temperature` | 3 | 3.318 | 9.953 | 9.953 |
| temperature, step 2 | `mean_temperature@t-1` | 1.5 | 3.318 | 4.977 | 4.977 |
| temperature, step 1 | `mean_temperature@t-2` | 0.75 | 3.318 | 2.488 | 2.488 |
| last observed count | `disease_cases@t-3` | 0.125 | 201.423 | 25.178 | 25.178 |

Weights are cases per background SD, within 5e-12 of the truth at every `n` from 30 to 5000 and for
seeds 0 to 2; the other 14 features (population at every offset, covariates in historic months, older
counts) come out at most 5e-12 in magnitude; `r2` is 1. The row-only scope finds only the
first and fourth rows. The effective sample size is 43 of 100 perturbations, 105 of 200, 312 of 500 and
3618 of 5000.

Time per explanation, in one process (the median of 19 repeats after a first, cold one that also starts
the thread pool), the model rebuilding its state from `historic` on every call (about 11 µs):

| `n` | auto (rayon here) | one thread | first call, auto |
|---|---|---|---|
| 100 | 0.48 ms | 1.22 ms | 2.1 ms |
| 200 | 0.76 ms | 2.30 ms | 2.4 ms |
| 500 | 1.50 ms | 5.59 ms | 5.5 ms |
| 5000 | 11.4 ms | 54.9 ms | 17.2 ms |

Every batch here is above the 75 µs threshold, so auto picks rayon. `make bench BENCH=xai_forecast`
times the same model on 20 locations with 36 historic months: 1.08 ms at `n` 100, 4.42 ms at 500 and
38.3 ms at 5000.

**How this differs from chap-core.** chap-core's time-series LIME splits each history series (the target's
included) into segments and each future step into a feature, switches whole segments off in a binary mask,
fills an "off" segment with a neutral replacement, weights perturbations by their distance in mask space
and fits a ridge surrogate on the mask. Its importances say how much switching a segment on matters, in
the model's output units after its transform; they need many perturbations to separate from noise, and
ridge shrinks them. chapkit-rs perturbs values continuously around the background, one feature per
period, with an unregularized fit on standardized values, so on a model that is locally linear it
recovers the slopes exactly with `n` just above the number of features, and `r2` says whether that is the
case. The cost of a call is what makes the difference in practice: chap-core runs a full predict per
perturbation in Python; here each is a direct call.

**Why not batch perturbations as pseudo-locations?** It is tempting to evaluate all perturbations in one
predict call by giving each its own location. Models with per-location state, including all three
baselines, would then forecast for an unknown location and give wrong answers. One in-process call per
perturbation costs microseconds anyway.
