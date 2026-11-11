# Proposal for chap-core: `$predict-splits`

A proposal for the chap-core team: an optional chapkit endpoint that returns the forecasts of every backtest split
in one call, so a backtest of a chapkit model makes one round trip per training instead of one per split. chap-core
keeps everything it does today: it splits the data, fills the future weather, scores the forecasts and stores them.
Services that do not offer the endpoint are run exactly as now.

**Status:** chapkit-rs implements the endpoint as described below (see [Implemented in
chapkit-rs](#implemented-in-chapkit-rs)); chap-core does not use it yet.

## The problem, measured

In the [full-stack run](../guide/full-stack.md) (chap-core 2.7.0 from `master`, 136 districts, weekly, 12 periods,
7 splits, stride 4), a backtest of a chapkit-rs baseline took 42 to 48 s through chap-core, against 18 to 21 s for
chap-core's own built-in version of the same model, which returns the same numbers. The model service's share of
that was well under a second:

| step | time |
|---|---|
| chap-core prepares the run before `$train` | about 9 s |
| `$train` in the service | 151 ms |
| each of the 7 `$predict` calls in the service | 40 to 90 ms |
| chap-core around each `$predict` (data conversion, request, job poll, artifact read) | about 4 s |

With one forecast per split, the per-split overhead is most of the backtest, and it grows with the number of
splits. The same backtest run in-process by chapkit-rs' `eval` takes 28 to 33 ms for the two medians and gives
chap-core's stored scores for all four baselines.

## The endpoint

`POST /api/v1/ml/$predict-splits`, advertised in `/api/v1/info` under `extensions` (chapkit-rs already lists
`$predict-sync` and `$explain` there; Python chapkit services list nothing and are unaffected).

Request: the trained artifact and the inputs chap-core already builds for each split, in the chapkit dataframe
format:

```json
{
  "artifact_id": "01K...",
  "config_id": "01K...",
  "splits": [
    {"historic": {"columns": [...], "data": [...]}, "future": {"columns": [...], "data": [...]}},
    {"historic": ..., "future": ...}
  ]
}
```

Response: one prediction frame per split, in order, in the same format `$predict` stores today (`time_period`,
`location`, `sample_0` ... `sample_n`). It answers directly when the work is quick, or as a job whose artifact holds
the list, like `$predict`.

To keep the request small, `historic` can be sent once with each split giving only the period it ends at; the
service cuts the history itself. The future frames are what chap-core's weather provider produced, so the provider
stays in chap-core.

## What chap-core keeps

- The split points, the retraining schedule (`n_retrain`: one `$train` per training, as now), the weather provider
  and target masking.
- Scoring and the stored backtest. The service returns forecasts only, so there is one place that computes metrics.
- The flow for services without the extension: unchanged.

## Expected effect

For the run above: 7 `$predict` round trips become 1, so the per-split overhead (about 28 s of the 42 s) mostly
goes, and the time left is chap-core's preparation, one training and one prediction call. The gain grows with
the number of splits.

## Implemented in chapkit-rs

`POST /api/v1/ml/$predict-splits` takes `artifact_id`, an optional `config_id` (checked against the training
artifact's config), `run_info` and `geo` as in `$predict`, and `splits`. Each split gives `future` and either its
own `historic` or `historic_end`, the last period of a shared top-level `historic` it sees. Every split runs with the
warm trained model in one blocking call, and its frame equals what `$predict` stores for that split alone (pinned
in `tests/contract.rs`).

- Done within the eager budget (`CHAPKIT_EAGER_WAIT_MS`, default 1 s): 200 with `{"predictions": [frame, ...]}`,
  nothing stored, like `$predict-sync`.
- Otherwise: 202 with `job_id` and `artifact_id`, like `$predict`; the job's `ml_prediction_splits` artifact,
  a child of the training artifact, holds the list in `data.content`.
- 404 for an unknown training artifact; 400 for an empty `splits`, a split without history, a `historic_end`
  that is not a period or cuts no rows, a frame without rows or without `time_period` and `location`, or another
  config; 422 for a body that is not chapkit dataframe JSON.

Measured in-process with criterion (`make bench BENCH=backtest_forecasts`; HTTP router in-process, no network),
136 locations x 300 weeks, 7 splits of 12 weeks, stride 4, 100 samples per forecast:

| | time | request size |
|---|---|---|
| 7 `$predict` calls, each with its job status and artifact read (chap-core's flow) | 302 ms | 19.1 MB |
| one `$predict-splits`, each split with its own history | 161 ms | 19.1 MB |
| one `$predict-splits`, one shared history | 106 ms | 3.6 MB |

In-process the gain is a factor 2 to 3, mostly from not storing and re-reading seven prediction artifacts and from
parsing the history once. The larger gain is outside the service: chap-core's roughly 4 s around each `$predict`
in the full-stack run is paid once instead of seven times.

## Questions for the chap-core team

1. Is a per-service extension the right shape, or should chap-core's REST wrapper batch the splits for every
   chapkit service, with a fallback to one call per split?
2. Should the endpoint also accept a model with several trainings in one call (`n_retrain > 1`), or is one call per
   training simpler to reason about?
3. Python chapkit could offer the same endpoint by looping over `$predict` internally; would chap-core want it
   there too, so the extension is not chapkit-rs only?
