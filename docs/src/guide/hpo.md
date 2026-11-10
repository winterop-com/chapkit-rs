# Hyperparameter search

`chapkit-hpo` (`chapkit::hpo`) searches a model's options. Each trial is an in-process backtest
([Backtests in-process](eval.md)), so trials cost milliseconds and run in parallel. It follows chap-core
2.7.0's `chap_core.hpo`: the same search space format, the same random sampling rules, the same metrics
as objectives and the same leaderboard.

## The search space

The model declares it with `Model::search_space`, in chap-core's format. This is the
`hpo_search_space` that `/api/v1/info` shows:

```rust,ignore
fn search_space(&self) -> Option<SearchSpace> {
    Some(
        SearchSpace::new()
            .with("spread_factor", Param::Float { low: 0.0, high: 3.0, step: None, log: false })
            .with("window", Param::Int { low: 2, high: 24, step: 2, log: false })
            .with("kind", Param::Categorical(vec![json!("mean"), json!("median")])),
    )
}
```

| parameter | random (chap-core's `RandomSearcher`) | grid |
|---|---|---|
| `Categorical` | one of the values | every value |
| `Int`, `step` | `low + k * step` | every `low + k * step` up to `high` |
| `Int`, `log` | `floor(exp(u))`, `u` uniform on `[ln low, ln(high + 1))` | every integer |
| `Float`, continuous | uniform | 5 evenly spaced points, ends included (`grid_points`) |
| `Float`, `step` | `low + k * step` | every `low + k * step` up to `high` |
| `Float`, `log` | log-uniform | 5 log-spaced points |

The grid is the cartesian product, with the last parameter varying fastest. chap-core's `GridSearcher` only
takes categorical lists; here ranges are enumerated too. chap-core's TPE searcher (Optuna) is not ported.

## From the command line

Every model binary has `hpo`. It takes the same data and backtest options as `eval`, plus the search:

```sh
configurable-baseline hpo --locations 10 --history 72 --trials 40 --seed 1
```

```text
configurable-baseline: random search, 40 trials (0 failed), crps, 7 splits of 3 periods, climatology weather
data: generated (seed 1), 10 locations x 72 months
search: 14.5 ms, 2767.1 trials/s, seed 1

rank    trial          crps      log_bias spread_factor
1          13      5.049862       -0.0526        2.2697
2           1      5.280308       -0.0738        2.1167
3          19      5.288790       -0.0721        2.2725
...
... 30 more

best: {"log_bias":-0.05255518007994586,"spread_factor":2.2697373422208056}
crps: 5.049861735911487 (CRPS 5.049862)
```

- `--metric` picks the objective. Only chap-core's proper scoring rules are objectives (CRPS and its
  forms, MAE, RMSE, the Winkler scores), and all are minimized. MAPE, coverage and the ratio above truth
  are rejected, as in chap-core's `get_optimization_direction`.
- `--config` gives options that every candidate is merged into. A candidate's values win, as in chap-core's
  `_make_configuration`.
- `--seed` seeds the generated data and the search. Without it, a seed is drawn and printed.
- `--json` prints the whole result: the backtest settings, the dataset, the base config, the search
  (`leaderboard`, `best_params`, `best_score`, `n_trials`, `stop_reason`, as in chap-core's
  `HyperparameterOptimization`), `best_config` (ready to store as a config), and `best`, the full
  `eval --json` result of the best config.

## As a job

`POST /api/v1/ml/$hpo` takes `config_id` or `config`, `data`, the backtest settings of `$backtest`, and
`trials`, `method` (`random`, `grid`), `metric`, `seed` (default 0), `grid_points`. It returns 202. The
result (the `--json` document) is stored as an `ml_hpo` artifact. With `"create_config": true`, the best
config is also created (named `config_name`, tagged `hpo`), and its id is written to the result as
`best_config_id`, ready for `$train`.

## From Rust

```rust,ignore
use chapkit::eval::{Climatology, Metric, Options as Backtest};
use chapkit::hpo::{Method, Options, search_model};

let options = Options { method: Method::Random, trials: 50, seed: 1, ..Options::default() };
let found = search_model(&model, &serde_json::json!({}), &data, &Backtest::default(), &Climatology, Metric::Crps, &options)?;
println!("{:?} {}", found.best_params, found.best_score);
```

`search` takes any objective (`Fn(&Params) -> Result<f64, String>`), and `candidates` returns what a search
would try. `search_model` splits the dataset and forecasts its weather once, then backtests every candidate
on those splits.

## How it runs

The candidates are drawn first, in order, from a seeded generator. The trials then run in parallel with
rayon, and every backtest is parallel inside too. A search therefore gives the same leaderboard for the
same seed, whatever the thread schedule. A trial fails when the model returns an error or the score is not
finite. Successful trials are ranked by score, ties in the order they were drawn, and failed trials come
last. When every trial fails, the search fails with the first failure. The budget is the trial count. A
grid smaller than the budget stops with `search_exhausted`.

## Speed

`make bench BENCH=hpo` runs a random search of 32 trials over a seasonal model's two options, on 20
locations x 120 months with chap-core's default backtest (7 splits of 3 periods, so each trial is one
training, 7 forecasts and the scores). It takes 14.2 ms (criterion's interval 12.2 to 16.3 ms), about
2250 trials per second, on an Apple Silicon laptop, measured 2026-11-10 while other builds were running.
Through chap-core, each trial is a backtest of tens of seconds.
