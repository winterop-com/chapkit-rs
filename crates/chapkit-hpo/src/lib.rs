//! Hyperparameter search for chapkit-rs models: chap-core's `chap_core.hpo`, with every trial an
//! in-process backtest.
//!
//! The search space is the model's [`SearchSpace`], in chap-core's format: integer and float
//! ranges (optionally on a `step` grid or log-uniform) and categorical choices. Two searchers:
//!
//! - [`Method::Random`]: chap-core's `RandomSearcher`, sampling with replacement by its rules
//!   (log-uniform, stepped grids, `floor(exp(u))` for log integers), from a seeded generator;
//! - [`Method::Grid`]: chap-core's `GridSearcher`: the cartesian product of every parameter's
//!   choices, the last parameter varying fastest. As in chap-core, it takes categorical
//!   parameters only and refuses a space with a range. With [`Options::grid_ranges`] it also
//!   steps through ranges: integer ranges and stepped floats value by value, a continuous float
//!   range in [`Options::grid_points`] points (log-spaced when `log`).
//!
//! Candidates are drawn first, then the trials run in parallel (rayon), so a search is the same
//! for the same seed however many threads run it. The budget is a trial count. The leaderboard
//! ranks successful trials by score, failed trials last, as chap-core's `HyperparameterOptimizer`.
//!
//! [`search`] takes any objective; [`search_model`] backtests a typed [`Model`] with
//! [`chapkit_eval`] and scores it by a [`Metric`] (chap-core's default for chapkit-rs: CRPS).
//!
//! Pure like `chapkit-core`: no IO, no async, no HTTP.

use std::time::Instant;

use chapkit_core::rng::Rng;
use chapkit_core::{DataFrame, Model, Param, PredictInput, RunInfo, SearchSpace, TrainInput};
use chapkit_eval::{
    Metric, Options as BacktestOptions, WeatherProvider, backtest_splits, train_test_splits,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};

/// One candidate: a value per search-space parameter.
pub type Params = Map<String, Value>;

/// A search that cannot run.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Bad search space, options, metric or data.
    #[error("{0}")]
    Invalid(String),
    /// Every trial failed (chap-core's error), with the first trial's failure.
    #[error("Hyperparameter optimization completed without any successful trials: {0}")]
    NoSuccessfulTrials(String),
}

/// How candidates are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// Sample each parameter at random (with replacement).
    #[default]
    Random,
    /// Every combination of the parameters' grid values, in order.
    Grid,
}

impl Method {
    /// `random` or `grid`.
    pub fn id(self) -> &'static str {
        match self {
            Method::Random => "random",
            Method::Grid => "grid",
        }
    }
}

/// Whether lower or higher scores are better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Minimize,
    Maximize,
}

/// chap-core's `get_optimization_direction`: a proper scoring rule with a direction. MAPE,
/// coverage and the ratio above truth are not objectives.
pub fn direction(metric: Metric) -> Result<Direction, Error> {
    match metric {
        Metric::Mape | Metric::Coverage10_90 | Metric::Coverage25_75 | Metric::RatioAboveTruth => {
            Err(Error::Invalid(format!(
                "Metric '{}' is not defined as a direct HPO objective. Choose a proper scoring rule \
             with an explicit optimization direction.",
                metric.id()
            )))
        }
        _ => Ok(Direction::Minimize),
    }
}

/// Search settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// The searcher.
    pub method: Method,
    /// The most trials to run (chap-core's `max_trials`).
    pub trials: usize,
    /// Seed of the random searcher.
    pub seed: u64,
    /// Let the grid searcher step through ranges (chap-core's grid refuses them).
    pub grid_ranges: bool,
    /// Points a continuous float range is split into by the grid searcher with `grid_ranges`.
    pub grid_points: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            method: Method::Random,
            trials: 20,
            seed: 0,
            grid_ranges: false,
            grid_points: 5,
        }
    }
}

/// Why a search stopped (chap-core's `HpoStopReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The trial budget was used.
    MaxTrials,
    /// The grid had fewer combinations than the budget.
    SearchExhausted,
}

/// One trial (chap-core's `Trial`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    /// Position in the order candidates were drawn.
    pub trial_nr: usize,
    pub params: Params,
    /// `None` when the trial failed.
    pub score: Option<f64>,
    pub seconds: f64,
    pub failure: Option<String>,
}

/// A finished search (chap-core's flattened `HyperparameterOptimization`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Search {
    pub method: Method,
    pub direction: Direction,
    pub seed: u64,
    pub max_trials: usize,
    pub search_space: SearchSpace,
    pub best_params: Params,
    pub best_score: f64,
    /// Successful trials best first, then failed ones.
    pub leaderboard: Vec<Trial>,
    pub n_trials: usize,
    pub n_successful_trials: usize,
    pub n_failed_trials: usize,
    pub seconds: f64,
    pub stop_reason: StopReason,
}

fn number(v: f64) -> Value {
    Number::from_f64(v).map_or(Value::Null, Value::Number)
}

/// chap-core's `RandomSearcher._sample_one`.
fn sample(param: &Param, rng: &mut Rng) -> Value {
    match *param {
        Param::Categorical(ref values) => {
            let i = rng.int_range(0, values.len() as i64 - 1) as usize;
            values.get(i).cloned().unwrap_or(Value::Null)
        }
        Param::Float {
            low,
            high,
            log: true,
            ..
        } => number(rng.uniform(low.ln(), high.ln()).exp()),
        Param::Float {
            low,
            high,
            step: None,
            ..
        } => number(rng.uniform(low, high)),
        Param::Float {
            low,
            high,
            step: Some(step),
            ..
        } => {
            let n = ((high - low) / step + 1e-12).floor() as i64;
            number(low + rng.int_range(0, n) as f64 * step)
        }
        Param::Int {
            low,
            high,
            log: true,
            ..
        } => {
            let u = rng.uniform((low as f64).ln(), ((high + 1) as f64).ln());
            Value::from((u.exp().floor() as i64).clamp(low, high))
        }
        Param::Int {
            low, high, step, ..
        } => Value::from(low + rng.int_range(0, (high - low) / step) * step),
    }
}

/// Number of grid values of `param`.
fn grid_len(param: &Param, grid_points: usize) -> usize {
    match *param {
        Param::Categorical(ref values) => values.len(),
        Param::Int {
            low, high, step, ..
        } => ((high - low) / step.max(1)) as usize + 1,
        Param::Float {
            low,
            high,
            step: Some(step),
            ..
        } => ((high - low) / step + 1e-12).floor() as usize + 1,
        Param::Float { step: None, .. } => grid_points.max(2),
    }
}

/// The `k`-th grid value of `param`.
fn grid_value(param: &Param, k: usize, grid_points: usize) -> Value {
    match *param {
        Param::Categorical(ref values) => values.get(k).cloned().unwrap_or(Value::Null),
        Param::Int { low, step, .. } => Value::from(low + k as i64 * step),
        Param::Float {
            low,
            step: Some(step),
            ..
        } => number(low + k as f64 * step),
        Param::Float { low, high, log, .. } => {
            let last = grid_points.max(2) - 1;
            let t = k as f64 / last as f64;
            // The ends exactly, whatever the rounding in between.
            if k == 0 {
                number(low)
            } else if k >= last {
                number(high)
            } else if log {
                number((low.ln() + t * (high.ln() - low.ln())).exp())
            } else {
                number(low + t * (high - low))
            }
        }
    }
}

/// The candidates a search tries, in order, and why it stops after them.
pub fn candidates(
    space: &SearchSpace,
    options: &Options,
) -> Result<(Vec<Params>, StopReason), Error> {
    space.validate().map_err(Error::Invalid)?;
    if options.trials == 0 {
        return Err(Error::Invalid("trials must be at least 1".into()));
    }
    match options.method {
        Method::Random => {
            let mut rng = Rng::new(options.seed);
            let out = (0..options.trials)
                .map(|_| {
                    space
                        .0
                        .iter()
                        .map(|(name, p)| (name.clone(), sample(p, &mut rng)))
                        .collect()
                })
                .collect();
            Ok((out, StopReason::MaxTrials))
        }
        Method::Grid => {
            if !options.grid_ranges
                && let Some((name, _)) = space
                    .0
                    .iter()
                    .find(|(_, p)| !matches!(p, Param::Categorical(_)))
            {
                return Err(Error::Invalid(format!(
                    "grid search takes choices only, as chap-core's does, and {name} is a range; \
                     use random search, or let the grid step through ranges"
                )));
            }
            let lens: Vec<usize> = space
                .0
                .values()
                .map(|p| grid_len(p, options.grid_points))
                .collect();
            let total = lens.iter().fold(1usize, |a, n| a.saturating_mul(*n));
            let out = (0..total.min(options.trials))
                .map(|i| {
                    // Mixed radix, the last parameter fastest (itertools.product).
                    let mut rest = i;
                    let mut picks = vec![0; lens.len()];
                    for (pick, n) in picks.iter_mut().zip(&lens).rev() {
                        *pick = rest % n;
                        rest /= n;
                    }
                    space
                        .0
                        .iter()
                        .zip(picks)
                        .map(|((name, p), k)| (name.clone(), grid_value(p, k, options.grid_points)))
                        .collect()
                })
                .collect();
            let stop = if total < options.trials {
                StopReason::SearchExhausted
            } else {
                StopReason::MaxTrials
            };
            Ok((out, stop))
        }
    }
}

/// Run `objective` on every candidate, in parallel, and rank the trials. A trial fails when the
/// objective returns an error or a non-finite score.
pub fn search<F>(
    space: &SearchSpace,
    options: &Options,
    direction: Direction,
    objective: F,
) -> Result<Search, Error>
where
    F: Fn(&Params) -> Result<f64, String> + Sync,
{
    let start = Instant::now();
    let (candidates, stop_reason) = candidates(space, options)?;
    let trials: Vec<Trial> = candidates
        .into_par_iter()
        .enumerate()
        .map(|(trial_nr, params)| {
            let t = Instant::now();
            let result = objective(&params).and_then(|score| {
                if score.is_finite() {
                    Ok(score)
                } else {
                    Err(format!("Objective returned non-finite score: {score}"))
                }
            });
            let (score, failure) = match result {
                Ok(s) => (Some(s), None),
                Err(e) => (None, Some(e)),
            };
            Trial {
                trial_nr,
                params,
                score,
                seconds: t.elapsed().as_secs_f64(),
                failure,
            }
        })
        .collect();
    let n_trials = trials.len();
    let (mut leaderboard, failed): (Vec<Trial>, Vec<Trial>) =
        trials.into_iter().partition(|t| t.score.is_some());
    if leaderboard.is_empty() {
        let first = failed
            .first()
            .and_then(|t| t.failure.clone())
            .unwrap_or_default();
        return Err(Error::NoSuccessfulTrials(first));
    }
    // A stable sort: ties keep the order the candidates were drawn in.
    leaderboard.sort_by(|a, b| {
        let (a, b) = (a.score.unwrap_or(f64::NAN), b.score.unwrap_or(f64::NAN));
        match direction {
            Direction::Minimize => a.total_cmp(&b),
            Direction::Maximize => b.total_cmp(&a),
        }
    });
    let n_successful_trials = leaderboard.len();
    let (best_params, best_score) = leaderboard
        .first()
        .map_or_default(|t| (t.params.clone(), t.score.unwrap_or(f64::NAN)));
    leaderboard.extend(failed);
    Ok(Search {
        method: options.method,
        direction,
        seed: options.seed,
        max_trials: options.trials,
        search_space: space.clone(),
        best_params,
        best_score,
        n_failed_trials: n_trials - n_successful_trials,
        n_trials,
        n_successful_trials,
        leaderboard,
        seconds: start.elapsed().as_secs_f64(),
        stop_reason,
    })
}

/// `base` (a config's `data`) with `params` set, as chap-core's `_make_configuration` merges a
/// candidate into the user's options: the candidate wins, other options are kept. A key also
/// under `user_option_values` is taken out there, so the candidate is what the model reads.
pub fn with_params(base: &Value, params: &Params) -> Value {
    let mut out = base.as_object().cloned().unwrap_or_default();
    if let Some(Value::Object(nested)) = out.get_mut("user_option_values") {
        for k in params.keys() {
            nested.remove(k);
        }
    }
    for (k, v) in params {
        out.insert(k.clone(), v.clone());
    }
    Value::Object(out)
}

/// Search `model`'s search space: each trial backtests the model with `base` and the candidate's
/// values on `data` (split once), scored by `metric`. Each call's [`RunInfo`] has the backtest's
/// horizon and the merged config's `additional_continuous_covariates`, as the service gives them.
pub fn search_model<M: Model>(
    model: &M,
    base: &Value,
    data: &DataFrame,
    backtest: &BacktestOptions,
    weather: &dyn WeatherProvider,
    metric: Metric,
    options: &Options,
) -> Result<Search, Error> {
    let space = model
        .search_space()
        .ok_or_else(|| Error::Invalid("the model has no search space".into()))?;
    let direction = direction(metric)?;
    let splits =
        train_test_splits(data, backtest, weather).map_err(|e| Error::Invalid(e.to_string()))?;
    let periods = u32::try_from(backtest.periods).unwrap_or(u32::MAX);
    search(&space, options, direction, |params| {
        let merged = with_params(base, params);
        let run = RunInfo::for_config(periods, &merged);
        let config: M::Config = serde_json::from_value(merged).map_err(|e| e.to_string())?;
        let bt = backtest_splits(
            &splits,
            backtest,
            |d| model.train(&config, TrainInput::new(d, &run, None)),
            |t, h, f| model.predict(t, &config, PredictInput::new(h, f, &run, None)),
        )
        .map_err(|e| e.to_string())?;
        Ok(bt.score(data).map_err(|e| e.to_string())?.get(metric))
    })
}
