//! `<bin> hpo`: a hyperparameter search of the service's model, each trial an in-process
//! backtest ([`chapkit_hpo`]).

use std::fmt::Write as _;

use chapkit_eval::Metric;
use serde_json::Value;

use crate::cli::{HpoArgs, Method};
use crate::erased::DynModel;
use crate::eval_cmd::{Prepared, prepared};
use crate::evaluate::{HpoReport, hpo};

/// Leaderboard rows shown in the table.
const SHOWN: usize = 10;

/// Run the search `args` describe.
pub(crate) fn report(model: &dyn DynModel, id: &str, args: &HpoArgs) -> Result<HpoReport, String> {
    let metric =
        Metric::from_id(&args.metric).ok_or_else(|| format!("unknown metric {}", args.metric))?;
    let Prepared {
        options,
        config,
        data,
        source,
        seed,
    } = prepared(model, &args.backtest)?;
    let search = chapkit_hpo::Options {
        method: match args.method {
            Method::Random => chapkit_hpo::Method::Random,
            Method::Grid => chapkit_hpo::Method::Grid,
        },
        trials: args.trials,
        seed,
        grid_ranges: args.grid_ranges,
        ..chapkit_hpo::Options::default()
    };
    hpo(
        model,
        id,
        &config,
        &data,
        source,
        &options,
        args.backtest.weather.id(),
        metric,
        &search,
    )
}

/// A parameter value, compactly.
fn value(v: Option<&Value>) -> String {
    match v {
        Some(Value::Number(n)) if n.is_f64() => {
            let x = n.as_f64().unwrap_or(f64::NAN);
            if x != 0.0 && (x.abs() < 1e-3 || x.abs() >= 1e5) {
                format!("{x:.4e}")
            } else {
                format!("{x:.4}")
            }
        }
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => "-".into(),
    }
}

/// Run a search and return the report: the leaderboard, or with `--json` the whole result.
pub(crate) fn run(model: &dyn DynModel, id: &str, args: &HpoArgs) -> Result<String, String> {
    let r = report(model, id, args)?;
    if args.json {
        return serde_json::to_string_pretty(&r).map_err(|e| e.to_string());
    }
    let s = &r.search;
    let names: Vec<&String> = s.search_space.0.keys().collect();
    let mut out = format!(
        "{id}: {} search, {} trials ({} failed), {}, {} splits of {} periods, {} weather\n\
         data: {}, {} locations x {} {}\n\
         search: {:.1} ms, {:.1} trials/s, seed {}\n\n{:<6}{:>7}{:>14}",
        s.method.id(),
        s.n_trials,
        s.n_failed_trials,
        r.metric,
        r.backtest.splits,
        r.backtest.periods,
        r.backtest.weather,
        r.data.source,
        r.data.locations,
        r.data.periods,
        r.data.unit,
        s.seconds * 1000.0,
        s.n_trials as f64 / s.seconds.max(1e-9),
        s.seed,
        "rank",
        "trial",
        r.metric,
    );
    for n in &names {
        let _ = write!(out, "{n:>14}");
    }
    for (rank, t) in s.leaderboard.iter().take(SHOWN).enumerate() {
        let score = t
            .score
            .map_or_else(|| "failed".into(), |v| format!("{v:.6}"));
        let _ = write!(out, "\n{:<6}{:>7}{score:>14}", rank + 1, t.trial_nr);
        for n in &names {
            let _ = write!(out, "{:>14}", value(t.params.get(*n)));
        }
    }
    if s.leaderboard.len() > SHOWN {
        let _ = write!(out, "\n... {} more", s.leaderboard.len() - SHOWN);
    }
    let best = Value::Object(s.best_params.clone());
    let _ = write!(
        out,
        "\n\nbest: {best}\n{}: {} (CRPS {:.6})",
        r.metric,
        s.best_score,
        r.best
            .aggregate
            .get("crps")
            .copied()
            .flatten()
            .unwrap_or(f64::NAN),
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erased::Erased;
    use crate::eval_cmd::tests::{Last, backtest_args};

    fn args(method: Method, trials: usize) -> HpoArgs {
        HpoArgs {
            backtest: backtest_args(),
            trials,
            method,
            grid_ranges: false,
            metric: "crps".into(),
            json: false,
        }
    }

    #[test]
    fn searches_the_models_space_reproducibly() {
        let a = args(Method::Random, 12);
        let r = report(&Erased(Last), "last", &a).unwrap();
        let s = &r.search;
        assert_eq!((s.n_trials, s.n_successful_trials), (12, 12));
        assert_eq!(s.seed, 7, "--seed seeds the search too");
        let spreads: Vec<f64> = s
            .leaderboard
            .iter()
            .map(|t| t.params["spread"].as_f64().unwrap())
            .collect();
        assert!(spreads.iter().all(|x| (0.1..=20.0).contains(x)));
        // The best config is the best trial's, and its backtest scores the same.
        assert_eq!(r.best_config["spread"], s.best_params["spread"]);
        assert_eq!(r.best_config["prediction_periods"], 2);
        assert_eq!(r.best.aggregate["crps"], Some(s.best_score));
        let again = report(&Erased(Last), "last", &a).unwrap();
        assert_eq!(again.search.best_params, s.best_params);

        let text = run(&Erased(Last), "last", &a).unwrap();
        assert!(
            text.starts_with(
                "last: random search, 12 trials (0 failed), crps, 3 splits of 2 periods"
            ),
            "{text}"
        );
        assert!(
            text.contains("seed 7")
                && text.contains("... 2 more")
                && text.contains("best: {\"spread\":"),
            "{text}"
        );
        let json: Value = serde_json::from_str(
            &run(&Erased(Last), "last", &HpoArgs { json: true, ..a }).unwrap(),
        )
        .unwrap();
        assert_eq!(json["metric"], "crps");
        assert_eq!(json["search"]["leaderboard"].as_array().unwrap().len(), 12);
        assert!(json["best"]["aggregate"]["crps"].is_f64());
    }

    #[test]
    fn grid_and_options_and_metric() {
        let mut a = args(Method::Grid, 100);
        a.grid_ranges = true;
        a.metric = "mae".into();
        a.backtest.config = Some(r#"{"spread": 3}"#.into());
        let r = report(&Erased(Last), "last", &a).unwrap();
        // A continuous log range: 5 points, then exhausted.
        assert_eq!(r.search.n_trials, 5);
        assert_eq!(r.metric, "mae");
        assert_eq!(r.config["spread"], 3);
        assert_eq!(
            r.search
                .leaderboard
                .iter()
                .filter(|t| t.params["spread"] == 20.0)
                .count(),
            1
        );
        a.metric = "mape".into();
        assert!(
            report(&Erased(Last), "last", &a)
                .unwrap_err()
                .contains("not defined as a direct HPO objective")
        );
    }
}
