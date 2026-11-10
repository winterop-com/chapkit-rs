#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Searchers, ranking and the backtest objective.

mod common;

use chapkit_core::synth::{Generator, Spec};
use chapkit_core::{Param, SearchSpace};
use chapkit_eval::{Climatology, Metric, Options as BacktestOptions};
use chapkit_hpo::{
    Direction, Error, Method, Options, Params, StopReason, candidates, direction, search,
    search_model, with_params,
};
use common::Seasonal;
use serde_json::{Value, json};

fn space() -> SearchSpace {
    SearchSpace::new()
        .with(
            "kind",
            Param::Categorical(vec![json!("a"), json!("b"), json!("c")]),
        )
        .with(
            "n",
            Param::Int {
                low: 2,
                high: 10,
                step: 4,
                log: false,
            },
        )
        .with(
            "k",
            Param::Int {
                low: 1,
                high: 100,
                step: 1,
                log: true,
            },
        )
        .with(
            "lr",
            Param::Float {
                low: 1e-3,
                high: 1.0,
                step: None,
                log: true,
            },
        )
        .with(
            "x",
            Param::Float {
                low: -1.0,
                high: 1.0,
                step: None,
                log: false,
            },
        )
        .with(
            "q",
            Param::Float {
                low: 0.0,
                high: 1.0,
                step: Some(0.25),
                log: false,
            },
        )
}

fn random(trials: usize, seed: u64) -> Options {
    Options {
        method: Method::Random,
        trials,
        seed,
        ..Options::default()
    }
}

#[test]
fn random_samples_stay_in_their_domains_and_follow_the_seed() {
    let (c, stop) = candidates(&space(), &random(500, 3)).unwrap();
    assert_eq!((c.len(), stop), (500, StopReason::MaxTrials));
    let f = |p: &Params, k: &str| p[k].as_f64().unwrap();
    for p in &c {
        assert!(["a", "b", "c"].contains(&p["kind"].as_str().unwrap()));
        assert!([2, 6, 10].contains(&p["n"].as_i64().unwrap()));
        assert!((1..=100).contains(&p["k"].as_i64().unwrap()));
        assert!((1e-3..=1.0).contains(&f(p, "lr")));
        assert!((-1.0..=1.0).contains(&f(p, "x")));
        assert!([0.0, 0.25, 0.5, 0.75, 1.0].contains(&f(p, "q")));
    }
    // Every value of the small domains is drawn; the log scales favour small values.
    for (k, n) in [("kind", 3), ("n", 3), ("q", 5)] {
        let distinct: std::collections::BTreeSet<String> =
            c.iter().map(|p| p[k].to_string()).collect();
        assert_eq!(distinct.len(), n, "{k}");
    }
    let below = c.iter().filter(|p| f(p, "lr") < 0.0316).count();
    assert!((200..300).contains(&below), "{below}");
    assert!(c.iter().any(|p| p["k"] == 1));
    assert!(c.iter().any(|p| p["k"].as_i64().unwrap() > 50));
    assert_eq!(candidates(&space(), &random(500, 3)).unwrap().0, c);
    assert_ne!(candidates(&space(), &random(500, 4)).unwrap().0, c);
}

#[test]
fn grid_is_the_product_last_parameter_fastest() {
    let space = SearchSpace::new()
        .with("a", Param::Categorical(vec![json!(1), json!(2)]))
        .with(
            "b",
            Param::Float {
                low: 0.0,
                high: 1.0,
                step: None,
                log: false,
            },
        )
        .with(
            "c",
            Param::Float {
                low: 0.01,
                high: 1.0,
                step: None,
                log: true,
            },
        );
    let grid = |trials, grid_points| Options {
        method: Method::Grid,
        trials,
        grid_points,
        seed: 0,
    };
    let (c, stop) = candidates(&space, &grid(100, 3)).unwrap();
    assert_eq!((c.len(), stop), (18, StopReason::SearchExhausted));
    let row = |p: &Params| {
        (
            p["a"].clone(),
            p["b"].as_f64().unwrap(),
            p["c"].as_f64().unwrap(),
        )
    };
    assert_eq!(row(&c[0]), (json!(1), 0.0, 0.01));
    assert!((row(&c[1]).2 - 0.1).abs() < 1e-15);
    assert_eq!(row(&c[2]).2, 1.0);
    assert_eq!(row(&c[3]).1, 0.5);
    assert_eq!(row(&c[17]).0, json!(2));
    let (c, stop) = candidates(&space, &grid(18, 3)).unwrap();
    assert_eq!((c.len(), stop), (18, StopReason::MaxTrials));
    let (c, _) = candidates(&space, &grid(4, 3)).unwrap();
    assert_eq!(c.len(), 4);
}

#[test]
fn trials_are_ranked_and_failures_go_last() {
    let space = SearchSpace::new().with(
        "x",
        Param::Int {
            low: -5,
            high: 5,
            step: 1,
            log: false,
        },
    );
    let grid = Options {
        method: Method::Grid,
        trials: 100,
        ..Options::default()
    };
    let s = search(&space, &grid, Direction::Minimize, |p| {
        let x = p["x"].as_i64().unwrap() as f64;
        match x {
            -5.0 => Err("boom".into()),
            -4.0 => Ok(f64::NAN),
            _ => Ok((x - 2.0).powi(2)),
        }
    })
    .unwrap();
    assert_eq!(s.best_params["x"], 2);
    assert_eq!(s.best_score, 0.0);
    assert_eq!(
        (s.n_trials, s.n_successful_trials, s.n_failed_trials),
        (11, 9, 2)
    );
    assert_eq!(s.stop_reason, StopReason::SearchExhausted);
    // Ties (x = 1 and x = 3) keep the order they were drawn in.
    assert_eq!(s.leaderboard[1].params["x"], 1);
    assert_eq!(s.leaderboard[2].params["x"], 3);
    // Failures in the order they were drawn.
    assert_eq!(s.leaderboard[9].failure.as_deref(), Some("boom"));
    assert!(
        s.leaderboard[10]
            .failure
            .as_deref()
            .unwrap()
            .contains("non-finite")
    );
    let up = search(&space, &grid, Direction::Maximize, |p| {
        Ok(p["x"].as_f64().unwrap())
    })
    .unwrap();
    assert_eq!(up.best_params["x"], 5);
    let none = search(&space, &grid, Direction::Minimize, |_| Err("no".into())).unwrap_err();
    assert!(
        matches!(none, Error::NoSuccessfulTrials(ref m) if m == "no"),
        "{none}"
    );
    let zero = Options { trials: 0, ..grid };
    assert!(search(&space, &zero, Direction::Minimize, |_| Ok(0.0)).is_err());
}

#[test]
fn only_proper_scoring_rules_are_objectives() {
    assert_eq!(direction(Metric::Crps).unwrap(), Direction::Minimize);
    assert_eq!(direction(Metric::Rmse).unwrap(), Direction::Minimize);
    assert_eq!(
        direction(Metric::Winkler10_90Log1p).unwrap(),
        Direction::Minimize
    );
    for m in [Metric::Mape, Metric::Coverage10_90, Metric::RatioAboveTruth] {
        assert!(
            direction(m)
                .unwrap_err()
                .to_string()
                .contains("not defined as a direct HPO objective")
        );
    }
}

#[test]
fn candidates_merge_over_the_users_options() {
    let base = json!({"bias": 1, "other": "kept", "user_option_values": {"width": 0.5, "x": 1}});
    let mut params = Params::new();
    params.insert("width".into(), json!(0.1));
    assert_eq!(
        with_params(&base, &params),
        json!({"bias": 1, "other": "kept", "width": 0.1, "user_option_values": {"x": 1}})
    );
    assert_eq!(with_params(&Value::Null, &params), json!({"width": 0.1}));
}

#[test]
fn searches_a_model_by_backtest_reproducibly() {
    let data = Generator::new(Some(5))
        .training_data(&Spec {
            num_locations: 4,
            num_periods: 48,
            ..Spec::default()
        })
        .unwrap();
    let backtest = BacktestOptions {
        periods: 3,
        splits: 4,
        ..BacktestOptions::default()
    };
    let options = random(16, 11);
    let run = || {
        search_model(
            &Seasonal,
            &json!({}),
            &data,
            &backtest,
            &Climatology,
            Metric::Crps,
            &options,
        )
        .unwrap()
    };
    let s = run();
    assert_eq!(s.n_successful_trials, 16);
    let scores: Vec<f64> = s.leaderboard.iter().map(|t| t.score.unwrap()).collect();
    assert!(scores.windows(2).all(|w| w[0] <= w[1]));
    // The same seed, the same search, whatever the thread schedule.
    let again = run();
    let key = |s: &chapkit_hpo::Search| {
        s.leaderboard
            .iter()
            .map(|t| (t.trial_nr, t.score.map(f64::to_bits)))
            .collect::<Vec<_>>()
    };
    assert_eq!(key(&s), key(&again));
    // The best trial's score is a plain backtest of its config.
    let config: common::SeasonalConfig =
        serde_json::from_value(Value::Object(s.best_params.clone())).unwrap();
    let bt =
        chapkit_eval::backtest_model(&Seasonal, &config, &data, &backtest, &Climatology).unwrap();
    assert_eq!(bt.score(&data).unwrap().get(Metric::Crps), s.best_score);
    let json = serde_json::to_value(&s).unwrap();
    assert_eq!(json["method"], "random");
    assert_eq!(json["direction"], "minimize");
    assert_eq!(json["stop_reason"], "max_trials");
    assert_eq!(json["search_space"]["width"]["log"], true);
    assert!(json["leaderboard"][0]["trial_nr"].is_u64());
}
