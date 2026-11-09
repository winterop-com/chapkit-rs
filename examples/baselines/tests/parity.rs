#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! The Rust baselines must reproduce chap-core's built-in baselines number for number.
//! The models are the example ports under `examples/`; fixtures come from
//! `scripts/gen_baseline_fixtures.py` (run against chap-core).

use std::collections::HashMap;

use chapkit::{DataFrame, Model, NoConfig, Period, PredictInput, RunInfo, TrainInput, cols};

use baselines::GlobalMedian;
use baselines::SeasonalMedian;
use baselines::SeasonalShiftPersistence;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    train: DataFrame,
    historic: DataFrame,
    future: DataFrame,
    predictions: HashMap<String, HashMap<String, Vec<f64>>>,
}

fn check<M: Model<Config = NoConfig>>(case: &str, name: &str, model: &M) {
    let raw = std::fs::read_to_string(format!(
        "{}/tests/fixtures/baselines_{case}.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let fx: Fixture = serde_json::from_str(&raw).unwrap();
    let run = RunInfo::with_periods(3);
    let trained = model
        .train(&NoConfig {}, TrainInput::new(&fx.train, &run, None))
        .unwrap();
    let pred = model
        .predict(
            &trained,
            &NoConfig {},
            PredictInput::new(&fx.historic, &fx.future, &run, None),
        )
        .unwrap();
    // chap-core keys periods by id (`202105`, `2021W15`); normalize both sides through `Period`.
    let norm = |loc: &str, tp: &str| format!("{loc}|{}", Period::parse(tp).unwrap());
    let expected: HashMap<String, &Vec<f64>> = fx.predictions[name]
        .iter()
        .map(|(k, v)| {
            let (l, t) = k.split_once('|').unwrap();
            (norm(l, t), v)
        })
        .collect();
    assert_eq!(pred.n_rows(), expected.len(), "{case}/{name}: row count");

    let tp = pred.require(cols::TIME_PERIOD).unwrap();
    let loc = pred.require(cols::LOCATION).unwrap();
    for row in 0..pred.n_rows() {
        let key = norm(loc.str_at(row).unwrap(), tp.str_at(row).unwrap());
        let want = expected[&key];
        let got: Vec<f64> = (0..want.len())
            .map(|i| pred.column(&cols::sample(i)).unwrap().f64_at(row).unwrap())
            .collect();
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-9 * w.abs().max(1.0),
                "{case}/{name} {key} sample_{i}: got {g}, want {w}"
            );
        }
    }
}

#[test]
fn global_median_matches_chap_core() {
    check("monthly", "global_median", &GlobalMedian);
    check("weekly", "global_median", &GlobalMedian);
}

#[test]
fn seasonal_median_matches_chap_core() {
    check("monthly", "seasonal_median", &SeasonalMedian);
    check("weekly", "seasonal_median", &SeasonalMedian);
}

#[test]
fn seasonal_shift_persistence_matches_chap_core() {
    check(
        "monthly",
        "seasonal_shift_persistence",
        &SeasonalShiftPersistence,
    );
    check(
        "weekly",
        "seasonal_shift_persistence",
        &SeasonalShiftPersistence,
    );
}

#[test]
fn quantile_levels_match_chap_core() {
    // Sampling [0, 1] returns the levels themselves.
    let l = baselines::empirical::quantile_samples(&[0.0, 1.0]).unwrap();
    assert_eq!(l.len(), baselines::empirical::N_SAMPLES);
    assert!((l[0] - 0.5 / 99.0).abs() < 1e-15);
    assert_eq!(l[49], 0.5);
    assert_eq!(l[50], 0.5);
    assert!((l[99] - 98.5 / 99.0).abs() < 1e-15);
}

#[test]
fn quantile_samples_edge_cases() {
    assert_eq!(
        baselines::empirical::quantile_samples(&[3.0]).unwrap(),
        vec![3.0; baselines::empirical::N_SAMPLES]
    );
    assert!(baselines::empirical::quantile_samples(&[]).is_err());
}

#[test]
fn median_per_level_even_and_odd() {
    let (a, b, c) = (vec![1.0, 10.0], vec![3.0, 20.0], vec![2.0, 30.0]);
    assert_eq!(
        baselines::empirical::median_per_level([&a, &b]).unwrap(),
        vec![2.0, 15.0]
    );
    assert_eq!(
        baselines::empirical::median_per_level([&a, &b, &c]).unwrap(),
        vec![2.0, 20.0]
    );
    assert!(baselines::empirical::median_per_level([&a, &vec![1.0]]).is_none());
}
