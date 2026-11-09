//! Run an example model in-process on the monthly fixture and time train and predict.
//! Stand-in `main` for the examples until the HTTP service layer exists.
#![allow(dead_code)]

use std::time::Instant;

use chapkit::{DataFrame, Model, PredictInput, RunInfo, TrainInput, cols};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    train: DataFrame,
    historic: DataFrame,
    future: DataFrame,
}

pub fn run<M: Model>(
    name: &str,
    model: &M,
    config: &M::Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/baselines_monthly.json"
    );
    let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let run = RunInfo::with_periods(6);

    let t = Instant::now();
    let trained = model.train(config, TrainInput::new(&fx.train, &run, None))?;
    let train_time = t.elapsed();
    let t = Instant::now();
    let pred = model.predict(
        &trained,
        config,
        PredictInput::new(&fx.historic, &fx.future, &run, None),
    )?;
    let predict_time = t.elapsed();

    println!(
        "{name}: train {train_time:?}, predict {predict_time:?}, {} rows",
        pred.n_rows()
    );
    let (tp, loc) = (
        pred.require(cols::TIME_PERIOD)?,
        pred.require(cols::LOCATION)?,
    );
    let samples: Vec<Vec<f64>> = pred
        .names()
        .iter()
        .filter(|n| n.starts_with("sample_"))
        .map(|n| pred.f64_column(n))
        .collect::<Result<_, _>>()?;
    for row in 0..pred.n_rows() {
        let mut v: Vec<f64> = samples.iter().filter_map(|s| s.get(row).copied()).collect();
        v.sort_by(f64::total_cmp);
        let median = v.get(v.len() / 2).copied().unwrap_or(f64::NAN);
        println!(
            "  {:<3} {}  median {median:>7.2}",
            loc.str_at(row).unwrap_or("?"),
            tp.str_at(row).unwrap_or("?")
        );
    }
    Ok(())
}
