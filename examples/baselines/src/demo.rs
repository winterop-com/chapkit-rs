//! Run a model once in-process on the monthly fixture and time train and predict.

use std::time::Instant;

use chapkit::{DataFrame, Model, PredictInput, RunInfo, TrainInput, cols};
use serde::Deserialize;

/// Train, historic and future frames from a fixture file.
#[derive(Deserialize)]
pub struct Fixture {
    pub train: DataFrame,
    pub historic: DataFrame,
    pub future: DataFrame,
}

impl Fixture {
    /// The monthly fixture shipped with this crate (3 locations plus an unseen one, 40 months).
    pub fn monthly() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_str(&std::fs::read_to_string(
            fixture_path(),
        )?)?)
    }
}

/// Train and predict once, print timings and the median forecast per row.
pub fn run<M: Model>(
    name: &str,
    model: &M,
    config: &M::Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let fx = Fixture::monthly()?;
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

/// Path of the monthly fixture shipped with this crate.
pub fn fixture_path() -> &'static str {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/baselines_monthly.json"
    )
}

/// Train on the monthly fixture, save a checkpoint with provenance, round-trip it
/// through safetensors bytes, load it, and check that the loaded model predicts
/// exactly what the freshly trained one does. Returns the loaded checkpoint.
pub fn checkpoint_round_trip<M: Model>(
    model: &M,
    config: &M::Config,
) -> Result<chapkit::Checkpoint, Box<dyn std::error::Error>> {
    let fx = Fixture::monthly()?;
    let run = RunInfo::with_periods(6);
    let trained = model.train(config, TrainInput::new(&fx.train, &run, None))?;
    let mut ck = model.save(&trained)?;
    ck.provenance = chapkit::Provenance {
        model: "test".into(),
        ..chapkit::Provenance::default()
    }
    .with_training_data(&fx.train);
    let loaded_ck = chapkit::Checkpoint::from_bytes(&ck.to_bytes()?)?;
    if loaded_ck != ck {
        return Err("checkpoint changed in a safetensors round trip".into());
    }
    let loaded = model.load(&loaded_ck)?;
    let input = PredictInput::new(&fx.historic, &fx.future, &run, None);
    let a = serde_json::to_value(model.predict(&trained, config, input)?)?;
    let b = serde_json::to_value(model.predict(&loaded, config, input)?)?;
    if a != b {
        return Err("loaded checkpoint predicts differently".into());
    }
    Ok(loaded_ck)
}
