#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Example models survive a safetensors round trip: predictions from a loaded
//! checkpoint equal predictions from the freshly trained state.

#[path = "../examples/common/empirical.rs"]
mod empirical;
#[path = "../examples/global_median/model.rs"]
mod global_median;
#[path = "../examples/ridge/linalg.rs"]
mod linalg;
#[path = "../examples/ridge/model.rs"]
mod ridge;
#[path = "../examples/seasonal_median/model.rs"]
mod seasonal_median;
#[path = "../examples/seasonal_shift_persistence/model.rs"]
mod seasonal_shift_persistence;

use chapkit::{Checkpoint, DataFrame, Model, PredictInput, Provenance, RunInfo, TrainInput};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    train: DataFrame,
    historic: DataFrame,
    future: DataFrame,
}

fn fixture() -> Fixture {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/baselines_monthly.json"
    ))
    .unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn round_trip<M: Model>(model: &M, config: &M::Config) -> Checkpoint {
    let fx = fixture();
    let run = RunInfo::with_periods(6);
    let trained = model
        .train(config, TrainInput::new(&fx.train, &run, None))
        .unwrap();
    let mut ck = model.save(&trained).unwrap();
    ck.provenance = Provenance {
        model: "test".into(),
        ..Provenance::default()
    }
    .with_training_data(&fx.train);
    let loaded_ck = Checkpoint::from_bytes(&ck.to_bytes().unwrap()).unwrap();
    assert_eq!(loaded_ck, ck);
    let loaded = model.load(&loaded_ck).unwrap();

    let input = PredictInput::new(&fx.historic, &fx.future, &run, None);
    let a = model.predict(&trained, config, input).unwrap();
    let b = model.predict(&loaded, config, input).unwrap();
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(&b).unwrap()
    );
    loaded_ck
}

#[test]
fn baselines_round_trip_as_json_state() {
    let ck = round_trip(&global_median::GlobalMedian, &chapkit::NoConfig {});
    assert!(ck.tensors.is_empty());
    assert_eq!(ck.provenance.trained_until.as_deref(), Some("2021-04"));
    round_trip(&seasonal_median::SeasonalMedian, &chapkit::NoConfig {});
    round_trip(
        &seasonal_shift_persistence::SeasonalShiftPersistence,
        &chapkit::NoConfig {},
    );
}

#[test]
fn ridge_round_trips_weights_as_tensors() {
    let ck = round_trip(&ridge::RidgeModel, &ridge::RidgeConfig::default());
    assert_eq!(ck.tensor("weights").unwrap().shape, vec![1]);
    assert!(ck.provenance.overlaps("2021-04"));
    assert!(!ck.provenance.overlaps("2021-05"));
}
