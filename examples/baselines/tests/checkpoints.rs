#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! The baselines survive a safetensors round trip; their state is stored as JSON.

use baselines::demo::checkpoint_round_trip;

#[test]
fn baselines_round_trip_as_json_state() {
    let ck = checkpoint_round_trip(&baselines::GlobalMedian, &chapkit::NoConfig {}).unwrap();
    assert!(ck.tensors.is_empty());
    assert_eq!(ck.provenance.trained_until.as_deref(), Some("2021-04"));
    checkpoint_round_trip(&baselines::SeasonalMedian, &chapkit::NoConfig {}).unwrap();
    checkpoint_round_trip(&baselines::SeasonalShiftPersistence, &chapkit::NoConfig {}).unwrap();
}
