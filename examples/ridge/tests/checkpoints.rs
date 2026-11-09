#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Ridge stores its weights as tensors and survives a safetensors round trip.

use baselines::demo::checkpoint_round_trip;
use ridge::{RidgeConfig, RidgeModel};

#[test]
fn ridge_round_trips_weights_as_tensors() {
    let ck = checkpoint_round_trip(&RidgeModel, &RidgeConfig::default()).unwrap();
    assert_eq!(ck.tensor("weights").unwrap().shape, vec![1]);
    assert!(ck.provenance.overlaps("2021-04"));
    assert!(!ck.provenance.overlaps("2021-05"));
}
