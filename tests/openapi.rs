#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! The committed `openapi.json` must match what the service generates, so every API
//! change shows up in review as a diff of that file. Regenerate with `make openapi`.

use chapkit::{DataFrame, Model, ModelInfo, NoConfig, PredictInput, Result, TrainInput};

struct Noop;

impl Model for Noop {
    type Config = NoConfig;
    type Trained = ();
    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }
    fn train(&self, _: &NoConfig, _: TrainInput<'_>) -> Result<()> {
        Ok(())
    }
    fn predict(&self, (): &(), _: &NoConfig, input: PredictInput<'_>) -> Result<DataFrame> {
        chapkit::prediction_frame(input.future, 1, |_, s| {
            s.fill(0.0);
            Ok(())
        })
    }
}

#[test]
fn committed_openapi_matches_generated() {
    let (_, doc, ..) = chapkit::Service::new("chapkit-rs", "chapkit-rs", Noop)
        .database(":memory:")
        .app()
        .unwrap()
        .build();
    let mut generated = serde_json::to_string_pretty(&doc).unwrap();
    generated.push('\n');
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/openapi.json");
    if std::env::var("UPDATE_OPENAPI").is_ok() {
        std::fs::write(path, &generated).unwrap();
        return;
    }
    let committed = std::fs::read_to_string(path).unwrap_or_default();
    assert!(
        committed == generated,
        "openapi.json is out of date; run `make openapi` and commit the result"
    );
}
