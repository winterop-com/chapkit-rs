//! Checks a model package can run in its own tests.

use crate::{
    Checkpoint, DataFrame, Error, Model, PredictInput, Provenance, Result, RunInfo, TrainInput,
};

/// Train on `train`, save a checkpoint with provenance, round-trip it through safetensors bytes,
/// load it, and check the loaded model predicts exactly what the freshly trained one does.
/// Returns the loaded checkpoint for further assertions.
pub fn checkpoint_round_trip<M: Model>(
    model: &M,
    config: &M::Config,
    train: &DataFrame,
    historic: &DataFrame,
    future: &DataFrame,
    run: &RunInfo,
) -> Result<Checkpoint> {
    let trained = model.train(config, TrainInput::new(train, run, None))?;
    let mut checkpoint = model.save(&trained)?;
    checkpoint.provenance = Provenance {
        model: "test".into(),
        ..Provenance::default()
    }
    .with_training_data(train);
    let loaded_checkpoint = Checkpoint::from_bytes(&checkpoint.to_bytes()?)?;
    if loaded_checkpoint != checkpoint {
        return Err(Error::model(
            "checkpoint changed in a safetensors round trip",
        ));
    }
    let loaded = model.load(&loaded_checkpoint)?;
    let input = PredictInput::new(historic, future, run, None);
    let fresh = model.predict(&trained, config, input)?;
    let reloaded = model.predict(&loaded, config, input)?;
    if fresh != reloaded {
        return Err(Error::model("loaded checkpoint predicts differently"));
    }
    Ok(loaded_checkpoint)
}
