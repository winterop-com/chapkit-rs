#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! `backtest_model_with` hands the caller's run info, with the options' horizon, to every train
//! and predict call, so models that read `additional_continuous_covariates` can be backtested.

use std::sync::Mutex;

use chapkit_core::synth::{Generator, Spec};
use chapkit_core::{
    DataFrame, Model, ModelInfo, NoConfig, PredictInput, Result, RunInfo, TrainInput,
    prediction_frame,
};
use chapkit_eval::{Climatology, Options, backtest_model, backtest_model_with};

/// Records the run info of every call and forecasts zeros.
#[derive(Default)]
struct Recorder {
    seen: Mutex<Vec<RunInfo>>,
}

impl Model for Recorder {
    type Config = NoConfig;
    type Trained = ();

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<()> {
        self.seen.lock().unwrap().push(input.run.clone());
        Ok(())
    }

    fn predict(&self, (): &(), _: &NoConfig, input: PredictInput<'_>) -> Result<DataFrame> {
        self.seen.lock().unwrap().push(input.run.clone());
        prediction_frame(input.future, 3, |_, out| {
            out.fill(0.0);
            Ok(())
        })
    }
}

fn data() -> DataFrame {
    let spec = Spec {
        num_locations: 2,
        num_periods: 36,
        ..Spec::default()
    };
    Generator::new(Some(7)).training_data(&spec).unwrap()
}

#[test]
fn passes_the_callers_covariates_with_the_options_horizon() {
    let model = Recorder::default();
    let options = Options {
        periods: 2,
        splits: 3,
        ..Options::default()
    };
    let run = RunInfo {
        prediction_periods: 99,
        additional_continuous_covariates: vec!["rainfall".into()],
        ..RunInfo::default()
    };
    backtest_model_with(&model, &NoConfig {}, &run, &data(), &options, &Climatology).unwrap();
    let seen = model.seen.lock().unwrap();
    assert!(!seen.is_empty());
    for r in seen.iter() {
        assert_eq!(r.prediction_periods, 2, "the options' horizon wins");
        assert_eq!(r.additional_continuous_covariates, ["rainfall"]);
    }
}

#[test]
fn backtest_model_passes_no_covariates() {
    let model = Recorder::default();
    let options = Options {
        periods: 2,
        splits: 3,
        ..Options::default()
    };
    backtest_model(&model, &NoConfig {}, &data(), &options, &Climatology).unwrap();
    let seen = model.seen.lock().unwrap();
    assert!(
        seen.iter()
            .all(|r| r.additional_continuous_covariates.is_empty())
    );
}
