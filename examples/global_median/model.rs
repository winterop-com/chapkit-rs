use chapkit::{DataFrame, Model, ModelInfo, NoConfig, PredictInput, Result, TrainInput, cols};

use super::empirical::{GlobalMedianSamples, baseline_info, frame_from_rows};

/// Forecasts every period with all observations of the same location in the
/// training data. Port of chap-core `global_median` (version 2).
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalMedian;

impl Model for GlobalMedian {
    type Config = NoConfig;
    type Trained = GlobalMedianSamples;

    fn info(&self) -> ModelInfo {
        baseline_info(
            "Baseline that uses all observed values for the same location as forecast samples for every period, \
             without smoothing or model fitting.",
        )
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<GlobalMedianSamples> {
        GlobalMedianSamples::from_data(input.data, &self.info().target)
    }

    fn predict(
        &self,
        trained: &GlobalMedianSamples,
        _: &NoConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let loc = input.future.require(cols::LOCATION)?;
        let rows: Vec<Vec<f64>> = (0..input.future.n_rows())
            .map(|r| {
                trained
                    .for_location(loc.str_at(r).unwrap_or_default())
                    .map(<[f64]>::to_vec)
            })
            .collect::<Result<_>>()?;
        frame_from_rows(input.future, &rows)
    }
}
