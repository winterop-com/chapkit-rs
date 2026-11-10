//! A seasonal model with two options, shared by the tests and the benchmark.

use std::collections::HashMap;

use chapkit_core::{
    DataFrame, Model, ModelInfo, Param, Period, PredictInput, Result, SearchSpace, TrainInput,
    cols, prediction_frame,
};
use serde::{Deserialize, Serialize};

/// The mean of each location and period of the year, plus `bias`; `samples` samples spread
/// `width` (relative) around it.
pub struct Seasonal;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SeasonalConfig {
    #[serde(default = "width")]
    pub width: f64,
    #[serde(default)]
    pub bias: i64,
}

fn width() -> f64 {
    0.2
}

impl Model for Seasonal {
    type Config = SeasonalConfig;
    type Trained = HashMap<(String, u32), f64>;

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn search_space(&self) -> Option<SearchSpace> {
        Some(
            SearchSpace::new()
                .with(
                    "width",
                    Param::Float {
                        low: 0.01,
                        high: 1.0,
                        step: None,
                        log: true,
                    },
                )
                .with(
                    "bias",
                    Param::Int {
                        low: -4,
                        high: 4,
                        step: 2,
                        log: false,
                    },
                ),
        )
    }

    fn train(&self, _: &SeasonalConfig, input: TrainInput<'_>) -> Result<Self::Trained> {
        let (tp, loc) = (
            input.data.require(cols::TIME_PERIOD)?,
            input.data.require(cols::LOCATION)?,
        );
        let y = input.data.f64_column(cols::DISEASE_CASES)?;
        let mut sums: HashMap<(String, u32), (f64, f64)> = HashMap::new();
        for (row, v) in y.iter().enumerate() {
            let (Some(l), Some(p)) = (loc.str_at(row), tp.str_at(row).and_then(Period::parse))
            else {
                continue;
            };
            if v.is_nan() {
                continue;
            }
            let e = sums.entry((l.to_owned(), p.period_of_year())).or_default();
            e.0 += v;
            e.1 += 1.0;
        }
        Ok(sums.into_iter().map(|(k, (s, n))| (k, s / n)).collect())
    }

    fn predict(
        &self,
        trained: &Self::Trained,
        config: &SeasonalConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let (tp, loc) = (
            input.future.require(cols::TIME_PERIOD)?,
            input.future.require(cols::LOCATION)?,
        );
        prediction_frame(input.future, 50, |row, out| {
            let key = (
                loc.str_at(row).unwrap_or_default().to_owned(),
                tp.str_at(row)
                    .and_then(Period::parse)
                    .map_or(0, Period::period_of_year),
            );
            let mean = trained.get(&key).copied().unwrap_or(0.0) + config.bias as f64;
            for (i, s) in out.iter_mut().enumerate() {
                *s = mean * (1.0 + config.width * (2.0 * i as f64 / 49.0 - 1.0));
            }
            Ok(())
        })
    }
}
