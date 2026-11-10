#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! One backtest of a simple seasonal model: 136 locations x 300 weekly periods, 7 splits of 12
//! periods, climatology weather, then the scores.

use std::collections::HashMap;
use std::hint::black_box;

use chapkit_core::synth::{Generator, Resolution, Spec};
use chapkit_core::{
    DataFrame, Model, ModelInfo, NoConfig, Period, PredictInput, Result, TrainInput, cols,
    prediction_frame,
};
use chapkit_eval::{Climatology, Options, backtest_model};
use criterion::{Criterion, criterion_group, criterion_main};

/// Mean of each location and week of year; 100 samples spread +-20% around it.
struct SeasonalMean;

impl Model for SeasonalMean {
    type Config = NoConfig;
    type Trained = HashMap<(String, u32), f64>;

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<Self::Trained> {
        let (tp, loc) = (
            input.data.require(cols::TIME_PERIOD)?,
            input.data.require(cols::LOCATION)?,
        );
        let y = input.data.f64_column(cols::DISEASE_CASES)?;
        let mut sums: HashMap<(String, u32), (f64, f64)> = HashMap::new();
        for (row, v) in y.iter().enumerate() {
            let p = Period::parse(tp.str_at(row).unwrap()).unwrap();
            let e = sums
                .entry((loc.str_at(row).unwrap().to_owned(), p.period_of_year()))
                .or_default();
            e.0 += v;
            e.1 += 1.0;
        }
        Ok(sums.into_iter().map(|(k, (s, n))| (k, s / n)).collect())
    }

    fn predict(
        &self,
        trained: &Self::Trained,
        _: &NoConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let (tp, loc) = (
            input.future.require(cols::TIME_PERIOD)?,
            input.future.require(cols::LOCATION)?,
        );
        prediction_frame(input.future, 100, |row, out| {
            let p = Period::parse(tp.str_at(row).unwrap()).unwrap();
            let mean = trained
                .get(&(loc.str_at(row).unwrap().to_owned(), p.period_of_year()))
                .copied()
                .unwrap_or(0.0);
            for (i, s) in out.iter_mut().enumerate() {
                *s = mean * (0.8 + 0.4 * i as f64 / 99.0);
            }
            Ok(())
        })
    }
}

fn bench(c: &mut Criterion) {
    let spec = Spec {
        num_locations: 136,
        num_periods: 300,
        resolution: Resolution::Weekly,
        ..Spec::default()
    };
    let data = Generator::new(Some(1)).training_data(&spec).unwrap();
    let options = Options {
        periods: 12,
        splits: 7,
        stride: 1,
        retrain: 1,
        ..Options::default()
    };
    c.bench_function("backtest_136x300_weekly_7x12", |b| {
        b.iter(|| {
            let bt =
                backtest_model(&SeasonalMean, &NoConfig {}, &data, &options, &Climatology).unwrap();
            black_box(bt.score(&data).unwrap())
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
