#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::hint::black_box;

use chapkit::{Column, DataFrame, Model, NoConfig, PredictInput, RunInfo, TrainInput};
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

use criterion::{Criterion, criterion_group, criterion_main};
use global_median::GlobalMedian;
use ridge::{RidgeConfig, RidgeModel};
use seasonal_median::SeasonalMedian;
use seasonal_shift_persistence::SeasonalShiftPersistence;

/// `locs` locations x `periods` monthly periods from 2015-01, plus a future frame of `horizon` periods.
fn data(locs: usize, periods: usize, horizon: usize) -> (DataFrame, DataFrame) {
    let build = |start: usize, n: usize, with_y: bool| {
        let (mut tp, mut loc, mut rain, mut temp, mut y) = (vec![], vec![], vec![], vec![], vec![]);
        for l in 0..locs {
            for p in start..start + n {
                tp.push(format!("{}-{:02}", 2015 + p / 12, p % 12 + 1));
                loc.push(format!("loc{l}"));
                let season = ((p % 12) as f64 / 12.0 * std::f64::consts::TAU).sin();
                rain.push(100.0 + 50.0 * season + l as f64);
                temp.push(25.0 + 3.0 * season);
                y.push((20.0 + 10.0 * season + 2.0 * l as f64).round());
            }
        }
        let mut cols = vec![
            ("time_period", Column::from_strs(tp)),
            ("location", Column::from_strs(loc)),
            ("rainfall", Column::from_f64(rain)),
            ("mean_temperature", Column::from_f64(temp)),
        ];
        if with_y {
            cols.push(("disease_cases", Column::from_f64(y)));
        }
        DataFrame::from_columns(cols).unwrap()
    };
    (build(0, periods, true), build(periods, horizon, false))
}

fn bench_model<M: Model>(c: &mut Criterion, name: &str, model: &M, config: &M::Config) {
    for (locs, periods) in [(5, 36), (50, 120)] {
        let (train, future) = data(locs, periods, 3);
        let run = RunInfo::with_periods(3);
        let trained = model
            .train(config, TrainInput::new(&train, &run, None))
            .unwrap();
        c.bench_function(&format!("{name}/train/{locs}x{periods}"), |b| {
            b.iter(|| {
                model
                    .train(config, TrainInput::new(black_box(&train), &run, None))
                    .unwrap()
            });
        });
        c.bench_function(&format!("{name}/predict/{locs}x{periods}"), |b| {
            b.iter(|| {
                model
                    .predict(
                        &trained,
                        config,
                        PredictInput::new(&train, black_box(&future), &run, None),
                    )
                    .unwrap()
            });
        });
    }
}

fn benches(c: &mut Criterion) {
    bench_model(c, "global_median", &GlobalMedian, &NoConfig {});
    bench_model(c, "seasonal_median", &SeasonalMedian, &NoConfig {});
    bench_model(
        c,
        "seasonal_shift_persistence",
        &SeasonalShiftPersistence,
        &NoConfig {},
    );
    bench_model(c, "ridge", &RidgeModel, &RidgeConfig::default());
}

criterion_group!(models, benches);
criterion_main!(models);
