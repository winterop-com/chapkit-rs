#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::hint::black_box;

use baselines::demo::Fixture;
use chapkit::{Model, PredictInput, RunInfo, TrainInput};
use criterion::{Criterion, criterion_group, criterion_main};
use ridge::{RidgeConfig, RidgeModel};

fn bench(c: &mut Criterion) {
    let fx = Fixture::monthly().unwrap();
    let run = RunInfo::with_periods(6);
    let cfg = RidgeConfig::default();
    let trained = RidgeModel
        .train(&cfg, TrainInput::new(&fx.train, &run, None))
        .unwrap();
    c.bench_function("ridge/train/fixture", |b| {
        b.iter(|| {
            RidgeModel
                .train(&cfg, TrainInput::new(black_box(&fx.train), &run, None))
                .unwrap()
        });
    });
    c.bench_function("ridge/predict/fixture", |b| {
        b.iter(|| {
            RidgeModel
                .predict(
                    &trained,
                    &cfg,
                    PredictInput::new(&fx.historic, black_box(&fx.future), &run, None),
                )
                .unwrap()
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
