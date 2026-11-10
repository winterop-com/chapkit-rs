#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! A random search of 32 trials over a seasonal model's two options: 20 locations x 120 months,
//! 7 splits of 3 periods (chap-core's defaults), each trial a full backtest and CRPS. Trials per
//! second = 32 / time.

#[path = "../tests/common/mod.rs"]
mod common;

use std::hint::black_box;

use chapkit_core::synth::{Generator, Spec};
use chapkit_eval::{Climatology, Metric, Options as BacktestOptions};
use chapkit_hpo::{Method, Options, search_model};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench(c: &mut Criterion) {
    let data = Generator::new(Some(1))
        .training_data(&Spec {
            num_locations: 20,
            num_periods: 120,
            ..Spec::default()
        })
        .unwrap();
    let backtest = BacktestOptions::default();
    let options = Options {
        method: Method::Random,
        trials: 32,
        seed: 1,
        ..Options::default()
    };
    let mut group = c.benchmark_group("hpo");
    group.sample_size(10);
    group.bench_function("random_32_trials_20x120_monthly_7x3", |b| {
        b.iter(|| {
            black_box(
                search_model(
                    &common::Seasonal,
                    &serde_json::json!({}),
                    &data,
                    &backtest,
                    &Climatology,
                    Metric::Crps,
                    &options,
                )
                .unwrap(),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
