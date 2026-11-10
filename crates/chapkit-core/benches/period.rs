//! `Period::parse` per form: called once per row when a dataset is loaded or split.

use std::hint::black_box;

use chapkit_core::Period;
use criterion::{Criterion, criterion_group, criterion_main};

fn parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("period_parse");
    for (name, input) in [
        ("monthly", "2024-03"),
        ("monthly_compact", "202403"),
        ("weekly", "2024W03"),
        ("weekly_dash", "2024-W03"),
        ("week_range", "2019-01-07/2019-01-13"),
    ] {
        group.bench_function(name, |b| b.iter(|| Period::parse(black_box(input))));
    }
    group.finish();
}

criterion_group!(benches, parse);
criterion_main!(benches);
