#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::hint::black_box;

use chapkit_dataframe::DataFrame;
use criterion::{Criterion, criterion_group, criterion_main};
use serde_json::{Value, json};

/// A chap-sized training frame: `locs` locations x `periods` monthly periods x 4 covariates.
fn wire(locs: usize, periods: usize) -> Value {
    let mut rows = Vec::with_capacity(locs * periods);
    for l in 0..locs {
        for p in 0..periods {
            let x = (l * periods + p) as f64;
            rows.push(json!([
                format!("{}-{:02}", 2020 + p / 12, p % 12 + 1),
                format!("loc{l}"),
                x * 0.5,
                x.sin() + 20.0,
                (p % 7) as i64,
                x.cos(),
                (l + p) as i64
            ]));
        }
    }
    json!({
        "columns": ["time_period", "location", "rainfall", "mean_temperature", "f1", "f2", "disease_cases"],
        "data": rows
    })
}

fn bench_codec(c: &mut Criterion) {
    for (locs, periods) in [(5, 36), (50, 120)] {
        let text = serde_json::to_string(&wire(locs, periods)).unwrap();
        let df: DataFrame = serde_json::from_str(&text).unwrap();
        let name = format!("{locs}x{periods}");
        c.bench_function(&format!("decode/{name}"), |b| {
            b.iter(|| serde_json::from_str::<DataFrame>(black_box(&text)).unwrap());
        });
        c.bench_function(&format!("encode/{name}"), |b| {
            b.iter(|| serde_json::to_string(black_box(&df)).unwrap());
        });
    }
}

fn bench_stack(c: &mut Criterion) {
    let df: DataFrame = serde_json::from_value(wire(1, 36)).unwrap();
    let copies: Vec<&DataFrame> = std::iter::repeat_n(&df, 1000).collect();
    c.bench_function("vstack/1000x36", |b| {
        b.iter(|| DataFrame::vstack(black_box(&copies)).unwrap());
    });
}

criterion_group!(benches, bench_codec, bench_stack);
criterion_main!(benches);
