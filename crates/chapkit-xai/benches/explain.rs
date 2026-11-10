#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! When parallel evaluation pays off, and what a temporal forecast explanation costs.
//!
//! `threads`: LIME over 3 features with a value function that busy-waits a fixed time per call,
//! 100 calls, on one thread and across cores; the crossover sets `PARALLEL_MIN_BATCH`.
//! `forecast`: temporal LIME of one cell of a recursive model (20 locations, 36 months of
//! history, 3 forecast steps, 3 covariates).

use std::hint::black_box;
use std::time::{Duration, Instant};

use chapkit_core::rng::Rng;
use chapkit_core::{Column, DataFrame, prediction_frame};
use chapkit_xai::{
    FeatureSpec, ForecastTarget, Options, Parallel, Result, explain, explain_forecast,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn spin(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {
        std::hint::spin_loop();
    }
}

fn threads(c: &mut Criterion) {
    let mut rng = Rng::new(1);
    let bg: Vec<Vec<f64>> = (0..100)
        .map(|_| vec![rng.normal(), rng.normal(), rng.normal()])
        .collect();
    let names: Vec<String> = ["a", "b", "c"].iter().map(|s| (*s).to_owned()).collect();
    let mut group = c.benchmark_group("xai_threads");
    for us in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
        let cost = Duration::from_secs_f64(us * 1e-6);
        let f = move |x: &[f64]| -> Result<f64> {
            spin(cost);
            Ok(x.iter().sum())
        };
        for (label, parallel) in [
            ("one_thread", Parallel::Never),
            ("rayon", Parallel::Always),
            ("auto", Parallel::Auto),
        ] {
            let opts = Options {
                n_samples: 100,
                parallel,
                ..Options::default()
            };
            group.bench_with_input(
                BenchmarkId::new(label, format!("100x{us}us")),
                &opts,
                |b, opts| b.iter(|| black_box(explain(&f, &names, &[0.1, 0.2, 0.3], &bg, opts))),
            );
        }
    }
    group.finish();
}

fn frames() -> (DataFrame, DataFrame) {
    let mut rng = Rng::new(2);
    let (mut h, mut f) = (Vec::new(), Vec::new());
    for l in 0..20 {
        for m in 0..39 {
            let row = (
                format!("{}-{:02}", 2020 + m / 12, m % 12 + 1),
                format!("L{l}"),
                50.0 + 20.0 * rng.normal(),
                25.0 + 2.0 * rng.normal(),
                1000.0 * f64::from(l + 1),
                200.0 + 40.0 * rng.normal(),
            );
            if m < 36 { h.push(row) } else { f.push(row) }
        }
    }
    let df = |rows: &[(String, String, f64, f64, f64, f64)], cases: bool| {
        let mut cols = vec![
            (
                "time_period",
                Column::from_strs(rows.iter().map(|r| r.0.as_str())),
            ),
            (
                "location",
                Column::from_strs(rows.iter().map(|r| r.1.as_str())),
            ),
            ("rainfall", Column::from_f64(rows.iter().map(|r| r.2))),
            (
                "mean_temperature",
                Column::from_f64(rows.iter().map(|r| r.3)),
            ),
            ("population", Column::from_f64(rows.iter().map(|r| r.4))),
        ];
        if cases {
            cols.push(("disease_cases", Column::from_f64(rows.iter().map(|r| r.5))));
        }
        DataFrame::from_columns(cols).unwrap()
    };
    (df(&h, true), df(&f, false))
}

/// y_t = 2 rain_t + 3 temp_t + 0.5 y_{t-1} + 10 per location, from the last observed count
/// (rows are in time order within each location).
fn recursive(historic: &DataFrame, future: &DataFrame) -> std::result::Result<DataFrame, String> {
    let err = |e: chapkit_core::FrameError| e.to_string();
    let loc_of = |df: &DataFrame| -> Vec<String> {
        let c = df.column("location").unwrap();
        (0..df.n_rows())
            .map(|r| c.str_at(r).unwrap().to_owned())
            .collect()
    };
    let cases = historic.f64_column("disease_cases").map_err(err)?;
    let mut prev: std::collections::HashMap<String, f64> =
        loc_of(historic).into_iter().zip(cases).collect();
    let rain = future.f64_column("rainfall").map_err(err)?;
    let temp = future.f64_column("mean_temperature").map_err(err)?;
    let values: Vec<f64> = loc_of(future)
        .iter()
        .enumerate()
        .map(|(r, l)| {
            let p = prev.get_mut(l).unwrap();
            *p = 2.0 * rain[r] + 3.0 * temp[r] + 0.5 * *p + 10.0;
            *p
        })
        .collect();
    prediction_frame(future, 10, |row, out| {
        out.fill(values[row]);
        Ok(())
    })
    .map_err(|e| e.to_string())
}

fn forecast(c: &mut Criterion) {
    let (historic, future) = frames();
    let target = ForecastTarget {
        location: "L0".into(),
        time_period: "2023-03".into(),
    };
    let mut group = c.benchmark_group("xai_forecast");
    group.sample_size(20);
    for n in [100, 500, 5000] {
        let opts = Options {
            n_samples: n,
            ..Options::default()
        };
        group.bench_with_input(BenchmarkId::new("temporal_lime", n), &opts, |b, opts| {
            b.iter(|| {
                black_box(
                    explain_forecast(
                        &recursive,
                        &historic,
                        &future,
                        &target,
                        &FeatureSpec::default(),
                        "disease_cases",
                        opts,
                    )
                    .unwrap(),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, threads, forecast);
criterion_main!(benches);
