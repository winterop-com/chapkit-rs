#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Parity with chap-core 2.7.0's evaluation, against fixtures it generated
//! (`scripts/gen_eval_fixtures.py`, `make eval-fixtures`).

use std::collections::HashMap;

use chapkit_core::{
    DataFrame, Model, ModelInfo, NoConfig, Period, PredictInput, Result, TrainInput, cols,
    prediction_frame,
};
use chapkit_eval::{
    Climatology, DampedPersistence, Forecast, MaskedRange, Metric, Options, Prepare, Scores,
    backtest_model, mask_target, prepare, score, train_test_splits, weather_provider,
};
use serde_json::Value;

const CASES: [&str; 6] = [
    "monthly",
    "weekly",
    "masked",
    "messy_monthly",
    "messy_weekly",
    "untrained",
];

struct Case {
    fixture: Value,
    data: DataFrame,
    options: Options,
}

fn case(name: &str) -> Case {
    let path = format!("{}/fixtures/eval_{name}.json", env!("CARGO_MANIFEST_DIR"));
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut data: DataFrame = serde_json::from_value(fixture["data"].clone()).unwrap();
    let p = &fixture["params"];
    let n = |k: &str| usize::try_from(p[k].as_u64().unwrap()).unwrap();
    if let Some(ranges) = p["masked_ranges"].as_array() {
        let period = |v: &Value| Period::parse(v.as_str().unwrap()).unwrap();
        let ranges: Vec<MaskedRange> = ranges
            .iter()
            .map(|r| MaskedRange {
                start: period(&r["start"]),
                end: period(&r["end"]),
            })
            .collect();
        data = mask_target(&data, &ranges).unwrap();
    }
    let options = Options {
        periods: n("n_periods"),
        splits: n("n_splits"),
        stride: n("stride"),
        retrain: n("n_retrain"),
        ..Options::default()
    };
    Case {
        fixture,
        data,
        options,
    }
}

fn period(v: &Value) -> Period {
    Period::parse(v.as_str().unwrap()).unwrap()
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got.is_nan() && want.is_nan()) || (got - want).abs() <= tol * want.abs().max(1.0)
}

/// Each split's windows exactly, and its future covariates to 1e-9 (the climatology's means are
/// summed in a different order than numpy's), MSTL + ARIMA's to 1e-3: the same orders, but some
/// of statsmodels' fits stop at `maxiter = 50` before converging, and where they stop follows
/// rounding (measured 3.8e-4).
#[test]
fn splits_match_chap_core() {
    let mut worst = 0.0f64;
    for name in CASES {
        let c = case(name);
        for provider in [
            "climatology",
            "damped_persistence",
            "observed",
            "mstl_arima",
        ] {
            let tol = if provider == "mstl_arima" { 1e-3 } else { 1e-9 };
            let want = &c.fixture["splits"][provider];
            let weather = weather_provider(provider).unwrap();
            let got = train_test_splits(&c.data, &c.options, weather.as_ref()).unwrap();
            let ctx = format!("{name}/{provider}");
            let dropped: Vec<&str> = c.fixture["dropped"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(got.dropped, dropped, "{ctx}");
            assert_eq!(
                Period::parse(&got.train_last),
                Some(period(&want["train_last"])),
                "{ctx}"
            );
            let want_splits = want["splits"].as_array().unwrap();
            assert_eq!(got.splits.len(), want_splits.len(), "{ctx}");
            for (g, w) in got.splits.iter().zip(want_splits) {
                assert_eq!(
                    Period::parse(&g.historic_last),
                    Some(period(&w["historic_last"])),
                    "{ctx}"
                );
                let periods: Vec<Period> = w["future_periods"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(period)
                    .collect();
                let got_periods: Vec<Period> = g
                    .periods
                    .iter()
                    .map(|p| Period::parse(p).unwrap())
                    .collect();
                assert_eq!(got_periods, periods, "{ctx}");
                let wf = &w["masked_future"];
                let names: Vec<&str> = wf["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect();
                assert_eq!(g.future.names(), names, "{ctx}");
                let rows = wf["data"].as_array().unwrap();
                assert_eq!(g.future.n_rows(), rows.len(), "{ctx}");
                let mut at = HashMap::new();
                for r in 0..g.future.n_rows() {
                    let row = g.future.row_json(r);
                    at.insert((row[1].as_str().unwrap().to_owned(), period(&row[0])), row);
                }
                for w in rows {
                    let key = (w[1].as_str().unwrap().to_owned(), period(&w[0]));
                    let row = &at[&key];
                    for (i, (got, want)) in
                        row.iter().zip(w.as_array().unwrap()).enumerate().skip(2)
                    {
                        let ok = match (got.as_f64(), want.as_f64()) {
                            (Some(a), Some(b)) => {
                                if provider == "mstl_arima" {
                                    worst = worst.max((a - b).abs() / b.abs().max(1.0));
                                }
                                close(a, b, tol)
                            }
                            _ => got == want,
                        };
                        assert!(ok, "{ctx} {key:?} {}: {got} vs {want}", names[i]);
                    }
                }
            }
        }
    }
    eprintln!("mstl_arima: worst relative difference {worst:.1e}");
}

fn forecasts(v: &Value) -> Vec<Forecast> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|f| Forecast {
            split: 0,
            location: f["location"].as_str().unwrap().to_owned(),
            time_period: f["time_period"].as_str().unwrap().to_owned(),
            horizon: usize::try_from(f["horizon"].as_u64().unwrap()).unwrap(),
            samples: f["samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect(),
        })
        .collect()
}

/// Assert `scores` equal chap-core's metrics in `want` to `tol` (relative, absolute below 1).
fn assert_scores(scores: &Scores, want: &Value, tol: f64, ctx: &str) {
    for m in Metric::ALL {
        let got = scores.get(m);
        match want["aggregate"][m.id()].as_f64() {
            Some(w) => assert!(close(got, w, tol), "{ctx} {}: {got} vs {w}", m.id()),
            // chap-core leaves an undefined aggregate out.
            None => assert!(got.is_nan(), "{ctx} {}: {got}", m.id()),
        }
    }
    let i = |m: Metric| Metric::ALL.iter().position(|x| *x == m).unwrap();
    for (key, got) in [
        (
            "by_horizon",
            scores
                .by_horizon
                .iter()
                .map(|(h, v)| (h.to_string(), v))
                .collect::<Vec<_>>(),
        ),
        (
            "by_location",
            scores
                .by_location
                .iter()
                .map(|(l, v)| (l.clone(), v))
                .collect(),
        ),
    ] {
        for m in Metric::ALL {
            let w = want[key][m.id()].as_object().unwrap();
            assert_eq!(got.len(), w.len(), "{ctx} {key} {}", m.id());
            for (group, values) in &got {
                let g = values[i(m)];
                let wv = w[group.as_str()].as_f64().unwrap_or(f64::NAN);
                assert!(
                    close(g, wv, tol),
                    "{ctx} {key} {group} {}: {g} vs {wv}",
                    m.id()
                );
            }
        }
    }
}

/// Every metric, aggregate and breakdowns, to 1e-12 given chap-core's own forecasts (`signed`:
/// samples below -1, whose log1p is NaN).
#[test]
fn metrics_match_chap_core() {
    for name in CASES {
        let c = case(name);
        for model in ["seasonal_median", "covariate", "since_last", "signed"] {
            let want = &c.fixture["models"][model];
            let scores = score(&forecasts(&want["forecasts"]), &c.data).unwrap();
            assert_scores(&scores, want, 1e-12, &format!("{name}/{model}"));
        }
    }
}

/// The fixture script's `Covariate` test model.
struct Covariate;

const OFFSETS: [f64; 9] = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0];

fn observed_by_location(df: &DataFrame) -> HashMap<String, Vec<(Period, f64)>> {
    let (tp, loc) = (
        df.require(cols::TIME_PERIOD).unwrap(),
        df.require(cols::LOCATION).unwrap(),
    );
    let y = df.f64_column(cols::DISEASE_CASES).unwrap();
    let mut out: HashMap<String, Vec<(Period, f64)>> = HashMap::new();
    for (r, v) in y.iter().enumerate() {
        if v.is_finite() {
            out.entry(loc.str_at(r).unwrap().to_owned())
                .or_default()
                .push((Period::parse(tp.str_at(r).unwrap()).unwrap(), *v));
        }
    }
    out
}

impl Model for Covariate {
    type Config = NoConfig;
    type Trained = HashMap<String, f64>;

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn train(&self, _: &NoConfig, input: TrainInput<'_>) -> Result<Self::Trained> {
        Ok(observed_by_location(input.data)
            .into_iter()
            .map(|(l, v)| (l, v.iter().map(|x| x.1).sum::<f64>() / v.len() as f64))
            .collect())
    }

    fn predict(
        &self,
        means: &Self::Trained,
        _: &NoConfig,
        input: PredictInput<'_>,
    ) -> Result<DataFrame> {
        let last: HashMap<String, f64> = observed_by_location(input.historic)
            .into_iter()
            .map(|(l, v)| (l, v.iter().max_by_key(|x| x.0).unwrap().1))
            .collect();
        let f = input.future;
        let loc = f.require(cols::LOCATION)?;
        let (rain, temp) = (f.f64_column("rainfall")?, f.f64_column("mean_temperature")?);
        prediction_frame(f, OFFSETS.len(), |r, out| {
            let l = loc.str_at(r).unwrap();
            let base =
                0.5 * means[l] + 0.5 * last[l] + 0.1 * (rain[r] - 120.0) + 0.5 * (temp[r] - 24.0);
            for (o, offset) in out.iter_mut().zip(OFFSETS) {
                *o = (base + offset).max(0.0);
            }
            Ok(())
        })
    }
}

/// The fixture script's `SinceLast` test model: counts the periods since the last observation.
struct SinceLast;

impl Model for SinceLast {
    type Config = NoConfig;
    type Trained = ();

    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn train(&self, _: &NoConfig, _: TrainInput<'_>) -> Result<()> {
        Ok(())
    }

    fn predict(&self, (): &(), _: &NoConfig, input: PredictInput<'_>) -> Result<DataFrame> {
        let h = input.historic;
        let (tp, loc) = (h.require(cols::TIME_PERIOD)?, h.require(cols::LOCATION)?);
        let y = h.f64_column(cols::DISEASE_CASES)?;
        let mut series: HashMap<&str, Vec<(Period, f64)>> = HashMap::new();
        for (r, v) in y.iter().enumerate() {
            let p = Period::parse(tp.str_at(r).unwrap()).unwrap();
            series
                .entry(loc.str_at(r).unwrap())
                .or_default()
                .push((p, *v));
        }
        // (last observed value, history rows from it to the end), per location.
        let state: HashMap<&str, (f64, f64)> = series
            .into_iter()
            .map(|(l, mut s)| {
                s.sort_by_key(|x| x.0);
                let i = s.iter().rposition(|x| x.1.is_finite()).unwrap();
                (l, (s[i].1, (s.len() - i) as f64))
            })
            .collect();
        let floc = input.future.require(cols::LOCATION)?;
        let mut step: HashMap<&str, f64> = HashMap::new();
        prediction_frame(input.future, OFFSETS.len(), |r, out| {
            let l = floc.str_at(r).unwrap();
            let k = step.entry(l).or_insert(0.0);
            let (last, since) = state[l];
            let base = last + 2.0 * (since + *k);
            *k += 1.0;
            for (o, offset) in out.iter_mut().zip(OFFSETS) {
                *o = (base + offset).max(0.0);
            }
            Ok(())
        })
    }
}

/// The first difference between a Rust backtest of `model` and chap-core's for the fixture's
/// `model_id`, if any: forecasts and metrics to 1e-12.
fn backtest_difference<M: Model<Config = NoConfig>>(
    model: &M,
    model_id: &str,
    c: &Case,
    options: &Options,
) -> Option<String> {
    let want = &c.fixture["models"][model_id];
    let weather = weather_provider("climatology").unwrap();
    let bt = match backtest_model(model, &NoConfig {}, &c.data, options, weather.as_ref()) {
        Ok(bt) => bt,
        Err(e) => return Some(e.to_string()),
    };
    let expected = forecasts(&want["forecasts"]);
    if bt.forecasts.len() != expected.len() {
        return Some(format!(
            "{} forecasts, chap-core {}",
            bt.forecasts.len(),
            expected.len()
        ));
    }
    let key = |f: &Forecast| {
        (
            f.location.clone(),
            Period::parse(&f.time_period).unwrap(),
            f.horizon,
        )
    };
    let got: HashMap<_, _> = bt.forecasts.iter().map(|f| (key(f), f)).collect();
    for e in &expected {
        let Some(g) = got.get(&key(e)) else {
            return Some(format!("no forecast {:?}", key(e)));
        };
        for (a, b) in g.samples.iter().zip(&e.samples) {
            if !close(*a, *b, 1e-12) {
                return Some(format!("{:?}: {a} vs {b}", key(e)));
            }
        }
    }
    let scores = bt.score(&c.data).unwrap();
    std::panic::catch_unwind(|| assert_scores(&scores, want, 1e-12, model_id))
        .err()
        .map(|_| "metrics differ".to_owned())
}

/// Whole backtests of the test models (climatology weather, retraining) give chap-core's
/// forecasts and metrics, messy datasets included.
#[test]
fn backtest_matches_chap_core() {
    for name in CASES {
        let c = case(name);
        for (id, diff) in [
            (
                "covariate",
                backtest_difference(&Covariate, "covariate", &c, &c.options),
            ),
            (
                "since_last",
                backtest_difference(&SinceLast, "since_last", &c, &c.options),
            ),
        ] {
            assert_eq!(diff, None, "{name}/{id}");
        }
    }
}

/// Without chap-core's gap filling the periods since the last observation are counted over fewer
/// rows: A's first forecast after its gap (2021-02, 2021-03) differs from chap-core's. The reason
/// preparation is on by default.
#[test]
fn since_last_differs_without_filling() {
    let c = case("messy_monthly");
    let weather = weather_provider("climatology").unwrap();
    let first_a = |options: &Options| {
        let bt =
            backtest_model(&SinceLast, &NoConfig {}, &c.data, options, weather.as_ref()).unwrap();
        bt.forecasts
            .into_iter()
            .find(|f| f.location == "A" && f.horizon == 1)
            .unwrap()
    };
    let chap_core = forecasts(&c.fixture["models"]["since_last"]["forecasts"])
        .into_iter()
        .find(|f| f.location == "A" && f.horizon == 1)
        .unwrap();
    let filled = first_a(&c.options);
    assert_eq!(filled.samples, chap_core.samples);
    let kept = first_a(&Options {
        prepare: Prepare {
            fill_gaps: false,
            pad: false,
        },
        ..c.options
    });
    assert_eq!(
        Period::parse(&kept.time_period),
        Period::parse(&chap_core.time_period)
    );
    // History rows since the last observation: 3 in chap-core (2021-01 observed, 2 filled), 1 kept.
    assert!((kept.samples[4] - chap_core.samples[4] + 4.0).abs() < 1e-12);
    assert!(backtest_difference(&SinceLast, "since_last", &c, &c.options).is_none());
}

/// The dataset as chap-core loads it (`from_pandas(fill_missing=True)`), row for row.
#[test]
fn prepared_matches_chap_core() {
    for name in CASES {
        let c = case(name);
        let got = prepare(&c.data, &Prepare::default()).unwrap();
        let want = &c.fixture["prepared"];
        let names: Vec<&str> = want["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(got.names(), names, "{name}");
        let rows = want["data"].as_array().unwrap();
        assert_eq!(got.n_rows(), rows.len(), "{name}");
        for (r, w) in rows.iter().enumerate() {
            let g = got.row_json(r);
            assert_eq!(period(&g[0]), period(&w[0]), "{name} row {r}");
            assert_eq!(g[1], w[1], "{name} row {r}");
            for (a, b) in g.iter().zip(w.as_array().unwrap()).skip(2) {
                assert_eq!(a.as_f64(), b.as_f64(), "{name} row {r}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn duplicate_rows_are_an_error() {
    let c = case("monthly");
    let first = c.data.take(&[0]).unwrap();
    let data = DataFrame::vstack(&[&c.data, &first]).unwrap();
    let weather = weather_provider("climatology").unwrap();
    let e = train_test_splits(&data, &c.options, weather.as_ref()).unwrap_err();
    assert!(
        e.to_string()
            .contains("location A has more than one row for 2018-01"),
        "{e}"
    );
}

#[test]
fn too_little_data_is_an_error() {
    let c = case("monthly");
    let options = Options {
        splits: 40,
        ..c.options
    };
    let weather = weather_provider("climatology").unwrap();
    let e = train_test_splits(&c.data, &options, weather.as_ref()).unwrap_err();
    assert!(e.to_string().contains("too few"), "{e}");
}

#[test]
fn masked_period_in_the_window_is_an_error() {
    let c = case("masked");
    let p = |s| Period::parse(s).unwrap();
    let raw: DataFrame = serde_json::from_value(c.fixture["data"].clone()).unwrap();
    let data = mask_target(
        &raw,
        &[MaskedRange {
            start: p("2020-11"),
            end: p("2020-11"),
        }],
    )
    .unwrap();
    let weather = weather_provider("climatology").unwrap();
    let e = train_test_splits(&data, &c.options, weather.as_ref()).unwrap_err();
    assert!(e.to_string().contains("evaluation window"), "{e}");
}

#[test]
fn incomplete_forecasts_are_an_error() {
    let c = case("monthly");
    let weather = weather_provider("climatology").unwrap();
    let e = chapkit_eval::backtest(
        &c.data,
        &c.options,
        weather.as_ref(),
        |_| Ok(()),
        |(), _, f| {
            let first = f.take(&[0]).unwrap();
            prediction_frame(&first, 1, |_, s| {
                s[0] = 1.0;
                Ok(())
            })
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("no forecast"), "{e}");
}

/// chap-core drops a location without a target before the test windows; kept on request.
#[test]
fn untrained_locations_are_dropped_as_chap_core_does() {
    let c = case("untrained");
    let weather = weather_provider("climatology").unwrap();
    let bt = backtest_model(
        &SinceLast,
        &NoConfig {},
        &c.data,
        &c.options,
        weather.as_ref(),
    )
    .unwrap();
    assert_eq!(bt.dropped, ["C"]);
    assert!(bt.forecasts.iter().all(|f| f.location != "C"));
    let kept = Options {
        drop_untrained: false,
        ..c.options
    };
    let splits = train_test_splits(&c.data, &kept, weather.as_ref()).unwrap();
    assert_eq!(splits.dropped, Vec::<String>::new());
    let truth = &splits.splits[0].truth;
    let loc = truth.require(cols::LOCATION).unwrap();
    assert!((0..truth.n_rows()).any(|r| loc.str_at(r) == Some("C")));
}

/// Damped persistence with `phi = 0` is the climatology, except where the last anomaly is missing.
#[test]
fn undamped_anomaly_zero_is_the_climatology() {
    let c = case("monthly");
    let zero = DampedPersistence { damping: Some(0.0) };
    let damped = train_test_splits(&c.data, &c.options, &zero).unwrap();
    let plain = train_test_splits(&c.data, &c.options, &Climatology).unwrap();
    let mut compared = 0;
    for (d, p) in damped.splits.iter().zip(&plain.splits) {
        for r in 0..d.future.n_rows() {
            for (a, b) in d.future.row_json(r).iter().zip(p.future.row_json(r)) {
                if !a.is_null() {
                    assert_eq!(*a, b);
                    compared += 1;
                }
            }
        }
    }
    assert!(compared > 100);
}
