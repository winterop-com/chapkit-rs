#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Parity with fable 0.5.0's `ARIMA(y)` (order search, coefficients, `sigma2`, forecasts and
//! `refit(reestimate = FALSE)`), against fixtures fable generated
//! (`scripts/gen_arima_fixtures.R fable`, in the auto_arima_chapkit image).

use std::time::Instant;

use chapkit_ts::{AutoArima, auto_arima};
use serde_json::Value;

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

/// The largest relative difference; infinite where one side is missing (`NaN`) and the other is
/// not, as `f64::max` would skip a `NaN` difference.
fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| match (x.is_nan(), y.is_nan()) {
            (true, true) => 0.0,
            (false, false) => (x - y).abs() / y.abs().max(1.0),
            _ => f64::INFINITY,
        })
        .fold(0.0, f64::max)
}

#[test]
fn worst_counts_a_missing_value() {
    assert_eq!(worst(&[f64::NAN], &[10.0]), f64::INFINITY);
    assert_eq!(worst(&[10.0], &[f64::NAN]), f64::INFINITY);
    assert_eq!(worst(&[f64::NAN, 1.0], &[f64::NAN, 0.5]), 0.5);
}

#[test]
fn fable_search_matches_r() {
    let path = format!("{}/fixtures/fable.json", env!("CARGO_MANIFEST_DIR"));
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = nums(&case["y"]);
        let n = nums(&case["n_train"])[0] as usize;
        let period = nums(&case["period"])[0] as usize;
        let start = Instant::now();
        let model = auto_arima(&y[..n], &AutoArima::new(period)).unwrap();
        let secs = start.elapsed().as_secs_f64();
        let s = model.spec;
        let order: Vec<f64> = [s.p, s.d, s.q, s.seasonal_p, s.seasonal_d, s.seasonal_q]
            .iter()
            .map(|v| *v as f64)
            .collect();
        assert_eq!(order, nums(&case["order"]), "{name}: {s}");
        assert_eq!(s.constant, case["constant"].as_bool().unwrap(), "{name}");
        let mean = nums(&case["mean"]);
        let fc = model.forecast(mean.len()).unwrap();
        let sd: Vec<f64> = fc.variance.iter().map(|v| v.sqrt()).collect();
        let later = model.filter(&y).unwrap();
        let lfc = later.forecast(mean.len()).unwrap();
        let lsd: Vec<f64> = lfc.variance.iter().map(|v| v.sqrt()).collect();
        let report = [
            ("coef", worst(&model.coef, &nums(&case["coef"]))),
            (
                "sigma2",
                worst(&[model.sigma2], &[case["sigma2"].as_f64().unwrap()]),
            ),
            (
                "aicc",
                worst(&[model.aicc], &[case["aicc"].as_f64().unwrap()]),
            ),
            ("mean", worst(&fc.mean, &mean)),
            ("sd", worst(&sd, &nums(&case["sd"]))),
            (
                "refit_sigma2",
                worst(&[later.sigma2], &[case["refit_sigma2"].as_f64().unwrap()]),
            ),
            ("refit_mean", worst(&lfc.mean, &nums(&case["refit_mean"]))),
            ("refit_sd", worst(&lsd, &nums(&case["refit_sd"]))),
        ];
        eprintln!(
            "{name} {s}: {secs:.3}s (fable {:.2}s) {report:?}",
            case["seconds"].as_f64().unwrap()
        );
        // Orders match on every series. Measured: 1e-14 to 4e-9 relative on the monthly series;
        // up to 6.5e-6 on the coefficients (1e-6 on forecasts) of the weekly ARIMA(2,0,2) models,
        // whose BFGS stops on a flat likelihood where rounding (x86 R against this port) moves
        // the stopping point. Ridges can be flatter: on one weekly series with near-cancelling
        // seasonal AR and MA terms, R on arm64 and R on x86 themselves differ by 1e-2.
        // Matagalpa is such a ridge and more: Gardner's initial covariance (R's getQ0) is not
        // even positive semi-definite there, which this port replaces by the exact covariance, so
        // only the order and the criterion agree closely (AICc 9e-6; coefficients 2.6e-2,
        // forecasts 1.5e-2).
        let ridge = name == "weekly_matagalpa";
        for (what, err) in report {
            let tol = match (ridge, what) {
                (false, _) => 1e-5,
                (true, "aicc") => 1e-4,
                (true, _) => 5e-2,
            };
            assert!(err < tol, "{name} {what}: {err}");
        }
    }
}

/// fable's `refit()` on a history shorter than the one trained on, where the model's seasonal
/// orders exceed a third of the new history in periods: `reestimate = FALSE` keeps the
/// coefficients and refilters; `reestimate = TRUE` fails in R here as in this port.
#[test]
fn fable_refit_on_a_shorter_history() {
    let path = format!("{}/fixtures/fable.json", env!("CARGO_MANIFEST_DIR"));
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for case in fixture["refits"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = nums(&case["y"]);
        let n = nums(&case["n_refit"])[0] as usize;
        let model = auto_arima(&y, &AutoArima::new(12)).unwrap();
        let s = model.spec;
        let order: Vec<f64> = [s.p, s.d, s.q, s.seasonal_p, s.seasonal_d, s.seasonal_q]
            .iter()
            .map(|v| *v as f64)
            .collect();
        assert_eq!(order, nums(&case["order"]), "{name}: {s}");
        assert_eq!(s.constant, case["constant"].as_bool().unwrap(), "{name}");
        let coef = worst(&model.coef, &nums(&case["coef"]));
        let want = &case["filter"];
        let later = model.filter(&y[..n]).unwrap();
        let mean = nums(&want["mean"]);
        let fc = later.forecast(mean.len()).unwrap();
        let sd: Vec<f64> = fc.variance.iter().map(|v| v.sqrt()).collect();
        let report = [
            ("coef", coef),
            (
                "sigma2",
                worst(&[later.sigma2], &[want["sigma2"].as_f64().unwrap()]),
            ),
            ("mean", worst(&fc.mean, &mean)),
            ("sd", worst(&sd, &nums(&want["sd"]))),
        ];
        eprintln!("{name} {s}: {report:?}");
        for (what, err) in report {
            assert!(err < 1e-5, "{name} {what}: {err}");
        }
        let re = model.reestimate(&y[..n]);
        assert_eq!(
            re.is_err(),
            case["reestimate"]["error"].as_bool().unwrap(),
            "{name}: {re:?}"
        );
    }
}
