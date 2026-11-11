#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Parity with R's `stats::arima` and `predict.Arima`, against fixtures R generated
//! (`Rscript scripts/gen_arima_fixtures.R stats`).

use chapkit_ts::{ArimaSpec, Method, fit_arima};
use serde_json::Value;

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

/// The largest relative difference; infinite where one side is not finite and the other is not the
/// same value (`f64::max` would skip a `NaN` difference). Missing on both sides counts as equal.
fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| match (x.is_finite(), y.is_finite()) {
            (true, true) => rel(*x, *y),
            (false, false) if (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits() => 0.0,
            _ => f64::INFINITY,
        })
        .fold(0.0, f64::max)
}

#[test]
fn worst_counts_a_missing_value() {
    assert_eq!(worst(&[f64::NAN], &[10.0]), f64::INFINITY);
    assert_eq!(worst(&[10.0], &[f64::NAN]), f64::INFINITY);
    assert_eq!(worst(&[f64::NAN, 1.0], &[f64::NAN, 0.5]), 0.5);
    assert_eq!(worst(&[10.0], &[f64::INFINITY]), f64::INFINITY);
    assert_eq!(worst(&[f64::INFINITY], &[f64::NEG_INFINITY]), f64::INFINITY);
    assert_eq!(worst(&[f64::INFINITY], &[f64::INFINITY]), 0.0);
}

#[test]
fn stats_arima_matches_r() {
    let path = format!("{}/fixtures/stats.json", env!("CARGO_MANIFEST_DIR"));
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = nums(&case["y"]);
        let o: Vec<usize> = nums(&case["order"]).iter().map(|v| *v as usize).collect();
        let spec = ArimaSpec {
            p: o[0],
            d: o[1],
            q: o[2],
            seasonal_p: o[3],
            seasonal_d: o[4],
            seasonal_q: o[5],
            period: o[6],
            constant: case["constant"].as_bool().unwrap(),
        };
        let method = match case["method"].as_str().unwrap() {
            "CSS-ML" => Method::CssMl,
            "ML" => Method::Ml,
            _ => Method::Css,
        };
        let got = fit_arima(&y, &spec, method);
        if case["error"].as_bool().unwrap() {
            assert!(got.is_err(), "{name}: R fails, got {got:?}");
            continue;
        }
        let got = got.unwrap();
        let fc = got.forecast(nums(&case["mean"]).len()).unwrap();
        let stats_sigma2 = case["sigma2"].as_f64().unwrap();
        let se: Vec<f64> = fc
            .variance
            .iter()
            .map(|v| (v / got.sigma2 * stats_sigma2).sqrt())
            .collect();
        let report = [
            ("coef", worst(&got.coef, &nums(&case["coef"]))),
            ("loglik", rel(got.loglik, case["loglik"].as_f64().unwrap())),
            (
                "residuals",
                worst(&got.residuals, &nums(&case["residuals"])),
            ),
            ("phi", worst(got.phi(), &nums(&case["phi"]))),
            ("mean", worst(&fc.mean, &nums(&case["mean"]))),
            ("se", worst(&se, &nums(&case["se"]))),
        ];
        eprintln!("{name}: {report:?}");
        // Measured: 1e-13 to 1e-9 relative on every case but the weekly ARIMA(0,1,1)(0,1,0)[52],
        // whose 53 diffuse states (kappa = 1e6) round differently, so BFGS stops at a slightly
        // different point of a flat optimum: 1.3e-6 on the coefficient, 2.4e-5 on residuals,
        // 4e-13 on the log-likelihood.
        for (what, err) in report {
            let tol = match what {
                "loglik" => 1e-10,
                "residuals" => 1e-4,
                _ => 1e-5,
            };
            assert!(err < tol, "{name} {what}: {err}");
        }
    }
}
