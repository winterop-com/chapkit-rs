//! `<bin> eval`: a backtest of the service's model in-process, with [`chapkit_eval`].

use std::fmt::Write as _;
use std::time::Instant;

use chapkit_core::synth::{Generator, Resolution, Spec};
use chapkit_core::{DataFrame, Period, PeriodType, PredictInput, RunInfo, TrainInput, cols};
use chapkit_eval::{Metric, Options, Scores, weather_provider};
use serde_json::{Value, json};

use crate::cli::EvalArgs;
use crate::erased::DynModel;

/// Read a dataset: chapkit's JSON (`{columns, data}`) for `.json`, else CSV.
fn read(path: &std::path::Path) -> Result<DataFrame, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let is_json = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("json"));
    let frame = if is_json {
        serde_json::from_str(&text).map_err(|e| e.to_string())
    } else {
        DataFrame::from_csv(&text).map_err(|e| e.to_string())
    };
    frame.map_err(|e| format!("{}: {e}", path.display()))
}

/// The dataset to backtest on and a line describing it.
fn dataset(model: &dyn DynModel, args: &EvalArgs) -> Result<(DataFrame, String), String> {
    if let Some(path) = &args.data {
        let data = read(path)?;
        return Ok((data, path.display().to_string()));
    }
    let info = model.info();
    let resolution = if info.period_type == PeriodType::Weekly {
        Resolution::Weekly
    } else {
        Resolution::Monthly
    };
    let seed = args
        .seed
        .unwrap_or_else(|| chapkit_core::rng::Rng::from_time().next_u64() % 1_000_000);
    let spec = Spec {
        num_locations: args.locations,
        num_periods: args.history,
        required_covariates: info.required_covariates,
        resolution,
        ..Spec::default()
    };
    let data = Generator::new(Some(seed))
        .training_data(&spec)
        .map_err(|e| e.to_string())?;
    Ok((data, format!("generated (seed {seed})")))
}

/// A score, compactly: `-` when undefined.
fn num(v: f64) -> String {
    match v.abs() {
        a if a.is_nan() => "-".into(),
        a if a < 10.0 => format!("{v:.4}"),
        a if a < 1000.0 => format!("{v:.2}"),
        _ => format!("{v:.0}"),
    }
}

/// The aggregate and per-horizon table.
fn table(scores: &Scores) -> String {
    let mut out = format!("{:<22}{:>10}", "metric", "all");
    for (h, _) in &scores.by_horizon {
        let _ = write!(out, "{:>10}", format!("h{h}"));
    }
    for (i, m) in Metric::ALL.iter().enumerate() {
        let _ = write!(out, "\n{:<22}{:>10}", m.name(), num(scores.get(*m)));
        for (_, values) in &scores.by_horizon {
            let _ = write!(
                out,
                "{:>10}",
                num(values.get(i).copied().unwrap_or(f64::NAN))
            );
        }
    }
    out
}

/// Scores by location: a few metrics, one row per location.
fn by_location(scores: &Scores) -> String {
    let shown = [
        Metric::Crps,
        Metric::Mae,
        Metric::Rmse,
        Metric::Mape,
        Metric::Coverage10_90,
        Metric::Coverage25_75,
    ];
    let index = |m: Metric| Metric::ALL.iter().position(|x| *x == m);
    let width = scores
        .by_location
        .iter()
        .map(|(l, _)| l.len())
        .max()
        .unwrap_or(0)
        .max(8)
        + 2;
    let mut out = format!("{:<width$}", "location");
    for m in shown {
        let _ = write!(out, "{:>16}", m.name());
    }
    for (location, values) in &scores.by_location {
        let _ = write!(out, "\n{location:<width$}");
        for m in shown {
            let v = index(m)
                .and_then(|i| values.get(i))
                .copied()
                .unwrap_or(f64::NAN);
            let _ = write!(out, "{:>16}", num(v));
        }
    }
    out
}

/// Run a backtest of `model` as `args` say, and return the report.
pub(crate) fn run(model: &dyn DynModel, id: &str, args: &EvalArgs) -> Result<String, String> {
    let options = Options {
        periods: args.periods,
        splits: args.splits,
        stride: args.stride,
        retrain: args.retrain,
    };
    options.validate().map_err(|e| e.to_string())?;
    let info = model.info();
    let periods = u32::try_from(args.periods).unwrap_or(u32::MAX);
    if periods < info.min_prediction_periods || periods > info.max_prediction_periods {
        return Err(format!(
            "--periods {periods} is outside the model's bounds [{}, {}]",
            info.min_prediction_periods, info.max_prediction_periods
        ));
    }
    let weather = weather_provider(args.weather.id())
        .ok_or_else(|| format!("unknown weather provider {}", args.weather.id()))?;
    let (data, source) = dataset(model, args)?;
    let config = model.normalize_config(json!({"prediction_periods": periods}))?;
    let run = RunInfo {
        prediction_periods: periods,
        additional_continuous_covariates: config
            .get("additional_continuous_covariates")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default(),
        future_covariate_origin: None,
    };
    let started = Instant::now();
    let backtest = chapkit_eval::backtest(
        &data,
        &options,
        weather.as_ref(),
        |d| {
            model
                .train(&config, TrainInput::new(d, &run, None))
                .map(|(trained, _)| trained)
        },
        |t, h, f| model.predict(t, &config, PredictInput::new(h, f, &run, None)),
    )
    .map_err(|e| e.to_string())?;
    let scores = backtest.score(&data).map_err(|e| e.to_string())?;
    let elapsed = started.elapsed();

    let locations = data
        .group_indices(cols::LOCATION)
        .map_err(|e| e.to_string())?
        .len();
    let unit = match backtest.split_last.first().and_then(|p| Period::parse(p)) {
        Some(Period::Week { .. }) => "weeks",
        _ => "months",
    };
    let n_periods = data
        .group_indices(cols::TIME_PERIOD)
        .map_err(|e| e.to_string())?
        .len();
    let mut out = format!(
        "{id}: {} splits of {} periods, stride {}, {} training{}, {} weather\n\
         data: {source}, {locations} locations x {n_periods} {unit}\n\
         backtest: {:.1} ms, {} forecasts scored\n\n{}",
        options.splits,
        options.periods,
        options.stride,
        backtest.trained_at.len(),
        if backtest.trained_at.len() == 1 {
            ""
        } else {
            "s"
        },
        args.weather.id(),
        elapsed.as_secs_f64() * 1000.0,
        scores.rows,
        table(&scores),
    );
    if args.by_location {
        let _ = write!(out, "\n\n{}", by_location(&scores));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Weather;
    use crate::erased::Erased;
    use chapkit_core::{Model, ModelInfo, NoConfig, Result as ModelResult, prediction_frame};

    /// Forecasts the history's last target, three samples around it.
    struct Last;

    impl Model for Last {
        type Config = NoConfig;
        type Trained = ();

        fn info(&self) -> ModelInfo {
            ModelInfo::default()
        }

        fn train(&self, _: &NoConfig, _: TrainInput<'_>) -> ModelResult<()> {
            Ok(())
        }

        fn predict(
            &self,
            (): &(),
            _: &NoConfig,
            input: PredictInput<'_>,
        ) -> ModelResult<DataFrame> {
            let y = input.historic.f64_column(cols::DISEASE_CASES)?;
            let last = y
                .iter()
                .rev()
                .find(|v| v.is_finite())
                .copied()
                .unwrap_or(0.0);
            prediction_frame(input.future, 3, |_, s| {
                s.copy_from_slice(&[last - 1.0, last, last + 1.0]);
                Ok(())
            })
        }
    }

    fn args() -> EvalArgs {
        EvalArgs {
            data: None,
            periods: 2,
            splits: 3,
            stride: 1,
            retrain: 2,
            weather: Weather::Climatology,
            locations: 3,
            history: 30,
            seed: Some(7),
            by_location: true,
        }
    }

    #[test]
    fn backtests_generated_data() {
        let report = run(&Erased(Last), "last", &args()).unwrap();
        assert!(
            report.starts_with(
                "last: 3 splits of 2 periods, stride 1, 2 trainings, climatology weather"
            ),
            "{report}"
        );
        assert!(
            report.contains("generated (seed 7), 3 locations x 30 months"),
            "{report}"
        );
        assert!(report.contains("18 forecasts scored"), "{report}");
        assert!(
            report.contains("h2") && report.contains("CRPS") && report.contains("location_2"),
            "{report}"
        );
        // The same seed gives the same scores.
        let again = run(&Erased(Last), "last", &args()).unwrap();
        let table = |r: &str| r.split_once("\n\n").unwrap().1.to_owned();
        assert_eq!(table(&report), table(&again));
    }

    #[test]
    fn backtests_a_csv_file_and_checks_its_settings() {
        let path = std::env::temp_dir().join(format!("chapkit-eval-{}.csv", std::process::id()));
        let mut csv = String::from("time_period,location,disease_cases,rainfall\n");
        for m in 1..=12 {
            let _ = write!(csv, "2023-{m:02},A,{m},1.5\n2023-{m:02},B,{},2\n", 2 * m);
        }
        std::fs::write(&path, csv).unwrap();
        let file = EvalArgs {
            data: Some(path.clone()),
            by_location: false,
            ..args()
        };
        let report = run(&Erased(Last), "last", &file).unwrap();
        assert!(report.contains("2 locations x 12 months"), "{report}");
        assert!(report.contains("12 forecasts scored"), "{report}");
        let too_many = EvalArgs { splits: 20, ..file };
        assert!(
            run(&Erased(Last), "last", &too_many)
                .unwrap_err()
                .contains("too few")
        );
        std::fs::remove_file(path).unwrap();
        let bounds = EvalArgs {
            periods: 500,
            ..args()
        };
        assert!(
            run(&Erased(Last), "last", &bounds)
                .unwrap_err()
                .contains("bounds")
        );
    }
}
