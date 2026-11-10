//! `<bin> eval`: a backtest of the service's model in-process, with [`chapkit_eval`].

use std::fmt::Write as _;

use chapkit_core::synth::{Generator, Resolution, Spec};
use chapkit_core::{DataFrame, PeriodType};
use chapkit_eval::{Metric, Options, Prepare, Scores};
use serde_json::Value;

use crate::cli::{BacktestArgs, EvalArgs};
use crate::erased::DynModel;
use crate::evaluate::{Report, backtest_config, check, evaluate};

/// Read a dataset: chapkit's JSON (`{columns, data}`) for `.json`, else CSV.
pub(crate) fn read(path: &std::path::Path) -> Result<DataFrame, String> {
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
fn dataset(
    model: &dyn DynModel,
    args: &BacktestArgs,
    seed: u64,
) -> Result<(DataFrame, String), String> {
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

/// `--config`: a JSON object, or `@path` to a file holding one.
pub(crate) fn config_arg(arg: Option<&str>) -> Result<Option<Value>, String> {
    let Some(arg) = arg else {
        return Ok(None);
    };
    let text = match arg.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?,
        None => arg.to_owned(),
    };
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| format!("--config: {e}"))
}

/// The backtest options of `args`.
pub(crate) fn options(args: &BacktestArgs) -> Options {
    Options {
        periods: args.periods,
        splits: args.splits,
        stride: args.stride,
        retrain: args.retrain,
        prepare: Prepare {
            fill_gaps: !args.keep_gaps,
            pad: !args.no_pad,
        },
        drop_untrained: !args.keep_untrained,
    }
}

/// What a backtest of `args` needs: its options, the model's config and the dataset.
pub(crate) struct Prepared {
    pub options: Options,
    pub config: Value,
    pub data: DataFrame,
    pub source: String,
    /// `--seed`, or one drawn at random.
    pub seed: u64,
}

/// Check `args` and load what the backtest needs.
pub(crate) fn prepared(model: &dyn DynModel, args: &BacktestArgs) -> Result<Prepared, String> {
    let options = options(args);
    check(model, &options, args.weather.id())?;
    let config = backtest_config(model, config_arg(args.config.as_deref())?, args.periods)?;
    let seed = args
        .seed
        .unwrap_or_else(|| chapkit_core::rng::Rng::from_time().next_u64() % 1_000_000);
    let (data, source) = dataset(model, args, seed)?;
    Ok(Prepared {
        options,
        config,
        data,
        source,
        seed,
    })
}

/// Run a backtest of `model` as `args` say.
pub(crate) fn report(
    model: &dyn DynModel,
    id: &str,
    args: &BacktestArgs,
) -> Result<Report, String> {
    let Prepared {
        options,
        config,
        data,
        source,
        ..
    } = prepared(model, args)?;
    evaluate(
        model,
        id,
        &config,
        &data,
        source,
        &options,
        args.weather.id(),
    )
}

/// Run a backtest of `model` as `args` say, and return the report: a table, or with `--json`
/// the whole result as JSON.
pub(crate) fn run(model: &dyn DynModel, id: &str, args: &EvalArgs) -> Result<String, String> {
    let r = report(model, id, &args.backtest)?;
    if args.json {
        return serde_json::to_string_pretty(&r).map_err(|e| e.to_string());
    }
    let options = &r.options;
    // Preparation is named only when it is not chap-core's.
    let preparation = match (args.backtest.keep_gaps, args.backtest.no_pad) {
        (false, false) => "",
        (true, false) => ", gaps kept",
        (false, true) => ", not padded",
        (true, true) => ", gaps kept, not padded",
    };
    let mut out = format!(
        "{id}: {} splits of {} periods, stride {}, {} training{}, {} weather{preparation}\n\
         data: {}, {} locations x {} {}{}\n\
         backtest: {:.1} ms, {} forecasts scored\n\n{}",
        options.splits,
        options.periods,
        options.stride,
        r.trained_at.len(),
        if r.trained_at.len() == 1 { "" } else { "s" },
        options.weather,
        r.data.source,
        r.data.locations,
        r.data.periods,
        r.data.unit,
        match r.data.dropped.len() {
            0 => String::new(),
            n => format!(", {n} dropped (no target before the test windows)"),
        },
        r.elapsed_ms,
        r.rows,
        table(&r.scores),
    );
    if args.by_location {
        let _ = write!(out, "\n\n{}", by_location(&r.scores));
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::cli::Weather;
    use crate::erased::Erased;
    use chapkit_core::{
        DataFrame, Model, ModelInfo, PredictInput, Result as ModelResult, TrainInput, cols,
        prediction_frame,
    };

    /// Forecasts the history's last target, three samples `spread` apart around it.
    pub(crate) struct Last;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    pub(crate) struct Spread {
        #[serde(default = "one")]
        spread: f64,
    }

    fn one() -> f64 {
        1.0
    }

    impl Model for Last {
        type Config = Spread;
        type Trained = ();

        fn info(&self) -> ModelInfo {
            ModelInfo::default()
        }

        fn search_space(&self) -> Option<chapkit_core::SearchSpace> {
            Some(chapkit_core::SearchSpace::new().with(
                "spread",
                chapkit_core::Param::Float {
                    low: 0.1,
                    high: 20.0,
                    step: None,
                    log: true,
                },
            ))
        }

        fn train(&self, _: &Spread, _: TrainInput<'_>) -> ModelResult<()> {
            Ok(())
        }

        fn predict(
            &self,
            (): &(),
            config: &Spread,
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
                let d = config.spread;
                s.copy_from_slice(&[last - d, last, last + d]);
                Ok(())
            })
        }
    }

    pub(crate) fn backtest_args() -> BacktestArgs {
        BacktestArgs {
            data: None,
            periods: 2,
            splits: 3,
            stride: 1,
            retrain: 2,
            weather: Weather::Climatology,
            keep_gaps: false,
            no_pad: false,
            keep_untrained: false,
            locations: 3,
            history: 30,
            seed: Some(7),
            config: None,
        }
    }

    fn args() -> EvalArgs {
        EvalArgs {
            backtest: backtest_args(),
            by_location: true,
            json: false,
        }
    }

    /// `args()` with its backtest settings changed by `f`.
    fn with(f: impl FnOnce(&mut BacktestArgs)) -> EvalArgs {
        let mut a = args();
        f(&mut a.backtest);
        a
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
    fn prepares_as_chap_core_unless_told_not_to() {
        let path =
            std::env::temp_dir().join(format!("chapkit-eval-gaps-{}.csv", std::process::id()));
        let mut csv = String::from("time_period,location,disease_cases\n");
        for m in 1..=12 {
            let _ = writeln!(csv, "2023-{m:02},A,{m}");
            // B misses 2023-06 and ends at 2023-11.
            if m != 6 && m != 12 {
                let _ = writeln!(csv, "2023-{m:02},B,{m}");
            }
        }
        std::fs::write(&path, csv).unwrap();
        let chap = EvalArgs {
            by_location: false,
            ..with(|b| b.data = Some(path.clone()))
        };
        let report = run(&Erased(Last), "last", &chap).unwrap();
        assert!(report.contains("climatology weather\n"), "{report}");
        let mut kept = chap.clone();
        kept.backtest.keep_gaps = true;
        kept.backtest.no_pad = true;
        let report = run(&Erased(Last), "last", &kept).unwrap();
        assert!(
            report.contains("climatology weather, gaps kept, not padded\n"),
            "{report}"
        );
        std::fs::remove_file(path).unwrap();
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
            by_location: false,
            ..with(|b| b.data = Some(path.clone()))
        };
        let report = run(&Erased(Last), "last", &file).unwrap();
        assert!(report.contains("2 locations x 12 months"), "{report}");
        assert!(report.contains("12 forecasts scored"), "{report}");
        let mut too_many = file;
        too_many.backtest.splits = 20;
        assert!(
            run(&Erased(Last), "last", &too_many)
                .unwrap_err()
                .contains("too few")
        );
        std::fs::remove_file(path).unwrap();
        let bounds = with(|b| b.periods = 500);
        assert!(
            run(&Erased(Last), "last", &bounds)
                .unwrap_err()
                .contains("bounds")
        );
    }

    #[test]
    fn json_has_every_score_at_full_precision_and_options_change_the_model() {
        let json = EvalArgs {
            json: true,
            ..args()
        };
        let out: Value = serde_json::from_str(&run(&Erased(Last), "last", &json).unwrap()).unwrap();
        let base = report(&Erased(Last), "last", &args().backtest).unwrap();
        assert_eq!(out["model"], "last");
        assert_eq!(out["options"]["weather"], "climatology");
        assert_eq!(out["options"]["splits"], 3);
        assert_eq!(out["config"]["spread"], 1.0);
        assert_eq!(out["config"]["prediction_periods"], 2);
        assert_eq!(out["data"]["source"], "generated (seed 7)");
        assert_eq!(out["data"]["locations"], 3);
        assert_eq!(out["rows"], 18);
        // Every bit of every value survives the JSON.
        let crps = out["aggregate"]["crps"].as_f64().unwrap();
        assert_eq!(crps.to_bits(), base.scores.get(Metric::Crps).to_bits());
        assert_eq!(out["by_horizon"].as_array().unwrap().len(), 2);
        assert_eq!(out["by_horizon"][1]["horizon"], 2);
        assert_eq!(out["by_location"][0]["location"], "location_0");
        assert!(out["by_location"][0]["metrics"]["mae"].is_f64());

        // --config, flat or nested as chap-core sends it, or from a file.
        let wide = |config: &str| {
            let a = with(|b| b.config = Some(config.into()));
            report(&Erased(Last), "last", &a.backtest).unwrap()
        };
        let flat = wide(r#"{"spread": 5}"#);
        assert_eq!(flat.config["spread"], 5);
        assert!(flat.scores.get(Metric::Coverage10_90) >= base.scores.get(Metric::Coverage10_90));
        assert_ne!(flat.aggregate["crps"], base.aggregate["crps"]);
        let nested = wide(r#"{"user_option_values": {"spread": 5}}"#);
        assert_eq!(nested.aggregate, flat.aggregate);
        let path =
            std::env::temp_dir().join(format!("chapkit-eval-config-{}.json", std::process::id()));
        std::fs::write(&path, r#"{"spread": 5}"#).unwrap();
        let file = wide(&format!("@{}", path.display()));
        assert_eq!(file.aggregate, flat.aggregate);
        std::fs::remove_file(path).unwrap();
        let bad = with(|b| b.config = Some(r#"{"spread": "wide"}"#.into()));
        assert!(report(&Erased(Last), "last", &bad.backtest).is_err());
        let not_json = with(|b| b.config = Some("{".into()));
        assert!(
            run(&Erased(Last), "last", &not_json)
                .unwrap_err()
                .contains("--config")
        );
    }

    #[test]
    fn drops_untrained_locations_unless_told_not_to() {
        let path =
            std::env::temp_dir().join(format!("chapkit-eval-untrained-{}.csv", std::process::id()));
        let mut csv = String::from("time_period,location,disease_cases\n");
        for m in 1..=12 {
            // C is observed only in the test windows.
            let c = if m > 8 { m.to_string() } else { String::new() };
            let _ = write!(
                csv,
                "2023-{m:02},A,{m}\n2023-{m:02},B,{m}\n2023-{m:02},C,{c}\n"
            );
        }
        std::fs::write(&path, csv).unwrap();
        let file = EvalArgs {
            by_location: false,
            ..with(|b| b.data = Some(path.clone()))
        };
        let text = run(&Erased(Last), "last", &file).unwrap();
        assert!(
            text.contains(
                "3 locations x 12 months, 1 dropped (no target before the test windows)\n"
            ),
            "{text}"
        );
        let r = report(&Erased(Last), "last", &file.backtest).unwrap();
        assert_eq!(r.data.dropped, ["C"]);
        assert!(r.by_location.iter().all(|l| l.location != "C"));
        let mut kept = file.backtest.clone();
        kept.keep_untrained = true;
        let r = report(&Erased(Last), "last", &kept).unwrap();
        assert!(r.data.dropped.is_empty() && !r.options.drop_untrained);
        assert!(r.by_location.iter().any(|l| l.location == "C"));
        std::fs::remove_file(path).unwrap();
    }
}
