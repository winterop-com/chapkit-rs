//! A model binary's command line, parsed by [`Service::serve`](crate::Service::serve).
//!
//! `<bin> serve` serves, `<bin> test` runs chap-core's flow against the running service,
//! `<bin> eval` backtests the model in-process, and `<bin> healthcheck` checks the service (the
//! Docker `HEALTHCHECK`); without arguments the binary prints its help. A binary with subcommands of its own adds `#[command(flatten)] Chapkit(Command)` to
//! its own clap subcommand enum, parses it with [`Service::parse_args`](crate::Service::parse_args)
//! and hands the [`Command`] to [`Service::run`](crate::Service::run).

use std::fmt::Display;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// A command's outcome as the binary's exit code: on error, one line `error: ...` on stderr and
/// exit code 1. Model binaries with commands of their own end `main` with it, so every command
/// fails the same way.
pub fn exit_code<E: Display>(result: Result<(), E>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The service's address: where `serve` binds and where `test` and `healthcheck` look.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct Address {
    /// Service host
    #[arg(
        long,
        env = "HOST",
        hide_env_values = true,
        default_value = "127.0.0.1"
    )]
    pub host: String,
    /// Service port
    #[arg(long, env = "PORT", hide_env_values = true, default_value_t = 8000)]
    pub port: u16,
}

/// `<bin> serve`.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct ServeArgs {
    #[command(flatten)]
    pub address: Address,
    /// Database URL
    #[arg(long, value_name = "URL", env = "DATABASE_URL", hide_env_values = true)]
    pub database: Option<String>,
    /// Jobs at once
    #[arg(
        long,
        value_name = "N",
        env = "CHAPKIT_MAX_CONCURRENCY",
        hide_env_values = true
    )]
    pub max_concurrency: Option<usize>,
}

/// `<bin> test`: chap-core's flow against the service on `--host`/`--port`.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct TestArgs {
    #[command(flatten)]
    pub address: Address,
    /// Service URL [default: from --host, --port]
    #[arg(long, short)]
    pub url: Option<String>,
    #[command(flatten)]
    pub flow: chapkit_flow::Options,
}

/// Where a backtest's future covariates come from.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Weather {
    // Seasonal means of the history. Plain comments: help on the values would switch clap to its
    // long help layout.
    #[default]
    Climatology,
    // As observed: looks ahead, for diagnostics only.
    Observed,
    // Climatology plus the last anomaly, damped with the horizon.
    #[value(alias = "damped_persistence")]
    DampedPersistence,
}

impl Weather {
    /// chap-core's provider id.
    pub fn id(self) -> &'static str {
        match self {
            Weather::Climatology => "climatology",
            Weather::Observed => "observed",
            Weather::DampedPersistence => "damped_persistence",
        }
    }
}

/// The data and settings of an in-process backtest (`eval`, `hpo`).
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct BacktestArgs {
    /// Dataset, CSV or JSON [default: generated]
    #[arg(long, value_name = "FILE")]
    pub data: Option<PathBuf>,
    /// Periods per forecast
    #[arg(long, default_value_t = 3)]
    pub periods: usize,
    /// Splits
    #[arg(long, default_value_t = 7)]
    pub splits: usize,
    /// Periods between splits
    #[arg(long, default_value_t = 1)]
    pub stride: usize,
    /// Trainings
    #[arg(long, default_value_t = 1)]
    pub retrain: usize,
    /// Future weather (observed looks ahead)
    #[arg(long, value_enum, default_value_t)]
    pub weather: Weather,
    /// Keep gaps in a location's periods
    #[arg(long)]
    pub keep_gaps: bool,
    /// Do not pad locations to the full range
    #[arg(long)]
    pub no_pad: bool,
    /// Keep locations with no target before the test windows
    #[arg(long)]
    pub keep_untrained: bool,
    /// Generated locations
    #[arg(long, default_value_t = 5)]
    pub locations: usize,
    /// Generated periods
    #[arg(long, default_value_t = 60)]
    pub history: usize,
    /// Seed (generated data, search) [default: random]
    #[arg(long)]
    pub seed: Option<u64>,
    /// Model options, JSON or @file
    #[arg(long, value_name = "JSON")]
    pub config: Option<String>,
}

/// `<bin> eval`: a backtest in-process, scored as chap-core scores it.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct EvalArgs {
    #[command(flatten)]
    pub backtest: BacktestArgs,
    /// Scores by location
    #[arg(long)]
    pub by_location: bool,
    /// Print the result as JSON
    #[arg(long)]
    pub json: bool,
}

/// A hyperparameter searcher.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    #[default]
    Random,
    Grid,
}

/// The metrics a search can minimize: chap-core's proper scoring rules.
pub const OBJECTIVES: [&str; 9] = [
    "crps",
    "crps_log1p",
    "crps_norm",
    "mae",
    "rmse",
    "winkler_score_10_90",
    "winkler_score_25_75",
    "winkler_score_10_90_log1p",
    "winkler_score_25_75_log1p",
];

/// `<bin> hpo`: a hyperparameter search over the model's search space, each trial an in-process
/// backtest.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct HpoArgs {
    #[command(flatten)]
    pub backtest: BacktestArgs,
    /// Trials
    #[arg(long, default_value_t = 20)]
    pub trials: usize,
    /// Searcher
    #[arg(long, value_enum, default_value_t)]
    pub method: Method,
    /// Let grid search step through ranges
    #[arg(long)]
    pub grid_ranges: bool,
    /// Metric to minimize
    #[arg(long, default_value = "crps", value_parser = clap::builder::PossibleValuesParser::new(OBJECTIVES))]
    pub metric: String,
    /// Print the result as JSON
    #[arg(long)]
    pub json: bool,
}

/// The library's subcommands.
#[derive(Subcommand, Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Serve the model
    Serve(ServeArgs),
    /// Test the running service
    Test(TestArgs),
    /// Backtest the model in-process
    Eval(EvalArgs),
    /// Search the model's hyperparameters
    Hpo(HpoArgs),
    /// Check the running service's health
    Healthcheck(Address),
}

/// The command line of a binary with only the library's subcommands.
#[derive(Parser, Debug, Clone, PartialEq, Eq)]
#[command(arg_required_else_help = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

impl Address {
    /// The service's URL as a client sees it (`0.0.0.0` and `::` mean this machine).
    pub fn url(&self) -> String {
        let port = self.port;
        match self.host.as_str() {
            "0.0.0.0" | "" => format!("http://127.0.0.1:{port}"),
            "::" => format!("http://[::1]:{port}"),
            h if h.contains(':') && !h.starts_with('[') => format!("http://[{h}]:{port}"),
            h => format!("http://{h}:{port}"),
        }
    }
}

/// The URL `test` calls: `--url`, else `--host`/`--port`.
pub(crate) fn test_url(test: &TestArgs) -> String {
    test.url.clone().unwrap_or_else(|| test.address.url())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, FromArgMatches};

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let m = Cli::command().try_get_matches_from(args)?;
        Cli::from_arg_matches(&m)
    }

    fn serve(args: &[&str]) -> ServeArgs {
        match parse(args).unwrap().command {
            Command::Serve(s) => s,
            other => panic!("expected serve, got {other:?}"),
        }
    }

    /// The env-var tests run in a child process (this test binary again), since setting
    /// variables in this process is unsafe and the crate forbids unsafe code.
    fn in_child(test: &str, env: &[(&str, &str)]) {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", test, "--nocapture"])
            .env("CHAPKIT_CLI_CHILD", "1")
            .env_remove("HOST")
            .env_remove("PORT");
        for (k, v) in env {
            cmd.env(k, v);
        }
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stdout)
        );
    }

    fn is_child() -> bool {
        std::env::var_os("CHAPKIT_CLI_CHILD").is_some()
    }

    #[test]
    fn no_arguments_shows_help() {
        let e = parse(&["bin"]).unwrap_err();
        assert_eq!(
            e.kind(),
            clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
    }

    #[test]
    fn serve_defaults() {
        if !is_child() {
            return in_child("cli::tests::serve_defaults", &[]);
        }
        let s = serve(&["bin", "serve"]);
        assert_eq!(s.address.host, "127.0.0.1");
        assert_eq!(s.address.port, 8000);
        assert_eq!(s.database, None);
        assert_eq!(s.max_concurrency, None);
    }

    #[test]
    fn env_is_read_and_flags_override_it() {
        if !is_child() {
            return in_child(
                "cli::tests::env_is_read_and_flags_override_it",
                &[("HOST", "0.0.0.0"), ("PORT", "9001")],
            );
        }
        let s = serve(&["bin", "serve"]);
        assert_eq!((s.address.host.as_str(), s.address.port), ("0.0.0.0", 9001));
        let s = serve(&["bin", "serve", "--host", "10.0.0.1", "--port", "8123"]);
        assert_eq!(
            (s.address.host.as_str(), s.address.port),
            ("10.0.0.1", 8123)
        );
        let Command::Healthcheck(a) = parse(&["service", "healthcheck"]).unwrap().command else {
            panic!("expected healthcheck")
        };
        assert_eq!(a.port, 9001);
    }

    #[test]
    fn healthcheck_parses_as_docker_calls_it() {
        let cli = parse(&["service", "healthcheck", "--port", "8123"]).unwrap();
        assert!(matches!(cli.command, Command::Healthcheck(a) if a.port == 8123));
    }

    #[test]
    fn database_and_concurrency_flags() {
        let s = serve(&[
            "bin",
            "serve",
            "--database",
            "data/x.db",
            "--max-concurrency",
            "4",
        ]);
        assert_eq!(s.database.as_deref(), Some("data/x.db"));
        assert_eq!(s.max_concurrency, Some(4));
        assert!(parse(&["bin", "serve", "--port", "nope"]).is_err());
        assert!(parse(&["bin", "unknown"]).is_err());
    }

    #[test]
    fn eval_defaults_as_chap_core() {
        let Command::Eval(e) = parse(&["bin", "eval"]).unwrap().command else {
            panic!("expected eval")
        };
        assert_eq!(
            (
                e.backtest.periods,
                e.backtest.splits,
                e.backtest.stride,
                e.backtest.retrain
            ),
            (3, 7, 1, 1)
        );
        assert_eq!(e.backtest.weather, Weather::Climatology);
        assert_eq!(e.backtest.data, None);
        assert!(!e.backtest.keep_gaps && !e.backtest.no_pad);
        let Command::Eval(e) = parse(&[
            "bin",
            "eval",
            "--data",
            "d.csv",
            "--weather",
            "observed",
            "--retrain",
            "2",
        ])
        .unwrap()
        .command
        else {
            panic!("expected eval")
        };
        assert_eq!(e.backtest.data, Some(PathBuf::from("d.csv")));
        assert_eq!(
            (e.backtest.weather.id(), e.backtest.retrain),
            ("observed", 2)
        );
        assert!(parse(&["bin", "eval", "--weather", "mstl"]).is_err());
        for name in ["damped-persistence", "damped_persistence"] {
            let Command::Eval(e) = parse(&["bin", "eval", "--weather", name]).unwrap().command
            else {
                panic!("expected eval")
            };
            assert_eq!(e.backtest.weather.id(), "damped_persistence");
        }
        let Command::Eval(e) = parse(&["bin", "eval", "--keep-gaps", "--no-pad"])
            .unwrap()
            .command
        else {
            panic!("expected eval")
        };
        assert!(e.backtest.keep_gaps && e.backtest.no_pad);
        assert!(!e.json && e.backtest.config.is_none());
        let Command::Eval(e) = parse(&["bin", "eval", "--json", "--config", "@c.json"])
            .unwrap()
            .command
        else {
            panic!("expected eval")
        };
        assert!(e.json);
        assert_eq!(e.backtest.config.as_deref(), Some("@c.json"));
    }

    #[test]
    fn test_finds_the_service_on_host_and_port() {
        let Command::Test(t) = parse(&["bin", "test", "--port", "8123", "--locations", "10"])
            .unwrap()
            .command
        else {
            panic!("expected test")
        };
        assert_eq!(t.flow.locations, 10);
        assert_eq!(test_url(&t), "http://127.0.0.1:8123");
        let mut any = t.clone();
        any.address.host = "0.0.0.0".into();
        assert_eq!(test_url(&any), "http://127.0.0.1:8123");
        let Command::Test(t) = parse(&["bin", "test", "--url", "http://h:1"])
            .unwrap()
            .command
        else {
            panic!("expected test")
        };
        assert_eq!(test_url(&t), "http://h:1");
    }
}
