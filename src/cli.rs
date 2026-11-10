//! A model binary's command line, parsed by [`Service::serve`](crate::Service::serve).
//!
//! `<bin> serve` serves, `<bin> test` runs chap-core's flow against the running service, and
//! `<bin> healthcheck` checks it (the Docker `HEALTHCHECK`); without arguments the binary prints
//! its help. A binary with subcommands of its own adds `#[command(flatten)] Chapkit(Command)` to
//! its own clap subcommand enum, parses it with [`Service::parse_args`](crate::Service::parse_args)
//! and hands the [`Command`] to [`Service::run`](crate::Service::run).

use std::fmt::Display;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

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
    /// Database URL [env: DATABASE_URL]
    #[arg(long, value_name = "URL")]
    pub database: Option<String>,
    /// Jobs at once [env: CHAPKIT_MAX_CONCURRENCY]
    #[arg(long, value_name = "N")]
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

/// The library's subcommands.
#[derive(Subcommand, Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Serve the model
    Serve(ServeArgs),
    /// Test the running service
    Test(TestArgs),
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
