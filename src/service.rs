//! [`Service`]: turn a [`Model`] into a chapkit-compatible HTTP service.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use chapkit_core::Model;
use chapkit_http::health::CheckResult;
use chapkit_http::{App, HealthState, Scheduler};
use chapkit_store::{Store, StoreError};
use serde_json::{Value, json};

use crate::cli::{self, Cli, Command, ServeArgs};
use crate::erased::{DynModel, Erased};
use crate::routes;
use crate::state::AppState;

/// The Python chapkit release whose HTTP API chapkit-rs matches.
pub const CHAPKIT_API_VERSION: &str = "2.6.0";

/// Builder for an ML service. Mirrors Python chapkit's `MLServiceBuilder`: one model,
/// configs, artifacts, jobs, health, system, info, and the ML endpoints.
pub struct Service {
    id: String,
    display_name: String,
    version: String,
    description: Option<String>,
    model: Arc<dyn DynModel>,
    database_url: Option<String>,
    max_concurrency: usize,
    eager_wait: Duration,
    cache_size: usize,
    max_body_bytes: Option<usize>,
}

impl Service {
    /// A service for `model`. `id` must be a slug (`^[a-z][a-z0-9]*(-[a-z0-9]+)*$`).
    ///
    /// Defaults read from the environment: `DATABASE_URL` (in-memory when unset),
    /// `CHAPKIT_EAGER_WAIT_MS` (1000), `CHAPKIT_MAX_CONCURRENCY` (1), `CHAPKIT_MAX_BODY_MB` (1024).
    pub fn new<M: Model>(id: impl Into<String>, display_name: impl Into<String>, model: M) -> Self {
        let env_u64 = |k: &str, d: u64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        Self {
            id: id.into(),
            display_name: display_name.into(),
            version: "1.0.0".into(),
            description: None,
            model: Arc::new(Erased(model)),
            database_url: std::env::var("DATABASE_URL").ok(),
            max_concurrency: usize::try_from(env_u64("CHAPKIT_MAX_CONCURRENCY", 1)).unwrap_or(1),
            eager_wait: Duration::from_millis(env_u64("CHAPKIT_EAGER_WAIT_MS", 1000)),
            cache_size: 16,
            max_body_bytes: None,
        }
    }

    /// Service version (default `1.0.0`).
    #[must_use]
    pub fn version(mut self, v: impl Into<String>) -> Self {
        self.version = v.into();
        self
    }

    /// Description.
    #[must_use]
    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }

    /// Database URL (`:memory:`, a path, `sqlite:///path`, or `turso:///path` with the
    /// `turso` feature). A default: `serve` uses `--database` or `DATABASE_URL` when given.
    #[must_use]
    pub fn database(mut self, url: impl Into<String>) -> Self {
        self.database_url = Some(url.into());
        self
    }

    /// Jobs allowed to run at once. A default: `serve` uses `--max-concurrency` or
    /// `CHAPKIT_MAX_CONCURRENCY` when given.
    #[must_use]
    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    /// How long `$train`/`$predict` wait for a job before answering 202.
    /// Jobs finished within this budget are already complete at the client's first poll.
    #[must_use]
    pub fn eager_wait(mut self, d: Duration) -> Self {
        self.eager_wait = d;
        self
    }

    /// Largest accepted request body in bytes (default `CHAPKIT_MAX_BODY_MB`, else 1 GiB).
    #[must_use]
    pub fn max_body_bytes(mut self, bytes: usize) -> Self {
        self.max_body_bytes = Some(bytes);
        self
    }

    /// Number of trained models kept warm in memory.
    #[must_use]
    pub fn cache_size(mut self, n: usize) -> Self {
        self.cache_size = n.max(1);
        self
    }

    /// The `/api/v1/info` document (chapkit's `MLServiceInfo`).
    pub fn info(&self) -> Value {
        let m = self.model.info();
        json!({
            "id": self.id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "model_metadata": m.model_metadata,
            "period_type": m.period_type,
            "min_prediction_periods": m.min_prediction_periods,
            "max_prediction_periods": m.max_prediction_periods,
            "allow_free_additional_continuous_covariates": m.allow_free_additional_continuous_covariates,
            "required_covariates": m.required_covariates,
            "requires_geo": m.requires_geo,
            "target": m.target,
            "hpo_search_space": self.model.search_space(),
            "git_revision": std::env::var("GIT_REVISION").ok().filter(|s| !s.is_empty()),
            // The Python chapkit API version this service is compatible with, with
            // chapkit-rs as semver build metadata, so version checks in chap tooling parse it.
            "chapkit_version": format!("{CHAPKIT_API_VERSION}+chapkit-rs.{}", env!("CARGO_PKG_VERSION")),
            "chapkit_rs_version": env!("CARGO_PKG_VERSION"),
            "servicekit_version": null,
            "extensions": ["$predict-sync", "$explain"],
        })
    }

    /// Build the app (routers, state, store). Use [`App::serve`] to run it, or
    /// [`App::build`] for an in-process router (tests, benchmarks).
    pub async fn app(self) -> Result<App, StoreError> {
        let store: Arc<dyn Store> = chapkit_store::open(self.database_url.as_deref()).await?;
        let info = self.info();
        let jobs = Scheduler::new(self.max_concurrency);
        let state = AppState::new(
            self.model,
            store.clone(),
            jobs.clone(),
            self.id.clone(),
            self.version.clone(),
            self.eager_wait,
            self.cache_size,
        );
        let check: chapkit_http::health::Check = Arc::new(move || {
            let store = store.clone();
            Box::pin(async move {
                match store.ping().await {
                    Ok(()) => CheckResult {
                        state: HealthState::Healthy,
                        message: None,
                    },
                    Err(e) => CheckResult {
                        state: HealthState::Unhealthy,
                        message: Some(format!("Check failed: {e}")),
                    },
                }
            })
        });
        let mut app = App::new(self.id, self.display_name, self.version, info);
        if let Some(bytes) = self.max_body_bytes {
            app = app.max_body_bytes(bytes);
        }
        Ok(app
            .health_check("database", check)
            .nest("/api/v1/configs", routes::configs::router(state.clone()))
            .nest(
                "/api/v1/artifacts",
                routes::artifacts::router(state.clone()),
            )
            .nest("/api/v1/ml", routes::ml::router(state))
            .nest("/api/v1/jobs", chapkit_http::jobs::router(jobs)))
    }

    /// The binary's `main`: parse the command line ([`Cli`]) and [`run`](Self::run) it. Without
    /// arguments the binary prints its help; an error is printed as one line `error: ...` and
    /// the exit code is 1.
    pub async fn serve(self) -> ExitCode {
        let cli: Cli = self.parse_args();
        cli::exit_code(self.run(cli.command).await)
    }

    /// Parse the command line into `P`, with this service's id, version and display name in
    /// `--help` and `--version`. Exits with clap's message on `--help`, `--version` or bad input.
    pub fn parse_args<P: clap::Parser>(&self) -> P {
        P::from_arg_matches(&self.command::<P>().get_matches()).unwrap_or_else(|e| e.exit())
    }

    /// The clap command for `P`, named after the service; `--version` also names the chapkit-rs
    /// version it was built with (`global-median 1.0.2 (chapkit-rs 0.1.0)`).
    fn command<P: clap::Parser>(&self) -> clap::Command {
        P::command()
            .name(self.id.clone())
            .version(format!(
                "{} (chapkit-rs {})",
                self.version,
                env!("CARGO_PKG_VERSION")
            ))
            .about(self.display_name.clone())
    }

    /// Apply `serve`'s `--database` and `--max-concurrency` over the builder. Each comes from its
    /// flag, else its environment variable (`DATABASE_URL`, `CHAPKIT_MAX_CONCURRENCY`), so the
    /// order is flag, environment variable, builder, default: a deployment's settings win over
    /// values written in the model's code.
    #[must_use]
    pub fn with_serve_args(mut self, args: &ServeArgs) -> Self {
        if let Some(url) = &args.database {
            self.database_url = Some(url.clone());
        }
        if let Some(n) = args.max_concurrency {
            self = self.max_concurrency(n);
        }
        self
    }

    /// Run a [`Command`]: `serve` until shutdown (`--database` and `--max-concurrency`, or their
    /// environment variables, override the builder), `test` the running service, or `healthcheck` it.
    pub async fn run(mut self, command: Command) -> std::io::Result<()> {
        match command {
            Command::Healthcheck(address) => chapkit_http::health::probe(&address.url()).await,
            Command::Test(test) => {
                let url = cli::test_url(&test);
                if !chapkit_flow::reachable(&url).await {
                    let bin = std::env::args().next().unwrap_or_else(|| self.id.clone());
                    return Err(std::io::Error::other(format!(
                        "no service at {url}; start it first with `{bin} serve`"
                    )));
                }
                chapkit_flow::run(&url, &test.flow)
                    .await
                    .map_err(std::io::Error::other)
            }
            Command::Serve(args) => {
                self = self.with_serve_args(&args);
                self.app()
                    .await
                    .map_err(std::io::Error::other)?
                    .serve(&args.address.host, args.address.port)
                    .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Address;
    use crate::{ModelInfo, NoConfig, PredictInput, Result as ModelResult, TrainInput};

    struct Nop;

    impl Model for Nop {
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
        ) -> ModelResult<chapkit_dataframe::DataFrame> {
            crate::prediction_frame(input.future, 1, |_, _| Ok(()))
        }
    }

    fn args(max_concurrency: Option<usize>) -> ServeArgs {
        ServeArgs {
            address: Address {
                host: "127.0.0.1".into(),
                port: 8000,
            },
            database: None,
            max_concurrency,
        }
    }

    /// Runs `test` again in a child process with `env`, since setting variables in this process
    /// is unsafe and the crate forbids unsafe code; `true` in the child.
    fn in_child(test: &str, env: &[(&str, &str)]) -> bool {
        if std::env::var_os("CHAPKIT_SERVICE_CHILD").is_some() {
            return true;
        }
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", test, "--nocapture"])
            .env("CHAPKIT_SERVICE_CHILD", "1")
            .env_remove("CHAPKIT_MAX_CONCURRENCY")
            .env_remove("DATABASE_URL");
        for (k, v) in env {
            cmd.env(k, v);
        }
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stdout)
        );
        false
    }

    #[test]
    fn version_names_the_chapkit_rs_version() {
        let service = Service::new("my-model", "My model", Nop).version("1.2.3");
        let version = service.command::<crate::cli::Cli>().render_version();
        assert_eq!(
            version.trim(),
            format!("my-model 1.2.3 (chapkit-rs {})", env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn max_concurrency_default_then_builder_then_flag() {
        if !in_child(
            "service::tests::max_concurrency_default_then_builder_then_flag",
            &[],
        ) {
            return;
        }
        assert_eq!(Service::new("m", "M", Nop).max_concurrency, 1);
        let built = Service::new("m", "M", Nop).max_concurrency(3);
        assert_eq!(built.max_concurrency, 3);
        assert_eq!(built.with_serve_args(&args(None)).max_concurrency, 3);
        let flagged = Service::new("m", "M", Nop)
            .max_concurrency(3)
            .with_serve_args(&args(Some(5)));
        assert_eq!(flagged.max_concurrency, 5);
    }

    /// `serve`'s settings as the binary parses them: the flag, else the environment variable,
    /// both over the builder (so `DATABASE_URL` set by Docker wins over `.database(...)`).
    #[test]
    fn serve_env_beats_builder_and_flag_beats_env() {
        if !in_child(
            "service::tests::serve_env_beats_builder_and_flag_beats_env",
            &[
                ("CHAPKIT_MAX_CONCURRENCY", "4"),
                ("DATABASE_URL", "/env.db"),
            ],
        ) {
            return;
        }
        let serve = |argv: &[&str]| match <crate::cli::Cli as clap::Parser>::try_parse_from(argv)
            .unwrap()
            .command
        {
            Command::Serve(args) => args,
            other => panic!("not serve: {other:?}"),
        };
        let built = || {
            Service::new("m", "M", Nop)
                .database("/code.db")
                .max_concurrency(3)
        };
        let from_env = built().with_serve_args(&serve(&["m", "serve"]));
        assert_eq!(from_env.database_url.as_deref(), Some("/env.db"));
        assert_eq!(from_env.max_concurrency, 4);
        let from_flags = built().with_serve_args(&serve(&[
            "m",
            "serve",
            "--database",
            "/flag.db",
            "--max-concurrency",
            "2",
        ]));
        assert_eq!(from_flags.database_url.as_deref(), Some("/flag.db"));
        assert_eq!(from_flags.max_concurrency, 2);
    }

    #[test]
    fn max_concurrency_flag_over_env_over_default() {
        if !in_child(
            "service::tests::max_concurrency_flag_over_env_over_default",
            &[("CHAPKIT_MAX_CONCURRENCY", "4")],
        ) {
            return;
        }
        assert_eq!(Service::new("m", "M", Nop).max_concurrency, 4);
        let flagged = Service::new("m", "M", Nop).with_serve_args(&args(Some(2)));
        assert_eq!(flagged.max_concurrency, 2);
    }
}
