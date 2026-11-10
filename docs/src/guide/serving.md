# Serving and configuration

```rust,ignore
chapkit::Service::new("id", "Display name", model)
    .version(env!("CARGO_PKG_VERSION")) // MLServiceInfo.version (default 1.0.0)
    .description("...")
    .database("data/chapkit.db")   // default: DATABASE_URL, else in memory
    .max_concurrency(1)            // jobs at once
    .eager_wait(Duration::from_millis(1000))
    .cache_size(16)                // trained models kept warm
    .serve()
    .await
```

`Service::app()` returns the app instead of serving it; `app.build()` gives an axum `Router` for in-process
tests and benchmarks.

`serve()` is the binary's `main`: it parses the command line, so every model binary has the same commands
([reference](../reference/cli.md#a-model-binary)):

```sh
my-model serve                      # serve on 127.0.0.1:8000
my-model serve --port 8001 --database data/chapkit.db
my-model test --locations 10        # in another terminal: chap-core's flow against it
my-model healthcheck                # exit 0 when the running service is healthy
my-model --help
```

Without arguments the binary prints its help. A failing command prints one line, such as
`error: no service at http://127.0.0.1:8000` or `error: cannot listen on 127.0.0.1:8000: Address already in use`,
and exits with code 1. Each option of `serve` falls back to its environment variable
(`HOST`, `PORT`, `DATABASE_URL`, `CHAPKIT_MAX_CONCURRENCY`), so containers and chap-devops can keep configuring it
with environment variables. The order is flag, then environment variable, then the builder
(`.database(...)`, `.max_concurrency(...)`), then the default: a value written in the model's code is only a
default, so the `DATABASE_URL` the Docker image sets always wins.

The binary prints `INFO:     Uvicorn running on http://HOST:PORT`, the line chap-core's MLproject runner
waits for. All environment variables are listed in the [reference](../reference/environment.md).

## Commands of your own

A binary with subcommands of its own (the baselines' `demo`) flattens chapkit's into its clap enum and hands
them to `Service::run`:

```rust,ignore
#[derive(clap::Parser)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    #[command(flatten)]
    Chapkit(chapkit::cli::Command),
    /// Run once on example data
    Demo,
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let service = chapkit::Service::new("my-model", "My model", MyModel);
    // --help and --version show the service's name and version
    let result = match service.parse_args::<Cli>().command {
        Command::Demo => demo(),
        Command::Chapkit(command) => service.run(command).await,
    };
    chapkit::cli::exit_code(result) // on error: one line `error: ...`, exit code 1
}
```

## Bump the version with every published change

chap-core stores a model template once per version, together with the git revision the service reports
(`GIT_REVISION`, set by the Docker build). A version is write-once: when an image built from a newer commit
reports the same version, chap-core keeps the old template and refuses to run the model ("A version is
write-once, so the new revision needs a new version label"). Take the version from the crate, as the `init`
template does, and bump it in `Cargo.toml` whenever you publish a changed model.

## Use a database file in deployments

Without `DATABASE_URL` the database is in memory, the same default as Python chapkit. Every config, trained
model and prediction then stays in RAM until the process exits, and a restart loses them. That suits tests
and benchmarks, not a service chap-core uses: memory grows by about 0.1 MB per stored prediction (the
benchmark's load test reached 1.2 GB after 12,000 predictions). Set `DATABASE_URL` to a file
(`data/chapkit.db`, or `turso:///data/chapkit.db` with the `turso` feature) so results live on disk and
survive restarts.

## Registering with chap-core

Set `SERVICEKIT_ORCHESTRATOR_URL` (for example `http://localhost:8700/v2/services/$register`), and
`SERVICEKIT_HOST`/`SERVICEKIT_PORT` to the address chap-core should call back on. The service registers
when it starts, sends a keepalive every 10 seconds, registers again if chap-core forgets it, and deregisters
on shutdown. `GIT_REVISION` (a commit sha) is reported as `git_revision`; chap-core 2.4 and later stores a
model template only for services that report one.
