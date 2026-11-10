# Command line: chapkit-rs

```sh
cargo install --git https://github.com/winterop-com/chapkit-rs chapkit-cli
chapkit-rs --help
```

From a checkout, `make release` builds it into `bin/chapkit-rs`; on macOS it is one binary for both Apple
Silicon and Intel. `make install` installs it into `~/.local/bin` (or
`PREFIX/bin`) with `crs` as its short form, as chap-devops installs itself with `cdx`; the docs use `chapkit-rs`
in full.

| command | |
|---|---|
| `chapkit-rs init NAME` | create a model package |
| `chapkit-rs test` | test a running service (Rust or Python chapkit) |
| `chapkit-rs artifact list` | list what a service stored, as a tree |
| `chapkit-rs artifact download ID` | save a trained model's checkpoint or a forecast |

Every model binary built with chapkit-rs has its own commands too: see [A model binary](#a-model-binary).

Every command has short help with `-h`, and the details with `--help`.

## init

```text
Usage: chapkit-rs init [OPTIONS] <NAME>

Arguments:
  <NAME>  Package and service id, e.g. my-model

Options:
      --dir <DIR>            Directory to create [default: NAME]
      --chapkit-path <PATH>  Use a local chapkit-rs checkout
      --title <TITLE>        Display name
```

Creates a model package from one template, a Rust model using chapkit-rs as a library:

| file | |
|---|---|
| `src/model.rs` | the calculation, as plain commented functions (an example to replace) |
| `src/lib.rs` | the chapkit `Model`: a configuration with one option, per-location training, the prediction table, and an example `$validate` check |
| `src/main.rs` | the service |
| `tests/model.rs` | trains and forecasts on generated data, and checks a checkpoint round trip |
| `Dockerfile`, `Dockerfile.dockerignore` | chapkit-rs's shared Dockerfile ([Docker](../guide/docker.md)) |
| `compose.yml` | the service in a container, with a volume, read-only filesystem and health check |
| `.github/workflows/publish.yml` | builds the image for amd64 and arm64 on GitHub and publishes it to the GitHub container registry, with build provenance |
| `README.md`, `.gitignore` | |

By default the package depends on chapkit-rs from GitHub. With `--chapkit-path` it uses a local checkout; then
there is no publishing workflow (GitHub cannot build against your checkout), and when the checkout sits next to
the package, `compose.yml` builds from the parent directory so Docker sees both. `init` refuses to write into a
directory that is not empty.

## test

```text
Usage: chapkit-rs test [OPTIONS]

Options:
  -u, --url <URL>                  Service URL [default: http://127.0.0.1:8000]
      --configs <CONFIGS>          Configurations to create [default: 1]
      --trainings <TRAININGS>      Trainings per configuration [default: 1]
      --predictions <PREDICTIONS>  Forecasts per trained model [default: 1]
      --locations <LOCATIONS>      Locations in the test data [default: 5]
      --periods <PERIODS>          Training periods [default: 50]
      --history <HISTORY>          History periods sent with a forecast [default: 30]
      --period-type <PERIOD_TYPE>  Period format [default: the service's] [possible values: monthly, weekly]
      --seed <SEED>                Seed for the test data [default: random]
      --timeout <TIMEOUT>          Seconds to wait per job [default: 300]
      --save-data <DIR>            Save the generated data in DIR
  -v, --verbose                    Show service details and diagnostics
      --keep                       Keep what the test creates
      --start-service              Start the package here (cargo run) first
```

Runs chap-core's flow against a service, the same way Python chapkit's `chapkit test` does, and works against
either:

1. **Discover**: health, service info (required covariates, geo, free covariates, period type, horizon bounds)
   and the configuration schema.
2. **Configure**: each configuration is built from the schema: every field's default, numbers shifted per
   configuration (the second gets `prediction_periods` 4), else a value of the field's type. The horizon is the
   configuration's `prediction_periods`, kept within the service's bounds.
3. **Generate data** here, shaped by what the service declares: its required covariates, the configuration's
   `additional_continuous_covariates`, two extra covariates when free ones are allowed, a GeoJSON when geo is
   required, monthly or weekly periods.
4. **For each training and forecast**: `$validate` first (skipped on services without it), then the job, then
   checks: the trained model is stored; the forecast has the keys chap-core reads, `time_period` and `location`
   first, sample columns, and one row per future period.

It prints one line per step with its time, keeps going after a failed job, ends with a summary, and exits
non-zero if anything failed. The seed is random unless given and printed, so a failing run can be repeated with
`--seed`. `--save-data` writes the configurations and data it sent. `--start-service` runs `cargo run --release -- serve`
for the package in the current directory, on the port of `--url`, and stops it afterwards. Unless `--keep` is
given, the configurations it created are deleted, and with them the trained models and forecasts.

## artifact

```text
Usage: chapkit-rs artifact list [OPTIONS] <--url <URL>|--database <DATABASE>>
  -u, --url <URL>            Service URL
  -d, --database <DATABASE>  SQLite database file
  -t, --type <TYPE>          Only this type, e.g. ml_prediction

Usage: chapkit-rs artifact download [OPTIONS] <--url <URL>|--database <DATABASE>> <ID>
  -u, --url <URL>            Service URL
  -d, --database <DATABASE>  SQLite database file
  -o, --output <OUTPUT>      Output file [default: artifact_ID.safetensors or .json]
      --force                Overwrite the output file
```

Both read from a running service (`--url`) or straight from its SQLite database file (`--database`), for
example a stopped container's volume. `list` shows each trained model with its forecasts under it:

```text
ID                               TYPE                  SIZE  CONFIG                      CREATED
01M72G9819B9GPTPPFX70DRHYW       ml_training         5.0 KB  01M72G9818NRTXHYJ4R6QYD9C7  2026-11-10T10:32:57
  01M72G981ACQ2R3J023GFA74BH     ml_prediction            -  01M72G9818NRTXHYJ4R6QYD9C7  2026-11-10T10:32:57
```

A forecast has no recorded size (as in Python chapkit), so a service shows `-`; from a database file the size
is measured.

`download` saves a trained model as its safetensors checkpoint (weights, state and provenance) and a forecast as
JSON records, one object per row.

## A model binary

A binary whose `main` is `chapkit::Service::serve()` (every package `init` creates, and the baselines) parses
its own command line. Without arguments it prints its help:

```text
Usage: my-model <COMMAND>

Commands:
  serve        Serve the model
  test         Test the running service
  eval         Backtest the model in-process
  hpo          Search the model's hyperparameters
  healthcheck  Check the running service's health
  help         Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

```text
Usage: my-model serve [OPTIONS]
      --host <HOST>          Service host [env: HOST] [default: 127.0.0.1]
      --port <PORT>          Service port [env: PORT] [default: 8000]
      --database <URL>       Database URL [env: DATABASE_URL]
      --max-concurrency <N>  Jobs at once [env: CHAPKIT_MAX_CONCURRENCY]
```

`test` is `chapkit-rs test` built into the binary: the same options, except that it finds the service on
`--host`/`--port` (the same defaults and variables as `serve`), so `my-model serve` in one terminal and
`my-model test` in another need no further arguments; `--url` overrides. When nothing answers there it says so
and exits non-zero. `healthcheck` checks the service on `--host`/`--port` and exits non-zero unless it is
healthy; the Docker image's `HEALTHCHECK` runs it. `--version` prints the service's id and version.

### eval

```text
Usage: my-model eval [OPTIONS]
      --data <FILE>            Dataset, CSV or JSON [default: generated]
      --periods <PERIODS>      Periods per forecast [default: 3]
      --splits <SPLITS>        Splits [default: 7]
      --stride <STRIDE>        Periods between splits [default: 1]
      --retrain <RETRAIN>      Trainings [default: 1]
      --weather <WEATHER>      Future weather (observed looks ahead) [default: climatology] [possible values: climatology, observed, damped-persistence]
      --keep-gaps              Keep gaps in a location's periods
      --no-pad                 Do not pad locations to the full range
      --keep-untrained         Keep locations with no target before the test windows
      --locations <LOCATIONS>  Generated locations [default: 5]
      --history <HISTORY>      Generated periods [default: 60]
      --seed <SEED>            Seed (generated data, search) [default: random]
      --config <JSON>          Model options, JSON or @file
      --by-location            Scores by location
      --json                   Print the result as JSON
```

A backtest of the model in-process, as chap-core runs one, with chap-core's metrics: no service, no
database. `--periods`, `--splits`, `--stride` and `--retrain` are chap-core's `n_periods`, `n_splits`,
`stride` and `n_retrain`, with its defaults. Without `--data` the dataset is generated as `test` generates
it (`--locations` x `--history` periods, the model's covariates and period type); the seed is printed. The
dataset is prepared as chap-core prepares it: sorted, a duplicate period in a location an error, gaps filled
and every location padded to the full period range; `--keep-gaps` and `--no-pad` turn filling and padding
off, and the first line of the output then says so. Locations with no target before the test windows are
left out, as chap-core does, unless `--keep-untrained` is given; the `data:` line counts them. `--config` sets
model options (JSON or `@file`, flat or as `user_option_values`). It prints the aggregate metrics and the same
by horizon, and with `--by-location` a table by location; `--json` prints everything at full precision. See
[Backtests in-process](../guide/eval.md).

### hpo

```text
Usage: my-model hpo [OPTIONS]
      (the data and backtest options of eval, then:)
      --trials <TRIALS>        Trials [default: 20]
      --method <METHOD>        Searcher [default: random] [possible values: random, grid]
      --grid-ranges            Let grid search step through ranges
      --metric <METRIC>        Metric to minimize [default: crps] [possible values: crps, crps_log1p, crps_norm, mae, rmse, winkler_score_10_90, winkler_score_25_75, winkler_score_10_90_log1p, winkler_score_25_75_log1p]
      --json                   Print the result as JSON
```

A random or grid search over the model's search space (`hpo_search_space`), each trial a backtest with the
eval options, every candidate merged into `--config`. It prints the leaderboard (the best ten), the best
options and their score; `--json` prints the whole search and the full backtest of the best config. See
[Hyperparameter search](../guide/hpo.md).
