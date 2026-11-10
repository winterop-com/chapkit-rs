# Command line: chapkit-rs

```sh
cargo install --git https://github.com/winterop-com/chapkit-rs chapkit-cli
chapkit-rs --help
```

From a checkout, `make release` builds it into `bin/chapkit-rs` for this machine, and `make release-universal`
builds one macOS binary for both Apple Silicon and Intel. `make install` installs it into `~/.local/bin` (or
`PREFIX/bin`) with `crs` as its short form, as chap-devops installs itself with `cdx`; the docs use `chapkit-rs`
in full.

| command | |
|---|---|
| `chapkit-rs init NAME` | create a model package |
| `chapkit-rs test` | test a running service (Rust or Python chapkit) |
| `chapkit-rs artifact list` | list what a service stored, as a tree |
| `chapkit-rs artifact download ID` | save a trained model's checkpoint or a forecast |

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
      --start-service              Start the package here (cargo run) first
  -v, --verbose                    Show service details and diagnostics
      --keep                       Keep what the test creates
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
`--seed`. `--save-data` writes the configurations and data it sent. `--start-service` runs `cargo run --release`
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
