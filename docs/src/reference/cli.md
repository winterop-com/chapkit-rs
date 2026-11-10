# Command line: chapkit-rs

```sh
cargo install --git https://github.com/winterop-com/chapkit-rs chapkit-cli
chapkit-rs --help
```

## chapkit-rs init

```text
Usage: chapkit-rs init [OPTIONS] <NAME>

Arguments:
  <NAME>  Package name, also the service id chap-core sees: lowercase words joined by hyphens

Options:
      --dir <DIR>            Directory to create (default: the name)
      --chapkit-path <PATH>  Depend on a local chapkit-rs checkout instead of the git repository
      --title <TITLE>        Display name (default: from the package name)
```

Creates a model package with one template, a Rust model using chapkit-rs as a library:

| file | |
|---|---|
| `src/model.rs` | the calculation, as plain commented functions (an example to replace) |
| `src/lib.rs` | the chapkit `Model`: a configuration with one option, per-location training, the prediction table |
| `src/main.rs` | the service |
| `tests/model.rs` | trains and forecasts on generated data, and checks a checkpoint round trip |
| `Dockerfile`, `Dockerfile.dockerignore` | chapkit-rs's shared Dockerfile ([Docker](../guide/docker.md)) |
| `compose.yml` | the service in a container, with a volume, read-only filesystem and health check |
| `README.md`, `.gitignore` | |

By default the package depends on chapkit-rs from GitHub. With `--chapkit-path`, it uses a local checkout instead;
when that checkout sits next to the package, `compose.yml` builds from the parent directory so Docker sees both.
`init` refuses to write into a directory that is not empty.

## chapkit-rs test

```text
Usage: chapkit-rs test [OPTIONS]

Options:
      --url <URL>              Base URL of the service [default: http://127.0.0.1:8000]
      --locations <LOCATIONS>  Locations in the generated data [default: 5]
      --periods <PERIODS>      Periods of training data [default: 36]
      --timeout <TIMEOUT>      Seconds to wait for each job [default: 300]
      --keep                   Keep the configuration and artifacts the test creates
```

Runs chap-core's flow against a running service and checks every response has the shape chap-core relies on:
health, service info, the configuration schema, creating a configuration, sample data, a training job, a
forecasting job, and the stored forecast (its keys, `time_period` and `location` first, sample columns, one row
per future period). It prints one line per step with its time and exits non-zero at the first failure. It uses
only the chapkit HTTP API, so it tests Python chapkit services too. Unless `--keep` is given it deletes the
configuration it created, and with it the trained model and forecast.
