# Your first model package

You create a model package, run its tests, serve it, and check it with the same calls chap-core makes. Then you
change how it forecasts.

## Before you start

- Rust and the `chapkit-rs` command ([Install the tools](install.md)).

## 1. Create the package

```sh
chapkit-rs init dengue-recent
cd dengue-recent
```

```text
Created Dengue recent in dengue-recent

  cd dengue-recent
  cargo test                  # the model's tests
  cargo run --release -- serve  # serve on http://127.0.0.1:8000
  cargo run --release -- test   # in another terminal: chap-core's flow against it

The calculation to replace with your model is in src/model.rs.
```

The package forecasts with an example method, *recent level*: each location's last 12 observed values, turned
into 100 forecast samples. Open `src/model.rs`: the method is described at the top, and the calculation is two
short functions, `recent_values` and `samples`. `src/lib.rs` connects them to chapkit.

## 2. Run the tests

```sh
cargo test
```

The first run downloads and compiles chapkit-rs and its dependencies, which takes a few minutes. Then every
`test result` line ends in `ok`: the calculation's unit tests, and a test that trains and forecasts on
generated data.

## 3. Serve it

```sh
cargo run --release -- serve
```

The service listens on <http://127.0.0.1:8000>, with interactive API documentation at
<http://127.0.0.1:8000/docs>. Leave it running.

## 4. Check it the way chap-core uses it

In a second terminal:

```sh
chapkit-rs test
```

```text
Testing http://127.0.0.1:8000

Random seed 114429000 (repeat this run with --seed 114429000)

  ok    health                                            0.7 ms
  ok    service info                                      0.3 ms
  ok    configuration schema                              0.4 ms
  ok    configuration 1                                   0.4 ms
  ok    configuration 1: train 1                          2.1 ms
  ok    configuration 1: train 1: forecast 1              1.8 ms

  configurations   1
  trainings        1 ok, 0 failed
  forecasts        1 ok, 0 failed
  validations      2 ok
  time             0.01 s
```

Every step chap-core takes with a model service passed: the package is a working Chap model. The test made up
its own data, shaped by what the service says it needs, built a configuration from the service's schema, had
each request checked by `$validate` first, and checked the forecast that came back. Try `--configs 2
--trainings 2 --predictions 2` for more, or `-v` for details; `chapkit-rs test --start-service` builds and starts
the package itself. The package's own binary runs the same test, `cargo run --release -- test`, against the
service on `--host`/`--port`.

## 5. Look at what it stored

```sh
chapkit-rs test --keep
chapkit-rs artifact list --url http://127.0.0.1:8000
```

```text
ID                               TYPE                  SIZE  CONFIG                      CREATED
01M72G9819B9GPTPPFX70DRHYW       ml_training         5.0 KB  01M72G9818NRTXHYJ4R6QYD9C7  2026-11-10T10:32:57
  01M72G981ACQ2R3J023GFA74BH     ml_prediction            -  01M72G9818NRTXHYJ4R6QYD9C7  2026-11-10T10:32:57
```

The trained model, and the forecast made with it. `chapkit-rs artifact download <ID> --url ...` saves the
trained model as a safetensors checkpoint, or the forecast as JSON.

## 6. Make it yours

The forecast logic lives in `src/model.rs`. As an exercise, forecast from the same month in earlier years
instead of the last 12 months. In `src/lib.rs`, `train` gets each location's series with its periods; the season
of each value is `series.seasons()` (month 1 to 12). Pass the seasons to a new function in `model.rs` that groups
the values by season, and in `predict` look up the samples for each future row's season
(`period.period_of_year()`).

`src/lib.rs` also has an example of the model's own checks for `$validate`: it warns about locations with fewer
observations than the model looks back over. Try `chapkit-rs test --periods 6 -v` to see them
([Validation](../guide/model-package.md#validation)).

Keep the rule of the package layout: **the calculation goes in `model.rs`, as functions over plain numbers**,
described in words at the top of the file, so that anyone can read how a forecast is made without reading
chapkit code. The baselines in [chapkit-rs-models](https://github.com/winterop-com/chapkit-rs-models) are
complete examples.

After each change, `cargo test`, restart the service, and `chapkit-rs test` again.

## If something goes wrong

- **`no service at http://127.0.0.1:8000`** from the test: the service is not running, or on another port; pass
  `--url http://127.0.0.1:PORT` (or `--port PORT` to the package's own `test`).
- **A step fails**: the line says which and why, for example a forecast with the wrong number of rows. The
  service's terminal shows its log.

## Next steps

- [Ship it as a container](ship.md)
- [Testing with chap-core](../guide/chap-core.md): register it with a real chap-core and backtest it.
