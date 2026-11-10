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
  cargo run --release         # serve on http://127.0.0.1:8000
  chapkit-rs test             # in another terminal: chap-core's flow against it

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
cargo run --release
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

  ok    health                                            1.5 ms
  ok    service info                                      0.3 ms
  ok    configuration schema                              0.5 ms
  ok    create a configuration                            0.5 ms
  ok    generate training data                            0.7 ms
  ok    generate prediction data                          0.8 ms
  ok    train (job)                                       2.4 ms
  ok    predict (job)                                     3.6 ms
  ok    forecast has the shape chap-core expects          2.9 ms

dengue-recent (chapkit 2.6.0+chapkit-rs.0.1.0): 5 locations, 36 periods.

All steps passed in 0.01 s.
```

Every step chap-core takes with a model service passed: the package is a working Chap model.

## 5. Make it yours

The forecast logic lives in `src/model.rs`. As an exercise, forecast from the same month in earlier years
instead of the last 12 months. In `src/lib.rs`, `train` gets each location's series with its periods; the season
of each value is `series.seasons()` (month 1 to 12). Pass the seasons to a new function in `model.rs` that groups
the values by season, and in `predict` look up the samples for each future row's season
(`period.period_of_year()`).

Keep the rule of the package layout: **the calculation goes in `model.rs`, as functions over plain numbers**,
described in words at the top of the file, so that anyone can read how a forecast is made without reading
chapkit code. The baselines in [chapkit-rs-models](https://github.com/winterop-com/chapkit-rs-models) are
complete examples.

After each change, `cargo test`, restart the service, and `chapkit-rs test` again.

## If something goes wrong

- **`Connection refused`** from `chapkit-rs test`: the service is not running, or on another port; pass
  `--url http://127.0.0.1:PORT`.
- **A step fails**: the line says which and why, for example a forecast with the wrong number of rows. The
  service's terminal shows its log.

## Next steps

- [Ship it as a container](ship.md)
- [Testing with chap-core](../guide/chap-core.md): register it with a real chap-core and backtest it.
