# Testing with chap-core

chap-devops (`cdx`) runs chap-core in Docker; a chapkit-rs model runs on the host and registers with it,
the "model on host" setup.

```sh
cdx init mychap --models none
cd mychap
cdx up --wait

# in the model package, one service per port
REV=$(git rev-parse HEAD)
SERVICEKIT_ORCHESTRATOR_URL='http://localhost:8700/v2/services/$register' \
SERVICEKIT_HOST=host.docker.internal SERVICEKIT_PORT=8001 \
HOST=0.0.0.0 PORT=8001 GIT_REVISION=$REV bin/baseline-global-median serve

cdx status                                   # the service is listed as unmanaged, with a recent ping
cdx models configs sync global-median        # chap-core configured model with the defaults
cdx models test global-median --backtest     # dataset, backtest and scores through chap-core
```

- `HOST=0.0.0.0` makes the service reachable from the chap-core container; `host.docker.internal` is the
  host as seen from inside it.
- `cdx models test --backtest` builds its dataset from the service's own `$generate-sample-data`, through
  chap-core's service proxy.
- Add `--seed N` to make the dataset repeatable and `--keep` to keep the dataset and backtests, for
  example to backtest chap-core's built-in baselines on the same data.

The results of running all three baselines this way are in [chap-core end to end](../results/end-to-end.md).
