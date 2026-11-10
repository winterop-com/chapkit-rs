# chap-core end to end

Rust services run on the host, registered with chap-core via SERVICEKIT_ORCHESTRATOR_URL.
`cdx models test ... --backtest` (dataset from the service's $generate-sample-data, 3 periods, 2 splits):

| model | result | time | crps | mae | rmse |
|---|---|---|---|---|---|
| global-median | pass | 8 s | 16.8 | 27.4 | 32.2 |
| seasonal-median | pass | 8 s | 2.5 | 3.1 | 4.8 |
| seasonal-shift-persistence | pass | 8 s | 4.2 | 4.7 | 6.1 |

Same datasets (seed 7) backtested with chap-core's built-in Python baselines (template v1 in the 2.6.0 image):

| model | metric | chap-core builtin | chapkit-rs |
|---|---|---|---|
| global_median | crps | 16.333646 | 16.333646 |
| global_median | mae | 25.633333 | 25.633333 |
| global_median | rmse | 28.211995 | 28.211995 |
| global_median | coverage_10_90 | 0.700000 | 0.700000 |
| global_median | winkler_10_90 | 75.309798 | 75.309798 |
| seasonal_median | crps | 2.996369 | 2.996369 |
| seasonal_median | mae | 3.516667 | 3.516667 |
| seasonal_median | rmse | 4.221966 | 4.221966 |
| seasonal_median | coverage_10_90 | 0.333333 | 0.333333 |
| seasonal_median | winkler_10_90 | 25.303771 | 25.303771 |

seasonal_shift_persistence is not a built-in in the chap-core 2.6.0 image; it is checked against the
chap-core checkout by the parity tests instead.

Fixes found by this run: chap-core's service proxy forwards `$` as `%24` (now decoded before routing);
cdx gates on `chapkit_version` (now `2.6.0+chapkit-rs.0.1.0`) and needs `$generate-sample-data`.
