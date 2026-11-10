# Benchmarks

Measured 2026-11-09 on an Apple Silicon laptop. All numbers come from scripts in `chapkit-rs-examples`:
`bench/run.py` (HTTP, real processes on localhost), `bench/builtin_inprocess.py` (chap-core's built-in
baseline in Python) and `make bench` (criterion, the Rust model in-process).

The model is the same everywhere: chap-core's `global_median` baseline. Python chapkit serves a numpy
version of it (`bench/python_global_median`), chapkit-rs serves the Rust port, chap-core runs its
built-in. The data is the same generated dataset: 20 locations x 120 months of training data (2400 rows,
10 columns), 3 future months.

## Through chap-core's job flow

One train plus predict, waiting for each job exactly as chap-core does (check, then sleep 2 s):

| | Python chapkit | chapkit-rs | chapkit-rs, eager wait off |
|---|---|---|---|
| train + predict, chap-core polling | 4.08 s | **0.013 s** | 4.03 s |
| train + predict, 5 ms polling (the work itself) | 74 ms | **6.9 ms** | 24 ms |

Two separate effects:

- **The polling trick** ([Finishing fast jobs](../design/eager-completion.md)) is what removes the
  4 seconds. With the eager wait turned off, chapkit-rs pays the same two 2-second sleeps as Python
  chapkit. This part is not about Rust: Python chapkit could adopt it.
- **The work itself** is about 10 times faster over HTTP (74 ms against 6.9 ms), and much more in CPU, below.

## Cost of a prediction

8 concurrent clients running predict jobs for 10 seconds (jobs run one at a time in both, chapkit's default):

| | Python chapkit | chapkit-rs |
|---|---|---|
| predictions per second | 6.4 | **1225** |
| CPU per prediction | 154 ms | **2.3 ms** |
| cores used | 0.99 | 2.77 |

Python chapkit is CPU bound on one core; each prediction costs 154 ms of CPU, mostly the workspace copy,
CSV and pickle handling around a model that itself takes well under a millisecond. chapkit-rs spends
2.3 ms per prediction, about 68 times less, most of it parsing the 2400-row history from JSON.

## The fast path

`$predict-sync` (chapkit-rs only): the forecast returned directly from a warm model, no job, no artifact.

| | single client | 8 concurrent (ApacheBench) |
|---|---|---|
| latency p50 | 2.7 ms | 2.0 ms |
| latency p99 | 4.5 ms | 3.0 ms |
| throughput | | **3808 requests/s** on 7 cores |
| CPU per request | | 1.9 ms |

This is what an external explainer or search loop sees per model call: about 2 ms, against 2.5 to 3
seconds through Python chapkit and chap-core.

## The model alone, in-process

| | chap-core built-in (Python, numpy) | chapkit-rs (Rust) |
|---|---|---|
| global_median train | 658 µs | **102 µs** |
| global_median predict | 276 µs | **28.5 µs** |
| seasonal_median train / predict | | 440 µs / 38 µs |
| seasonal_shift_persistence train / predict | | 731 µs / 335 µs |

Even against chap-core's vectorised numpy with no HTTP at all, the Rust model is 6 to 10 times faster.
This is the cost explainers and search pay per evaluation when they run inside the service
(`$explain`).

## Footprint

| | Python chapkit | chapkit-rs |
|---|---|---|
| startup until `/health` answers | 908 ms | **19 ms** |
| memory at idle | 150 MB | **16 MB** |
| memory after 100 stored predictions | 212 MB | **35 MB** |
| memory per stored prediction (in-memory database) | about 0.6 MB | about 0.1 MB |
| install size | 522 MB (Python environment) | **11.9 MB** (one static binary) |

Peak memory during the load test is not comparable: chapkit-rs stored 12,000 predictions in its in-memory
database in 10 seconds, Python chapkit 64. With a file database (`DATABASE_URL`) stored artifacts live on
disk instead.

## Caveats

- One machine, one run per number; the medians are stable to within about 10 percent between runs.
- The HTTP benchmark client is Python (standard library) with pre-encoded request bodies; the `$predict-sync`
  load test uses ApacheBench so the client is not the bottleneck.
- The criterion and chap-core in-process numbers use datasets of the same size generated differently.
- Container image sizes were not measured: no chapkit-rs image exists yet. The Python environment size is the
  chapkit checkout's `.venv`; a published chapkit image also carries a Python base image.
