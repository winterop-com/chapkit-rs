# Benchmarks

Measured 2026-11-09 on an Apple Silicon laptop. All numbers come from scripts in `chapkit-rs-models`:
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

## Scaling with locations and periods

`make bench-scale` in `chapkit-rs-models` runs chap-core's three built-in baselines and the Rust ports
in-process on identical generated monthly data (100 to 5000 locations, 120 to 5000 periods), one process per
size. Times are milliseconds for train plus predict; "convert" is chap-core turning pandas into its `DataSet`
before calling a model, "decode" is chapkit-rs parsing the same data from JSON as it arrives over HTTP. CPU
time is measured inside each process; cores used is CPU time over wall time. Peak memory is the whole process.

**Same output.** `make bench-compare` runs both on the same data and compares all 100 samples of every forecast
row: global_median and seasonal_median are bit-for-bit identical at every size (5.8 million samples in all),
seasonal_shift_persistence differs by at most 3.5e-16 relative (the last bit of a float).

**chap-core uses one core**: its CPU time equals its wall time in every run.

global_median:

| locations x periods | chap-core convert | chap-core train + predict | chap-core peak MB | Rust 1 core | Rust all cores (cores used) | Rust peak MB | Rust decode |
|---|---|---|---|---|---|---|---|
| 500 x 120 | 229 | 22.5 | 149 | 3.0 | 3.5 (4.5) | 25 | 10 |
| 5000 x 120 | 2,296 | 223 | 315 | 37.5 | 35.5 (2.7) | 159 | 105 |
| 100 x 5000 | 398 | 14.3 | 258 | 6.1 | 4.6 (3.1) | 92 | 90 |
| 2000 x 2000 | 2,112 | 141 | 1,206 | 65.8 | 49.8 (2.1) | 703 | 693 |

seasonal_median:

| locations x periods | chap-core convert | chap-core train + predict | chap-core peak MB | Rust 1 core | Rust all cores (cores used) | Rust peak MB | Rust decode |
|---|---|---|---|---|---|---|---|
| 500 x 120 | 222 | 152 | 151 | 7.2 | 4.8 (5.2) | 35 | 10 |
| 5000 x 120 | 2,265 | 1,530 | 380 | 84.0 | 55.4 (4.6) | 226 | 105 |
| 100 x 5000 | 401 | 97.6 | 259 | 20.6 | 9.8 (5.0) | 103 | 89 |
| 2000 x 2000 | 2,120 | 1,185 | 1,232 | 220 | 94.5 (3.7) | 799 | 691 |

seasonal_shift_persistence:

| locations x periods | chap-core convert | chap-core train + predict | chap-core peak MB | Rust 1 core | Rust all cores (cores used) | Rust peak MB | Rust decode |
|---|---|---|---|---|---|---|---|
| 500 x 120 | 224 | 165 | 178 | 14.8 | 7.2 (6.2) | 61 | 10 |
| 5000 x 120 | 2,281 | 1,784 | 586 | 180 | 103 (4.5) | 399 | 105 |
| 100 x 5000 | 443 | 223 | 370 | 53.4 | 27.7 (4.2) | 227 | 92 |
| 2000 x 2000 | 2,156 | 1,736 | 2,122 | 557 | 265 (3.7) | 1,974 | 723 |

What this shows:

- **Faster at every size.** On one core against chap-core's one core, the Rust models are 2 to 24 times faster;
  on all cores 3 to 31 times. Counting the data step too (chap-core's pandas conversion against chapkit-rs's
  JSON decode), a train and predict is 3 to 25 times faster.
- **Many locations favour Rust most** (seasonal_median at 5000 locations: 1,530 ms against 55 ms). Long histories
  with few locations favour it least, because numpy already handles one long series well; there JSON decoding
  is most of chapkit-rs's time.
- **chap-core's pandas-to-`DataSet` conversion costs more than its models**: 2.3 s at 5000 locations, against
  0.1 s for chapkit-rs to decode the same data from JSON.
- **Less memory at every size**, from about a tenth at small sizes to slightly less at 4 million rows.
- **All cores trade CPU for latency.** Splitting locations over cores shortens a call but costs more CPU in
  total than one core (seasonal_shift_persistence at 5000 x 120: 103 ms wall but about 460 ms CPU, against
  180 ms on one core). For a service handling many calls at once, `RAYON_NUM_THREADS=1` gives the most work
  per core.

The models were profiled before this run: period parsing, grouping by location and per-ratio map lookups were
the hot spots, and fixing them made the single-core times about twice as fast. polars was measured as an
alternative for the per-location work and was 3 to 4 times slower than the plain Rust here (its conversion
alone costs as much as a whole training); it remains an optional tool for writing complex models.
`bench/scale_results.json` holds every run.

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
