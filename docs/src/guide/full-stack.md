# Full stack: chap-core, DHIS2, Modeling App

You run the Rust models the way a Chap deployment runs them: chap-core (latest `master`), DHIS2 with demo data, and
the Modeling App inside DHIS2, with the models registered as services. Then you evaluate and forecast with them
through the Modeling App, next to a Python chapkit model and chap-core's own built-in baselines.

This page is also the runbook for testing chapkit-rs against the full stack: each step says what it printed when
it was last run (2026-11-10, chap-core `2.7.0.dev0` from `master`, DHIS2 2.42.7.1, Modeling App 7.4.0).
[Findings](#findings) lists what the run turned up.

`cdx` below is short for `chap-devops`; make it with `alias cdx=chap-devops` (or a symlink named `cdx` on your
`PATH`), or type `chap-devops` instead.

## Before you start

- Docker with at least 8 GB of memory for containers (`docker info` shows it as `MemTotal`).
- [chap-devops](https://github.com/dhis2-chap/chap-devops), which sets up the whole stack:

```sh
curl -fsSL https://raw.githubusercontent.com/dhis2-chap/chap-devops/main/install.sh | sh
```

- The chapkit-rs model images: the published baselines (`ghcr.io/winterop-com/chapkit-rs-models/...`), or images
  built locally with `make docker-build` in chapkit-rs-models.

## 1. Create the deployment

In a folder outside the repositories:

```sh
cdx init chap-stack --models chapkit_simple_multistep_model --with dhis2 --chap-tag master
cd chap-stack
cdx doctor
```

- `--chap-tag master` runs chap-core's latest `master` build.
- `--with dhis2` adds a DHIS2 (2.42) seeded with the Laos demo database: weekly dengue cases 2019 to 2024 for 18
  provinces, with rainfall, temperature and population.
- `--models chapkit_simple_multistep_model` adds a Python chapkit model from the marketplace, as a reference.

```text
created a deployment in .../chap-stack
components: chap-core, dhis2
chap-core: master, API on http://localhost:8700
DHIS2: http://localhost:8780
enabled chapkit_simple_multistep_model v0.1.5 at http://localhost:8700/v2/services/chapkit-simple-multistep-model/run/
```

`cdx doctor` ended with `26 checks: 22 ok, 2 warn, 0 fail, 2 skipped`. One of the warnings is that the
marketplace pins an older build of the Python model than its newest; harmless here.

## 2. Start it

```sh
cdx up
cdx status
```

The first start pulls the images: about 6 GB for Chap and 1 to 1.5 GB for DHIS2. DHIS2 then restores the demo
database, which takes a few minutes; `cdx status` says `starting` for DHIS2 until it answers.

```text
chap-core   up         http://localhost:8700   2.7.0.dev0   master: running 69049dc8ffdd, revision 4ad3a6fb4260   auth: off
dhis2       starting   http://localhost:8780

MODEL                           STATE                       REACH          LAST PING
chapkit-simple-multistep-model  registered, not configured  via chap-core  3s ago
```

The Python model's image is amd64-only, so on an Apple Silicon Mac it runs under emulation: fine for testing, not
for timing.

## 3. Add the chapkit-rs models

Each model image is added with the service id the binary registers under (`global-median`, ...):

```sh
for m in global-median seasonal-median seasonal-shift-persistence; do
  cdx models add ghcr.io/winterop-com/chapkit-rs-models/baseline-$m:latest --service-id $m
done
cdx up --wait
```

`cdx models add` names each model after its image (`baseline_global_median`, ...). `cdx up --wait` returns when
every model has registered with chap-core, and creates a configured model for each, which chap-core needs before
anything can run a model:

```text
  chap-core                       up          http://localhost:8700
  global-median                   registered  http://localhost:8700/v2/services/global-median/run/
  seasonal-median                 registered  http://localhost:8700/v2/services/seasonal-median/run/
  seasonal-shift-persistence      registered  http://localhost:8700/v2/services/seasonal-shift-persistence/run/
  chapkit-simple-multistep-model  registered  http://localhost:8700/v2/services/chapkit-simple-multistep-model/run/
  dhis2                           up          http://localhost:8780
created configured models in chap-core for baseline_global_median (default), baseline_seasonal_median (default),
baseline_seasonal_shift_persistence (default), chapkit_simple_multistep_model (monthly_climate, monthly_selfhistory)
```

To use images built locally instead, `make docker-build` in chapkit-rs-models and add
`chapkit-rs-models/baseline-global-median:latest` and so on.

## 4. Connect DHIS2 to Chap

```sh
cdx dhis2 connect
```

It points DHIS2's `chap` route at this chap-core, installs the Modeling App and the Climate App, and generates the
analytics tables the Modeling App reads:

```text
DHIS2 2.42.7.1 at http://localhost:8780, as `admin`
repointed the `chap` route at http://chap:8000/**; chap-core answered through it
installed Modeling App 7.4.0
installed DHIS2 Climate App 1.16.2
analytics finished in 41 seconds (job Y3PUxvkDzDs)
the Modeling App can reach Chap; open DHIS2 with `chap-devops open dhis2`
```

## 5. What chap-core knows about the models

chap-core stores a model template for each registered service, from its `/api/v1/info`; the Modeling App shows it.
To see them:

```sh
cdx api GET /v1/crud/model-templates
```

| template | display name | author | status | period type |
|---|---|---|---|---|
| `global-median` (chapkit-rs) | Global median | Morten Hansen | green | any |
| `global_median` (chap-core built-in) | Global median | CHAP team | red | any |
| `seasonal-median` (chapkit-rs) | Seasonal median | Morten Hansen | green | any |
| `seasonal_median` (built-in) | Seasonal median | CHAP team | red | any |
| `seasonal-shift-persistence` (chapkit-rs) | Seasonal shift persistence | Morten Hansen | green | any |
| `seasonal_shift_persistence` (built-in) | Seasonal shift persistence | CHAP team | red | any |
| `chapkit-simple-multistep-model` (Python chapkit) | Simple Multistep Model (chapkit) | CHAP team | orange | month |

chap-core `master` ships all three baselines built in (2.6.0 had only the two medians), so each port can be
compared with its original through chap-core.

## Findings

| # | finding | where | status |
|---|---|---|---|
| 1 | The ports register with the same display names as chap-core's built-in baselines ("Global median", ...), so the Modeling App's model list shows two entries with one name, told apart only by the author. | chapkit-rs-models | open |
| 2 | The ports declare `author_assessed_status: green`; chap-core rates its identical built-ins `red` (baselines are yardsticks, not forecasting models). | chapkit-rs-models | open |
| 3 | `organization` and `documentation_url` are empty; the models book is the obvious documentation URL. The library supports both. | chapkit-rs-models | open |
| 4 | The Python reference model ships an amd64-only image; the chapkit-rs images are built for amd64 and arm64, so they run natively on Apple Silicon. | marketplace | noted |
