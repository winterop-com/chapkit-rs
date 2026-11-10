# Full stack: chap-core, DHIS2, Modeling App

You run the chapkit-rs models (the baseline ports from chapkit-rs-models) the way a Chap deployment runs them: chap-core (latest `master`), DHIS2 with demo data, and
the Modeling App inside DHIS2, with the models registered as services. Then you evaluate and forecast with them
through the Modeling App, next to a Python chapkit model and chap-core's own built-in baselines.

This page is also the runbook for testing chapkit-rs against the full stack: each step says what it printed when
it was last run (2026-11-10, chap-core 2.7.0 from `master`, DHIS2 2.42.7.1, Modeling App 7.4.0).
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

Each model image is added with the service id the binary registers under (`global-median`, ...). Persistence
registers as `persistence-crs`, since `persistence` is the name of chap-core's own built-in. The images are not in
the marketplace, so `cdx models add` asks for confirmation; `--yes` gives it:

```sh
for m in global-median seasonal-median seasonal-shift-persistence; do
  cdx models add ghcr.io/winterop-com/chapkit-rs-models/baseline-$m:latest --service-id $m --yes
done
cdx models add ghcr.io/winterop-com/chapkit-rs-models/baseline-persistence:latest --service-id persistence-crs --yes
cdx up --wait
```

The Python reference model runs emulated on Apple Silicon and costs the most memory of the model services;
`cdx models disable chapkit_simple_multistep_model` switches it off when it is not needed (its volume is kept).

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

| template | version | display name | author | status | period type |
|---|---|---|---|---|---|
| `global-median` (chapkit-rs) | 1.0.2 | Global median (crs) | Morten Hansen | red | any |
| `global_median` (chap-core built-in) | 2 | Global median | CHAP team | red | any |
| `seasonal-median` (chapkit-rs) | 1.0.2 | Seasonal median (crs) | Morten Hansen | red | any |
| `seasonal_median` (built-in) | 2 | Seasonal median | CHAP team | red | any |
| `seasonal-shift-persistence` (chapkit-rs) | 1.0.2 | Seasonal shift persistence (crs) | Morten Hansen | red | any |
| `seasonal_shift_persistence` (built-in) | 1 | Seasonal shift persistence | CHAP team | red | any |
| `persistence-crs` (chapkit-rs) | 1.0.2 | Persistence (crs) | Morten Hansen | red | any |
| `persistence` (built-in) | 2 | Persistence | CHAP team | red | any |
| `chapkit-simple-multistep-model` (Python chapkit) | 0.1.5 | Simple Multistep Model (chapkit) | CHAP team | orange | month |

chap-core 2.7.0 ships all four baselines built in (2.6.0 had only the two medians), so each port can be compared
with its original through chap-core. (The first run of this page registered the ports with the built-ins' names
and status green; findings 1 and 2.)

## 6. Swap in a new build of a model

After changing a model, bump its version (the workspace `version` in chapkit-rs-models' `Cargo.toml`; chap-core
refuses a new build under an old version, finding 6), publish its images (push to `main`; the image workflow builds
them, once per version) and replace the running ones:

```sh
for m in global-median seasonal-median seasonal-shift-persistence persistence; do
  docker pull --platform linux/amd64 ghcr.io/winterop-com/chapkit-rs-models/baseline-$m:latest
done
cdx up --wait
```

`cdx up` recreates the services from the pulled images; with a new version each registers as a new model template
and gets a configured model. (`cdx models remove` retires a model in chap-core and archives its configured models;
it is only needed to drop a model.) A locally built image only
works if it was built for linux/amd64 (finding 5).

## 7. Open the Modeling App

`cdx open dhis2`, log in as `admin` / `district`, and pick **Modeling** in the app menu, or go straight to
<http://localhost:8780/apps/dhis2-chapmodeling-app>. The Models page took about 10 seconds to load its 19
configured models.

## 8. Give a district its missing geometry

The demo database's district "0710 Xon" (`X8kOf41v0rx`) has no geometry. The Modeling App needs a geometry for every
location, so the district datasets below cannot be built until it has one. Give it a small square at the centre of its
province (Houaphan, `hdeC7uX9Cko`); it is fake, but any polygon will do for a test:

```sh
curl -s -u admin:district 'http://localhost:8780/api/organisationUnits/hdeC7uX9Cko.json?fields=geometry' > houaphan.json
python3 - > xon-patch.json <<'EOF'
import json
g = json.load(open("houaphan.json"))["geometry"]
def pts(c):
    if isinstance(c[0], (int, float)): yield c
    else:
        for x in c: yield from pts(x)
xs, ys = zip(*pts(g["coordinates"]))
cx, cy, d = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.025
square = [[cx-d, cy-d], [cx+d, cy-d], [cx+d, cy+d], [cx-d, cy+d], [cx-d, cy-d]]
print(json.dumps([{"op": "add", "path": "/geometry", "value": {"type": "Polygon", "coordinates": [square]}}]))
EOF
curl -s -u admin:district -X PATCH -H 'Content-Type: application/json-patch+json' \
  --data @xon-patch.json http://localhost:8780/api/organisationUnits/X8kOf41v0rx
```

## 9. Create datasets

The Modeling App's **Datasets** page imports data from DHIS2 once and reuses it across evaluations: **New dataset**,
pick the data element, the org units, the period type and range, and the target column. Two were used here, both from
the dengue cases data element (`SK9a8nJJTAI`):

| id | org units | periods | locations | note |
|---|---|---|---|---|
| 1 | provinces (level 2) | monthly, 2019-01 to 2024-12 | 18 | small; enough for a first check |
| 2 | districts (level 3) | weekly, 2019W02 to 2024W52 | 136 | "Laos districts weekly 2019-2024"; needs step 8 |

The facility level has no dengue data in the demo database, so districts are the largest set it offers; the district
series are sparse (many zero or missing weeks). To list what chap-core stored:

```sh
cdx api GET /v1/crud/datasets
```

## 10. Run an evaluation

**Evaluate**, **New evaluation**, tab **Use saved dataset**:

1. Name it after what it compares, such as "Baselines (crs) 1.0.2, Laos districts weekly" (each model's backtest
   is named `<name>/<service id>`), and pick the dataset (the target column comes from the dataset).
2. **Select models**: tick the models, then **Use selected models**.
3. Backtest parameters: for weekly data the defaults became 12 forecast periods, 7 splits, step 4, 1 training run.
4. Future-weather provider: **Seasonal climatology**.
5. **Start evaluation**; the app moves to **Jobs**.

chap-core also runs its built-in baselines (global median, persistence, seasonal median, seasonal shift persistence)
with every evaluation, as the reference the app compares against (finding 7). Each chapkit-rs model is a separate
backtest job. Follow them with:

```sh
cdx api GET /v1/jobs
```

To compare the ports with the built-ins, read the metrics of the backtest specification; its `backtests` are the
selected models and its `baselines` the built-ins. An evaluation with the same dataset and backtest parameters as an
earlier one joins that one's specification, and chap-core reuses the built-ins' results instead of running them
again:

```sh
cdx api GET /v1/crud/backtest-specifications/2
```

To sample CPU and memory while it runs, poll `docker stats` for the worker, chap-core and the model containers:

```sh
while :; do
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' \
    $(docker ps --format '{{.Names}}' | grep -E 'worker|chap-1|median|persistence') >> stats.csv
done
```

## 11. Run a prediction

A prediction starts from a completed evaluation:

1. **Evaluate**, open a model's backtest (for example ".../global-median"), **Create prediction setup**, give the
   setup a name ("Global median (crs), Laos districts weekly") and **Save**; the app opens the setup.
2. **Run prediction**, then pick the **Last training period** (it has no default; the Laos data ends in week 52 of
   2024: year 2024, then "Week 52") and **Run prediction** again.
3. The job appears as `create_prediction` in `cdx api GET /v1/jobs`.

Evaluations are renamed in the app (a backtest's **More** menu, **Rename**) or through chap-core's API:

```sh
cdx api PATCH /v1/crud/backtests/17 --data '{"name": "Baselines (crs) 1.0.2, Laos districts weekly/global-median"}'
```

## 12. Results

All on an Apple Silicon laptop, with the whole stack emulated as linux/amd64 (finding 5), so the times are only
comparable with each other.

**Accuracy.** The ports give the built-ins' metrics to ten decimals:

| evaluation | model | crps | mae | rmse | coverage 10-90 |
|---|---|---|---|---|---|
| 1: 18 provinces monthly, 3 periods, 7 splits | seasonal median | 27.1944240520 | 30.4953271028 | 46.2400448156 | 0.6261682243 |
| 2: 136 districts weekly, 12 periods, 7 splits | global median | 2.2518619012 | 2.6640580535 | 5.0445017319 | 0.9408081958 |
| 2 | seasonal median | 4.5017012082 | 5.3167330677 | 10.1044558714 | 0.3918611269 |
| 2 | seasonal shift persistence | 3.2495319309 | 4.0936462610 | 8.3621875534 | 0.5412635174 |
| 2 | persistence | 2.5537512713 | 3.0051223677 | 5.3147382626 | 0.8215708594 |

Evaluation 2 ran twice: with the 1.0.0 ports and again with 1.0.2 (persistence added); both runs gave the same
numbers.

Every metric of every model is identical except one: seasonal shift persistence's `ratio_above_truth` is 0.5116505407
for the port and 0.5116476949 for the built-in. One sample in about 350,000 that equals the observation exactly in
Python lands one rounding step above it in the port (finding 9, fixed in 1.0.3).

**Time, CPU and memory** (evaluation 2, all 11 jobs running at once):

| | wall time per backtest | CPU peak | memory peak |
|---|---|---|---|
| chap-core built-in baseline | 18-21 s | in the worker | in the worker |
| chapkit-rs model, through chap-core | 47-48 s | 9-20 % of a core | 76-114 MiB |
| chap-core worker (all jobs) | | 1143 % | 5.7 GiB |

The chapkit-rs services did very little of that time: `$train` took 151 ms and each `$predict` 40-90 ms, JSON decoding
included, and the first job poll found each job done. The rest is chap-core's worker: 9 s before it posts `$train`
and about 4 s around each of the 7 predictions (finding 8). With the four ports alone (1.0.2 run, the built-ins
reused), each backtest took 42 s.

**Prediction.** "Global median (crs), Laos districts weekly prediction" (136 districts, training to 2024W52) took
14.4 s as a chap-core job; the service's part was `$train` 22 ms, `$predict` 46 ms and reading the result 12 ms.

## Findings

| # | finding | where | status |
|---|---|---|---|
| 1 | The ports register with the same display names as chap-core's built-in baselines ("Global median", ...), so the Modeling App's model list shows two entries with one name, told apart only by the author. | chapkit-rs-models | fixed: names end in "(crs)" |
| 2 | The ports declare `author_assessed_status: green`; chap-core rates its identical built-ins `red` (baselines are yardsticks, not forecasting models). | chapkit-rs-models | fixed: red |
| 3 | `organization` and `documentation_url` are empty; the models book is the obvious documentation URL. The library supports both. | chapkit-rs-models | fixed: documentation and repository URLs |
| 4 | The Python reference model ships an amd64-only image; the chapkit-rs images are built for amd64 and arm64. | marketplace | noted |
| 5 | chap-devops pins every model, and chap-core, to `linux/amd64` (most marketplace images are amd64-only), so on Apple Silicon the whole stack runs emulated, the chapkit-rs images included, although they ship arm64. It also only accepts locally built model images for linux/amd64: a native arm64 build from `make docker-build` is refused ("not in the local image store for linux/amd64"). Running natively needs `cdx run` outside a deployment. | chap-devops | open (chap-devops' call) |
| 6 | Retiring a model (`cdx models remove`) and registering a new build does not change what chap-core already stored: its model templates kept the old display names and status. chap-core does this by design: a template version is write-once, and since the new build reported the same version (1.0.0, the `Service` default) from a new git revision, chap-core logged a revision conflict and refuses to run the model until the version changes. The ports now report their crate version. | chapkit-rs-models | fixed: version 1.0.1 |
| 7 | chap-core runs its four built-in baselines with every evaluation, so ticking the built-ins as models as well runs them twice. | Modeling App | noted |
| 8 | A chapkit model backtest took 47 s against 19 s for a built-in, although the service itself answered in under a second in total: the time is chap-core's REST wrapper (9 s before `$train`, about 4 s per split). This is the overhead the in-process fast paths avoid; it is on chap-core's side. | chap-core | open |
| 9 | Seasonal shift persistence differs from the built-in in the last bit of some samples (at most 3.5e-16), enough to flip one tie with the observation in about 350,000 samples (`ratio_above_truth` off by 2.8e-6). The cause was the quantile interpolation shared by every port: numpy computes `b - (b - a) * (1 - t)` for `t >= 0.5`. | chapkit-rs-models | fixed in 1.0.3: bit-identical, and the parity tests compare bitwise |
| 10 | District "0710 Xon" has no geometry in the demo database, which blocks district datasets (step 8). | DHIS2 demo data | worked around |
| 11 | Running a prediction needs a **Last training period**, which has no default although the setup knows where its data ends; the picker opens on the current year, two years after the data. | Modeling App | noted |
| 12 | The deployment was taken down twice mid-run from outside the session running it: once from the Docker Desktop dashboard (a bulk action, which runs `docker compose --project-name <project> down --remove-orphans`), once by another session running `chap-devops down` in the same folder. The deployment folder and its compose project are shared state. Docker Desktop's backend log (`~/Library/Containers/com.docker.docker/Data/log/host/com.docker.backend.log*`) records who did it: the client, the compose command and the project. | local setup | noted |
