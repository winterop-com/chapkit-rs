# Docker

chapkit models usually run in containers, next to chap-core. chapkit-rs has one Dockerfile for every model
package, `docker/Dockerfile`; a package does not need its own.

## Building an image

Model packages depend on chapkit-rs by path, so the build context is the directory that holds both checkouts:

```sh
cd chap-workspace/rs        # contains chapkit-rs/ and chapkit-rs-models/
docker build -f chapkit-rs/docker/Dockerfile \
    --build-arg WORKSPACE=chapkit-rs-models \
    --build-arg BIN=baseline-global-median \
    --build-arg GIT_REVISION=$(git -C chapkit-rs-models rev-parse HEAD) \
    -t baseline-global-median .
```

| build argument | |
|---|---|
| `WORKSPACE` | the cargo workspace to build in, a directory of the build context (required) |
| `BIN` | the binary to build (required) |
| `FEATURES` | cargo features, for example `turso` |
| `GIT_REVISION` | the commit, reported as `git_revision` in `/api/v1/info`; chap-core stores a model template only for services that report one |
| `RUST_IMAGE`, `RUNTIME_IMAGE` | the build and runtime base images |

`docker/Dockerfile.dockerignore` keeps build output (`target/`, `bin/`), git data and local environments out of
the build context.

## What the image is

| | |
|---|---|
| base | `gcr.io/distroless/cc-debian13`: glibc and certificates, no shell, no package manager |
| contents | the one binary, `/usr/local/bin/service` |
| user | uid 1000, as in Python chapkit's images |
| listens on | `0.0.0.0:8000` (`HOST`, `PORT`) |
| database | `DATABASE_URL=sqlite:////work/data/chapkit.db`: mount a volume on `/work/data` so results survive restarts |
| filesystem | can be read-only; only `/work/data` is written |
| health | `HEALTHCHECK` runs `service healthcheck`, which calls the running service's `/health` (minimal images have no curl) |

Every chapkit-rs binary accepts `healthcheck` as its first argument: it checks the service on `PORT` and exits
with an error unless the service reports `healthy`.

## Several models: compose

`chapkit-rs-models/compose.yml` runs the baselines from the shared Dockerfile, each with its own volume, a
read-only root filesystem and a tmpfs `/tmp`:

```sh
cd chapkit-rs-models
make docker-up        # build and start, on ports 8001-8003
make docker-down
```

To register them with chap-core, set `SERVICEKIT_ORCHESTRATOR_URL`; each service registers under its compose
service name (`SERVICEKIT_HOST`), which chap-core uses to reach it on a shared network.
