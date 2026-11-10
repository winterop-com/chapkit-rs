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
| `RUST_IMAGE` | the build image, `rust:1.99-alpine` (it must build for musl) |

`docker/Dockerfile.dockerignore` keeps build output (`target/`, `bin/`), git data and local environments out of
the build context.

## What the image is

| | |
|---|---|
| base | `scratch`: no shell, no libc, no package manager |
| contents | the one binary, `/usr/local/bin/service`, statically linked against musl; CA certificates (for registering with chap-core over https); `/etc/passwd`; an empty `/tmp` |
| size | 6.5 MB content, 19.9 MB on disk (`docker image ls`) for the seasonal median baseline; 16.8 MB and 71.9 MB on the earlier distroless base |
| user | uid 1000, as in Python chapkit's images |
| listens on | `0.0.0.0:8000` (`HOST`, `PORT`) |
| database | `DATABASE_URL=sqlite:////work/data/chapkit.db`: mount a volume on `/work/data` so results survive restarts |
| filesystem | can be read-only; only `/work/data` is written, and `/tmp` by SQLite for large temporary tables (mount a tmpfs there) |
| health | `HEALTHCHECK` runs `service healthcheck`, which calls the running service's `/health` (the image has no curl) |

musl's own allocator is slow under multithreading: training a baseline with rayon took four to six times as long.
chapkit therefore uses mimalloc as the global allocator in every musl build, which makes the image as fast as or
faster than a glibc build (seasonal median, 5000 locations x 120 periods: train 25-30 ms against 104-134 ms). A
model binary that sets its own `#[global_allocator]` cannot be built for musl.

The image's command is `service serve`; `docker run <image> test` runs chap-core's flow against a running
service instead. Every chapkit-rs binary has a `healthcheck` subcommand: it checks the service on `PORT` and exits
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
