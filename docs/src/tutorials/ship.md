# Ship it as a container

You build your model package into a container image, run it the way it runs next to chap-core, and publish it.

## Before you start

- A model package ([Your first model package](first-model.md)) that you have built once with `cargo test` or
  `cargo build`: the image builds from its `Cargo.lock`.
- Docker running.

## 1. Build and start

```sh
docker compose up --build -d
docker compose ps --format '{{.Service}} {{.Status}}'
```

The first build compiles everything inside Docker and takes a few minutes. When the service reports
`(healthy)`, it is ready.

The image is chapkit-rs's shared Dockerfile, which `init` copied into the package: the one binary on a minimal
base image, running as uid 1000, with its database in a volume and a health check ([Docker](../guide/docker.md)).

## 2. Check it

```sh
chapkit-rs test
```

passes exactly as against `cargo run`: the container serves on port 8000.

## 3. Publish it

`init` also wrote `.github/workflows/publish.yml`. Push the package to a GitHub repository, and on every push to
`main` it builds the image natively for amd64 and arm64, publishes it to the GitHub container registry as
`ghcr.io/<owner>/<repository>:latest` (and `1.2.3` for a `v1.2.3` tag), and attests where it was built. Nothing to
configure: the built-in `GITHUB_TOKEN` is enough. A new package on the registry starts out private; make it public
in its settings so chap-core deployments can pull it.

## 4. Clean up

```sh
docker compose down
```

The database stays in the `data` volume; `docker compose down -v` removes it too.

## Next steps

- [Testing with chap-core](../guide/chap-core.md): register the container with chap-core by setting
  `SERVICEKIT_ORCHESTRATOR_URL`.
