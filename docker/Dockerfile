# syntax=docker/dockerfile:1
#
# One Dockerfile for every chapkit-rs model package.
#
# Model packages depend on chapkit-rs by path, so the build context is the directory holding both
# checkouts (for example chap-workspace/rs, with chapkit-rs/ and chapkit-rs-models/ in it):
#
#     docker build -f chapkit-rs/docker/Dockerfile \
#         --build-arg WORKSPACE=chapkit-rs-models \
#         --build-arg BIN=baseline-global-median \
#         --build-arg GIT_REVISION=$(git -C chapkit-rs-models rev-parse HEAD) \
#         -t baseline-global-median .
#
# The image holds the one binary on a minimal base (no shell, no package manager), runs as uid 1000,
# serves (`service serve`) on 0.0.0.0:8000, and keeps its SQLite database in /work/data (mount a volume there). The
# root filesystem can be read-only; only /work/data is written.

ARG RUST_IMAGE=rust:1.99-slim-trixie
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian13:nonroot

FROM ${RUST_IMAGE} AS build
# The cargo workspace to build in (a directory of the build context), the binary, and optional
# cargo features (for example turso).
ARG WORKSPACE
ARG BIN
ARG FEATURES=""
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=/cargo-target,sharing=locked \
    test -n "$WORKSPACE" && test -n "$BIN" \
    && cd "$WORKSPACE" \
    && CARGO_TARGET_DIR=/cargo-target cargo build --release --locked --bin "$BIN" ${FEATURES:+--features "$FEATURES"} \
    && mkdir -p /out/bin /out/work/data \
    && cp "/cargo-target/release/$BIN" /out/bin/service \
    && chown -R 1000:1000 /out/work

FROM ${RUNTIME_IMAGE}
# Commit the image was built from, reported as git_revision in /api/v1/info (chap-core stores a
# model template only for services that report one).
ARG GIT_REVISION=""
ENV GIT_REVISION=${GIT_REVISION} \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATABASE_URL=sqlite:////work/data/chapkit.db
COPY --from=build /out/bin/service /usr/local/bin/service
COPY --from=build --chown=1000:1000 /out/work /work
WORKDIR /work
USER 1000:1000
EXPOSE 8000
# The binary checks its own /health: minimal images have no curl.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/service", "healthcheck"]
ENTRYPOINT ["/usr/local/bin/service"]
CMD ["serve"]
