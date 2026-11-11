.PHONY: help install release-tag check test-turso test-polars build release lint fmt test bench doc docs docs-serve openapi compare-openapi eval-fixtures arima-fixtures models-check models-release ui clean

MODELS ?= ../chapkit-rs-models
CHAP_CORE ?= ../../chap-core
ARIMA_IMAGE ?= ghcr.io/chap-models/auto_arima_chapkit@sha256:49f2b0ecce17cdb908606f4797bc93b6ac8b8c19e444a1f768226f764ff3314f


help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install bin/chapkit-rs and its short form crs into PREFIX/bin (PREFIX defaults to ~/.local)"
	@echo "  check     lint + test; run before every commit"
	@echo "  test-turso  Run the store suite and the HTTP contract on the turso backend"
	@echo "  test-polars  Lint and test the polars conversion (feature polars)"
	@echo "  build     Build the workspace, tests and benches (debug)"
	@echo "  release   The chapkit-rs command into ./bin, universal on macOS"
	@echo "  release-tag  Bump the version, check, commit and tag (VERSION=x.y.z, PUSH=1 to push)"
	@echo "  lint      Format and run clippy (warnings are errors)"
	@echo "  fmt       Format the code"
	@echo "  test      Run all tests"
	@echo "  bench     Run criterion benchmarks (BENCH=<filter> to narrow)"
	@echo "  doc       Build API docs (rustdoc)"
	@echo "  docs      Build the mdBook (docs/book)"
	@echo "  docs-serve  Serve the mdBook with live reload"
	@echo "  openapi   Regenerate the committed openapi.json"
	@echo "  compare-openapi PY=<url> RS=<url>  Diff a Python chapkit service's OpenAPI against a chapkit-rs one"
	@echo "  eval-fixtures   Regenerate chapkit-eval's parity fixtures from chap-core (CHAP_CORE=$(CHAP_CORE))"
	@echo "  arima-fixtures  Regenerate chapkit-ts' parity fixtures from R and fable (ARIMA_IMAGE)"
	@echo "  models-check    Run make check in $(MODELS)"
	@echo "  models-release  Build the model binaries into $(MODELS)/bin"
	@echo "  ui        Install and build the web UI into crates/chapkit-ui/dist (needs bun and NPM_TOKEN)"
	@echo "  clean     Remove build output"

PREFIX ?= $(HOME)/.local

# As chap-devops installs itself and cdx: crs is a link, left alone if it is another program.
install: release
	@mkdir -p $(PREFIX)/bin
	@install -m 0755 bin/chapkit-rs $(PREFIX)/bin/chapkit-rs
	@if [ -e $(PREFIX)/bin/crs ] && [ "$$(readlink $(PREFIX)/bin/crs)" != "chapkit-rs" ]; then \
		echo "installed $(PREFIX)/bin/chapkit-rs; $(PREFIX)/bin/crs is another program, so it was left alone"; \
	else ln -sf chapkit-rs $(PREFIX)/bin/crs && echo "installed $(PREFIX)/bin/chapkit-rs, and $(PREFIX)/bin/crs as its short form"; fi

check: lint test

# Bump the version, run the checks, commit and tag (as chap-devops does). Pushing the tag starts the release
# workflow, which builds the chapkit-rs archives and creates the GitHub release.
release-tag:
	@test -n "$(VERSION)" || { echo "usage: make release-tag VERSION=x.y.z [PUSH=1]"; exit 2; }
	@MAKE="$(MAKE)" scripts/release.sh $(VERSION) $(if $(PUSH),--push,)

test-polars:
	@cargo clippy -p chapkit-dataframe --features polars --all-targets -- -D warnings
	@cargo test -p chapkit-dataframe --features polars

test-turso:
	@cargo test -p chapkit-store --features turso
	@CHAPKIT_TEST_DATABASE_URL=turso::memory: cargo test --features turso --test contract

build:
	@cargo build --workspace --all-targets

# On macOS, both architectures joined with lipo into one binary (as chap-devops ships its macOS
# build); elsewhere a native build.
release:
ifeq ($(shell uname -s),Darwin)
	@cargo build --release -p chapkit-cli --target aarch64-apple-darwin
	@cargo build --release -p chapkit-cli --target x86_64-apple-darwin
	@mkdir -p bin && lipo -create -output bin/chapkit-rs \
		target/aarch64-apple-darwin/release/chapkit-rs target/x86_64-apple-darwin/release/chapkit-rs
	@lipo -info bin/chapkit-rs
else
	@cargo build --release -p chapkit-cli
	@mkdir -p bin && cp target/release/chapkit-rs bin/chapkit-rs
endif
	@ls -lh bin | awk 'NR>1 {print $$5, "bin/"$$9}'

lint:
	@echo ">>> Formatting"
	@cargo fmt --all
	@echo ">>> Clippy"
	@cargo clippy --workspace --all-targets -- -D warnings

fmt:
	@cargo fmt --all

test:
	@cargo test --workspace

bench:
	@cargo bench --workspace -- $(BENCH)

doc:
	@cargo doc --workspace --no-deps

docs:
	@mdbook build docs

docs-serve:
	@mdbook serve docs --open

openapi:
	@UPDATE_OPENAPI=1 cargo test -q --test openapi
	@echo "wrote openapi.json"

compare-openapi:
	@if [ -z "$(PY)" ] || [ -z "$(RS)" ]; then echo "usage: make compare-openapi PY=<python chapkit url> RS=<chapkit-rs url>"; exit 2; fi
	@python3 scripts/compare_openapi.py $(PY) $(RS)

eval-fixtures:
	@$(CHAP_CORE)/.venv/bin/python scripts/gen_eval_fixtures.py
	@$(CHAP_CORE)/.venv/bin/python -I scripts/gen_mstl_arima_fixtures.py

arima-fixtures:
	@Rscript scripts/gen_arima_fixtures.R stats
	@docker run --rm --platform linux/amd64 -v "$(CURDIR)":/w -w /w $(ARIMA_IMAGE) Rscript scripts/gen_arima_fixtures.R fable

models-check:
	@$(MAKE) -C $(MODELS) check

models-release:
	@$(MAKE) -C $(MODELS) release

# The web UI (ui/, on sokkel-ui from GitHub Packages) into the dist/ chapkit-ui embeds.
ui:
	@cd ui && bun install --frozen-lockfile && bun run lint && bun run test && bun run build

clean:
	@cargo clean
	@rm -rf bin
