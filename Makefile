.PHONY: help install check test-turso test-polars build release lint fmt test bench doc docs docs-serve openapi compare-openapi models-check models-release clean

MODELS ?= ../chapkit-rs-models


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
	@echo "  lint      Format and run clippy (warnings are errors)"
	@echo "  fmt       Format the code"
	@echo "  test      Run all tests"
	@echo "  bench     Run criterion benchmarks (BENCH=<filter> to narrow)"
	@echo "  doc       Build API docs (rustdoc)"
	@echo "  docs      Build the mdBook (docs/book)"
	@echo "  docs-serve  Serve the mdBook with live reload"
	@echo "  openapi   Regenerate the committed openapi.json"
	@echo "  compare-openapi PY=<url> RS=<url>  Diff a Python chapkit service's OpenAPI against a chapkit-rs one"
	@echo "  models-check    Run make check in $(MODELS)"
	@echo "  models-release  Build the model binaries into $(MODELS)/bin"
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

models-check:
	@$(MAKE) -C $(MODELS) check

models-release:
	@$(MAKE) -C $(MODELS) release

clean:
	@cargo clean
	@rm -rf bin
