.PHONY: help check build release lint fmt test bench doc openapi compare-openapi clean


help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  check     lint + test; run before every commit"
	@echo "  build     Build the workspace, examples and benches (debug)"
	@echo "  release   Build the workspace in release mode"
	@echo "  lint      Format and run clippy (warnings are errors)"
	@echo "  fmt       Format the code"
	@echo "  test      Run all tests"
	@echo "  bench     Run criterion benchmarks (BENCH=<filter> to narrow)"
	@echo "  doc       Build API docs"
	@echo "  openapi   Regenerate the committed openapi.json"
	@echo "  compare-openapi PY=<url> RS=<url>  Diff a Python chapkit service's OpenAPI against a chapkit-rs one"
	@echo "  clean     Remove build output"

check: lint test

build:
	@cargo build --workspace --all-targets

release:
	@cargo build --workspace --release

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

openapi:
	@UPDATE_OPENAPI=1 cargo test -q --test openapi
	@echo "wrote openapi.json"

compare-openapi:
	@python3 scripts/compare_openapi.py $(PY) $(RS)

clean:
	@cargo clean
