.PHONY: help check build release examples-release lint fmt test bench examples fixtures doc clean

EXAMPLES := global_median seasonal_median seasonal_shift_persistence ridge
CHAP_CORE ?= ../chap-core

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  check     lint + test; run before every commit"
	@echo "  build     Build the workspace, examples and benches (debug)"
	@echo "  release   Build the workspace in release mode"
	@echo "  examples-release  Build every example in release mode into ./bin"
	@echo "  lint      Format and run clippy (warnings are errors)"
	@echo "  fmt       Format the code"
	@echo "  test      Run all tests"
	@echo "  bench     Run criterion benchmarks (BENCH=<filter> to narrow)"
	@echo "  examples  Run every example model in release mode"
	@echo "  fixtures  Regenerate baseline golden outputs from chap-core (CHAP_CORE=$(CHAP_CORE))"
	@echo "  doc       Build API docs"
	@echo "  clean     Remove build output"

check: lint test

build:
	@cargo build --workspace --all-targets

release:
	@cargo build --workspace --release

examples-release:
	@cargo build --release --examples
	@mkdir -p bin
	@for e in $(EXAMPLES); do cp target/release/examples/$$e bin/$$e; done
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

examples:
	@for e in $(EXAMPLES); do cargo run -q --release --example $$e -- demo || exit 1; echo; done

fixtures:
	@$(CHAP_CORE)/.venv/bin/python scripts/gen_baseline_fixtures.py

doc:
	@cargo doc --workspace --no-deps

clean:
	@cargo clean
	@rm -rf bin
