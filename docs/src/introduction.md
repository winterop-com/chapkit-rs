# Introduction

chapkit-rs serves forecasting models to Chap from Rust. A model is a Rust type that implements the `Model`
trait; one line turns it into an HTTP service that speaks the same API as Python
[chapkit](https://github.com/dhis2-chap/chapkit), so chap-core, chap-devops and chap-marketplace use it
unchanged.

```rust,ignore
#[tokio::main]
async fn main() -> std::io::Result<()> {
    chapkit::Service::new("my-model", "My model", MyModel).serve().await
}
```

It exists for one reason: speed. A call to a Python chapkit model costs 2.5 to 3 seconds before the model
does any work. That is acceptable for one backtest and impossible for explainers, hyperparameter search and
AutoML, which call a model hundreds or thousands of times. In chapkit-rs a model call is a function call,
measured in microseconds, and a train plus predict through chap-core's normal job flow takes milliseconds.

This book explains the approach, the decisions behind it and how to use it:

- **The approach**: why it exists, what it promises to stay compatible with, and how it is built.
- **Design decisions**: each significant choice and the reasoning, including what was rejected.
- **Guide**: writing a model package, serving it, and testing it through chap-core.
- **Reference**: endpoints, configuration and crates.
- **Results**: the chap-core end-to-end run and the benchmarks against Python chapkit.

chapkit-rs is an experiment. Shell, R and Python models stay on Python chapkit; chapkit-rs only covers models
written in Rust.
