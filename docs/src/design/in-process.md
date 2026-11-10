# In-process models only

**Decision:** a model is a Rust type implementing `Model`, and the service calls it directly. There is no
shell runner and no subprocess.

**Why:** the reason chapkit-rs exists is that a model call must cost what the model costs. A shell runner
reintroduces process start-up, file IO and serialization on every call, which is exactly the overhead being
removed. Models written in R, Python or as shell scripts are well served by Python chapkit's
`ShellModelRunner`; there is no value in a second implementation of it.

**The trait:**

```rust,ignore
pub trait Model: Send + Sync + 'static {
    type Config: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync;
    type Trained: Serialize + DeserializeOwned + Send + Sync;

    fn info(&self) -> ModelInfo;
    fn search_space(&self) -> Option<SearchSpace> { None }
    fn train(&self, config: &Self::Config, input: TrainInput<'_>) -> Result<Self::Trained>;
    fn predict(&self, trained: &Self::Trained, config: &Self::Config, input: PredictInput<'_>) -> Result<DataFrame>;

    // Checkpoints, with JSON-state defaults:
    fn save(&self, trained: &Self::Trained) -> Result<Checkpoint> { ... }
    fn load(&self, checkpoint: &Checkpoint) -> Result<Self::Trained> { ... }
    fn pretrained(&self) -> Option<Result<Checkpoint>> { None }
}
```

- `train` and `predict` are plain synchronous functions with no IO. The service runs them on a blocking
  thread; explainers call them directly in tight loops.
- A pure statistical model whose fitting happens at predict time uses `type Trained = ()`; its `train` is a
  no-op and nothing is stored.
- `Config` must accept unknown fields: configs also carry chapkit's reserved keys (`prediction_periods`,
  `additional_continuous_covariates`).
- `TrainInput` and `PredictInput` are `#[non_exhaustive]` structs, so new inputs can be added without
  breaking models.
