# Checkpoints, provenance and pre-trained models

**Decision:** trained state is stored as a [safetensors](https://github.com/huggingface/safetensors) file:
numeric weights as tensors, everything else as JSON in the file's `__metadata__` header, together with
provenance.

**Why safetensors:**

- Loading cannot execute code, unlike Python pickle.
- It is read by the Python, Rust and JavaScript safetensors libraries, so a checkpoint is not tied to Rust.
- Weights load fast and stay exact (little-endian f32, f64 or i64).

**The defaults:** `Model::save` stores the trained state as JSON state with no tensors, which suits the
baselines (quantile tables) and most statistical models. A model with real weights overrides `save` and
`load`; the ridge example stores its coefficients, means and standard deviations as tensors.

**Provenance:** every checkpoint written by the service records the model id and version, the chapkit-rs
version, when it was written, the config, the locations, and `trained_until`: the last period in the
training data. `Provenance::overlaps(first_test_period)` answers whether a backtest's test periods start
inside the training data. Unparseable periods count as overlapping, because for an evaluation the safe
answer is "cannot rule it out".

**Pre-trained models:** this is the main objection to them: a model pre-trained on data that overlaps a
backtest scores better than it should. Recording `trained_until` turns that into something a backtest can
check automatically instead of something to argue about. `Model::pretrained` lets a model ship a
checkpoint; its `train` can then load the shipped weights and optionally fine-tune, which takes milliseconds,
so chap-core's train-then-predict flow is unchanged.

**Exactness:** serde_json's `float_roundtrip` feature is enabled. Without it, JSON state could change in the
last digit after a save and load (a test caught `24.735294117647054` coming back as `...058`).
