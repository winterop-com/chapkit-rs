# Explainers

`chapkit-xai` implements LIME and KernelSHAP, ported from `chap-rs`, generic over any value function
`f(features) -> value`. `POST /api/v1/ml/$explain` uses them to explain one forecast cell, a location and a
period, of a trained model:

- the features are the numeric covariates of the `future` frame (or a chosen list);
- each perturbation replaces the target row's covariates, the model predicts the whole future frame, and the
  value is the mean of the samples at the target cell;
- every perturbation is a direct model call on a warm model, run in parallel across cores;
- LIME importances are local coefficients per standard deviation of each feature; SHAP importances are
  additive contributions that sum exactly to `prediction - baseline`.

Both methods are tested against linear models, where the right answer is known exactly.

**Limitation: lagged effects.** Only the target month's covariates are perturbed. Climate-driven models
often react to earlier months (rain one to three months before), which they read from `historic` and earlier
`future` rows; those stay fixed, so such effects do not show up. chap-core's own time-series LIME perturbs
segments of each location's history instead. Lagged features (perturbing a covariate at `t`, `t-1`, `t-2`
for the same location, with a background built the same way) are the planned extension.

**Why not batch perturbations as pseudo-locations?** It is tempting to evaluate all perturbations in one
predict call by giving each its own location. Models with per-location state, including all three
baselines, would then forecast for an unknown location and give wrong answers. One in-process call per
perturbation costs microseconds anyway.
