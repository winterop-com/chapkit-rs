//! In-process backtests: chap-core's evaluation, run against a [`Model`](chapkit_core::Model) by
//! plain function calls.
//!
//! A port of chap-core 2.7.0's `chap_core.assessment`, checked against fixtures chap-core generates
//! (`scripts/gen_eval_fixtures.py`):
//!
//! - [`split`]: `train_test_generator`'s expanding windows and `backtest`'s retraining points;
//! - [`weather`]: the future covariates of each window ([`Climatology`], chap-core's default, and
//!   [`Observed`], which looks ahead and is for diagnostics only);
//! - [`metrics`]: CRPS and its log and normalised forms, MAE, RMSE, MAPE, coverage, Winkler and
//!   ratio above truth, aggregated as chap-core's `aggregate_metrics`, and by horizon and location;
//! - [`mask`]: target masking (`target_masked`);
//! - [`backtest()`]: all of it with a model's `train` and `predict`.
//!
//! Pure like `chapkit-core`: no IO, no async, no HTTP.

mod backtest;
pub mod mask;
pub mod metrics;
pub mod split;
pub mod weather;

pub use backtest::{Backtest, Forecast, backtest, backtest_model};
pub use mask::{MASK_INDICATOR, MaskedRange, mask_target};
pub use metrics::{Metric, Scores, score};
pub use split::{Options, Split, Splits, retrain_points, train_test_splits};
pub use weather::{Climatology, Observed, WeatherProvider, weather_provider};
