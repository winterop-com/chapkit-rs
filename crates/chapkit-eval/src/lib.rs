//! In-process backtests: chap-core's evaluation, run against a [`Model`](chapkit_core::Model) by
//! plain function calls.
//!
//! A port of chap-core 2.7.0's `chap_core.assessment`, checked against fixtures chap-core generates
//! (`scripts/gen_eval_fixtures.py`):
//!
//! - [`prepare()`]: the dataset as chap-core loads it (`fill_missing=True`): sorted, gaps filled,
//!   every location padded to the same period range;
//! - [`split`]: `train_test_generator`'s expanding windows and `backtest`'s retraining points;
//! - [`weather`]: the future covariates of each window ([`Climatology`], chap-core's default, and
//!   [`DampedPersistence`], chap-core's recommended reference, [`MstlArima`], MSTL plus an ARIMA
//!   chosen by AICc, and [`Observed`], which looks ahead and is for diagnostics only);
//! - [`metrics`]: CRPS and its log and normalised forms, MAE, RMSE, MAPE, coverage, Winkler and
//!   ratio above truth, aggregated as chap-core's `aggregate_metrics`, and by horizon and location;
//! - [`mask`]: target masking (`target_masked`);
//! - [`backtest()`]: all of it with a model's `train` and `predict`;
//! - [`forecast_ahead()`]: the history and future frame of a forecast past the dataset's end.
//!
//! Pure like `chapkit-core`: no IO, no async, no HTTP.

mod ahead;
mod backtest;
mod budget;
pub mod mask;
pub mod metrics;
pub mod prepare;
pub mod split;
pub mod weather;

pub use ahead::{Ahead, forecast_ahead};
pub use backtest::{
    Backtest, Forecast, backtest, backtest_model, backtest_model_with, backtest_splits,
};
pub use mask::{MASK_INDICATOR, MaskedRange, mask_target};
pub use metrics::{ForecastSummary, Metric, Quantiles, Scores, score, summarize};
pub use prepare::{Prepare, prepare};
pub use split::{Options, Split, Splits, retrain_points, train_test_splits};
pub use weather::{
    Climatology, DampedPersistence, MstlArima, Observed, WeatherProvider, weather_provider,
};
