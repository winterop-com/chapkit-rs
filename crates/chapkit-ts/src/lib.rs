//! Time-series building blocks for chapkit-rs models and evaluation, ported for parity with the R
//! and Python code Chap's models use:
//!
//! - [`stl`]: STL decomposition (Cleveland et al.), the netlib algorithm behind R's `stats::stl` and
//!   statsmodels' `STL`;
//! - [`unitroot`]: the KPSS test (urca's and statsmodels' lag choices), feasts' STL seasonal
//!   strength, and the number of differences fable chooses from them;
//! - [`arima`]: R's `stats::arima` (CSS, ML and CSS-ML estimation, the Kalman filter, forecasts with
//!   their variance) and fable's automatic order search (`fable::ARIMA`);
//! - [`statsmodels`]: statsmodels' ARIMA (`tsa.arima.model.ARIMA(...).fit()`), what chap-core's
//!   MSTL + ARIMA weather provider fits.
//!
//! Pure: no IO, no async.

// Line-by-line ports of R, netlib and scipy numerics keep the originals' loops, names, exact float
// comparisons and long routines, so they can be read against them.
#![allow(
    clippy::similar_names,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::explicit_counter_loop,
    clippy::manual_midpoint,
    clippy::manual_clamp,
    clippy::nonminimal_bool,
    clippy::single_match_else
)]

pub mod arima;
mod optim;
pub mod poly;
pub mod statsmodels;
pub mod stl;
pub mod unitroot;

pub use arima::{
    Arima, ArimaSpec, AutoArima, Forecast, Method, auto_arima, fit_arima, fit_arima_fixed,
};

/// What can go wrong.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// The input cannot be used (too short, wrong shape, all missing).
    #[error("{0}")]
    Invalid(String),
    /// Estimation failed (non-stationary start, non-finite likelihood, no model found).
    #[error("{0}")]
    Estimation(String),
}

impl Error {
    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }

    pub(crate) fn estimation(msg: impl Into<String>) -> Self {
        Error::Estimation(msg.into())
    }
}

/// `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
