//! chapkit-rs: in-process ML services for Chap.
//!
//! This is the crate model authors depend on. Implement [`Model`], then hand it
//! to [`Service`] to get the full chapkit HTTP API (configs, artifacts, jobs,
//! `$train`, `$predict`, health, info), wire-compatible with Python chapkit so
//! chap-core can use it unchanged.
//!
//! Model packages, starting with ports of chap-core's baselines, live in the
//! separate chapkit-rs-models repository and are not part of the library.

/// musl's allocator is slow under multithreading (rayon); every musl build of a chapkit binary, such as
/// the Docker image, uses mimalloc instead.
#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod analysis;
pub mod cli;
mod erased;
mod eval_cmd;
mod evaluate;
mod hpo_cmd;
mod ml;
mod routes;
mod service;
mod state;
pub mod wire;

pub use chapkit_core::*;
pub use chapkit_dataframe as dataframe;
/// polars, at the version `DataFrame::to_polars` / `from_polars` use (feature `polars`).
#[cfg(feature = "polars")]
pub use chapkit_dataframe::polars;
pub use chapkit_eval as eval;
pub use chapkit_flow as flow;
pub use chapkit_hpo as hpo;
pub use chapkit_http as http;
pub use chapkit_store as store;
pub use chapkit_xai as xai;
pub use erased::{DynModel, Erased, RESERVED_KEYS, Trained};
pub use service::{CHAPKIT_API_VERSION, Service};

/// Compiles the README's code blocks as doctests, so the example there cannot rot.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
