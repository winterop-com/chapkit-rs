//! chapkit-rs: in-process ML services for Chap.
//!
//! This is the crate model authors depend on. Implement [`Model`], then hand it
//! to [`Service`] to get the full chapkit HTTP API (configs, artifacts, jobs,
//! `$train`, `$predict`, health, info), wire-compatible with Python chapkit so
//! chap-core can use it unchanged.
//!
//! Example models, starting with ports of chap-core's baselines, are under
//! `examples/` and are not part of the library.

mod erased;
mod ml;
mod routes;
mod service;
mod state;
pub mod wire;

pub use chapkit_core::*;
pub use chapkit_dataframe as dataframe;
pub use chapkit_http as http;
pub use chapkit_store as store;
pub use chapkit_xai as xai;
pub use erased::{DynModel, Erased, RESERVED_KEYS, Trained};
pub use service::Service;

/// Compiles the README's code blocks as doctests, so the example there cannot rot.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
