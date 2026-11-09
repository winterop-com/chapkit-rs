//! chapkit-rs: in-process ML services for Chap.
//!
//! This is the crate model authors depend on. It re-exports the model contract
//! from `chapkit-core` and the table type from `chapkit-dataframe`; the HTTP
//! service layer (chapkit-compatible API) will live here too.
//!
//! Example models, starting with ports of chap-core's baselines, are under
//! `examples/` and are not part of the library.

pub use chapkit_core::*;
pub use chapkit_dataframe as dataframe;

/// Compiles the README's code blocks as doctests, so the example there cannot rot.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
