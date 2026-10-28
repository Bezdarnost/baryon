//! The media-agnostic runtime shared by every baryon media crate.
//!
//! * [`Format`] detects containers from file signatures and says what kind of media they hold.
//! * [`Error`] is the error type used throughout baryon.
//! * [`Executor`] owns the thread pool that all decoding and processing runs on.
//!
//! Media-specific decoding, lazy plans and operations live in their own crates
//! (`baryon-image` today; video and audio later), which build on these pieces.

mod error;
mod executor;
mod format;

pub use error::{BoxError, Error, Result};
pub use executor::Executor;
pub use format::{Format, MediaType};
