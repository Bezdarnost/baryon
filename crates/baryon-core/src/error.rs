//! Error type shared by every part of the engine.

use crate::format::ImageFormat;

/// Convenience alias for results produced by this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Boxed error from a codec backend, kept as the [`std::error::Error::source`] of [`Error::Decode`].
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Everything that can go wrong while opening, planning or decoding an image.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The input does not start with the signature of any known image format.
    #[error("unrecognized image format")]
    UnrecognizedFormat,

    /// The format was recognized but cannot be decoded (yet).
    #[error("{format} is not supported: {reason}")]
    Unsupported {
        /// The detected container format.
        format: ImageFormat,
        /// Human-readable explanation.
        reason: &'static str,
    },

    /// The codec backend rejected the data.
    #[error("failed to decode {format} image: {source}")]
    Decode {
        /// The format whose decoder failed.
        format: ImageFormat,
        /// The backend error.
        #[source]
        source: BoxError,
    },

    /// A transformation was requested with arguments that make no sense for the image.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Reading the input failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// An invariant inside baryon was violated. Always a bug; please report it.
    #[error("internal error (this is a bug in baryon): {0}")]
    Internal(String),
}

impl Error {
    pub(crate) fn decode(format: ImageFormat, source: impl Into<BoxError>) -> Self {
        Self::Decode {
            format,
            source: source.into(),
        }
    }
}
