//! Error type shared by every media type.

use crate::format::Format;

/// Convenience alias for results produced by this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Boxed error from a codec backend, kept as the [`std::error::Error::source`] of [`Error::Decode`].
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Everything that can go wrong while opening, planning or decoding media.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The input does not start with the signature of any known format.
    #[error("unrecognized format")]
    UnrecognizedFormat,

    /// The format was recognized but cannot be decoded (yet).
    #[error("{format} is not supported: {reason}")]
    Unsupported {
        /// The detected container format.
        format: Format,
        /// Human-readable explanation.
        reason: &'static str,
    },

    /// The codec backend rejected the data.
    #[error("failed to decode {format} data: {source}")]
    Decode {
        /// The format whose decoder failed.
        format: Format,
        /// The backend error.
        #[source]
        source: BoxError,
    },

    /// The encoder rejected the data or the settings.
    #[error("failed to encode {format}: {source}")]
    Encode {
        /// The format being written.
        format: Format,
        /// The encoder error.
        #[source]
        source: BoxError,
    },

    /// An operation was requested with arguments that do not fit the input.
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
    /// Builds [`Error::Decode`] from any error-like value, including strings.
    pub fn decode(format: Format, source: impl Into<BoxError>) -> Self {
        Self::Decode {
            format,
            source: source.into(),
        }
    }

    /// Builds [`Error::Encode`] from any error-like value, including strings.
    pub fn encode(format: Format, source: impl Into<BoxError>) -> Self {
        Self::Encode {
            format,
            source: source.into(),
        }
    }
}
