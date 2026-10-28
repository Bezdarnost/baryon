//! Format-specific decoders behind a common, region-aware interface.
//!
//! Every codec answers two questions: what does the image look like (cheap, header only), and
//! how do I produce the pixels of a given source region. Codecs are free to decode more than was
//! asked for; they report the region they actually produced and the executor trims the rest. This
//! lets each format exploit whatever partial-decoding ability it has without the planner needing
//! to know the details.

mod jpeg;
mod png;

use std::fmt;

use crate::error::{Error, Result};
use crate::format::ImageFormat;
use crate::geometry::Rect;
use crate::image::{ColorType, Image, SampleType};

/// Header-level facts about an encoded image, known without decoding pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    /// Container format.
    pub format: ImageFormat,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Channel layout the decoder will produce.
    pub color: ColorType,
    /// Sample type the decoder will produce.
    pub sample_type: SampleType,
    /// Whether the data is stored in multiple passes (progressive JPEG, Adam7 PNG), which
    /// limits how much partial decoding can save.
    pub progressive: bool,
}

impl ImageInfo {
    /// The full image area.
    pub const fn bounds(&self) -> Rect {
        Rect::from_size(self.width, self.height)
    }
}

/// How a codec intends to satisfy a region request; reported by `explain()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionStrategy {
    /// The whole image is decoded and then trimmed.
    FullDecode,
    /// Rows are decoded in order and decoding stops after the last requested row; only the
    /// requested columns are kept.
    RowStreaming {
        /// Number of rows that must be decoded.
        rows_decoded: u32,
    },
}

impl fmt::Display for RegionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullDecode => f.write_str("full decode"),
            Self::RowStreaming { rows_decoded } => {
                write!(f, "row streaming, stops after {rows_decoded} rows")
            }
        }
    }
}

/// Pixels produced by a codec together with the source region they cover.
pub(crate) struct Decoded {
    pub(crate) image: Image,
    pub(crate) region: Rect,
}

/// A decoder for one container format.
pub(crate) trait Codec: Sync {
    /// Reads only the headers.
    fn probe(&self, data: &[u8]) -> Result<ImageInfo>;

    /// Describes how [`Codec::decode`] will handle `region`.
    fn region_strategy(&self, info: &ImageInfo, region: Rect) -> RegionStrategy;

    /// Decodes at least `region`, which must lie within the image bounds.
    fn decode(&self, data: &[u8], info: &ImageInfo, region: Rect) -> Result<Decoded>;
}

/// Returns the codec for `format`, or an error if this build cannot decode it.
pub(crate) fn codec_for(format: ImageFormat) -> Result<&'static dyn Codec> {
    match format {
        ImageFormat::Jpeg => Ok(&jpeg::JpegCodec),
        ImageFormat::Png => Ok(&png::PngCodec),
        ImageFormat::JpegXl => Err(Error::Unsupported {
            format,
            reason: "JPEG XL decoding is planned but not implemented yet",
        }),
        ImageFormat::Avif => Err(Error::Unsupported {
            format,
            reason: "AVIF decoding is planned but not implemented yet",
        }),
    }
}
