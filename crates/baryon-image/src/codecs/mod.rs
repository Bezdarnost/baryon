//! Format-specific decoders behind a common, region-aware interface.
//!
//! Every codec answers two questions: what does the image look like (cheap, header only), and
//! how do I produce the pixels of a given source region. Codecs are free to decode more than was
//! asked for; they report the region they actually produced and the executor trims the rest. This
//! lets each format exploit whatever partial-decoding ability it has without the planner needing
//! to know the details.

mod jpeg;
mod jxl;
mod jxl_full;
mod png;

use std::fmt;

use crate::geometry::Rect;
use crate::image::{ColorType, Image, SampleType};
use baryon_core::Format;
use baryon_core::{Error, Result};

/// Header-level facts about an encoded image, known without decoding pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    /// Container format.
    pub format: Format,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Channel layout the decoder will produce.
    pub color: ColorType,
    /// Sample type the decoder will produce.
    pub sample_type: SampleType,
    /// Whether the data is stored in multiple passes (progressive JPEG, Adam7 PNG).
    pub progressive: bool,
    /// How much work the decoder can skip when only part of this image is needed.
    pub region_support: RegionSupport,
}

impl ImageInfo {
    /// The full image area.
    pub const fn bounds(&self) -> Rect {
        Rect::from_size(self.width, self.height)
    }
}

/// Partial-decoding capability of one particular encoded image.
///
/// This is a property of the file, not just the format: an Adam7-interlaced PNG cannot stream
/// rows, for example.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionSupport {
    /// The whole image is always decoded.
    None,
    /// Rows are decoded top to bottom and decoding can stop after the last needed row.
    Rows,
    /// The image is coded in independent tiles (JPEG XL groups); only the tiles overlapping the
    /// requested region are decoded.
    Tiles,
}

/// How a region request will be satisfied; reported by `explain()`.
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
    /// Only the tiles overlapping the region are decoded.
    TileDecode,
}

impl RegionStrategy {
    /// The strategy for decoding `region` (which must lie inside the image) of `info`.
    pub const fn new(info: &ImageInfo, region: Rect) -> Self {
        let full = region.x == 0
            && region.y == 0
            && region.width == info.width
            && region.height == info.height;
        match info.region_support {
            _ if full => Self::FullDecode,
            RegionSupport::None => Self::FullDecode,
            RegionSupport::Rows => Self::RowStreaming {
                rows_decoded: region.y + region.height,
            },
            RegionSupport::Tiles => Self::TileDecode,
        }
    }
}

impl fmt::Display for RegionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullDecode => f.write_str("full decode"),
            Self::RowStreaming { rows_decoded } => {
                write!(f, "row streaming, stops after {rows_decoded} rows")
            }
            Self::TileDecode => f.write_str("tile decoding, skips tiles outside the region"),
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

    /// Decodes at least `region`, which must lie within the image bounds, following
    /// [`RegionStrategy::new`] for `info.region_support`.
    fn decode(&self, data: &[u8], info: &ImageInfo, region: Rect) -> Result<Decoded>;
}

/// Returns the codec for `format`, or an error if this build cannot decode it.
pub(crate) fn codec_for(format: Format) -> Result<&'static dyn Codec> {
    match format {
        Format::Jpeg => Ok(&jpeg::JpegCodec),
        Format::Png => Ok(&png::PngCodec),
        Format::JpegXl => Ok(&jxl::JxlCodec),
        Format::Avif => Err(Error::Unsupported {
            format,
            reason: "AVIF decoding is planned but not implemented yet",
        }),
        // Formats of other media types, added as baryon grows beyond images.
        _ => Err(Error::Unsupported {
            format,
            reason: "not an image format",
        }),
    }
}
