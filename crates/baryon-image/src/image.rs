//! Decoded, in-memory images.

use std::fmt;

use baryon_core::{Error, Result};

/// Channel layout of an image. Channels are always interleaved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorType {
    /// Single luminance channel.
    L,
    /// Luminance followed by alpha.
    La,
    /// Red, green, blue.
    Rgb,
    /// Red, green, blue, alpha.
    Rgba,
}

impl ColorType {
    /// Number of interleaved channels.
    pub const fn channels(self) -> usize {
        match self {
            Self::L => 1,
            Self::La => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    /// Whether the last channel is alpha.
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::La | Self::Rgba)
    }

    /// Conventional mode string, e.g. `"RGB"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::L => "L",
            Self::La => "LA",
            Self::Rgb => "RGB",
            Self::Rgba => "RGBA",
        }
    }
}

impl fmt::Display for ColorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Storage type of a single channel value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleType {
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer, native endian.
    U16,
}

impl SampleType {
    /// Size of one sample in bytes.
    pub const fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
        }
    }

    /// NumPy-style dtype name, e.g. `"uint8"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::U16 => "uint16",
        }
    }
}

impl fmt::Display for SampleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Owned sample storage, typed so that 16-bit data is always correctly aligned.
#[derive(Clone, PartialEq, Eq)]
pub enum PixelBuffer {
    /// 8-bit samples.
    U8(Vec<u8>),
    /// 16-bit samples.
    U16(Vec<u16>),
}

impl PixelBuffer {
    /// Allocates a zero-filled buffer holding `len` samples.
    pub fn zeroed(sample_type: SampleType, len: usize) -> Self {
        match sample_type {
            SampleType::U8 => Self::U8(vec![0; len]),
            SampleType::U16 => Self::U16(vec![0; len]),
        }
    }

    /// The sample type of this buffer.
    pub const fn sample_type(&self) -> SampleType {
        match self {
            Self::U8(_) => SampleType::U8,
            Self::U16(_) => SampleType::U16,
        }
    }

    /// Number of samples (not bytes).
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
        }
    }

    /// Whether the buffer holds no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Raw bytes in native endianness.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::U8(v) => v,
            Self::U16(v) => bytemuck::cast_slice(v),
        }
    }

    /// Mutable raw bytes in native endianness.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::U8(v) => v,
            Self::U16(v) => bytemuck::cast_slice_mut(v),
        }
    }
}

impl fmt::Debug for PixelBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never dump pixel data into debug output.
        write!(
            f,
            "PixelBuffer::{:?}(len = {})",
            self.sample_type(),
            self.len()
        )
    }
}

/// A fully decoded image held in memory, with interleaved channels in row-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    width: u32,
    height: u32,
    color: ColorType,
    pixels: PixelBuffer,
}

impl Image {
    /// Wraps an existing buffer, checking that its length matches the dimensions exactly.
    pub fn new(width: u32, height: u32, color: ColorType, pixels: PixelBuffer) -> Result<Self> {
        let expected = sample_count(width, height, color)?;
        if pixels.len() != expected {
            return Err(Error::InvalidArgument(format!(
                "pixel buffer holds {} samples, expected {expected} for {width}x{height} {color}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            color,
            pixels,
        })
    }

    /// Allocates a zero-filled image.
    pub fn zeroed(
        width: u32,
        height: u32,
        color: ColorType,
        sample_type: SampleType,
    ) -> Result<Self> {
        let len = sample_count(width, height, color)?;
        Ok(Self {
            width,
            height,
            color,
            pixels: PixelBuffer::zeroed(sample_type, len),
        })
    }

    /// Width in pixels.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Channel layout.
    pub const fn color(&self) -> ColorType {
        self.color
    }

    /// Per-channel storage type.
    pub const fn sample_type(&self) -> SampleType {
        self.pixels.sample_type()
    }

    /// Bytes occupied by one pixel.
    pub const fn bytes_per_pixel(&self) -> usize {
        self.color.channels() * self.pixels.sample_type().size()
    }

    /// Bytes occupied by one row. Rows are tightly packed.
    pub const fn row_stride(&self) -> usize {
        self.width as usize * self.bytes_per_pixel()
    }

    /// The sample storage.
    pub const fn pixels(&self) -> &PixelBuffer {
        &self.pixels
    }

    /// Mutable sample storage.
    pub const fn pixels_mut(&mut self) -> &mut PixelBuffer {
        &mut self.pixels
    }

    /// Consumes the image and returns its storage.
    pub fn into_pixels(self) -> PixelBuffer {
        self.pixels
    }
}

/// Total samples for an image, rejecting sizes that do not fit in memory addressing.
pub(crate) fn sample_count(width: u32, height: u32, color: ColorType) -> Result<usize> {
    usize::try_from(width)
        .ok()
        .and_then(|w| w.checked_mul(usize::try_from(height).ok()?))
        .and_then(|px| px.checked_mul(color.channels()))
        .ok_or_else(|| {
            Error::InvalidArgument(format!("image of {width}x{height} {color} is too large"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_buffer() {
        let err = Image::new(2, 2, ColorType::Rgb, PixelBuffer::U8(vec![0; 11])).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn strides_account_for_sample_size() {
        let img = Image::zeroed(5, 3, ColorType::Rgba, SampleType::U16).unwrap();
        assert_eq!(img.bytes_per_pixel(), 8);
        assert_eq!(img.row_stride(), 40);
        assert_eq!(img.pixels().as_bytes().len(), 120);
    }
}
