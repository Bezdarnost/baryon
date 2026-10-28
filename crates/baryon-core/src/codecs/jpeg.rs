//! JPEG decoding via `zune-jpeg`.
//!
//! Region requests are currently satisfied by a full decode. Skipping entropy-coded MCUs outside
//! the region and DCT-domain downscaling are the planned next steps.

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

use super::{Codec, Decoded, ImageInfo, RegionStrategy};
use crate::error::{Error, Result};
use crate::format::ImageFormat;
use crate::geometry::Rect;
use crate::image::{ColorType, Image, PixelBuffer, SampleType};

/// Baseline and progressive JPEG decoder.
pub(crate) struct JpegCodec;

/// JPEG dimensions are 16-bit, so this lifts `zune-jpeg`'s default 16384px cap to the format limit.
const MAX_DIMENSION: usize = u16::MAX as usize;

impl JpegCodec {
    fn options(color: ColorType) -> DecoderOptions {
        let out = match color {
            ColorType::L => ColorSpace::Luma,
            _ => ColorSpace::RGB,
        };
        DecoderOptions::default()
            .set_max_width(MAX_DIMENSION)
            .set_max_height(MAX_DIMENSION)
            .jpeg_set_out_colorspace(out)
    }
}

impl Codec for JpegCodec {
    fn probe(&self, data: &[u8]) -> Result<ImageInfo> {
        let mut decoder =
            JpegDecoder::new_with_options(ZCursor::new(data), Self::options(ColorType::Rgb));
        decoder.decode_headers().map_err(jpeg_error)?;
        let info = decoder
            .info()
            .ok_or_else(|| Error::Internal("JPEG headers decoded but no info available".into()))?;
        // Grayscale stays grayscale; YCbCr, RGB, CMYK and YCCK are all delivered as RGB.
        let color = if info.components == 1 {
            ColorType::L
        } else {
            ColorType::Rgb
        };
        Ok(ImageInfo {
            format: ImageFormat::Jpeg,
            width: u32::from(info.width),
            height: u32::from(info.height),
            color,
            sample_type: SampleType::U8,
            progressive: info.sof.is_progressive(),
        })
    }

    fn region_strategy(&self, _info: &ImageInfo, _region: Rect) -> RegionStrategy {
        RegionStrategy::FullDecode
    }

    fn decode(&self, data: &[u8], info: &ImageInfo, _region: Rect) -> Result<Decoded> {
        let mut decoder =
            JpegDecoder::new_with_options(ZCursor::new(data), Self::options(info.color));
        let pixels = decoder.decode().map_err(jpeg_error)?;
        let image = Image::new(info.width, info.height, info.color, PixelBuffer::U8(pixels))?;
        Ok(Decoded {
            image,
            region: info.bounds(),
        })
    }
}

fn jpeg_error(err: zune_jpeg::errors::DecodeErrors) -> Error {
    Error::decode(ImageFormat::Jpeg, err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(width: u16, height: u16, color: jpeg_encoder::ColorType, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 95)
            .encode(data, width, height, color)
            .unwrap();
        out
    }

    #[test]
    fn probes_without_decoding() {
        let data = encode(33, 17, jpeg_encoder::ColorType::Rgb, &[128; 33 * 17 * 3]);
        let info = JpegCodec.probe(&data).unwrap();
        assert_eq!((info.width, info.height), (33, 17));
        assert_eq!(info.color, ColorType::Rgb);
    }

    #[test]
    fn grayscale_stays_single_channel() {
        let data = encode(8, 8, jpeg_encoder::ColorType::Luma, &[200; 64]);
        let info = JpegCodec.probe(&data).unwrap();
        assert_eq!(info.color, ColorType::L);
        let decoded = JpegCodec.decode(&data, &info, info.bounds()).unwrap();
        assert_eq!(decoded.image.pixels().len(), 64);
        assert!(
            decoded
                .image
                .pixels()
                .as_bytes()
                .iter()
                .all(|&v| v.abs_diff(200) <= 1)
        );
    }

    #[test]
    fn truncated_data_is_a_decode_error() {
        let data = encode(16, 16, jpeg_encoder::ColorType::Rgb, &[50; 16 * 16 * 3]);
        let err = JpegCodec.probe(&data[..20]).unwrap_err();
        assert!(matches!(
            err,
            Error::Decode {
                format: ImageFormat::Jpeg,
                ..
            }
        ));
    }
}
