//! JPEG XL decoding via `jxl-oxide`.
//!
//! JPEG XL codes images in independent groups (256x256 pixels by default), so a region request is
//! handed to the decoder, which only renders the groups overlapping it. The orientation stored in
//! the header is applied by the decoder, so all dimensions here are as displayed. Colour
//! management (needed for ICC-tagged lossy images) uses the pure-Rust `moxcms` backend.
//!
//! Only the first frame of an animation is decoded.

use jxl_oxide::image::BitDepth;
use jxl_oxide::{CropInfo, JxlImage, PixelFormat};

use super::{Codec, Decoded, ImageInfo, RegionSupport};
use crate::error::{BoxError, Error, Result};
use crate::format::ImageFormat;
use crate::geometry::Rect;
use crate::image::{ColorType, Image, PixelBuffer, SampleType};

/// JPEG XL codestreams and ISOBMFF containers.
pub(crate) struct JxlCodec;

/// The orientation value meaning "stored as displayed".
const IDENTITY_ORIENTATION: u32 = 1;

impl JxlCodec {
    fn open(data: &[u8]) -> Result<JxlImage> {
        JxlImage::builder().read(data).map_err(jxl_error)
    }
}

impl Codec for JxlCodec {
    fn probe(&self, data: &[u8]) -> Result<ImageInfo> {
        let image = Self::open(data)?;
        let color = match image.pixel_format() {
            PixelFormat::Gray => ColorType::L,
            PixelFormat::Graya => ColorType::La,
            PixelFormat::Rgb => ColorType::Rgb,
            PixelFormat::Rgba => ColorType::Rgba,
            PixelFormat::Cmyk | PixelFormat::Cmyka => {
                return Err(Error::Unsupported {
                    format: ImageFormat::JpegXl,
                    reason: "CMYK JPEG XL images are not supported yet",
                });
            }
        };
        let metadata = &image.image_header().metadata;
        // Higher integer depths and float images are delivered as full-range 16-bit samples.
        let sample_type = match metadata.bit_depth {
            BitDepth::IntegerSample { bits_per_sample } if bits_per_sample <= 8 => SampleType::U8,
            _ => SampleType::U16,
        };
        // The decoder accepts regions in displayed coordinates for every orientation, but only
        // the unrotated case is covered by fixtures so far; rotated files are decoded in full.
        let region_support = if metadata.orientation == IDENTITY_ORIENTATION {
            RegionSupport::Tiles
        } else {
            RegionSupport::None
        };
        Ok(ImageInfo {
            format: ImageFormat::JpegXl,
            width: image.width(),
            height: image.height(),
            color,
            sample_type,
            progressive: false,
            region_support,
        })
    }

    fn decode(&self, data: &[u8], info: &ImageInfo, region: Rect) -> Result<Decoded> {
        let mut image = Self::open(data)?;
        if image.num_loaded_keyframes() == 0 {
            return Err(Error::decode(
                ImageFormat::JpegXl,
                "the first frame is incomplete",
            ));
        }

        let region = if info.region_support == RegionSupport::Tiles {
            image.set_image_region(CropInfo {
                width: region.width,
                height: region.height,
                left: region.x,
                top: region.y,
            });
            region
        } else {
            info.bounds()
        };

        let render = image.render_frame(0).map_err(jxl_error)?;
        let mut stream = render.stream();
        let channels = info.color.channels();
        if (stream.width(), stream.height()) != (region.width, region.height)
            || stream.channels() as usize != channels
        {
            return Err(Error::Internal(format!(
                "JPEG XL decoder produced {}x{}x{} for region {region} of a {} image",
                stream.width(),
                stream.height(),
                stream.channels(),
                info.color
            )));
        }

        let len = crate::image::sample_count(region.width, region.height, info.color)?;
        let pixels = match info.sample_type {
            SampleType::U8 => {
                let mut buf = vec![0u8; len];
                stream.write_to_buffer(&mut buf);
                PixelBuffer::U8(buf)
            }
            SampleType::U16 => {
                let mut buf = vec![0u16; len];
                stream.write_to_buffer(&mut buf);
                PixelBuffer::U16(buf)
            }
        };
        let image = Image::new(region.width, region.height, info.color, pixels)?;
        Ok(Decoded { image, region })
    }
}

fn jxl_error(err: BoxError) -> Error {
    Error::Decode {
        format: ImageFormat::JpegXl,
        source: err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;

    // Pixel formulas are documented in `tests/fixtures/generate.py`.
    const RGB8: &[u8] = include_bytes!("../../tests/fixtures/gradient-rgb8-lossless.jxl");
    const RGBA16: &[u8] = include_bytes!("../../tests/fixtures/gradient-rgba16-lossless.jxl");
    const GRAY8: &[u8] = include_bytes!("../../tests/fixtures/gradient-gray8-lossless.jxl");
    const LOSSY: &[u8] = include_bytes!("../../tests/fixtures/smooth-rgb8-lossy.jxl");

    fn decode_full(data: &[u8]) -> (ImageInfo, Image) {
        let info = JxlCodec.probe(data).unwrap();
        let decoded = JxlCodec.decode(data, &info, info.bounds()).unwrap();
        assert_eq!(decoded.region, info.bounds());
        (info, decoded.image)
    }

    fn per_pixel<T>(width: u32, height: u32, f: impl Fn(u32, u32) -> Vec<T>) -> Vec<T> {
        let mut out = Vec::new();
        for y in 0..height {
            for x in 0..width {
                out.extend(f(x, y));
            }
        }
        out
    }

    #[test]
    fn lossless_rgb8_is_exact() {
        let (info, image) = decode_full(RGB8);
        assert_eq!((info.width, info.height), (40, 30));
        assert_eq!(
            (info.color, info.sample_type),
            (ColorType::Rgb, SampleType::U8)
        );
        assert_eq!(info.region_support, RegionSupport::Tiles);
        let expected = per_pixel(40, 30, |x, y| {
            [x, y, x ^ y].map(|v| u8::try_from(v).unwrap()).to_vec()
        });
        assert_eq!(image.pixels(), &PixelBuffer::U8(expected));
    }

    #[test]
    fn lossless_rgba16_is_exact() {
        let (info, image) = decode_full(RGBA16);
        assert_eq!(
            (info.color, info.sample_type),
            (ColorType::Rgba, SampleType::U16)
        );
        let expected = per_pixel(40, 30, |x, y| {
            [x * 1000, y * 1000, (x ^ y) * 1000, 65535 - 10 * x * y]
                .map(|v| u16::try_from(v).unwrap())
                .to_vec()
        });
        assert_eq!(image.pixels(), &PixelBuffer::U16(expected));
    }

    #[test]
    fn lossless_gray8_is_exact() {
        let (info, image) = decode_full(GRAY8);
        assert_eq!(info.color, ColorType::L);
        let expected = per_pixel(40, 30, |x, y| vec![u8::try_from(5 * x + y).unwrap()]);
        assert_eq!(image.pixels(), &PixelBuffer::U8(expected));
    }

    #[test]
    fn lossy_decode_is_close_to_the_source() {
        let (info, image) = decode_full(LOSSY);
        assert_eq!((info.width, info.height), (600, 450));
        let PixelBuffer::U8(pixels) = image.pixels() else {
            panic!("expected 8-bit samples")
        };
        let source = per_pixel(600, 450, |x, y| {
            let (fx, fy) = (f64::from(x), f64::from(y));
            vec![
                128.0 + 100.0 * (fx / 37.0).sin() * (fy / 23.0).cos(),
                fx * 255.0 / 599.0,
                fy * 255.0 / 449.0,
            ]
        });
        let total: f64 = source
            .iter()
            .zip(pixels)
            .map(|(&s, &p)| (s.round() - f64::from(p)).abs())
            .sum();
        #[expect(clippy::cast_precision_loss, reason = "small test image")]
        let mean = total / source.len() as f64;
        assert!(mean < 1.5, "mean absolute error {mean}");
    }

    #[test]
    fn region_decode_matches_full_decode() {
        let (info, full) = decode_full(LOSSY);
        let regions = [
            Rect::new(0, 0, 1, 1),
            Rect::new(599, 449, 1, 1),
            Rect::new(200, 200, 150, 100), // straddles four 256x256 groups
            Rect::new(250, 10, 20, 400),
            Rect::new(256, 256, 256, 194),
        ];
        for region in regions {
            let part = JxlCodec.decode(LOSSY, &info, region).unwrap();
            assert_eq!(part.region, region);
            assert!(
                part.image == ops::crop(&full, region).unwrap(),
                "region {region} differs from the same area of a full decode"
            );
        }
    }

    #[test]
    fn truncated_pixel_data_fails_at_decode_time() {
        let data = &LOSSY[..LOSSY.len() / 2];
        let info = JxlCodec.probe(data).unwrap();
        assert_eq!((info.width, info.height), (600, 450));
        let err = JxlCodec.decode(data, &info, info.bounds()).err().unwrap();
        assert!(matches!(
            err,
            Error::Decode {
                format: ImageFormat::JpegXl,
                ..
            }
        ));
    }

    #[test]
    fn garbage_after_signature_is_a_decode_error() {
        let err = JxlCodec.probe(&[0xFF, 0x0A, 0xDE, 0xAD]).unwrap_err();
        assert!(matches!(err, Error::Decode { .. }));
    }
}
