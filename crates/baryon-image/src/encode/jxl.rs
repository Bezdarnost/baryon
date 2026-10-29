//! JPEG XL encoding with the vendored libjxl, and restoring transcoded JPEGs.

use baryon_core::{Error, Executor, Format, Result};

use crate::image::{Image, SampleType};
use crate::plan::LazyImage;

/// How [`LazyImage::to_jxl`] compresses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JxlQuality {
    /// Mathematically lossless.
    Lossless,
    /// Lossy at a Butteraugli distance from 0.0 to 25.0; 1.0 is visually lossless.
    Distance(f32),
}

/// Options for [`LazyImage::to_jxl`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JxlOptions {
    /// Lossless (the default) or lossy.
    pub quality: JxlQuality,
    /// Encoder effort from 1 (fastest) to 10 (smallest); libjxl's default is 7.
    pub effort: u8,
    /// Transcode untransformed JPEG sources instead of encoding their pixels, when lossless.
    ///
    /// Transcoding re-encodes the JPEG's DCT coefficients: the result is typically 20% smaller
    /// than the JPEG and the original file can be restored bit for bit with
    /// [`jxl_to_jpeg`]. Encoding the decoded pixels losslessly instead produces files several
    /// times larger than the JPEG.
    pub transcode_jpeg: bool,
}

impl Default for JxlOptions {
    fn default() -> Self {
        Self {
            quality: JxlQuality::Lossless,
            effort: 7,
            transcode_jpeg: true,
        }
    }
}

impl LazyImage {
    /// Encodes the result of this pipeline as JPEG XL.
    ///
    /// An untransformed JPEG source is transcoded losslessly when `options` allow it; a JPEG
    /// that libjxl cannot transcode is reported as an error rather than silently encoded from
    /// pixels. Everything else is decoded, transformed and encoded from pixels.
    pub fn to_jxl(&self, options: &JxlOptions) -> Result<Vec<u8>> {
        if options.transcode_jpeg
            && options.quality == JxlQuality::Lossless
            && self.ops().is_empty()
            && self.info().format == Format::Jpeg
        {
            return baryon_libjxl::encode_jpeg(self.source().data(), options.effort).map_err(
                |err| {
                    if err.is_untranscodable_jpeg() {
                        Error::encode(
                            Format::JpegXl,
                            format!(
                                "this JPEG cannot be transcoded losslessly ({err}); keep it as \
                                 JPEG, or disable JPEG transcoding to encode its pixels"
                            ),
                        )
                    } else {
                        Error::encode(Format::JpegXl, err)
                    }
                },
            );
        }
        encode_pixels(&self.execute()?, options)
    }

    /// Encodes many pipelines as JPEG XL in parallel, the largest first. Results are in input
    /// order, and one failure does not affect the others.
    pub fn to_jxl_all(
        executor: &Executor,
        images: &[Self],
        options: &JxlOptions,
    ) -> Vec<Result<Vec<u8>>> {
        Self::encode_all(executor, images, |image| image.to_jxl(options))
    }
}

/// Encodes decoded pixels as JPEG XL.
pub fn encode_pixels(image: &Image, options: &JxlOptions) -> Result<Vec<u8>> {
    let quality = match options.quality {
        JxlQuality::Lossless => baryon_libjxl::Quality::Lossless,
        JxlQuality::Distance(distance) => {
            if !(0.0..=25.0).contains(&distance) {
                return Err(Error::InvalidArgument(format!(
                    "JPEG XL distance must be between 0.0 and 25.0, got {distance}"
                )));
            }
            baryon_libjxl::Quality::Distance(distance)
        }
    };
    if !(1..=10).contains(&options.effort) {
        return Err(Error::InvalidArgument(format!(
            "JPEG XL effort must be between 1 and 10, got {}",
            options.effort
        )));
    }
    let bits = match image.sample_type() {
        SampleType::U8 => 8,
        SampleType::U16 => 16,
    };
    let channels = u32::try_from(image.color().channels())
        .map_err(|_| Error::Internal("channel count overflows".into()))?;
    baryon_libjxl::encode_pixels(
        image.pixels().as_bytes(),
        image.width(),
        image.height(),
        channels,
        bits,
        quality,
        options.effort,
    )
    .map_err(|err| Error::encode(Format::JpegXl, err))
}

/// Restores the original JPEG file from a JPEG XL file made by lossless JPEG transcoding,
/// bit for bit.
///
/// Uses libjxl's reconstruction, which handles every JPEG structure the encoder accepts.
pub fn jxl_to_jpeg(jxl: &[u8]) -> Result<Vec<u8>> {
    if Format::detect(jxl) != Some(Format::JpegXl) {
        return Err(Error::UnrecognizedFormat);
    }
    baryon_libjxl::reconstruct_jpeg(jxl).map_err(|err| {
        if err.is_missing_jpeg() {
            Error::decode(
                Format::JpegXl,
                "the file holds no reconstructable JPEG; it was not made by lossless JPEG \
                 transcoding",
            )
        } else {
            Error::decode(Format::JpegXl, err)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::image::PixelBuffer;

    const LOSSY: &[u8] = include_bytes!("../../tests/fixtures/smooth-rgb8-lossy.jxl");
    const RGBA16: &[u8] = include_bytes!("../../tests/fixtures/gradient-rgba16-lossless.jxl");
    const ADAM7: &[u8] = include_bytes!("../../tests/fixtures/gradient-adam7.png");

    fn jpeg(width: u16, height: u16) -> Vec<u8> {
        let pixels: Vec<u8> = (0..u32::from(width) * u32::from(height) * 3)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 90)
            .encode(&pixels, width, height, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        out
    }

    #[test]
    fn lossless_pixels_round_trip_exactly() {
        for source in [ADAM7, RGBA16, LOSSY] {
            let image = LazyImage::from_bytes(source).unwrap();
            let expected = image.execute().unwrap();
            let jxl = image.to_jxl(&JxlOptions::default()).unwrap();
            let decoded = LazyImage::from_bytes(jxl).unwrap().execute().unwrap();
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn jpeg_transcoding_is_reversible() {
        let original = jpeg(96, 64);
        let image = LazyImage::from_bytes(original.clone()).unwrap();
        let jxl = image.to_jxl(&JxlOptions::default()).unwrap();
        assert_eq!(jxl_to_jpeg(&jxl).unwrap(), original);
    }

    #[test]
    fn transformed_jpegs_are_encoded_from_pixels() {
        let image = LazyImage::from_bytes(jpeg(96, 64))
            .unwrap()
            .crop(Rect::new(8, 8, 32, 16))
            .unwrap();
        let jxl = image.to_jxl(&JxlOptions::default()).unwrap();
        let decoded = LazyImage::from_bytes(jxl).unwrap().execute().unwrap();
        assert_eq!(decoded, image.execute().unwrap());
        assert!(jxl_to_jpeg(&image.to_jxl(&JxlOptions::default()).unwrap()).is_err());
    }

    #[test]
    fn lossy_encoding_is_close() {
        let image = LazyImage::from_bytes(LOSSY).unwrap();
        let options = JxlOptions {
            quality: JxlQuality::Distance(1.0),
            ..JxlOptions::default()
        };
        let original = image.execute().unwrap();
        let decoded = LazyImage::from_bytes(image.to_jxl(&options).unwrap())
            .unwrap()
            .execute()
            .unwrap();
        let (PixelBuffer::U8(a), PixelBuffer::U8(b)) = (original.pixels(), decoded.pixels()) else {
            panic!("expected 8-bit images")
        };
        let mean = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(x.abs_diff(*y)))
            .sum::<f64>()
            / f64::from(u32::try_from(a.len()).unwrap());
        assert!(mean < 2.0, "mean absolute error {mean}");
    }

    #[test]
    fn rejects_out_of_range_options() {
        let image = LazyImage::from_bytes(ADAM7).unwrap();
        for options in [
            JxlOptions {
                effort: 0,
                ..JxlOptions::default()
            },
            JxlOptions {
                quality: JxlQuality::Distance(30.0),
                ..JxlOptions::default()
            },
        ] {
            assert!(matches!(
                image.to_jxl(&options),
                Err(Error::InvalidArgument(_))
            ));
        }
        assert!(matches!(jxl_to_jpeg(ADAM7), Err(Error::UnrecognizedFormat)));
    }
}
