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
    /// How to encode a JPEG that libjxl cannot transcode: `None` (the default) reports an
    /// error, otherwise its pixels are encoded at this quality.
    ///
    /// About 1% of real-world JPEGs cannot be transcoded, mostly large progressive files whose
    /// reconstruction data exceeds a limit of the JPEG XL format. Only applies when
    /// transcoding, i.e. with lossless quality and `transcode_jpeg`.
    pub jpeg_fallback: Option<JxlQuality>,
}

impl Default for JxlOptions {
    fn default() -> Self {
        Self {
            quality: JxlQuality::Lossless,
            effort: 7,
            transcode_jpeg: true,
            jpeg_fallback: None,
        }
    }
}

impl JxlOptions {
    /// Checks that every option is in range and that they fit together.
    pub fn validate(&self) -> Result<()> {
        if !(1..=10).contains(&self.effort) {
            return Err(Error::InvalidArgument(format!(
                "JPEG XL effort must be between 1 and 10, got {}",
                self.effort
            )));
        }
        for quality in [Some(self.quality), self.jpeg_fallback]
            .into_iter()
            .flatten()
        {
            if let JxlQuality::Distance(distance) = quality
                && !(0.0..=25.0).contains(&distance)
            {
                return Err(Error::InvalidArgument(format!(
                    "JPEG XL distance must be between 0.0 and 25.0, got {distance}"
                )));
            }
        }
        if self.jpeg_fallback.is_some() && !self.transcodes() {
            return Err(Error::InvalidArgument(
                "a JPEG fallback only applies when transcoding JPEGs, which needs lossless \
                 quality and transcode_jpeg"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Whether JPEG sources are transcoded rather than encoded from pixels.
    fn transcodes(&self) -> bool {
        self.transcode_jpeg && self.quality == JxlQuality::Lossless
    }
}

/// A JPEG XL file from [`LazyImage::encode_jxl`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedJxl {
    /// The encoded file.
    pub data: Vec<u8>,
    /// Whether the source was a JPEG that libjxl could not transcode, encoded from its pixels
    /// as [`JxlOptions::jpeg_fallback`] asks.
    pub fell_back: bool,
}

impl LazyImage {
    /// Encodes the result of this pipeline as JPEG XL.
    ///
    /// An untransformed JPEG source is transcoded losslessly when `options` allow it. A JPEG
    /// that libjxl cannot transcode is encoded from its pixels if
    /// [`JxlOptions::jpeg_fallback`] is set, and reported as an error otherwise. Everything
    /// else is decoded, transformed and encoded from pixels.
    pub fn to_jxl(&self, options: &JxlOptions) -> Result<Vec<u8>> {
        self.encode_jxl(options).map(|encoded| encoded.data)
    }

    /// Like [`to_jxl`](Self::to_jxl), but also tells whether a JPEG fallback was used.
    pub fn encode_jxl(&self, options: &JxlOptions) -> Result<EncodedJxl> {
        options.validate()?;
        if options.transcodes() && self.ops().is_empty() && self.info().format == Format::Jpeg {
            match baryon_libjxl::encode_jpeg(self.source().data(), options.effort) {
                Ok(data) => {
                    return Ok(EncodedJxl {
                        data,
                        fell_back: false,
                    });
                }
                Err(err) if err.is_untranscodable_jpeg() => {
                    let Some(quality) = options.jpeg_fallback else {
                        return Err(Error::encode(
                            Format::JpegXl,
                            format!(
                                "libjxl cannot transcode this JPEG losslessly: {err}. Keep it \
                                 as JPEG, or set jpeg_fallback to encode its pixels instead"
                            ),
                        ));
                    };
                    let fallback = JxlOptions {
                        quality,
                        jpeg_fallback: None,
                        ..*options
                    };
                    return Ok(EncodedJxl {
                        data: encode_pixels(&self.execute()?, &fallback)?,
                        fell_back: true,
                    });
                }
                Err(err) => return Err(Error::encode(Format::JpegXl, err)),
            }
        }
        Ok(EncodedJxl {
            data: encode_pixels(&self.execute()?, options)?,
            fell_back: false,
        })
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
    options.validate()?;
    let quality = match options.quality {
        JxlQuality::Lossless => baryon_libjxl::Quality::Lossless,
        JxlQuality::Distance(distance) => baryon_libjxl::Quality::Distance(distance),
    };
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
    const UNTRANSCODABLE: &[u8] = include_bytes!("../../tests/fixtures/untranscodable.jpg");

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
    fn untranscodable_jpegs_fail_without_a_fallback() {
        let image = LazyImage::from_bytes(UNTRANSCODABLE).unwrap();
        let err = image.to_jxl(&JxlOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Encode { .. }), "{err:?}");
        let message = err.to_string();
        assert!(message.contains("cannot transcode this JPEG"), "{message}");
        assert_eq!(message.matches("cannot transcode").count(), 1, "{message}");
    }

    #[test]
    fn untranscodable_jpegs_fall_back_to_pixels() {
        let image = LazyImage::from_bytes(UNTRANSCODABLE).unwrap();
        let pixels = image.execute().unwrap();

        let lossless = JxlOptions {
            jpeg_fallback: Some(JxlQuality::Lossless),
            ..JxlOptions::default()
        };
        let encoded = image.encode_jxl(&lossless).unwrap();
        assert!(encoded.fell_back);
        let decoded = LazyImage::from_bytes(encoded.data)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(decoded, pixels);

        let lossy = JxlOptions {
            jpeg_fallback: Some(JxlQuality::Distance(1.0)),
            ..JxlOptions::default()
        };
        let encoded = image.encode_jxl(&lossy).unwrap();
        assert!(encoded.fell_back);
        assert!(jxl_to_jpeg(&encoded.data).is_err());

        // Transcodable JPEGs and other sources ignore the fallback.
        for source in [jpeg(96, 64), ADAM7.to_vec()] {
            let encoded = LazyImage::from_bytes(source)
                .unwrap()
                .encode_jxl(&lossy)
                .unwrap();
            assert!(!encoded.fell_back);
        }
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
            JxlOptions {
                jpeg_fallback: Some(JxlQuality::Distance(-1.0)),
                ..JxlOptions::default()
            },
            JxlOptions {
                quality: JxlQuality::Distance(1.0),
                jpeg_fallback: Some(JxlQuality::Distance(1.0)),
                ..JxlOptions::default()
            },
            JxlOptions {
                transcode_jpeg: false,
                jpeg_fallback: Some(JxlQuality::Lossless),
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
