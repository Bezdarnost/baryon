//! JPEG encoding with `jpeg-encoder`, which uses AVX2 when the CPU has it.

use baryon_core::{Error, Format, Result};

use crate::image::{ColorType, Image, PixelBuffer};

/// Encodes an 8-bit grayscale or RGB image as a baseline JPEG at `quality` (1 to 100).
///
/// Below quality 90 the colour planes are subsampled 2x2 (4:2:0), as most encoders do.
pub fn encode_jpeg(image: &Image, quality: u8) -> Result<Vec<u8>> {
    if !(1..=100).contains(&quality) {
        return Err(Error::InvalidArgument(format!(
            "JPEG quality must be between 1 and 100, got {quality}"
        )));
    }
    let color = match image.color() {
        ColorType::L => jpeg_encoder::ColorType::Luma,
        ColorType::Rgb => jpeg_encoder::ColorType::Rgb,
        ColorType::La | ColorType::Rgba => {
            return Err(Error::InvalidArgument(
                "JPEG cannot store an alpha channel; use PNG or JPEG XL".into(),
            ));
        }
    };
    let PixelBuffer::U8(samples) = image.pixels() else {
        return Err(Error::InvalidArgument(
            "JPEG stores 8-bit samples; use PNG or JPEG XL for 16-bit images".into(),
        ));
    };
    let dimension = |value: u32| {
        u16::try_from(value).map_err(|_| {
            Error::InvalidArgument(format!(
                "JPEG images are at most 65535 pixels wide and high, got {}x{}",
                image.width(),
                image.height()
            ))
        })
    };
    let (width, height) = (dimension(image.width())?, dimension(image.height())?);

    let mut out = Vec::new();
    jpeg_encoder::Encoder::new(&mut out, quality)
        .encode(samples, width, height, color)
        .map_err(|err| Error::encode(Format::Jpeg, err))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::LazyImage;

    const RGB8: &[u8] = include_bytes!("../../tests/fixtures/gradient-rgb8-lossless.jxl");
    const RGBA16: &[u8] = include_bytes!("../../tests/fixtures/gradient-rgba16-lossless.jxl");
    const GRAY8: &[u8] = include_bytes!("../../tests/fixtures/gradient-gray8-lossless.jxl");

    fn max_error(a: &Image, b: &Image) -> u8 {
        a.pixels()
            .as_bytes()
            .iter()
            .zip(b.pixels().as_bytes())
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap()
    }

    #[test]
    fn high_quality_is_close() {
        for source in [RGB8, GRAY8] {
            let image = LazyImage::from_bytes(source).unwrap().execute().unwrap();
            let jpeg = encode_jpeg(&image, 100).unwrap();
            let decoded = LazyImage::from_bytes(jpeg).unwrap().execute().unwrap();
            assert_eq!(decoded.color(), image.color());
            // The fixtures have sharp per-pixel patterns, the hardest case for JPEG.
            assert!(max_error(&decoded, &image) <= 24);
        }
    }

    #[test]
    fn rejects_what_jpeg_cannot_store() {
        let rgba16 = LazyImage::from_bytes(RGBA16).unwrap().execute().unwrap();
        assert!(matches!(
            encode_jpeg(&rgba16, 90),
            Err(Error::InvalidArgument(message)) if message.contains("alpha")
        ));
        let rgb = LazyImage::from_bytes(RGB8).unwrap().execute().unwrap();
        for quality in [0, 101] {
            assert!(matches!(
                encode_jpeg(&rgb, quality),
                Err(Error::InvalidArgument(_))
            ));
        }
    }
}
