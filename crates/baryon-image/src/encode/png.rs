//! PNG encoding with the `png` crate, whose `fdeflate` backend makes the fast levels very fast.

use baryon_core::{Error, Format, Result};

use crate::image::{ColorType, Image, PixelBuffer};

/// How hard the PNG encoder compresses. Every level is lossless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PngCompression {
    /// Several times faster than `Balanced`, with somewhat larger files.
    Fast,
    /// A good default.
    #[default]
    Balanced,
    /// Slower, for slightly smaller files.
    High,
}

/// Encodes an image as PNG. 16-bit images stay 16-bit.
pub fn encode_png(image: &Image, compression: PngCompression) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, image.width(), image.height());
    encoder.set_color(match image.color() {
        ColorType::L => png::ColorType::Grayscale,
        ColorType::La => png::ColorType::GrayscaleAlpha,
        ColorType::Rgb => png::ColorType::Rgb,
        ColorType::Rgba => png::ColorType::Rgba,
    });
    encoder.set_compression(match compression {
        PngCompression::Fast => png::Compression::Fast,
        PngCompression::Balanced => png::Compression::Balanced,
        PngCompression::High => png::Compression::High,
    });
    let big_endian;
    let data = match image.pixels() {
        PixelBuffer::U8(samples) => {
            encoder.set_depth(png::BitDepth::Eight);
            samples.as_slice()
        }
        PixelBuffer::U16(samples) => {
            encoder.set_depth(png::BitDepth::Sixteen);
            // PNG stores 16-bit samples big-endian.
            big_endian = samples
                .iter()
                .flat_map(|v| v.to_be_bytes())
                .collect::<Vec<u8>>();
            big_endian.as_slice()
        }
    };
    let mut writer = encoder.write_header().map_err(png_error)?;
    writer.write_image_data(data).map_err(png_error)?;
    writer.finish().map_err(png_error)?;
    Ok(out)
}

fn png_error(err: png::EncodingError) -> Error {
    Error::encode(Format::Png, err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::LazyImage;

    const ADAM7: &[u8] = include_bytes!("../../tests/fixtures/gradient-adam7.png");
    const RGBA16: &[u8] = include_bytes!("../../tests/fixtures/gradient-rgba16-lossless.jxl");
    const GRAY8: &[u8] = include_bytes!("../../tests/fixtures/gradient-gray8-lossless.jxl");

    #[test]
    fn round_trips_exactly_at_every_level() {
        for source in [ADAM7, RGBA16, GRAY8] {
            let image = LazyImage::from_bytes(source).unwrap().execute().unwrap();
            for level in [
                PngCompression::Fast,
                PngCompression::Balanced,
                PngCompression::High,
            ] {
                let png = encode_png(&image, level).unwrap();
                let decoded = LazyImage::from_bytes(png).unwrap().execute().unwrap();
                assert_eq!(decoded, image, "{level:?}");
            }
        }
    }
}
