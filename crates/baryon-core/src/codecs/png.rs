//! PNG decoding via the `png` crate.
//!
//! Non-interlaced images are decoded row by row: decoding stops right after the last requested
//! row and only the requested columns are copied, so a crop near the top of a large PNG costs a
//! fraction of a full decode. Adam7-interlaced images need every pass and are decoded fully.

use std::io::Cursor;

use png::{BitDepth, Decoder, Limits, Reader, Transformations};

use super::{Codec, Decoded, ImageInfo, RegionStrategy};
use crate::error::{Error, Result};
use crate::format::ImageFormat;
use crate::geometry::Rect;
use crate::image::{ColorType, Image, PixelBuffer, SampleType};

/// PNG and the first frame of APNG.
pub(crate) struct PngCodec;

impl PngCodec {
    fn reader(data: &[u8]) -> Result<Reader<Cursor<&[u8]>>> {
        // Allocation is bounded by the image dimensions, which the caller already knows; the
        // crate's default 64 MiB cap would reject ordinary large photos.
        let mut decoder = Decoder::new_with_limits(Cursor::new(data), Limits { bytes: usize::MAX });
        // Expand palettes, sub-byte depths and tRNS into plain 8/16-bit channels.
        decoder.set_transformations(Transformations::EXPAND);
        decoder.read_info().map_err(png_error)
    }
}

impl Codec for PngCodec {
    fn probe(&self, data: &[u8]) -> Result<ImageInfo> {
        let reader = Self::reader(data)?;
        let (color, depth) = reader.output_color_type();
        let color = match color {
            png::ColorType::Grayscale => ColorType::L,
            png::ColorType::GrayscaleAlpha => ColorType::La,
            png::ColorType::Rgb => ColorType::Rgb,
            png::ColorType::Rgba => ColorType::Rgba,
            png::ColorType::Indexed => {
                return Err(Error::Internal("palette survived EXPAND".into()));
            }
        };
        let sample_type = match depth {
            BitDepth::Sixteen => SampleType::U16,
            _ => SampleType::U8,
        };
        let info = reader.info();
        Ok(ImageInfo {
            format: ImageFormat::Png,
            width: info.width,
            height: info.height,
            color,
            sample_type,
            progressive: info.interlaced,
        })
    }

    fn region_strategy(&self, info: &ImageInfo, region: Rect) -> RegionStrategy {
        if info.progressive {
            RegionStrategy::FullDecode
        } else {
            RegionStrategy::RowStreaming {
                // `region` lies inside a u32-sized image, so its bottom edge fits in a u32.
                rows_decoded: region.y + region.height,
            }
        }
    }

    fn decode(&self, data: &[u8], info: &ImageInfo, region: Rect) -> Result<Decoded> {
        let mut reader = Self::reader(data)?;
        if info.progressive {
            return decode_full(&mut reader, info);
        }

        let bytes_per_sample = info.sample_type.size();
        let bytes_per_pixel = info.color.channels() * bytes_per_sample;
        let column_start = region.x as usize * bytes_per_pixel;
        let column_end = column_start + region.width as usize * bytes_per_pixel;

        let mut out = Image::zeroed(region.width, region.height, info.color, info.sample_type)?;
        let mut rows_written = 0u32;
        for row_index in 0..region.bottom() {
            let row = reader
                .next_row()
                .map_err(png_error)?
                .ok_or_else(|| truncated(row_index))?;
            if row_index < u64::from(region.y) {
                continue;
            }
            let src = &row.data()[column_start..column_end];
            write_row(out.pixels_mut(), rows_written as usize, src);
            rows_written += 1;
        }
        Ok(Decoded { image: out, region })
    }
}

fn decode_full(reader: &mut Reader<Cursor<&[u8]>>, info: &ImageInfo) -> Result<Decoded> {
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| Error::decode(ImageFormat::Png, "image is too large to decode"))?;
    let mut raw = vec![0; size];
    let frame = reader.next_frame(&mut raw).map_err(png_error)?;
    raw.truncate(frame.buffer_size());

    let pixels = match info.sample_type {
        SampleType::U8 => PixelBuffer::U8(raw),
        SampleType::U16 => PixelBuffer::U16(be_to_u16(&raw)),
    };
    let image = Image::new(info.width, info.height, info.color, pixels)?;
    Ok(Decoded {
        image,
        region: info.bounds(),
    })
}

/// Copies one decoded row (big-endian for 16-bit) into row `index` of `dst`.
fn write_row(dst: &mut PixelBuffer, index: usize, src: &[u8]) {
    match dst {
        PixelBuffer::U8(buf) => {
            let start = index * src.len();
            buf[start..start + src.len()].copy_from_slice(src);
        }
        PixelBuffer::U16(buf) => {
            let len = src.len() / 2;
            let start = index * len;
            for (d, s) in buf[start..start + len].iter_mut().zip(src.chunks_exact(2)) {
                *d = u16::from_be_bytes([s[0], s[1]]);
            }
        }
    }
}

fn be_to_u16(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2)
        .map(|s| u16::from_be_bytes([s[0], s[1]]))
        .collect()
}

fn truncated(row: u64) -> Error {
    Error::decode(
        ImageFormat::Png,
        format!("image data ended before row {row}"),
    )
}

fn png_error(err: png::DecodingError) -> Error {
    Error::decode(ImageFormat::Png, err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;

    /// 40x30 RGB8 Adam7 image whose pixel `(x, y)` is `[x, y, x ^ y]`; see `gradient`.
    const ADAM7_GRADIENT: &[u8] = include_bytes!("../../tests/fixtures/gradient-adam7.png");

    fn encode(
        width: u32,
        height: u32,
        color: png::ColorType,
        depth: BitDepth,
        data: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(color);
        encoder.set_depth(depth);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(data)
            .unwrap();
        out
    }

    #[allow(clippy::cast_possible_truncation, reason = "values wrap by design")]
    fn gradient(width: u32, height: u32) -> Vec<u8> {
        (0..height)
            .flat_map(|y| (0..width).flat_map(move |x| [x as u8, y as u8, (x ^ y) as u8]))
            .collect()
    }

    #[test]
    fn region_decode_matches_full_decode_crop() {
        let (w, h) = (40, 30);
        let sequential = encode(w, h, png::ColorType::Rgb, BitDepth::Eight, &gradient(w, h));
        for (data, interlaced) in [(sequential.as_slice(), false), (ADAM7_GRADIENT, true)] {
            let info = PngCodec.probe(data).unwrap();
            assert_eq!(info.progressive, interlaced);
            let full = PngCodec.decode(data, &info, info.bounds()).unwrap();
            assert_eq!(full.image.pixels().as_bytes(), gradient(w, h).as_slice());

            let region = Rect::new(5, 7, 20, 10);
            let part = PngCodec.decode(data, &info, region).unwrap();
            let got = ops::crop(&part.image, part.region.relative(&region)).unwrap();
            assert_eq!(
                got,
                ops::crop(&full.image, region).unwrap(),
                "interlaced={interlaced}"
            );
        }
    }

    #[test]
    fn sixteen_bit_is_preserved_in_native_order() {
        let samples: [u16; 4] = [0x0102, 0xA0B0, 0xFFFF, 0x0000];
        let be: Vec<u8> = samples.iter().flat_map(|s| s.to_be_bytes()).collect();
        let data = encode(2, 2, png::ColorType::Grayscale, BitDepth::Sixteen, &be);
        let info = PngCodec.probe(&data).unwrap();
        assert_eq!(
            (info.color, info.sample_type),
            (ColorType::L, SampleType::U16)
        );
        let decoded = PngCodec.decode(&data, &info, info.bounds()).unwrap();
        assert_eq!(decoded.image.pixels(), &PixelBuffer::U16(samples.to_vec()));
    }

    #[test]
    fn palette_expands_to_rgb() {
        let mut out = Vec::new();
        let mut encoder = png::Encoder::new(&mut out, 2, 1);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_palette(vec![255, 0, 0, 0, 0, 255]);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&[0, 1])
            .unwrap();

        let info = PngCodec.probe(&out).unwrap();
        assert_eq!(info.color, ColorType::Rgb);
        let decoded = PngCodec.decode(&out, &info, info.bounds()).unwrap();
        assert_eq!(decoded.image.pixels().as_bytes(), &[255, 0, 0, 0, 0, 255]);
    }
}
