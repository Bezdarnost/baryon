//! Whole-image JPEG XL decoding with `jxl-rs`.
//!
//! jxl-rs decodes full images faster per core than jxl-oxide, but it has no region API, so the
//! JPEG XL codec uses it only when the whole image is needed and keeps jxl-oxide for regions.
//! Its parallel work runs on the current rayon pool, i.e. the executor the decode runs on.

use jxl::api::{
    Endianness, Event, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlParallelRunner, JxlParallelRunnerFun, JxlPixelFormat,
};
use rayon::prelude::*;

use super::ImageInfo;
use crate::image::{ColorType, Image, PixelBuffer, SampleType};
use baryon_core::{Error, Format, Result};

/// Runs jxl-rs's parallel tasks on the current rayon pool.
struct RayonRunner;

impl JxlParallelRunner for RayonRunner {
    fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun<'_>) -> jxl::error::Result<()> {
        (0..num).into_par_iter().try_for_each(fun)
    }

    fn num_threads(&self) -> usize {
        rayon::current_num_threads()
    }
}

/// Decodes the first frame of an unrotated image.
///
/// Returns `Ok(None)` when jxl-rs would deliver a different colour space than jxl-oxide (lossy
/// images with an embedded ICC profile, which jxl-rs converts to sRGB), so that full decodes
/// and region decodes always agree; the caller then falls back to jxl-oxide.
pub(super) fn decode(data: &[u8], info: &ImageInfo) -> Result<Option<Image>> {
    let width = info.width as usize;
    let height = info.height as usize;
    let row_bytes = width * info.color.channels() * info.sample_type.size();

    let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());
    let mut input = data;
    let mut pixels: Option<PixelBuffer> = None;
    loop {
        let mut outputs;
        let buffers = match pixels.as_mut() {
            Some(buffer) => {
                outputs = [JxlOutputBuffer::new(
                    buffer.as_bytes_mut(),
                    height,
                    row_bytes,
                )];
                Some(&mut outputs[..])
            }
            None => None,
        };
        match decoder
            .process(&mut input, buffers, Some(&mut RayonRunner))
            .map_err(jxl_rs_error)?
        {
            Event::BasicInfo => {
                let basic = decoder
                    .basic_info()
                    .ok_or_else(|| Error::Internal("jxl-rs reported no basic info".into()))?;
                if basic.size != (width, height) {
                    return Err(Error::Internal(format!(
                        "jxl-rs reports {}x{}, header parsing reported {width}x{height}",
                        basic.size.0, basic.size.1
                    )));
                }
                let extra_channels = basic.extra_channels.len();
                decoder
                    .set_pixel_format(pixel_format(info, extra_channels))
                    .map_err(jxl_rs_error)?;
                if decoder.output_color_profile() != decoder.embedded_color_profile() {
                    return Ok(None);
                }
                let len = crate::image::sample_count(info.width, info.height, info.color)?;
                pixels = Some(PixelBuffer::zeroed(info.sample_type, len));
            }
            Event::FrameComplete { .. } | Event::Complete => break,
            Event::NeedMoreInput { .. } if input.is_empty() => {
                return Err(Error::decode(
                    Format::JpegXl,
                    "the first frame is incomplete",
                ));
            }
            Event::NeedMoreInput { .. } | Event::FrameHeader => {}
        }
    }

    let pixels = pixels.ok_or_else(|| {
        Error::decode(
            Format::JpegXl,
            "the image ended before any frame was decoded",
        )
    })?;
    Image::new(info.width, info.height, info.color, pixels).map(Some)
}

/// Interleaved output in the layout baryon reports for the image; alpha is part of the
/// interleaved color buffer and other extra channels are skipped.
fn pixel_format(info: &ImageInfo, extra_channels: usize) -> JxlPixelFormat {
    let color_type = match info.color {
        ColorType::L => JxlColorType::Grayscale,
        ColorType::La => JxlColorType::GrayscaleAlpha,
        ColorType::Rgb => JxlColorType::Rgb,
        ColorType::Rgba => JxlColorType::Rgba,
    };
    let data_format = match info.sample_type {
        SampleType::U8 => JxlDataFormat::U8 { bit_depth: 8 },
        SampleType::U16 => JxlDataFormat::U16 {
            endianness: Endianness::native(),
            bit_depth: 16,
        },
    };
    JxlPixelFormat {
        color_type,
        color_data_format: Some(data_format),
        extra_channel_format: vec![None; extra_channels],
    }
}

fn jxl_rs_error(err: jxl::error::Error) -> Error {
    Error::decode(Format::JpegXl, err)
}
