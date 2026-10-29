//! Logical transformations and their eager, in-memory implementations.

use std::fmt;
use std::str::FromStr;

use fast_image_resize::{self as fir, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use rayon::prelude::*;

use crate::geometry::{Orientation, Rect, Rotation};
use crate::image::{ColorType, Image, PixelBuffer, SampleType};
use baryon_core::{Error, Result};

/// A single transformation recorded in a lazy pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Keep only the given rectangle.
    Crop(Rect),
    /// Mirror left to right.
    FlipHorizontal,
    /// Mirror top to bottom.
    FlipVertical,
    /// Rotate clockwise by a multiple of 90 degrees.
    Rotate(Rotation),
    /// Resample to new dimensions.
    Resize {
        /// Target width.
        width: u32,
        /// Target height.
        height: u32,
        /// Resampling filter.
        filter: ResizeFilter,
    },
    /// Change the channel layout; see [`ColorConversion`].
    Convert(ColorType),
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Crop(rect) => write!(f, "crop {rect}"),
            Self::FlipHorizontal => f.write_str("flip horizontal"),
            Self::FlipVertical => f.write_str("flip vertical"),
            Self::Rotate(r) => write!(f, "rotate {r}"),
            Self::Resize {
                width,
                height,
                filter,
            } => write!(f, "resize {width}x{height} ({filter})"),
            Self::Convert(color) => write!(f, "convert {color}"),
        }
    }
}

/// A change of channel layout, possibly the composition of several [`Op::Convert`]s.
///
/// * Colour to gray computes luma with the ITU-R BT.601 weights (0.299, 0.587, 0.114), as
///   JPEG and Pillow do; gray to colour copies the value into every channel.
/// * Dropping alpha discards it without blending; adding alpha makes every pixel opaque.
///
/// Composition remembers what was lost on the way: converting RGBA to RGB and back to RGBA
/// yields an opaque image, and RGB to L to RGB yields a gray one, exactly as running the
/// conversions one by one would.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorConversion {
    from: ColorType,
    to: ColorType,
    /// A conversion to gray happened along the way.
    gray: bool,
    /// A conversion without alpha happened along the way.
    alpha_lost: bool,
}

impl ColorConversion {
    /// The conversion that changes nothing.
    pub const fn identity(color: ColorType) -> Self {
        Self {
            from: color,
            to: color,
            gray: false,
            alpha_lost: false,
        }
    }

    /// This conversion followed by a conversion to `to`.
    #[must_use]
    pub const fn then(self, to: ColorType) -> Self {
        Self {
            from: self.from,
            to,
            gray: self.gray || to.is_gray(),
            alpha_lost: self.alpha_lost || !to.has_alpha(),
        }
    }

    /// Input layout.
    pub const fn from(self) -> ColorType {
        self.from
    }

    /// Output layout.
    pub const fn to(self) -> ColorType {
        self.to
    }

    /// Whether colour information is replaced by luma.
    pub const fn collapses_to_gray(self) -> bool {
        self.gray && !self.from.is_gray()
    }

    /// Whether the output alpha channel, if any, comes from the input.
    pub const fn keeps_alpha(self) -> bool {
        self.from.has_alpha() && !self.alpha_lost
    }

    /// Whether pixels are left unchanged.
    pub const fn is_identity(self) -> bool {
        self.to as u8 == self.from as u8
            && !self.collapses_to_gray()
            && (self.keeps_alpha() || !self.from.has_alpha())
    }

    /// Whether the output has fewer channels than the input, so the conversion is best run
    /// before other work rather than after it.
    pub const fn shrinks(self) -> bool {
        self.to.channels() < self.from.channels()
    }

    /// The same conversion applied to the luma plane of a colour input instead, which the
    /// caller guarantees is what this conversion would compute: used when a decoder can
    /// produce luma directly.
    pub(crate) const fn with_luma_input(self) -> Self {
        Self {
            from: ColorType::L,
            to: self.to,
            gray: true,
            alpha_lost: self.alpha_lost,
        }
    }
}

impl fmt::Display for ColorConversion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.from, self.to)?;
        if self.collapses_to_gray() && !self.to.is_gray() {
            f.write_str(" (gray)")?;
        }
        if self.to.has_alpha() && !self.keeps_alpha() {
            f.write_str(" (opaque)")?;
        }
        Ok(())
    }
}

/// Resampling filter used by [`Op::Resize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ResizeFilter {
    /// Nearest neighbour; fastest, blocky.
    Nearest,
    /// Box (area averaging).
    Box,
    /// Bilinear (triangle) filter.
    Bilinear,
    /// Hamming filter; bilinear speed with sharper downscaling.
    Hamming,
    /// Catmull-Rom bicubic.
    Bicubic,
    /// Mitchell-Netravali bicubic.
    Mitchell,
    /// Lanczos with a 3-lobe window; highest quality.
    #[default]
    Lanczos3,
}

impl ResizeFilter {
    /// Every filter, in order of increasing cost.
    pub const ALL: [Self; 7] = [
        Self::Nearest,
        Self::Box,
        Self::Bilinear,
        Self::Hamming,
        Self::Bicubic,
        Self::Mitchell,
        Self::Lanczos3,
    ];

    /// Canonical lowercase name, e.g. `"lanczos3"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Box => "box",
            Self::Bilinear => "bilinear",
            Self::Hamming => "hamming",
            Self::Bicubic => "bicubic",
            Self::Mitchell => "mitchell",
            Self::Lanczos3 => "lanczos3",
        }
    }

    const fn algorithm(self) -> ResizeAlg {
        let filter = match self {
            Self::Nearest => return ResizeAlg::Nearest,
            Self::Box => FilterType::Box,
            Self::Bilinear => FilterType::Bilinear,
            Self::Hamming => FilterType::Hamming,
            Self::Bicubic => FilterType::CatmullRom,
            Self::Mitchell => FilterType::Mitchell,
            Self::Lanczos3 => FilterType::Lanczos3,
        };
        ResizeAlg::Convolution(filter)
    }
}

impl fmt::Display for ResizeFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for ResizeFilter {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|f| f.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| {
                let names: Vec<_> = Self::ALL.iter().map(|f| f.name()).collect();
                Error::InvalidArgument(format!(
                    "unknown resize filter {s:?}; expected one of {}",
                    names.join(", ")
                ))
            })
    }
}

/// Copies `rect` out of `image`. The rectangle must lie inside the image.
pub fn crop(image: &Image, rect: Rect) -> Result<Image> {
    if !Rect::from_size(image.width(), image.height()).contains(&rect) {
        return Err(Error::InvalidArgument(format!(
            "crop {rect} exceeds image bounds {}x{}",
            image.width(),
            image.height()
        )));
    }
    let bpp = image.bytes_per_pixel();
    let src_stride = image.row_stride();
    let row_len = rect.width as usize * bpp;
    let x_offset = rect.x as usize * bpp;

    let mut out = Image::zeroed(rect.width, rect.height, image.color(), image.sample_type())?;
    let src = image.pixels().as_bytes();
    let dst = out.pixels_mut().as_bytes_mut();
    dst.par_chunks_exact_mut(row_len)
        .with_min_len(rows_per_task(row_len))
        .enumerate()
        .for_each(|(row, dst_row)| {
            let start = (rect.y as usize + row) * src_stride + x_offset;
            dst_row.copy_from_slice(&src[start..start + row_len]);
        });
    Ok(out)
}

/// Applies a flip/rotation combination in a single pass.
pub fn orient(image: Image, orientation: Orientation) -> Result<Image> {
    if orientation.is_identity() {
        return Ok(image);
    }
    let (sw, sh) = (image.width(), image.height());
    let (ow, oh) = orientation.output_size(sw, sh);
    let bpp = image.bytes_per_pixel();
    let src_stride = image.row_stride();

    let mut out = Image::zeroed(ow, oh, image.color(), image.sample_type())?;
    let src = image.pixels().as_bytes();
    let dst_stride = out.row_stride();
    let dst = out.pixels_mut().as_bytes_mut();
    (0..oh)
        .into_par_iter()
        .zip(dst.par_chunks_exact_mut(dst_stride))
        .with_min_len(rows_per_task(dst_stride))
        .for_each(|(y, dst_row)| {
            for (x, dst_px) in (0..ow).zip(dst_row.chunks_exact_mut(bpp)) {
                let (u, v) = orientation.source_pixel(x, y, sw, sh);
                let start = v as usize * src_stride + u as usize * bpp;
                dst_px.copy_from_slice(&src[start..start + bpp]);
            }
        });
    Ok(out)
}

/// Converts `image` to `to`; see [`ColorConversion`].
pub fn convert(image: &Image, to: ColorType) -> Result<Image> {
    convert_color(image, ColorConversion::identity(image.color()).then(to))
}

/// Applies a composed conversion, whose input must match the image.
pub fn convert_color(image: &Image, conversion: ColorConversion) -> Result<Image> {
    if conversion.from() != image.color() {
        return Err(Error::Internal(format!(
            "conversion {conversion} applied to a {} image",
            image.color()
        )));
    }
    if conversion.is_identity() {
        return Ok(image.clone());
    }
    let mut out = Image::zeroed(
        image.width(),
        image.height(),
        conversion.to(),
        image.sample_type(),
    )?;
    let pixels = image.width() as usize;
    match (image.pixels(), out.pixels_mut()) {
        (PixelBuffer::U8(src), PixelBuffer::U8(dst)) => {
            convert_samples(src, dst, pixels, conversion);
        }
        (PixelBuffer::U16(src), PixelBuffer::U16(dst)) => {
            convert_samples(src, dst, pixels, conversion);
        }
        _ => return Err(Error::Internal("sample types differ".into())),
    }
    Ok(out)
}

/// An 8- or 16-bit sample.
trait Sample: Copy + Send + Sync + Into<u64> {
    const MAX: Self;
    fn from_u64(value: u64) -> Self;
}

impl Sample for u8 {
    const MAX: Self = Self::MAX;
    fn from_u64(value: u64) -> Self {
        Self::try_from(value).unwrap_or(Self::MAX)
    }
}

impl Sample for u16 {
    const MAX: Self = Self::MAX;
    fn from_u64(value: u64) -> Self {
        Self::try_from(value).unwrap_or(Self::MAX)
    }
}

/// BT.601 luma in 16-bit fixed point; the weights sum to 65536, so white stays white.
fn luma<T: Sample>(r: T, g: T, b: T) -> T {
    T::from_u64((19_595 * r.into() + 38_470 * g.into() + 7_471 * b.into() + 32_768) >> 16)
}

/// Converts rows of `row_pixels` pixels in parallel.
fn convert_samples<T: Sample>(
    src: &[T],
    dst: &mut [T],
    row_pixels: usize,
    conversion: ColorConversion,
) {
    let (from, to) = (conversion.from(), conversion.to());
    let (src_channels, dst_channels) = (from.channels(), to.channels());
    let gray = conversion.collapses_to_gray();
    let keep_alpha = conversion.keeps_alpha();
    let dst_row = row_pixels * dst_channels;
    dst.par_chunks_exact_mut(dst_row)
        .zip(src.par_chunks_exact(row_pixels * src_channels))
        .with_min_len(rows_per_task(dst_row * size_of::<T>()))
        .for_each(|(dst_row, src_row)| {
            for (out, px) in dst_row
                .chunks_exact_mut(dst_channels)
                .zip(src_row.chunks_exact(src_channels))
            {
                let (r, g, b) = if from.is_gray() {
                    (px[0], px[0], px[0])
                } else {
                    (px[0], px[1], px[2])
                };
                let alpha = if keep_alpha {
                    px[src_channels - 1]
                } else {
                    T::MAX
                };
                let value = if gray { luma(r, g, b) } else { r };
                match to {
                    ColorType::L => out[0] = value,
                    ColorType::La => {
                        out[0] = value;
                        out[1] = alpha;
                    }
                    ColorType::Rgb | ColorType::Rgba => {
                        if gray {
                            out[..3].fill(value);
                        } else {
                            out[..3].copy_from_slice(&[r, g, b]);
                        }
                        if to == ColorType::Rgba {
                            out[3] = alpha;
                        }
                    }
                }
            }
        });
}

/// Rows handed to one parallel task: enough bytes that scheduling overhead stays negligible,
/// so small images are processed on a single thread.
fn rows_per_task(row_bytes: usize) -> usize {
    const BYTES_PER_TASK: usize = 256 * 1024;
    (BYTES_PER_TASK / row_bytes.max(1)).max(1)
}

/// Resamples `image` to `width` x `height`. Alpha channels are handled premultiplied.
pub fn resize(image: &Image, width: u32, height: u32, filter: ResizeFilter) -> Result<Image> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidArgument(format!(
            "resize target must be non-empty, got {width}x{height}"
        )));
    }
    if (width, height) == (image.width(), image.height()) {
        return Ok(image.clone());
    }

    let pixel_type = pixel_type(image.color(), image.sample_type());
    let src = fir::images::ImageRef::new(
        image.width(),
        image.height(),
        image.pixels().as_bytes(),
        pixel_type,
    )
    .map_err(resize_error)?;

    let mut out = Image::zeroed(width, height, image.color(), image.sample_type())?;
    let mut dst = fir::images::Image::from_slice_u8(
        width,
        height,
        out.pixels_mut().as_bytes_mut(),
        pixel_type,
    )
    .map_err(resize_error)?;

    let options = ResizeOptions::new()
        .resize_alg(filter.algorithm())
        .use_alpha(image.color().has_alpha());
    Resizer::new()
        .resize(&src, &mut dst, &options)
        .map_err(resize_error)?;
    drop(dst);
    Ok(out)
}

const fn pixel_type(color: ColorType, sample: SampleType) -> PixelType {
    match (sample, color) {
        (SampleType::U8, ColorType::L) => PixelType::U8,
        (SampleType::U8, ColorType::La) => PixelType::U8x2,
        (SampleType::U8, ColorType::Rgb) => PixelType::U8x3,
        (SampleType::U8, ColorType::Rgba) => PixelType::U8x4,
        (SampleType::U16, ColorType::L) => PixelType::U16,
        (SampleType::U16, ColorType::La) => PixelType::U16x2,
        (SampleType::U16, ColorType::Rgb) => PixelType::U16x3,
        (SampleType::U16, ColorType::Rgba) => PixelType::U16x4,
    }
}

fn resize_error(err: impl std::error::Error) -> Error {
    // Buffers are sized and aligned by construction, so any failure here is a bug in baryon.
    Error::Internal(format!("resize failed: {err}"))
}

/// Builds an image whose pixel at `(x, y)` encodes its own coordinates. Test helper.
#[cfg(test)]
#[allow(clippy::cast_possible_truncation, reason = "test images are small")]
pub(crate) fn coordinate_image(width: u32, height: u32) -> Image {
    let pixels = (0..height)
        .flat_map(|y| (0..width).flat_map(move |x| [x as u16, y as u16]))
        .collect();
    Image::new(width, height, ColorType::La, PixelBuffer::U16(pixels)).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(image: &Image, x: u32, y: u32) -> (u16, u16) {
        let PixelBuffer::U16(p) = image.pixels() else {
            unreachable!()
        };
        let i = (y * image.width() + x) as usize * 2;
        (p[i], p[i + 1])
    }

    #[test]
    fn crop_copies_the_requested_window() {
        let img = coordinate_image(6, 4);
        let out = crop(&img, Rect::new(2, 1, 3, 2)).unwrap();
        assert_eq!((out.width(), out.height()), (3, 2));
        assert_eq!(pixel(&out, 0, 0), (2, 1));
        assert_eq!(pixel(&out, 2, 1), (4, 2));
    }

    #[test]
    fn crop_rejects_out_of_bounds() {
        let img = coordinate_image(4, 4);
        assert!(crop(&img, Rect::new(2, 2, 3, 1)).is_err());
        assert!(crop(&img, Rect::new(u32::MAX, 0, 2, 1)).is_err());
    }

    #[test]
    fn orient_rotates_clockwise() {
        let img = coordinate_image(3, 2);
        let out = orient(img, Orientation::IDENTITY.then_rotate(Rotation::Cw90)).unwrap();
        assert_eq!((out.width(), out.height()), (2, 3));
        assert_eq!(pixel(&out, 1, 0), (0, 0));
        assert_eq!(pixel(&out, 0, 0), (0, 1));
        assert_eq!(pixel(&out, 0, 2), (2, 1));
    }

    #[test]
    fn resize_handles_every_layout() {
        for color in [ColorType::L, ColorType::La, ColorType::Rgb, ColorType::Rgba] {
            for sample in [SampleType::U8, SampleType::U16] {
                let img = Image::zeroed(16, 9, color, sample).unwrap();
                for filter in ResizeFilter::ALL {
                    let out = resize(&img, 7, 5, filter).unwrap();
                    assert_eq!((out.width(), out.height()), (7, 5));
                    assert_eq!((out.color(), out.sample_type()), (color, sample));
                }
            }
        }
    }

    #[test]
    fn resize_preserves_constant_images() {
        let img = Image::new(8, 8, ColorType::Rgb, PixelBuffer::U8(vec![77; 8 * 8 * 3])).unwrap();
        let out = resize(&img, 3, 5, ResizeFilter::Lanczos3).unwrap();
        assert!(out.pixels().as_bytes().iter().all(|&v| v == 77));
    }

    fn rgba8(pixels: &[[u8; 4]]) -> Image {
        let width = u32::try_from(pixels.len()).unwrap();
        Image::new(width, 1, ColorType::Rgba, PixelBuffer::U8(pixels.concat())).unwrap()
    }

    #[test]
    fn luma_uses_bt601_weights() {
        let image = rgba8(&[[255, 0, 0, 9], [0, 255, 0, 9], [0, 0, 255, 9], [255; 4]]);
        let gray = convert(&image, ColorType::La).unwrap();
        // 0.299, 0.587 and 0.114 of 255, rounded; alpha is kept.
        assert_eq!(gray.pixels().as_bytes(), [76, 9, 150, 9, 29, 9, 255, 255]);
        let l16 = convert(
            &Image::new(1, 1, ColorType::Rgb, PixelBuffer::U16(vec![65535; 3])).unwrap(),
            ColorType::L,
        )
        .unwrap();
        assert_eq!(l16.pixels(), &PixelBuffer::U16(vec![65535]));
    }

    #[test]
    fn conversions_compose_like_sequential_ones() {
        let image = rgba8(&[[10, 200, 30, 40], [255, 128, 0, 7]]);
        for path in [
            [ColorType::Rgb, ColorType::Rgba, ColorType::La],
            [ColorType::L, ColorType::Rgba, ColorType::Rgb],
            [ColorType::La, ColorType::Rgb, ColorType::Rgba],
            [ColorType::Rgba, ColorType::Rgba, ColorType::Rgba],
        ] {
            let sequential = path
                .iter()
                .fold(image.clone(), |img, &to| convert(&img, to).unwrap());
            let composed = path
                .iter()
                .fold(ColorConversion::identity(ColorType::Rgba), |c, &to| {
                    c.then(to)
                });
            assert_eq!(
                convert_color(&image, composed).unwrap(),
                sequential,
                "{path:?}"
            );
        }
        assert!(
            ColorConversion::identity(ColorType::Rgba)
                .then(ColorType::Rgba)
                .is_identity()
        );
        assert!(
            !ColorConversion::identity(ColorType::Rgba)
                .then(ColorType::Rgb)
                .then(ColorType::Rgba)
                .is_identity()
        );
    }

    #[test]
    fn gray_expands_to_every_channel() {
        let image = Image::new(2, 1, ColorType::L, PixelBuffer::U8(vec![3, 250])).unwrap();
        let rgba = convert(&image, ColorType::Rgba).unwrap();
        assert_eq!(rgba.pixels().as_bytes(), [3, 3, 3, 255, 250, 250, 250, 255]);
    }

    #[test]
    fn filter_names_round_trip() {
        for f in ResizeFilter::ALL {
            assert_eq!(f.name().parse::<ResizeFilter>().unwrap(), f);
        }
        assert!("sinc".parse::<ResizeFilter>().is_err());
    }
}
