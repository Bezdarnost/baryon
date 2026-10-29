//! End-to-end checks that the optimized physical plan is indistinguishable from running every
//! recorded op naively, one after another, on a fully decoded image.

use baryon_image::{
    ColorType, Image, LazyImage, Op, PixelBuffer, Rect, ResizeFilter, Rotation, ops,
};

/// A 16-bit grey+alpha PNG whose pixel `(x, y)` stores `[x, y]`, so every output pixel says
/// exactly which source pixel it came from.
fn coordinate_png(width: u32, height: u32) -> Vec<u8> {
    oriented_coordinate_png(width, height, None)
}

/// [`coordinate_png`] with an `eXIf` chunk holding the given EXIF orientation.
fn oriented_coordinate_png(width: u32, height: u32, exif_orientation: Option<u16>) -> Vec<u8> {
    let samples = (0..height).flat_map(|y| (0..width).flat_map(move |x| [x, y]));
    sixteen_bit_png(
        width,
        height,
        png::ColorType::GrayscaleAlpha,
        samples,
        exif_orientation,
    )
}

/// A 16-bit RGBA PNG whose pixel `(x, y)` is `[x, y, x * y, 1000 + x + y]`: distinct colour
/// channels, so conversions to gray compute real luma values.
fn colour_png(width: u32, height: u32, exif_orientation: Option<u16>) -> Vec<u8> {
    let samples =
        (0..height).flat_map(|y| (0..width).flat_map(move |x| [x, y, x * y, 1000 + x + y]));
    sixteen_bit_png(
        width,
        height,
        png::ColorType::Rgba,
        samples,
        exif_orientation,
    )
}

fn sixteen_bit_png(
    width: u32,
    height: u32,
    color: png::ColorType,
    samples: impl Iterator<Item = u32>,
    exif_orientation: Option<u16>,
) -> Vec<u8> {
    let samples: Vec<u8> = samples
        .flat_map(|v| u16::try_from(v).unwrap().to_be_bytes())
        .collect();
    let mut info = png::Info::with_size(width, height);
    info.color_type = color;
    info.bit_depth = png::BitDepth::Sixteen;
    info.exif_metadata = exif_orientation.map(|value| exif_block(value).into());
    let mut out = Vec::new();
    png::Encoder::with_info(&mut out, info)
        .unwrap()
        .write_header()
        .unwrap()
        .write_image_data(&samples)
        .unwrap();
    out
}

/// A little-endian EXIF (TIFF) block whose IFD0 holds only the orientation tag.
fn exif_block(orientation: u16) -> Vec<u8> {
    let mut out = b"II".to_vec();
    out.extend(42_u16.to_le_bytes());
    out.extend(8_u32.to_le_bytes());
    out.extend(1_u16.to_le_bytes());
    out.extend(0x0112_u16.to_le_bytes());
    out.extend(3_u16.to_le_bytes());
    out.extend(1_u32.to_le_bytes());
    out.extend([orientation.to_le_bytes(), [0, 0]].concat());
    out.extend(0_u32.to_le_bytes());
    out
}

/// The ops that turn a stored image into the displayed one for an EXIF orientation value,
/// spelled out from the EXIF specification.
fn exif_ops(orientation: u16) -> Vec<Op> {
    use Rotation::{Cw90, Cw180, Cw270};
    match orientation {
        1 => vec![],
        2 => vec![Op::FlipHorizontal],
        3 => vec![Op::Rotate(Cw180)],
        4 => vec![Op::FlipVertical],
        5 => vec![Op::FlipHorizontal, Op::Rotate(Cw270)],
        6 => vec![Op::Rotate(Cw90)],
        7 => vec![Op::FlipHorizontal, Op::Rotate(Cw90)],
        8 => vec![Op::Rotate(Cw270)],
        _ => unreachable!("EXIF orientations are 1 to 8"),
    }
}

/// Deliberately simple reference implementations, independent of the optimizer's group algebra.
fn naive(image: &Image, op: Op) -> Image {
    let (w, h) = (image.width(), image.height());
    let channels = image.color().channels();
    let remap = |ow: u32, oh: u32, src: &dyn Fn(u32, u32) -> (u32, u32)| {
        let PixelBuffer::U16(px) = image.pixels() else {
            unreachable!("test images are 16-bit")
        };
        let mut out = Vec::with_capacity(ow as usize * oh as usize * channels);
        for y in 0..oh {
            for x in 0..ow {
                let (sx, sy) = src(x, y);
                let idx = (sy * w + sx) as usize * channels;
                out.extend_from_slice(&px[idx..idx + channels]);
            }
        }
        Image::new(ow, oh, image.color(), PixelBuffer::U16(out)).unwrap()
    };
    match op {
        Op::Crop(rect) => ops::crop(image, rect).unwrap(),
        Op::FlipHorizontal => remap(w, h, &|x, y| (w - 1 - x, y)),
        Op::FlipVertical => remap(w, h, &|x, y| (x, h - 1 - y)),
        Op::Rotate(Rotation::Cw90) => remap(h, w, &|x, y| (y, h - 1 - x)),
        Op::Rotate(Rotation::Cw180) => remap(w, h, &|x, y| (w - 1 - x, h - 1 - y)),
        Op::Rotate(Rotation::Cw270) => remap(h, w, &|x, y| (w - 1 - y, x)),
        Op::Resize {
            width,
            height,
            filter,
        } => ops::resize(image, width, height, filter).unwrap(),
        Op::Convert(color) => ops::convert(image, color).unwrap(),
    }
}

/// Small deterministic PRNG so failures are reproducible without extra dependencies.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform-ish value in `lo..=hi`.
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + u32::try_from(self.next() % u64::from(hi - lo + 1)).unwrap()
    }
}

fn random_op(rng: &mut XorShift, image: &LazyImage, allow_resize: bool) -> LazyImage {
    let (w, h) = (image.width(), image.height());
    match rng.range(0, if allow_resize { 6 } else { 5 }) {
        0 if w > 1 && h > 1 => {
            let cw = rng.range(1, w);
            let ch = rng.range(1, h);
            let rect = Rect::new(rng.range(0, w - cw), rng.range(0, h - ch), cw, ch);
            image.crop(rect).unwrap()
        }
        1 => image.flip_horizontal(),
        2 => image.flip_vertical(),
        3 => image
            .rotate([Rotation::Cw90, Rotation::Cw180, Rotation::Cw270][rng.range(0, 2) as usize]),
        4 => image.flip_horizontal().rotate(Rotation::Cw90),
        5 => image.convert(ColorType::ALL[rng.range(0, 3) as usize]),
        _ => image
            .resize(
                rng.range(1, 2 * w),
                rng.range(1, 2 * h),
                ResizeFilter::Bilinear,
            )
            .unwrap(),
    }
}

fn check_random_pipelines(allow_resize: bool, seed: u64, exif_orientation: u16, colour: bool) {
    let png = if colour {
        colour_png
    } else {
        oriented_coordinate_png
    };
    let source = LazyImage::from_bytes(png(23, 17, Some(exif_orientation))).unwrap();
    let stored = LazyImage::from_bytes(png(23, 17, None))
        .unwrap()
        .execute()
        .unwrap();
    let mut rng = XorShift(seed);

    for case in 0..200 {
        let mut image = source.clone();
        for _ in 0..rng.range(0, 8) {
            image = random_op(&mut rng, &image, allow_resize);
        }
        let expected = exif_ops(exif_orientation)
            .iter()
            .chain(image.ops())
            .fold(stored.clone(), |img, &op| naive(&img, op));
        let actual = image.execute().unwrap();
        assert_eq!(
            (actual.width(), actual.height()),
            (image.width(), image.height()),
            "case {case}: declared size disagrees with executed size\n{}",
            image.explain(false)
        );
        assert!(
            actual == expected,
            "case {case}: optimized result differs\nlogical:\n{}\nphysical:\n{}",
            image.explain(false),
            image.explain(true)
        );
    }
}

#[test]
fn geometric_pipelines_match_naive_execution() {
    for orientation in 1..=8 {
        check_random_pipelines(false, 0x5EED_0001, orientation, false);
    }
    check_random_pipelines(false, 0x5EED_0003, 7, true);
}

/// Large enough that crops and orientations are split across threads.
#[test]
fn parallel_geometry_matches_naive_execution() {
    let source = LazyImage::from_bytes(coordinate_png(700, 500)).unwrap();
    let full = source.execute().unwrap();
    let pipelines = [
        source.rotate(Rotation::Cw90),
        source.flip_horizontal().rotate(Rotation::Cw270),
        source
            .resize(350, 250, ResizeFilter::Bilinear)
            .unwrap()
            .flip_vertical(),
        source
            .resize(640, 480, ResizeFilter::Nearest)
            .unwrap()
            .crop(Rect::new(10, 20, 600, 400))
            .unwrap()
            .rotate(Rotation::Cw180),
    ];
    for image in pipelines {
        let expected = image
            .ops()
            .iter()
            .fold(full.clone(), |img, &op| naive(&img, op));
        assert!(
            image.execute().unwrap() == expected,
            "{}",
            image.explain(false)
        );
    }
}

#[test]
fn pipelines_with_resize_match_naive_execution() {
    for orientation in 1..=8 {
        check_random_pipelines(true, 0x5EED_0002, orientation, false);
    }
    check_random_pipelines(true, 0x5EED_0004, 1, true);
    check_random_pipelines(true, 0x5EED_0005, 6, true);
}

#[test]
fn exif_orientation_is_applied_and_reported() {
    let image = LazyImage::from_bytes(oriented_coordinate_png(64, 48, Some(6))).unwrap();
    assert_eq!((image.width(), image.height()), (48, 64));
    assert!(image.explain(false).contains("from EXIF"));
    // Cropping the top-left of the displayed image reads the bottom-left of the stored one,
    // and the decoder is asked for just that region.
    let corner = image.crop(Rect::new(0, 0, 10, 5)).unwrap();
    assert_eq!(corner.plan().region(), Rect::new(0, 38, 5, 10));
    assert_eq!(
        LazyImage::from_bytes(oriented_coordinate_png(8, 8, Some(1)))
            .unwrap()
            .plan()
            .steps(),
        []
    );
}

#[test]
fn leading_crop_is_pushed_into_the_decoder() {
    let image = LazyImage::from_bytes(coordinate_png(64, 48))
        .unwrap()
        .rotate(Rotation::Cw90)
        .crop(Rect::new(0, 0, 10, 20))
        .unwrap()
        .flip_horizontal();
    let plan = image.plan();
    // Output rows 0..20 of the rotated image are source columns 0..20, read bottom-up; output
    // columns 0..10 are the bottom 10 source rows.
    assert_eq!(plan.region(), Rect::new(0, 38, 20, 10));
    assert_eq!(
        plan.steps().len(),
        1,
        "flips and rotation fuse into one step"
    );
    assert_eq!(
        plan.strategy(),
        baryon_image::RegionStrategy::RowStreaming { rows_decoded: 48 }
    );
}

#[test]
fn identity_ops_produce_no_steps() {
    let image = LazyImage::from_bytes(coordinate_png(8, 8))
        .unwrap()
        .flip_horizontal()
        .flip_horizontal()
        .rotate(Rotation::Cw180)
        .rotate(Rotation::Cw180)
        .resize(8, 8, ResizeFilter::Lanczos3)
        .unwrap();
    let plan = image.plan();
    assert!(plan.steps().is_empty(), "{plan}");
    assert_eq!(
        image.execute().unwrap(),
        LazyImage::from_bytes(coordinate_png(8, 8))
            .unwrap()
            .execute()
            .unwrap()
    );
}

#[test]
fn invalid_arguments_are_rejected_eagerly() {
    let image = LazyImage::from_bytes(coordinate_png(8, 8)).unwrap();
    assert!(image.crop(Rect::new(4, 4, 5, 1)).is_err());
    assert!(image.crop(Rect::new(0, 0, 0, 1)).is_err());
    assert!(image.resize(0, 4, ResizeFilter::Nearest).is_err());
    assert!(LazyImage::from_bytes(b"not an image".to_vec()).is_err());
}

#[test]
fn conversions_fuse_and_move_to_the_cheaper_side() {
    let image = LazyImage::from_bytes(colour_png(40, 30, None))
        .unwrap()
        .convert(ColorType::Rgb)
        .rotate(Rotation::Cw90)
        .convert(ColorType::L);
    let plan = image.plan();
    assert_eq!(plan.steps().len(), 2, "{plan}");
    assert!(
        matches!(plan.steps()[0], baryon_image::Step::Convert(c) if c.to() == ColorType::L),
        "the shrinking conversion runs before the orientation pass:\n{plan}"
    );

    let grown = LazyImage::from_bytes(coordinate_png(40, 30))
        .unwrap()
        .convert(ColorType::Rgba)
        .flip_vertical();
    let plan = grown.plan();
    assert!(
        matches!(plan.steps().last(), Some(baryon_image::Step::Convert(_))),
        "the growing conversion runs after the orientation pass:\n{plan}"
    );
    assert_eq!(grown.color(), ColorType::Rgba);
}

#[test]
fn gray_jpegs_are_decoded_as_luma() {
    let (width, height) = (64_u16, 48_u16);
    let pixels: Vec<u8> = (0..u32::from(height))
        .flat_map(|y| (0..u32::from(width)).flat_map(move |x| [x * 4, y * 5, (x ^ y) * 3]))
        .map(|v| u8::try_from(v % 256).unwrap())
        .collect();
    let mut jpeg = Vec::new();
    jpeg_encoder::Encoder::new(&mut jpeg, 95)
        .encode(&pixels, width, height, jpeg_encoder::ColorType::Rgb)
        .unwrap();

    let source = LazyImage::from_bytes(jpeg).unwrap();
    let gray = source
        .convert(ColorType::L)
        .crop(Rect::new(4, 4, 32, 16))
        .unwrap();
    let plan = gray.plan();
    assert_eq!(plan.decode_color(), ColorType::L);
    assert!(plan.steps().is_empty(), "{plan}");

    // The JPEG's own luma plane and luma computed from its decoded RGB agree to rounding.
    let direct = gray.execute().unwrap();
    let via_rgb = ops::convert(
        &source
            .crop(Rect::new(4, 4, 32, 16))
            .unwrap()
            .execute()
            .unwrap(),
        ColorType::L,
    )
    .unwrap();
    let worst = direct
        .pixels()
        .as_bytes()
        .iter()
        .zip(via_rgb.pixels().as_bytes())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(
        worst <= 3,
        "luma plane differs from converted RGB by {worst}"
    );
}
