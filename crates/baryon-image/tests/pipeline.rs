//! End-to-end checks that the optimized physical plan is indistinguishable from running every
//! recorded op naively, one after another, on a fully decoded image.

use baryon_image::{
    ColorType, Image, LazyImage, Op, PixelBuffer, Rect, ResizeFilter, Rotation, ops,
};

/// A 16-bit grey+alpha PNG whose pixel `(x, y)` stores `[x, y]`, so every output pixel says
/// exactly which source pixel it came from.
fn coordinate_png(width: u32, height: u32) -> Vec<u8> {
    let samples: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).flat_map(move |x| [x, y]))
        .flat_map(|v| u16::try_from(v).unwrap().to_be_bytes())
        .collect();
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, width, height);
    encoder.set_color(png::ColorType::GrayscaleAlpha);
    encoder.set_depth(png::BitDepth::Sixteen);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&samples)
        .unwrap();
    out
}

/// Deliberately simple reference implementations, independent of the optimizer's group algebra.
fn naive(image: &Image, op: Op) -> Image {
    let (w, h) = (image.width(), image.height());
    let remap = |ow: u32, oh: u32, src: &dyn Fn(u32, u32) -> (u32, u32)| {
        let PixelBuffer::U16(px) = image.pixels() else {
            unreachable!("test images are 16-bit")
        };
        let mut out = Vec::with_capacity((ow * oh * 2) as usize);
        for y in 0..oh {
            for x in 0..ow {
                let (sx, sy) = src(x, y);
                let idx = ((sy * w + sx) * 2) as usize;
                out.extend_from_slice(&px[idx..idx + 2]);
            }
        }
        Image::new(ow, oh, ColorType::La, PixelBuffer::U16(out)).unwrap()
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
    match rng.range(0, if allow_resize { 5 } else { 4 }) {
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
        _ => image
            .resize(
                rng.range(1, 2 * w),
                rng.range(1, 2 * h),
                ResizeFilter::Bilinear,
            )
            .unwrap(),
    }
}

fn check_random_pipelines(allow_resize: bool, seed: u64) {
    let source = LazyImage::from_bytes(coordinate_png(23, 17)).unwrap();
    let full = source.execute().unwrap();
    let mut rng = XorShift(seed);

    for case in 0..500 {
        let mut image = source.clone();
        for _ in 0..rng.range(0, 8) {
            image = random_op(&mut rng, &image, allow_resize);
        }
        let expected = image
            .ops()
            .iter()
            .fold(full.clone(), |img, &op| naive(&img, op));
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
    check_random_pipelines(false, 0x5EED_0001);
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
    check_random_pipelines(true, 0x5EED_0002);
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
