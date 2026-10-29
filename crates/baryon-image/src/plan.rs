//! Lazy pipelines: recording transformations, optimizing them and executing the result.
//!
//! A [`LazyImage`] is an encoded [`Source`] plus a list of logical [`Op`]s. Nothing is decoded
//! until [`LazyImage::execute`]. Before execution the ops are lowered to a [`PhysicalPlan`]:
//!
//! * Every run of crops, flips and rotations collapses into one *window* (a rectangle of the
//!   run's input) and one [`Orientation`]. Crops are moved in front of flips and rotations by
//!   mapping their rectangles back through the orientation, so they can be applied first.
//! * The window of the first run becomes the region requested from the decoder, which lets
//!   codecs skip work for pixels that would be thrown away.
//! * Later runs (after a resize) become an in-memory crop followed by a single orientation pass.

use std::fmt::{self, Write as _};
use std::path::Path;
use std::sync::Arc;

use crate::codecs::{self, ImageInfo, RegionStrategy};
use crate::geometry::{Orientation, Rect, Rotation};
use crate::image::{ColorType, Image, SampleType};
use crate::ops::{self, ColorConversion, Op, ResizeFilter};
use baryon_core::Format;
use baryon_core::{Error, Result};

/// Encoded image bytes together with their parsed header.
pub struct Source {
    data: Box<[u8]>,
    info: ImageInfo,
}

impl Source {
    /// Detects the format and parses the header. Pixels are not decoded.
    pub fn from_bytes(data: impl Into<Box<[u8]>>) -> Result<Self> {
        let data = data.into();
        let format = Format::detect(&data).ok_or(Error::UnrecognizedFormat)?;
        let info = codecs::codec_for(format)?.probe(&data)?;
        if info.width == 0 || info.height == 0 {
            return Err(Error::decode(format, "image has zero width or height"));
        }
        Ok(Self { data, info })
    }

    /// Reads a file and parses its header.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// Header information.
    pub const fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// The encoded bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

impl fmt::Debug for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Source")
            .field("info", &self.info)
            .field("encoded_len", &self.data.len())
            .finish()
    }
}

/// An image whose transformations are recorded rather than performed.
///
/// Cloning is cheap: the encoded source is shared and only the op list is copied. Every builder
/// method returns a new value and leaves `self` untouched.
#[derive(Debug, Clone)]
pub struct LazyImage {
    source: Arc<Source>,
    ops: Vec<Op>,
    width: u32,
    height: u32,
    color: ColorType,
}

impl LazyImage {
    /// Starts an empty pipeline over `source`.
    pub fn new(source: Arc<Source>) -> Self {
        let (width, height) = source.info().display_size();
        let color = source.info().color;
        Self {
            source,
            ops: Vec::new(),
            width,
            height,
            color,
        }
    }

    /// Opens a file lazily (reads the bytes and the header only).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Source::open(path).map(|s| Self::new(Arc::new(s)))
    }

    /// Wraps encoded bytes lazily.
    pub fn from_bytes(data: impl Into<Box<[u8]>>) -> Result<Self> {
        Source::from_bytes(data).map(|s| Self::new(Arc::new(s)))
    }

    /// The underlying source.
    pub fn source(&self) -> &Arc<Source> {
        &self.source
    }

    /// Header information of the source.
    pub fn info(&self) -> &ImageInfo {
        self.source.info()
    }

    /// Recorded logical operations, in application order.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Output width after all recorded operations.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Output height after all recorded operations.
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Output channel layout.
    pub const fn color(&self) -> ColorType {
        self.color
    }

    /// Output sample type.
    pub fn sample_type(&self) -> SampleType {
        self.info().sample_type
    }

    /// Records a crop. The rectangle is relative to the current output and must lie within it.
    pub fn crop(&self, rect: Rect) -> Result<Self> {
        if rect.width == 0 || rect.height == 0 {
            return Err(Error::InvalidArgument(format!("crop {rect} is empty")));
        }
        if !Rect::from_size(self.width, self.height).contains(&rect) {
            return Err(Error::InvalidArgument(format!(
                "crop {rect} exceeds image bounds {}x{}",
                self.width, self.height
            )));
        }
        Ok(self.push(Op::Crop(rect), rect.width, rect.height))
    }

    /// Records a left-right mirror.
    #[must_use]
    pub fn flip_horizontal(&self) -> Self {
        self.push(Op::FlipHorizontal, self.width, self.height)
    }

    /// Records a top-bottom mirror.
    #[must_use]
    pub fn flip_vertical(&self) -> Self {
        self.push(Op::FlipVertical, self.width, self.height)
    }

    /// Records a clockwise rotation.
    #[must_use]
    pub fn rotate(&self, rotation: Rotation) -> Self {
        let (w, h) = match rotation {
            Rotation::Cw180 => (self.width, self.height),
            Rotation::Cw90 | Rotation::Cw270 => (self.height, self.width),
        };
        self.push(Op::Rotate(rotation), w, h)
    }

    /// Records a resize to exactly `width` x `height`.
    pub fn resize(&self, width: u32, height: u32, filter: ResizeFilter) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidArgument(format!(
                "resize target must be non-empty, got {width}x{height}"
            )));
        }
        Ok(self.push(
            Op::Resize {
                width,
                height,
                filter,
            },
            width,
            height,
        ))
    }

    /// Records a change of channel layout; see [`ColorConversion`] for the rules.
    #[must_use]
    pub fn convert(&self, color: ColorType) -> Self {
        let mut image = self.push(Op::Convert(color), self.width, self.height);
        image.color = color;
        image
    }

    fn push(&self, op: Op, width: u32, height: u32) -> Self {
        let mut ops = Vec::with_capacity(self.ops.len() + 1);
        ops.extend_from_slice(&self.ops);
        ops.push(op);
        Self {
            source: Arc::clone(&self.source),
            ops,
            width,
            height,
            color: self.color,
        }
    }

    /// Lowers the recorded ops into an optimized physical plan.
    pub fn plan(&self) -> PhysicalPlan {
        PhysicalPlan::new(self.info(), &self.ops)
    }

    /// Human-readable description of the pipeline.
    ///
    /// With `optimized` set, shows the physical plan that [`LazyImage::execute`] will run;
    /// otherwise lists the logical ops as recorded.
    pub fn explain(&self, optimized: bool) -> String {
        if optimized {
            return self.plan().to_string();
        }
        let info = self.info();
        let mut out = format!(
            "source {} {}x{} {} {}\n",
            info.format, info.width, info.height, info.color, info.sample_type
        );
        if !info.orientation.is_identity() {
            let _ = writeln!(out, "orient {} (from EXIF)", info.orientation);
        }
        for op in &self.ops {
            // Writing to a String cannot fail.
            let _ = writeln!(out, "{op}");
        }
        out
    }

    /// Decodes and transforms the image.
    pub fn execute(&self) -> Result<Image> {
        self.plan().execute(self.source.data())
    }
}

/// One step of a physical plan, applied to an in-memory image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Keep only this rectangle.
    Crop(Rect),
    /// Apply all flips and rotations in one pass.
    Orient(Orientation),
    /// Resample.
    Resize {
        /// Target width.
        width: u32,
        /// Target height.
        height: u32,
        /// Resampling filter.
        filter: ResizeFilter,
    },
    /// Change the channel layout.
    Convert(ColorConversion),
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Crop(rect) => write!(f, "crop {rect}"),
            Self::Orient(o) => write!(f, "orient {o}"),
            Self::Resize {
                width,
                height,
                filter,
            } => write!(f, "resize {width}x{height} ({filter})"),
            Self::Convert(conversion) => write!(f, "convert {conversion}"),
        }
    }
}

/// An optimized, executable pipeline: a region-restricted decode followed by in-memory steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPlan {
    info: ImageInfo,
    region: Rect,
    strategy: RegionStrategy,
    decode_color: ColorType,
    steps: Vec<Step>,
}

impl PhysicalPlan {
    fn new(info: &ImageInfo, ops: &[Op]) -> Self {
        // The stored orientation is the first transform of the first run, so crops after it
        // still become a decode region in stored coordinates.
        let mut geometry = Geometry::new(info.width, info.height, info.color);
        geometry.orientation = info.orientation;
        let mut region = None;
        let mut decode_color = info.color;
        let mut steps = Vec::new();

        for op in ops {
            match *op {
                Op::Crop(rect) => geometry.crop(rect),
                Op::FlipHorizontal => {
                    geometry.orientation = geometry.orientation.then_flip_horizontal();
                }
                Op::FlipVertical => {
                    geometry.orientation = geometry.orientation.then_flip_vertical();
                }
                Op::Rotate(r) => geometry.orientation = geometry.orientation.then_rotate(r),
                Op::Convert(color) => geometry.conversion = geometry.conversion.then(color),
                Op::Resize {
                    width,
                    height,
                    filter,
                } => {
                    let (w, h) = geometry.output_size();
                    let color = geometry.conversion.to();
                    if region.is_none() {
                        decode_color = geometry.push_conversion_into_decoder(info);
                    }
                    geometry.flush(&mut region, &mut steps);
                    if (w, h) != (width, height) {
                        steps.push(Step::Resize {
                            width,
                            height,
                            filter,
                        });
                    }
                    geometry = Geometry::new(width, height, color);
                }
            }
        }
        if region.is_none() {
            decode_color = geometry.push_conversion_into_decoder(info);
        }
        geometry.flush(&mut region, &mut steps);

        let region = region.unwrap_or_else(|| info.bounds());
        Self {
            info: *info,
            region,
            strategy: RegionStrategy::new(info, region),
            decode_color,
            steps,
        }
    }

    /// Header of the image being decoded.
    pub const fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// Source region requested from the decoder.
    pub const fn region(&self) -> Rect {
        self.region
    }

    /// How the decoder will satisfy [`PhysicalPlan::region`].
    pub const fn strategy(&self) -> RegionStrategy {
        self.strategy
    }

    /// Channel layout requested from the decoder; differs from the source's when the decoder
    /// can produce a conversion directly, such as a JPEG's luma plane.
    pub const fn decode_color(&self) -> ColorType {
        self.decode_color
    }

    /// In-memory steps run after decoding.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Runs the plan against the encoded bytes it was built for.
    pub fn execute(&self, data: &[u8]) -> Result<Image> {
        let codec = codecs::codec_for(self.info.format)?;
        let decoded = codec.decode(data, &self.info, self.region, self.decode_color)?;
        if decoded.image.color() != self.decode_color {
            return Err(Error::Internal(format!(
                "{} decoder produced {} for a request of {}",
                self.info.format,
                decoded.image.color(),
                self.decode_color
            )));
        }
        if !decoded.region.contains(&self.region) {
            return Err(Error::Internal(format!(
                "{} decoder returned region {} for request {}",
                self.info.format, decoded.region, self.region
            )));
        }
        let mut image = if decoded.region == self.region {
            decoded.image
        } else {
            ops::crop(&decoded.image, decoded.region.relative(&self.region))?
        };
        for step in &self.steps {
            image = match *step {
                Step::Crop(rect) => ops::crop(&image, rect)?,
                Step::Orient(o) => ops::orient(image, o)?,
                Step::Resize {
                    width,
                    height,
                    filter,
                } => ops::resize(&image, width, height, filter)?,
                Step::Convert(conversion) => ops::convert_color(&image, conversion)?,
            };
        }
        Ok(image)
    }
}

impl fmt::Display for PhysicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let info = &self.info;
        writeln!(
            f,
            "decode {} {}x{} {} {}",
            info.format, info.width, info.height, info.color, info.sample_type
        )?;
        if self.region != info.bounds() {
            writeln!(f, "  region {} ({})", self.region, self.strategy)?;
        }
        if self.decode_color != info.color {
            writeln!(f, "  decoded directly as {}", self.decode_color)?;
        }
        for step in &self.steps {
            writeln!(f, "{step}")?;
        }
        Ok(())
    }
}

/// Accumulated crops, orientation and colour conversions for one run of ops between resizes.
///
/// `window` is a rectangle of the run's input image and `orientation` maps output coordinates
/// into that window. Output pixel `(x, y)` is input pixel `window.origin + orientation(x, y)`.
/// Conversions act on each pixel independently, so they commute with the geometry and the
/// run's conversions fuse into one, placed wherever it moves the least data.
struct Geometry {
    input_width: u32,
    input_height: u32,
    window: Rect,
    orientation: Orientation,
    conversion: ColorConversion,
}

impl Geometry {
    const fn new(width: u32, height: u32, color: ColorType) -> Self {
        Self {
            input_width: width,
            input_height: height,
            window: Rect::from_size(width, height),
            orientation: Orientation::IDENTITY,
            conversion: ColorConversion::identity(color),
        }
    }

    /// For the first run: lets the decoder produce the gray conversion itself when it can,
    /// and returns the layout to request from it.
    fn push_conversion_into_decoder(&mut self, info: &ImageInfo) -> ColorType {
        if self.conversion.collapses_to_gray() && codecs::decodes_luma(info) {
            self.conversion = self.conversion.with_luma_input();
            ColorType::L
        } else {
            info.color
        }
    }

    const fn output_size(&self) -> (u32, u32) {
        self.orientation
            .output_size(self.window.width, self.window.height)
    }

    /// Narrows the window to `rect`, given in current output coordinates (already validated).
    fn crop(&mut self, rect: Rect) {
        let local = self
            .orientation
            .source_rect(rect, self.window.width, self.window.height);
        self.window = Rect::new(
            self.window.x + local.x,
            self.window.y + local.y,
            local.width,
            local.height,
        );
    }

    /// Emits this run: as the decode region if it is the first run, otherwise as steps.
    fn flush(&self, region: &mut Option<Rect>, steps: &mut Vec<Step>) {
        let full = Rect::from_size(self.input_width, self.input_height);
        if region.is_none() {
            *region = Some(self.window);
        } else if self.window != full {
            steps.push(Step::Crop(self.window));
        }
        let convert = (!self.conversion.is_identity()).then_some(Step::Convert(self.conversion));
        // Shrinking conversions run before the orientation pass, growing ones after it.
        if self.conversion.shrinks() {
            steps.extend(convert);
        }
        if !self.orientation.is_identity() {
            steps.push(Step::Orient(self.orientation));
        }
        if !self.conversion.shrinks() {
            steps.extend(convert);
        }
    }
}
