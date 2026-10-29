//! Lazy, region-aware image decoding and processing.
//!
//! ```no_run
//! use baryon_image::{Executor, LazyImage, Rect, ResizeFilter, Rotation};
//!
//! # fn main() -> baryon_image::Result<()> {
//! let image = LazyImage::open("photo.png")?          // header only
//!     .crop(Rect::new(100, 100, 512, 512))?           // recorded, not performed
//!     .rotate(Rotation::Cw90)
//!     .resize(256, 256, ResizeFilter::Lanczos3)?;
//! println!("{}", image.explain(true));                // the optimized plan
//! let executor = Executor::new(0)?;                   // one thread per CPU
//! let pixels = image.execute_on(&executor)?;          // decode only what is needed
//! # Ok(())
//! # }
//! ```

mod batch;
mod codecs;
#[cfg(feature = "jxl-encode")]
mod encode;
mod exif;
mod geometry;
mod image;
pub mod ops;
mod plan;

pub use baryon_core::{BoxError, Error, Executor, Format, MediaType, Result};
pub use codecs::{ImageInfo, RegionStrategy, RegionSupport};
#[cfg(feature = "jxl-encode")]
pub use encode::{JxlOptions, JxlQuality, encode_pixels, jxl_to_jpeg};
pub use geometry::{Orientation, Rect, Rotation};
pub use image::{ColorType, Image, PixelBuffer, SampleType};
pub use ops::{Op, ResizeFilter};
pub use plan::{LazyImage, PhysicalPlan, Source, Step};
