//! The engine behind the `baryon` Python library: lazy, region-aware image decoding.
//!
//! ```no_run
//! use baryon_core::{LazyImage, Rect, ResizeFilter, Rotation};
//!
//! # fn main() -> baryon_core::Result<()> {
//! let image = LazyImage::open("photo.png")?          // header only
//!     .crop(Rect::new(100, 100, 512, 512))?           // recorded, not performed
//!     .rotate(Rotation::Cw90)
//!     .resize(256, 256, ResizeFilter::Lanczos3)?;
//! println!("{}", image.explain(true));               // the optimized plan
//! let pixels = image.execute()?;                      // decode only what is needed
//! # Ok(())
//! # }
//! ```

mod codecs;
mod error;
mod executor;
mod format;
mod geometry;
mod image;
pub mod ops;
mod plan;

pub use codecs::{ImageInfo, RegionStrategy, RegionSupport};
pub use error::{BoxError, Error, Result};
pub use executor::Executor;
pub use format::ImageFormat;
pub use geometry::{Orientation, Rect, Rotation};
pub use image::{ColorType, Image, PixelBuffer, SampleType};
pub use ops::{Op, ResizeFilter};
pub use plan::{LazyImage, PhysicalPlan, Source, Step};
