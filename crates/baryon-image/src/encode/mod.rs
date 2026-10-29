//! Writing images: PNG and JPEG with pure-Rust encoders, JPEG XL with the vendored libjxl.
//!
//! Every encoder takes a decoded [`Image`](crate::Image); the [`LazyImage`](crate::LazyImage)
//! methods run the pipeline first.

mod jpeg;
#[cfg(feature = "jxl-encode")]
mod jxl;
mod png;

pub use jpeg::encode_jpeg;
#[cfg(feature = "jxl-encode")]
pub use jxl::{JxlOptions, JxlQuality, encode_pixels, jxl_to_jpeg};
pub use png::{PngCompression, encode_png};

use baryon_core::{Executor, Result};

use crate::plan::LazyImage;

impl LazyImage {
    /// Runs the pipeline and encodes the result as PNG.
    pub fn to_png(&self, compression: PngCompression) -> Result<Vec<u8>> {
        encode_png(&self.execute()?, compression)
    }

    /// Runs the pipeline and encodes the result as JPEG at `quality` (1 to 100).
    pub fn to_jpeg(&self, quality: u8) -> Result<Vec<u8>> {
        encode_jpeg(&self.execute()?, quality)
    }

    /// Runs `encode` on many pipelines in parallel, the largest first. Results are in input
    /// order, and one failure does not affect the others.
    pub fn encode_all(
        executor: &Executor,
        images: &[Self],
        encode: impl Fn(&Self) -> Result<Vec<u8>> + Send + Sync,
    ) -> Vec<Result<Vec<u8>>> {
        executor.map_by_cost(images, Self::cost, encode)
    }
}
