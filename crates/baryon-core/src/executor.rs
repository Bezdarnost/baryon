//! Parallel execution on a dedicated thread pool.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::image::Image;
use crate::plan::LazyImage;

/// Runs pipelines on a dedicated thread pool.
///
/// Everything parallel inside baryon (JPEG XL tile decoding, resizing, crops and rotations) runs
/// on the pool it was started from, so an executor bounds the CPU used by all the work it runs.
/// Batches are parallelized across images as well as within each image.
///
/// Work started outside any executor, such as a bare [`LazyImage::execute`], runs on rayon's
/// global pool.
#[derive(Clone)]
pub struct Executor {
    pool: Arc<rayon::ThreadPool>,
}

impl Executor {
    /// Starts a pool with `num_threads` workers; `0` means one per logical CPU (or the value of
    /// `RAYON_NUM_THREADS`, if set).
    pub fn new(num_threads: usize) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|index| format!("baryon-{index}"))
            .build()
            .map_err(|err| Error::Internal(format!("failed to start thread pool: {err}")))?;
        Ok(Self {
            pool: Arc::new(pool),
        })
    }

    /// Number of worker threads.
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Runs `op` on the pool, so that any parallel work it starts uses this executor's threads.
    /// Blocks the calling thread until `op` returns.
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        self.pool.install(op)
    }

    /// Decodes and transforms one image, using all of the pool's threads where the format and
    /// operations allow it.
    pub fn execute(&self, image: &LazyImage) -> Result<Image> {
        self.install(|| image.execute())
    }

    /// Decodes and transforms many images in parallel. Results are in input order.
    pub fn execute_all(&self, images: &[LazyImage]) -> Vec<Result<Image>> {
        self.install(|| images.par_iter().map(LazyImage::execute).collect())
    }

    /// Reads and parses the headers of many files in parallel. Results are in input order.
    pub fn open_all<P: AsRef<Path> + Sync>(&self, paths: &[P]) -> Vec<Result<LazyImage>> {
        self.install(|| paths.par_iter().map(LazyImage::open).collect())
    }
}

impl fmt::Debug for Executor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Executor")
            .field("num_threads", &self.num_threads())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Rect, Rotation};
    use crate::ops::ResizeFilter;

    const LOSSY_JXL: &[u8] = include_bytes!("../tests/fixtures/smooth-rgb8-lossy.jxl");
    const ADAM7_PNG: &[u8] = include_bytes!("../tests/fixtures/gradient-adam7.png");

    fn pipelines() -> Vec<LazyImage> {
        let jxl = LazyImage::from_bytes(LOSSY_JXL).unwrap();
        let png = LazyImage::from_bytes(ADAM7_PNG).unwrap();
        vec![
            jxl.clone(),
            jxl.crop(Rect::new(100, 50, 300, 200)).unwrap(),
            jxl.rotate(Rotation::Cw90)
                .resize(150, 200, ResizeFilter::Lanczos3)
                .unwrap(),
            png.clone(),
            png.flip_horizontal(),
        ]
    }

    #[test]
    fn results_do_not_depend_on_thread_count() {
        let images = pipelines();
        let single = Executor::new(1).unwrap();
        let many = Executor::new(4).unwrap();
        assert_eq!(single.num_threads(), 1);
        assert_eq!(many.num_threads(), 4);

        let expected: Vec<Image> = images.iter().map(|i| single.execute(i).unwrap()).collect();
        for (got, want) in many.execute_all(&images).into_iter().zip(&expected) {
            assert_eq!(&got.unwrap(), want);
        }
        for (image, want) in images.iter().zip(&expected) {
            assert_eq!(&many.execute(image).unwrap(), want);
        }
    }

    #[test]
    fn batch_errors_stay_in_place() {
        let truncated = LazyImage::from_bytes(&LOSSY_JXL[..LOSSY_JXL.len() / 2]).unwrap();
        let ok = LazyImage::from_bytes(ADAM7_PNG).unwrap();
        let results = Executor::new(2)
            .unwrap()
            .execute_all(&[ok.clone(), truncated, ok]);
        assert!(results[0].is_ok());
        assert!(matches!(results[1], Err(Error::Decode { .. })));
        assert!(results[2].is_ok());
    }

    #[test]
    fn open_all_preserves_order() {
        let dir = std::env::temp_dir().join(format!("baryon-open-all-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("a.png");
        let jxl = dir.join("b.jxl");
        std::fs::write(&png, ADAM7_PNG).unwrap();
        std::fs::write(&jxl, LOSSY_JXL).unwrap();

        let results = Executor::new(2).unwrap().open_all(&[
            png.clone(),
            dir.join("missing.png"),
            jxl.clone(),
        ]);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(results[0].as_ref().unwrap().width(), 40);
        assert!(matches!(results[1], Err(Error::Io(_))));
        assert_eq!(results[2].as_ref().unwrap().width(), 600);
    }
}
