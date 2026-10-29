//! Running image pipelines on an [`Executor`], one at a time or in batches.

use std::path::Path;

use baryon_core::{Executor, Result};

use crate::image::Image;
use crate::plan::LazyImage;

impl LazyImage {
    /// Decodes and transforms this image on `executor`, using all of its threads where the
    /// format and operations allow it.
    pub fn execute_on(&self, executor: &Executor) -> Result<Image> {
        executor.install(|| self.execute())
    }

    /// Decodes and transforms many images in parallel. Results are in input order, and one
    /// failure does not affect the others.
    ///
    /// The largest decodes start first, so that one big image does not run alone at the end
    /// of a batch.
    pub fn execute_all(executor: &Executor, images: &[Self]) -> Vec<Result<Image>> {
        executor.map_by_cost(images, Self::cost, Self::execute)
    }

    /// Relative cost of executing this pipeline: the number of source pixels decoded.
    pub(crate) fn cost(&self) -> u64 {
        let region = self.plan().region();
        u64::from(region.width) * u64::from(region.height)
    }

    /// Reads and parses the headers of many files in parallel. Results are in input order.
    pub fn open_all<P: AsRef<Path> + Sync>(executor: &Executor, paths: &[P]) -> Vec<Result<Self>> {
        executor.map(paths, |path| Self::open(path))
    }
}

#[cfg(test)]
mod tests {
    use baryon_core::Error;

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

        let expected: Vec<Image> = images
            .iter()
            .map(|i| i.execute_on(&single).unwrap())
            .collect();
        for (got, want) in LazyImage::execute_all(&many, &images)
            .into_iter()
            .zip(&expected)
        {
            assert_eq!(&got.unwrap(), want);
        }
        for (image, want) in images.iter().zip(&expected) {
            assert_eq!(&image.execute_on(&many).unwrap(), want);
        }
    }

    #[test]
    fn batch_errors_stay_in_place() {
        let truncated = LazyImage::from_bytes(&LOSSY_JXL[..LOSSY_JXL.len() / 2]).unwrap();
        let ok = LazyImage::from_bytes(ADAM7_PNG).unwrap();
        let executor = Executor::new(2).unwrap();
        let results = LazyImage::execute_all(&executor, &[ok.clone(), truncated, ok]);
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

        let executor = Executor::new(2).unwrap();
        let results = LazyImage::open_all(&executor, &[png, dir.join("missing.png"), jxl]);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(results[0].as_ref().unwrap().width(), 40);
        assert!(matches!(results[1], Err(Error::Io(_))));
        assert_eq!(results[2].as_ref().unwrap().width(), 600);
    }
}
