# Changelog

All notable changes to baryon are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Lazy image pipelines: `crop`, `resize`, `rotate`, `flip_horizontal`, `flip_vertical`, with
  an optimizer that fuses geometry and pushes crops into the decoder; `explain()` shows the plan.
- Decoding of JPEG, PNG and JPEG XL. JPEG XL crops decode only the tiles they touch (jxl-oxide);
  whole JPEG XL images decode with jxl-rs, as fast as libjxl; PNG crops stop after the last
  needed row.
- JPEG XL encoding with the vendored libjxl: `Image.to_jxl`, `to_jxl_all` and `jxl_to_jpeg`.
  Untransformed JPEGs are transcoded losslessly (about 20% smaller, restorable bit for bit).
  `jpeg_fallback="lossy"` or `"lossless"` encodes the ~1% of JPEGs libjxl cannot transcode
  from their pixels; `save_all` returns their positions.
- PNG and JPEG encoding (`Image.to_png`, `Image.to_jpeg`), `Image.save` with the format taken
  from the extension, and `save_all` to convert many files on all cores.
- `Image.convert(mode)` for `L`, `LA`, `RGB` and `RGBA`, planned with the other operations;
  colour JPEGs converted to gray are decoded from their luma plane alone.
- EXIF orientation of JPEG and PNG files is applied, fused with the pipeline's own rotations.
  JPEGs and their JPEG XL transcodes now display identically.
- Multi-core execution on a shared thread pool: `open_all`, `to_numpy_all`,
  `set_num_threads` / `get_num_threads` (`BARYON_NUM_THREADS`); fork-safe for data loaders.
  Batches start the largest images first, so one big image does not finish alone. A failing
  batch raises its first error with every failed position listed, and a `failures` attribute
  holding each `(position, exception)`.
- Rotation is clockwise by default; `rotate(degrees, clockwise=False)` matches Pillow and
  `numpy.rot90`.
- Zero-copy NumPy output, `numpy.asarray(image)` support, typed stubs, free-threaded Python
  support.
- Benchmark suite comparing against Pillow, OpenCV, pyvips, imagecodecs and pillow-jxl-plugin.
- Documentation site at https://bezdarnost.github.io/baryon/.
