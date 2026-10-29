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
- Multi-core execution on a shared thread pool: `open_all`, `to_numpy_all`,
  `set_num_threads` / `get_num_threads` (`BARYON_NUM_THREADS`); fork-safe for data loaders.
- Rotation is clockwise by default; `rotate(degrees, clockwise=False)` matches Pillow and
  `numpy.rot90`.
- Zero-copy NumPy output, `numpy.asarray(image)` support, typed stubs, free-threaded Python
  support.
- Benchmark suite comparing against Pillow, OpenCV, pyvips, imagecodecs and pillow-jxl-plugin.
- Documentation site at https://bezdarnost.github.io/baryon/.
