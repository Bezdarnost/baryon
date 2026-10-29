# baryon

Fast, lazy, region-aware image decoding and processing for Python, written in Rust.

**[Documentation](https://bezdarnost.github.io/baryon/)** ·
[Benchmarks](https://bezdarnost.github.io/baryon/benchmarks/) ·
[Roadmap](https://bezdarnost.github.io/baryon/roadmap/)

> **Status: pre-alpha.** The architecture is in place; codec coverage and performance work are
> just starting. APIs may change.

```python
import baryon

image = (
    baryon.open("scan.jxl")  # parses the header only
    .crop(0, 0, 1024, 1024)  # recorded, not performed
    .rotate(90)  # clockwise; see the note below
    .resize(256, 256, filter="lanczos3")
)
print(image.explain())  # the optimized plan
pixels = image.to_numpy()  # decodes only what the pipeline needs; GIL released
```

## Why

Most Python image libraries decode the whole image before you can do anything with it. baryon
records transformations lazily, optimizes the pipeline, and pushes work *into the decoder*, so
pixels you would throw away are never decoded in the first place.

```text
>>> print(image.explain(optimized=False))   # as recorded
source png 8000x6000 RGB uint8
crop 1024x1024+0+0
rotate 90° clockwise
resize 256x256 (lanczos3)

>>> print(image.explain())                  # as executed
decode png 8000x6000 RGB uint8
  region 1024x1024+0+0 (row streaming, stops after 1024 rows)
orient transpose+flip_y
resize 256x256 (lanczos3)
```

How it works:

- **Lazy pipelines.** `Image` is an immutable handle. `crop`, `resize`, `rotate` and `flip_*`
  return new handles instantly.
- **Geometric fusion.** Any run of crops, flips and rotations collapses into one source window
  and one orientation. Crops are moved in front of flips and rotations so they can be applied
  first.
- **Decoder pushdown.** The fused window becomes the region requested from the codec. Each
  codec uses whatever partial decoding its format allows, and the executor trims the rest.
- **Zero-copy NumPy.** Decoded buffers are moved into NumPy arrays without copying. Arrays are
  always `(height, width, channels)` with dtype `uint8` or `uint16`.
- **All cores.** Batches are decoded in parallel with `baryon.to_numpy_all`, and a single large
  image is split across threads where possible (JPEG XL decoding, resize, crop, rotate, flip).
  One setting, `baryon.set_num_threads(n)` or `BARYON_NUM_THREADS`, bounds all of it, and
  forked worker processes (PyTorch `DataLoader`, `multiprocessing`) get a fresh thread pool.
- **Free-threading ready.** All work runs with the GIL released, and the extension declares
  free-threaded support.

```python
images = baryon.open_all(paths)  # read files and parse headers in parallel
batch = baryon.to_numpy_all([im.crop(0, 0, 512, 512) for im in images])  # decode on all cores
```

> [!WARNING]
> **`rotate()` turns clockwise by default**, like CSS and photo viewers. This is the opposite
> of Pillow's `Image.rotate` and `numpy.rot90`, which turn counter-clockwise for positive
> angles. When porting code from those libraries, use `rotate(degrees, clockwise=False)`.

## Format support

| Format  | Decode | Partial decode today                              | Planned                                                     |
|---------|:------:|---------------------------------------------------|-------------------------------------------------------------|
| JPEG XL | ✅ + write | tile decoding: only groups overlapping the region | progressive previews, region decode for rotated files |
| PNG     | ✅     | row streaming: stops after the last needed row    | —                                                           |
| JPEG    | ✅     | full decode, then trim                            | MCU skipping, DCT-domain downscale (1/2–1/8)                |
| AVIF    | 🔜     | —                                                 | grid/tile-aware decode                                      |

### JPEG XL

JPEG XL is baryon's flagship format, for reading and writing:

- **Reading:** whole images decode with [jxl-rs](https://github.com/libjxl/jxl-rs) (the
  decoder in Chrome and Firefox), as fast as the libjxl reference decoder. Crops decode with
  [jxl-oxide](https://github.com/tirr-c/jxl-oxide), which renders only the 256×256 groups the
  crop touches: 6.8× faster than libjxl's full decode on 16-megapixel photos.
- **Writing:** `image.to_jxl()` uses the vendored libjxl. JPEGs are transcoded losslessly,
  about 20% smaller and restorable bit for bit with `baryon.jxl_to_jpeg`; PNGs shrink by
  about 30% with every pixel preserved.

```python
jxl = baryon.open("photo.jpg").to_jxl()  # lossless, ~20% smaller
assert baryon.jxl_to_jpeg(jxl) == open("photo.jpg", "rb").read()
```

8-bit images decode to `uint8`; higher bit depths and float images to full-range `uint16`.
Orientation is applied automatically. See the
[JPEG XL guide](https://bezdarnost.github.io/baryon/guide/jpeg-xl/) for measurements on a
47 GB dataset and when to convert, and the
[benchmarks](https://bezdarnost.github.io/baryon/benchmarks/) for comparisons with Pillow,
OpenCV, pyvips, imagecodecs and pillow-jxl.

## Roadmap

Images come first, with JPEG XL as the flagship format: JPEG region decoding and DCT-domain
downscaling, region decoding for rotated JPEG XL files, progressive previews, then AVIF and
WebP.
**Video and audio** are planned on the same foundation: lazy frame and time ranges pushed into
the decoder, batched across cores, returned as NumPy arrays. See the
[roadmap](https://bezdarnost.github.io/baryon/roadmap/).

## Development

Requires a Rust toolchain (stable, MSRV 1.89), [uv](https://docs.astral.sh/uv/), CMake and a
C++17 compiler (JPEG XL encoding builds the vendored libjxl).

```bash
git submodule update --init --depth 1 crates/baryon-libjxl/libjxl
git -C crates/baryon-libjxl/libjxl submodule update --init --depth 1 \
    third_party/highway third_party/brotli third_party/skcms
uv sync                      # builds the extension in development mode
uv run pytest                # Python tests
cargo test --workspace       # Rust tests
```

Lints, all enforced in CI:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
uv run ruff check && uv run ruff format --check
uv run mypy
```

### Layout

```
crates/baryon-core     media-agnostic runtime: errors, format detection, thread-pool executor
crates/baryon-image    images: codecs, lazy plans, optimizer, operations, batches
crates/baryon-libjxl   JPEG XL encoding via the vendored libjxl (the only crate with unsafe code)
crates/baryon-python   PyO3 bindings (the baryon._baryon extension module)
python/baryon          Python package: public API and type stubs
tests/python           Python test suite
benchmarks             reproducible comparisons with other libraries
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
