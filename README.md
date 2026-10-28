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
| JPEG XL | ✅     | tile decoding: only groups overlapping the region | progressive previews, lossless JPEG transcoding, region decode for rotated files |
| PNG     | ✅     | row streaming: stops after the last needed row    | —                                                           |
| JPEG    | ✅     | full decode, then trim                            | MCU skipping, DCT-domain downscale (1/2–1/8)                |
| AVIF    | 🔜     | —                                                 | grid/tile-aware decode                                      |

### JPEG XL

JPEG XL is baryon's flagship format. It is coded in independent groups (256×256 pixels by
default), so a crop is handed straight to the decoder, which renders only the groups it
touches. Decoding uses [jxl-oxide](https://github.com/tirr-c/jxl-oxide) (pure Rust) with
`moxcms` colour management. 8-bit images decode to `uint8`; higher bit depths and float images
decode to full-range `uint16`. The orientation stored in the file is applied automatically.
Only the first frame of an animation is decoded, and CMYK images are not supported yet.

What a 256×256 crop costs compared with decoding the whole image (synthetic 4096×4096 image,
release build, best of three runs on one machine; treat these as illustrative):

| Source            | Full decode | Crop at the center | Crop at the top |
|-------------------|------------:|-------------------:|----------------:|
| JPEG XL (lossy)   | 344 ms      | 11.6 ms (30×)      | 8.6 ms (40×)    |
| PNG               | 80 ms       | 37 ms (2×)         | 4.5 ms (18×)    |

Files with a non-default orientation are currently decoded in full and then cropped, until test
fixtures for rotated files are in place.

## Roadmap

Images come first, with JPEG XL as the flagship format: JPEG XL encoding and lossless JPEG
transcoding, faster full-image JPEG XL decoding, JPEG region decoding, then AVIF and WebP.
**Video and audio** are planned on the same foundation: lazy frame and time ranges pushed into
the decoder, batched across cores, returned as NumPy arrays. See the
[roadmap](https://bezdarnost.github.io/baryon/roadmap/).

## Development

Requires a Rust toolchain (stable, MSRV 1.89) and [uv](https://docs.astral.sh/uv/).

```bash
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
crates/baryon-python   PyO3 bindings (the baryon._baryon extension module)
python/baryon          Python package: public API and type stubs
tests/python           Python test suite
benchmarks             reproducible comparisons with other libraries
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
