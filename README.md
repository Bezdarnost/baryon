# baryon

Fast, lazy, region-aware image decoding and processing for Python, written in Rust.

> **Status: pre-alpha.** The architecture is in place; codec coverage and performance work are
> just starting. APIs may change.

```python
import baryon

image = (
    baryon.open("scan.png")  # parses the header only
    .crop(0, 0, 1024, 1024)  # recorded, not performed
    .rotate(90)
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
rotate 90° counter-clockwise
resize 256x256 (lanczos3)

>>> print(image.explain())                  # as executed
decode png 8000x6000 RGB uint8
  region 1024x1024+0+0 (row streaming, stops after 1024 rows)
orient transpose+flip_x
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
- **Free-threading ready.** All work runs with the GIL released, and the extension declares
  free-threaded support.

## Format support

| Format   | Decode | Partial decode today                     | Planned                                   |
|----------|:------:|------------------------------------------|-------------------------------------------|
| PNG      | ✅     | row streaming: stops after the last needed row | —                                   |
| JPEG     | ✅     | full decode, then trim                   | MCU skipping, DCT-domain downscale (1/2–1/8) |
| JPEG XL  | 🔜     | —                                        | group-level region decode, progressive previews, lossless JPEG transcoding |
| AVIF     | 🔜     | —                                        | grid/tile-aware decode                    |

JPEG XL is the flagship target. Its independently decodable groups and progressive structure
suit region-aware decoding better than any other mainstream format.

## Development

Requires a Rust toolchain (stable, MSRV 1.87) and [uv](https://docs.astral.sh/uv/).

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
crates/baryon-core     pure-Rust engine: codecs, lazy plans, optimizer, ops
crates/baryon-python   PyO3 bindings (the baryon._baryon extension module)
python/baryon          Python package: public API and type stubs
tests/python           Python test suite
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
