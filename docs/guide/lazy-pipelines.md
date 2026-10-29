# Lazy pipelines

An `Image` is an immutable recipe: the encoded source plus a list of transformations. Calling
`crop`, `resize`, `rotate`, `flip_*` or `convert` returns a new recipe and does no work. Pixels are
produced only by `to_numpy()`, `numpy.asarray(image)` or `baryon.to_numpy_all(...)`.

Because the whole pipeline is known before anything is decoded, baryon can plan it.

## What the optimizer does

1. **Fuses geometry.** Any run of crops, flips and rotations collapses into one rectangle of
   the source plus one orientation. A crop after a rotation is mapped back through the
   rotation, so it can be applied first.
2. **Pushes the crop into the decoder.** The fused rectangle becomes the region the decoder is
   asked for. Each format skips as much as it can:

    | Format | What a region request skips |
    |---|---|
    | JPEG XL | every tile (group) outside the region |
    | PNG | all rows after the region's last row |
    | JPEG | nothing yet; the image is decoded and trimmed |

3. **Orients once.** All flips and rotations, including the file's EXIF orientation, become a
   single pass over the pixels.
4. **Converts once, on the cheaper side.** Channel conversions fuse into one. A conversion to
   fewer channels runs before the orientation pass and a conversion to more channels after
   it, so the pass moves as little data as possible. A colour JPEG converted to `"L"` is
   decoded from its luma plane alone.
5. **Drops no-ops.** Two flips cancel, a resize to the current size disappears, and converting
   to the current mode does nothing.

## Seeing the plan

`explain()` prints the plan that will run; `explain(optimized=False)` prints the
transformations as you recorded them:

```python
>>> image = baryon.open("scan.png").crop(0, 0, 1024, 1024).rotate(90).resize(256, 256)
>>> print(image.explain(optimized=False))
source png 8000x6000 RGB uint8
crop 1024x1024+0+0
rotate 90° clockwise
resize 256x256 (lanczos3)

>>> print(image.explain())
decode png 8000x6000 RGB uint8
  region 1024x1024+0+0 (row streaming, stops after 1024 rows)
orient transpose+flip_y
resize 256x256 (lanczos3)
```

Rectangles are written `WIDTHxHEIGHT+X+Y`.

A grayscale thumbnail of a JPEG shows the conversion moving into the decoder:

```python
>>> print(baryon.open("photo.jpg").convert("L").rotate(90).resize(224, 224).explain())
decode jpeg 4000x3000 RGB uint8
  decoded directly as L
orient transpose+flip_y
resize 224x224 (lanczos3)
```

## Correctness

The optimizer is tested against a deliberately naive implementation that applies every
recorded step one at a time: thousands of random pipelines must produce identical pixels.

## Channel conversions

`convert(mode)` changes the channel layout to `"L"`, `"LA"`, `"RGB"` or `"RGBA"`:

- colour to gray uses the ITU-R BT.601 luma weights, exactly like Pillow's `convert("L")`;
- gray to colour copies the value into every channel;
- dropping alpha discards it without blending; adding alpha makes every pixel opaque.

Use it to normalize a mixed dataset before batching: `[im.convert("RGB") for im in images]`
gives every array three channels, whatever the files contain.

## Resizing

`resize(width, height, filter=...)` resamples to exact dimensions with one of `nearest`,
`box`, `bilinear`, `hamming`, `bicubic`, `mitchell` or `lanczos3` (the default). Images with
alpha are resized with premultiplied alpha, so transparent pixels do not bleed colour. Results
closely match Pillow's filters of the same name.
