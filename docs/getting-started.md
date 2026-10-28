# Getting started

## Install

```bash
pip install baryon
```

baryon needs Python 3.10 or newer and NumPy. Wheels include everything; there are no system
libraries to install.

!!! note "Before the first release"
    Until baryon is published on PyPI, build it from a checkout with a Rust toolchain
    (1.88 or newer) and [uv](https://docs.astral.sh/uv/):

    ```bash
    git clone https://github.com/Bezdarnost/baryon && cd baryon
    uv sync
    ```

## Open an image

`baryon.open` accepts a path or the encoded bytes, and reads only the header:

```python
import baryon

image = baryon.open("photo.jpg")
image.size  # (6000, 4000)
image.mode  # "RGB"
image.format  # "jpeg"
```

Formats are detected from the file contents, so a PNG named `.jpg` opens correctly.

## Get pixels

```python
pixels = image.to_numpy()  # numpy.ndarray, shape (4000, 6000, 3), dtype uint8
pixels = numpy.asarray(image)  # the same, through the NumPy array protocol
```

Arrays are always `(height, width, channels)`, also for grayscale (`channels == 1`). 16-bit
images give `uint16` arrays.

## Transform lazily

Every transformation returns a new image instantly; nothing is decoded until you ask for
pixels:

```python
thumb = image.crop(1000, 500, 2048, 2048).rotate(90).resize(256, 256)
thumb.shape  # (256, 256, 3), known without decoding
thumb.to_numpy()  # decodes, crops, rotates and resizes in one optimized pass
```

!!! warning "Rotation direction"
    `rotate()` turns **clockwise** by default, like CSS and photo viewers. Pillow's
    `Image.rotate` and `numpy.rot90` turn counter-clockwise; use
    `rotate(degrees, clockwise=False)` to match them.

## Many images at once

```python
images = baryon.open_all(paths)  # parallel file reads and header parsing
batch = baryon.to_numpy_all(images)  # parallel decoding on all cores
```

See [Using all cores](guide/parallelism.md) for thread control and data loaders.

## Errors

| Exception | Raised when |
|---|---|
| `baryon.UnsupportedFormatError` | the format is unknown, or known but not decodable yet |
| `baryon.DecodeError` | the data is corrupt or truncated |
| `ValueError` | a transformation does not fit the image, e.g. a crop past the edge |
| `OSError` | a file cannot be read |

`UnsupportedFormatError` and `DecodeError` derive from `baryon.BaryonError`.
