# Formats

Formats are recognized from file signatures, never from extensions.

| Format | Read | Write | Bit depths | Partial decoding | Notes |
|---|:---:|:---:|---|---|---|
| JPEG XL | ✅ | ✅ | 8-bit → `uint8`; higher, and float, → `uint16` | tiles (groups) | first frame of animations; CMYK not yet |
| PNG | ✅ | ✅ | 8, 16 (palette and low bit depths expanded) | rows | first frame of APNG; Adam7 decoded fully |
| JPEG | ✅ | ✅ | 8 | — | baseline and progressive; CMYK/YCCK converted to RGB |
| AVIF | detected | — | — | — | decoding planned |

Output channel layouts are `L`, `LA`, `RGB` and `RGBA`.

## Colour

JPEG XL images are delivered in their original colour space; ICC-tagged lossy images are
converted with the pure-Rust `moxcms` colour management system. Other formats return the
stored samples unchanged; colour profiles are not applied yet.

## Orientation

Images are delivered upright: the orientation in a JPEG XL header, and the EXIF orientation
tag of JPEG and PNG files, are applied automatically. `width`, `height` and every crop
coordinate refer to the upright image; `source_size` is the size as stored.

The EXIF rotation costs nothing extra in most pipelines: it joins your own flips and rotations
in a single pass, and crops are still pushed into the decoder. This also keeps JPEG and JPEG XL
consistent, because libjxl moves the EXIF orientation into the JPEG XL header when it
transcodes a JPEG.

## Writing images

`save` picks the format from the file extension; `to_jxl`, `to_png` and `to_jpeg` return
bytes. Every writer runs the image's pending transformations first.

```python
image = baryon.open("photo.jpg")
image.save("photo.jxl")  # lossless JPEG transcoding, ~20% smaller
image.crop(0, 0, 512, 512).save("crop.png", compression="fast")
image.resize(320, 240).save("thumb.jpg", quality=85)

jpeg = baryon.jxl_to_jpeg("photo.jxl")  # the original JPEG, bit for bit
jxl = baryon.open("scan.png").to_jxl(effort=7)  # lossless pixels
jxl = image.to_jxl(lossless=False, distance=1.0)  # lossy, visually lossless
```

| Format | Encoder | Options |
|---|---|---|
| JPEG XL | libjxl (built into the wheel) | `lossless`, `distance`, `effort`, `transcode_jpeg`, `jpeg_fallback` |
| PNG | `png` crate with `fdeflate` | `compression`: `"fast"`, `"balanced"`, `"high"` |
| JPEG | `jpeg-encoder` (AVX2 when available) | `quality` 1–100; 8-bit L or RGB only |

An option that does not apply to the chosen format, such as `quality` for a PNG, raises
`ValueError` instead of being ignored.

To convert many files, `save_all` decodes, encodes and writes on every core, largest images
first:

```python
images = baryon.open_all(sources)
baryon.save_all(images, [p.with_suffix(".jxl") for p in sources])
```

About 1% of JPEGs cannot be transcoded losslessly; pass `jpeg_fallback="lossy"` to encode
those from their pixels instead of raising (`save_all` returns their positions). See the
[JPEG XL guide](guide/jpeg-xl.md#recommendation) for the details.

See the [JPEG XL guide](guide/jpeg-xl.md) for when to convert to JPEG XL.
