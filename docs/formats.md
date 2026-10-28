# Formats

Formats are recognized from file signatures, never from extensions.

| Format | Read | Bit depths | Partial decoding | Notes |
|---|:---:|---|---|---|
| JPEG XL | ✅ | 8-bit → `uint8`; higher, and float, → `uint16` | tiles (groups) | first frame of animations; orientation applied; CMYK not yet |
| PNG | ✅ | 8, 16 (palette and low bit depths expanded) | rows | first frame of APNG; Adam7 decoded fully |
| JPEG | ✅ | 8 | — | baseline and progressive; CMYK/YCCK converted to RGB |
| AVIF | detected | — | — | decoding planned |

Output channel layouts are `L`, `LA`, `RGB` and `RGBA`.

## Colour

JPEG XL images are delivered in their original colour space; ICC-tagged lossy images are
converted with the pure-Rust `moxcms` colour management system. Other formats return the
stored samples unchanged; colour profiles are not applied yet.

## Orientation

JPEG XL orientation is applied automatically. EXIF orientation in JPEG files is not applied
yet.

## Writing images

Encoding, starting with lossless JPEG XL conversion, is on the [roadmap](roadmap.md).
