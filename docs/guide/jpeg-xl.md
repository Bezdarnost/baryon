# JPEG XL

JPEG XL is baryon's flagship format: it compresses better than JPEG and PNG, can convert
existing JPEGs **without any loss**, and is coded in independent tiles, which is exactly what
region-aware decoding needs.

## Why it fits data pipelines

- **Lossless JPEG transcoding.** A JPEG can be re-encoded as JPEG XL about 20% smaller, and
  the original file reconstructed bit for bit. No generation loss, ever.
- **Better lossless compression.** PNG-style lossless images shrink by about 30%.
- **Tiles.** Images are coded in independent groups of 256×256 pixels. baryon decodes only
  the groups a crop touches, so random crops (the bread and butter of training) cost a
  fraction of a full decode.
- **One format for everything.** 8–32-bit, float, alpha, HDR, animation and huge dimensions.

## Measured on a real dataset

A 47 GB dataset of 6,835 photos and renders (median 16 megapixels), converted with libjxl and
every file verified:

| Source | Files | Before | After | Saving | Verified |
|---|---:|---:|---:|---:|---|
| JPEG, lossless transcoding | 6,676 | 46.15 GB | 36.65 GB | **20.6%** | every file reconstructs bit-exact |
| PNG, lossless (effort 7) | 78 | 0.81 GB | 0.56 GB | **31.1%** | every pixel identical |

- Per-file JPEG savings ranged from 4.9% to 58% (median 18.3%).
- 81 JPEGs (1.2%) were rejected by libjxl's transcoder; keep those as JPEG.
- Re-encoding JPEG *pixels* losslessly instead of transcoding made files **2.6× larger** than
  the JPEGs. Always transcode JPEGs.

Encoding cost, per CPU core: JPEG transcoding ran at 7.7 MB/s of input (median 0.57 s per
file; the whole 46 GB took 9 minutes on 16 cores). For PNG, effort controls the trade-off:

| Effort | Saving vs PNG | Speed per core |
|---:|---:|---:|
| 1 | 13.1% | 345 MP/s |
| 3 | 27.0% | 16 MP/s |
| 7 (default) | 31.1% | 3.2 MP/s |

## Reading speed

One thread, median 16-megapixel photo (see [Benchmarks](../benchmarks.md) for all numbers):

| Read | Full image | 512×512 crop |
|---|---:|---:|
| the JPEG (baryon) | 85 ms | 84 ms |
| the same photo as lossless JPEG XL (baryon) | 152 ms | **23 ms** |
| the same JPEG XL with libjxl (imagecodecs) | 156 ms | 158 ms |

- Full-image JPEG XL decoding in baryon is as fast as libjxl, the reference decoder. It does
  more work per pixel than JPEG, so a full decode costs about 1.8× the JPEG's.
- Crops decode only the tiles they touch: 3.6× faster than decoding the JPEG and 6.8× faster
  than libjxl, which has no region decoding.
- Decoding scales across cores: 47 full 16-megapixel JPEG XL images per second on 16 threads.

## Recommendation

**Store datasets as JPEG XL when:**

- storage or bandwidth matters: transcoding saves about 20% on JPEGs at zero quality cost and
  is reversible, so the original JPEGs can always be restored;
- training reads crops or tiles of large images: decoding only the touched tiles is several
  times faster than decoding the JPEG;
- the data is lossless (PNG, scans, renders, medical, satellite): about 30% smaller at the
  default effort, with every pixel preserved.

**Keep JPEG when** a pipeline decodes every image in full at high throughput and CPU, not
storage or I/O, is the bottleneck: a full JPEG XL decode costs about 1.8× the CPU of the JPEG.

**About 1% of JPEGs cannot be transcoded.** In a 6,757-photo dataset, libjxl refused 82: 79
progressive photos, mostly large (median 30 megapixels), whose restore data outgrows a limit of
the JPEG XL format, one with 6 MB of data after the image, and two with an ICC profile libjxl
cannot parse. By default these raise `EncodeError`. With `jpeg_fallback="lossy"` they are
encoded from their pixels at `distance` instead (visually lossless at the default 1.0; 29%
smaller than the JPEGs in that dataset, but no longer restorable bit for bit), and
`jpeg_fallback="lossless"` keeps every pixel at several times the JPEG's size. Keeping them as
JPEG is the other option.

## Converting with baryon

```python
import baryon

jxl = baryon.open("photo.jpg").to_jxl()  # lossless transcoding, about 20% smaller
assert baryon.jxl_to_jpeg(jxl) == open("photo.jpg", "rb").read()  # bit-exact restore

jxl = baryon.open("render.png").to_jxl()  # lossless pixels, about 30% smaller
jxl = baryon.open("render.png").to_jxl(effort=3)  # faster, slightly larger
jxl = baryon.open("photo.jpg").resize(1024, 768).to_jxl(lossless=False)  # lossy thumbnail
```

Converting a whole directory on every core:

```python
from pathlib import Path

import baryon

paths = sorted(Path("dataset").rglob("*.jpg"))
for start in range(0, len(paths), 256):
    chunk = paths[start : start + 256]
    targets = [p.with_suffix(".jxl") for p in chunk]
    lossy = baryon.save_all(baryon.open_all(chunk), targets, jpeg_fallback="lossy")
    for i in lossy:
        print(f"{chunk[i]} could not be transcoded; encoded lossy instead")
```

`save_all` decodes, encodes and writes on every thread, largest files first, and returns the
positions of the files that used the fallback. If any file fails, the rest of the chunk is
still written and the first failure is raised with the positions of all the others in its
message; its `failures` attribute holds every `(position, exception)` pair, so you can retry
or skip exactly those files. `benchmarks/bench_convert.py` converts a sample of your data and
verifies every file, so you can check the savings before converting everything.
