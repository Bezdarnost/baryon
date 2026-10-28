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

JPEG XL decoding does more work per pixel than JPEG, and full-image decoding in baryon is
currently slower than libjxl's. Crops are where JPEG XL wins:

| Read, one thread, median 16 MP image | Full image | 512×512 crop |
|---|---:|---:|
| JPEG (baryon) | 83 ms | 85 ms |
| JPEG XL transcoded from that JPEG (baryon) | 503 ms | **25 ms** |
| JPEG XL (libjxl via imagecodecs) | 180 ms | 182 ms |

Faster full-image JPEG XL decoding is the top item on the [roadmap](../roadmap.md); the
[benchmarks](../benchmarks.md) page tracks multi-core numbers.

## Recommendation

**Store datasets as JPEG XL when:**

- storage or bandwidth matters: transcoding saves ~20% on JPEGs at zero quality cost and is
  reversible, so the originals can always be restored;
- training reads crops or tiles of large images: decoding only the touched tiles is several
  times faster than any full decode;
- the data is lossless (PNG, scans, renders, medical, satellite): ~30% smaller at effort 7.

**Keep JPEG for now when** the pipeline decodes every image in full at high throughput and
CPU, not storage, is the bottleneck; full-image JPEG XL decoding costs more CPU today.

**How to convert today:** baryon does not encode yet (it is on the roadmap). Use libjxl's
`cjxl` (lossless JPEG transcoding is its default for JPEG input), or from Python:

```python
import imagecodecs

jxl = imagecodecs.jpegxl_encode_jpeg(open("photo.jpg", "rb").read())  # lossless, ~20% smaller
original = imagecodecs.jpegxl_decode_jpeg(jxl)  # bit-exact JPEG back
```

`benchmarks/bench_convert.py` converts a directory sample and verifies every file, so you can
check the savings on your own data before converting everything.
