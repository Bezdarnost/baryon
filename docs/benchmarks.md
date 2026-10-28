# Benchmarks

All numbers come from the scripts in
[`benchmarks/`](https://github.com/Bezdarnost/baryon/tree/main/benchmarks), run on a real
dataset of photos and renders (JPEG median 16 megapixels). Each library runs in its own
process, files are read into memory first, and every number is the best of two runs per
image. Run them on your own images; results depend on content and hardware.

**Machine:** AMD Ryzen 9 9950X (16 cores, 32 threads), Linux, Python 3.12.
**Versions:** baryon 0.1.0, Pillow 13.0, OpenCV 5.0, pyvips 3.2 (libvips 8.18),
imagecodecs 2026.10.10 (libjpeg-turbo 3.2, libjxl 0.12), pillow-jxl-plugin 1.3.8.

## Decoding, one thread

500 random JPEGs and all 78 PNGs from the dataset.

| Library | JPEG full | JPEG MP/s | PNG full | PNG MP/s | PNG 512px crop |
|---|---:|---:|---:|---:|---:|
| **baryon** | 83.4 ms | 214 | **27.9 ms** | **296** | **16.0 ms** |
| imagecodecs (libjpeg-turbo) | **69.0 ms** | **254** | 34.7 ms | 218 | 34.7 ms |
| pyvips | 77.8 ms | 226 | 40.5 ms | 204 | 41.6 ms |
| OpenCV | 77.3 ms | 225 | 58.1 ms | 140 | 58.8 ms |
| Pillow | 98.9 ms | 174 | 64.6 ms | 126 | 60.6 ms |

- **PNG:** baryon is the fastest: 1.4× the next library and 2.3× Pillow, and its crops
  stop decoding after the last needed row.
- **JPEG:** baryon's pure-Rust decoder is within 10% of OpenCV and pyvips, 17% behind raw
  libjpeg-turbo, and 16% ahead of Pillow (which also pays for an array copy).

## JPEG XL, one thread

The same JPEGs transcoded losslessly to JPEG XL, plus the PNGs at effort 7. Pillow, OpenCV and
pyvips wheels cannot read JPEG XL.

| Library | Full image | 512px crop |
|---|---:|---:|
| **baryon** (jxl-oxide) | 503 ms | **25 ms** |
| imagecodecs (libjxl) | **180 ms** | 182 ms |

Region decoding makes baryon's crops 7× faster than libjxl's, but its full decodes are 2.8×
slower. Closing that gap is in progress.

## Multi-core and conversion

Batch throughput with all cores and JPEG XL conversion speed per library are being measured
with the new suite; this page will be updated with the results.
