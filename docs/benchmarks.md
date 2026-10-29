# Benchmarks

All numbers come from the scripts in
[`benchmarks/`](https://github.com/Bezdarnost/baryon/tree/main/benchmarks), run on a real
dataset of photos and renders. Each library runs in its own process, files are read into
memory first (disk speed is excluded), and latency is the best of two runs per image. Run them
on your own images; results depend on content and hardware.

- **Images:** 200 random JPEGs (median 16 megapixels, up to 100 MP), all 78 PNGs, and the
  same JPEGs and PNGs converted losslessly to JPEG XL.
- **Machine:** AMD Ryzen 9 9950X (16 cores, 32 threads), Linux, Python 3.12.
- **Versions:** baryon 0.1.0, Pillow 13.0, OpenCV 5.0, pyvips 3.2 (libvips 8.18),
  imagecodecs 2026.10.10 (libjpeg-turbo 3.2, libjxl 0.12), pillow-jxl-plugin 1.3.8,
  NumPy 2.5.
- **Batches** decode 64 images at a time with every thread: `baryon.to_numpy_all` for
  baryon, a thread pool for the others (they all release the GIL while decoding).
- Pillow, OpenCV and pyvips wheels cannot read JPEG XL; pillow-jxl is the Pillow plugin.

## Summary

- **PNG:** baryon is the fastest single-threaded decoder, 1.4x imagecodecs and 2.3x Pillow,
  and its crops stop decoding after the last needed row (16.5 ms vs 35-59 ms).
- **JPEG XL:** baryon decodes full images as fast as libjxl (84 vs 79 MP/s on one thread)
  and crops **6.8x faster** than libjxl's full decode, because only the touched tiles are
  decoded.
- **JPEG:** baryon's pure-Rust decoder matches OpenCV and pyvips, trails raw libjpeg-turbo
  (imagecodecs) by about 14%, and scales on 16 threads to 2.4x Pillow's batch throughput.
- **Conversion to JPEG XL:** baryon, imagecodecs and pillow-jxl all use libjxl, so sizes are
  identical (20.7% smaller JPEGs, 31.1% smaller PNGs) and speed is within 10%. Every output
  was verified lossless by decoding it with libjxl.

## Reading

### JPEG: one image at a time, 1 thread

| Library | Full image (median) | Throughput | 512x512 crop (median) |
|---|---:|---:|---:|
| baryon | 85.0 ms | 211 MP/s | 84.2 ms |
| imagecodecs | 73.8 ms | 245 MP/s | 73.5 ms |
| pyvips | 84.0 ms | 217 MP/s | 86.5 ms |
| opencv | 83.9 ms | 213 MP/s | 82.9 ms |
| pillow | 101.5 ms | 175 MP/s | 79.8 ms |

### JPEG: batches on 16 threads

| Library | Images/s | Throughput | Speed-up vs 1 thread |
|---|---:|---:|---:|
| baryon | 74.4 | 1461 MP/s | 7.2x |
| imagecodecs | 85.7 | 1682 MP/s | 7.0x |
| pyvips | 56.9 | 1116 MP/s | 5.3x |
| opencv | 67.6 | 1327 MP/s | 6.9x |
| pillow | 30.6 | 600 MP/s | 3.3x |

### JXL: one image at a time, 1 thread

| Library | Full image (median) | Throughput | 512x512 crop (median) |
|---|---:|---:|---:|
| baryon | 151.6 ms | 84 MP/s | 23.3 ms |
| imagecodecs | 155.9 ms | 79 MP/s | 157.8 ms |
| pillow-jxl | 451.5 ms | 32 MP/s | 435.2 ms |

### JXL: batches on 16 threads

| Library | Images/s | Throughput | Speed-up vs 1 thread |
|---|---:|---:|---:|
| baryon | 47.1 | 883 MP/s | 10.6x |
| imagecodecs | 54.7 | 1025 MP/s | 13.1x |
| pillow-jxl | 10.0 | 187 MP/s | 5.8x |

### PNG: one image at a time, 1 thread

| Library | Full image (median) | Throughput | 512x512 crop (median) |
|---|---:|---:|---:|
| baryon | 28.3 ms | 300 MP/s | 16.5 ms |
| imagecodecs | 34.4 ms | 219 MP/s | 34.5 ms |
| pyvips | 39.9 ms | 204 MP/s | 40.6 ms |
| opencv | 60.0 ms | 140 MP/s | 59.3 ms |
| pillow | 61.1 ms | 128 MP/s | 58.3 ms |

### PNG: batches on 16 threads

| Library | Images/s | Throughput | Speed-up vs 1 thread |
|---|---:|---:|---:|
| baryon | 114.3 | 1605 MP/s | 5.9x |
| imagecodecs | 122.7 | 1724 MP/s | 8.1x |
| pyvips | 87.2 | 1225 MP/s | 6.2x |
| opencv | 78.3 | 1100 MP/s | 7.9x |
| pillow | 44.5 | 626 MP/s | 4.8x |

## Converting to JPEG XL

JPEGs are transcoded losslessly (the original file is restorable bit for bit); PNGs are
encoded pixel-lossless at libjxl's default effort 7. "Speed per thread" divides input bytes by
the time each file took, so with 16 files in flight it shows per-thread cost under load; the
last table shows the wall-clock throughput of whole runs. One JPEG in the sample cannot be
transcoded by libjxl and is reported as failed by every library.

### JPEG to JPEG XL (lossless transcoding)

| Library | Threads | Files | Saving | Speed per thread | Lossless | Failed |
|---|---:|---:|---:|---:|---:|---:|
| baryon | 1 | 199 | 20.7% | 13.2 MB/s | 199/199 | 1 |
| imagecodecs | 1 | 199 | 20.7% | 13.1 MB/s | 199/199 | 1 |
| pillow-jxl | 1 | 199 | 20.7% | 11.3 MB/s | 199/199 | 1 |
| baryon | 16 | 199 | 20.7% | 3.1 MB/s | 199/199 | 1 |
| imagecodecs | 16 | 199 | 20.7% | 7.2 MB/s | 199/199 | 1 |
| pillow-jxl | 16 | 199 | 20.7% | 5.5 MB/s | 199/199 | 1 |

### PNG to JPEG XL (lossless)

| Library | Threads | Files | Saving | Speed per thread | Lossless | Failed |
|---|---:|---:|---:|---:|---:|---:|
| baryon | 1 | 78 | 31.1% | 2.8 MB/s | 78/78 | 0 |
| imagecodecs | 1 | 78 | 31.1% | 2.7 MB/s | 78/78 | 0 |
| pillow-jxl | 1 | 78 | 31.1% | 2.7 MB/s | 78/78 | 0 |
| baryon | 16 | 78 | 31.1% | 3.9 MB/s | 78/78 | 0 |
| imagecodecs | 16 | 78 | 31.1% | 2.2 MB/s | 78/78 | 0 |
| pillow-jxl | 16 | 78 | 31.1% | 2.1 MB/s | 78/78 | 0 |

### Whole conversion runs (all formats, wall clock)

| Library | Threads | Input | Wall time | Throughput |
|---|---:|---:|---:|---:|
| baryon | 1 | 2331 MB | 399.3 s | 5.8 MB/s |
| imagecodecs | 1 | 2331 MB | 419.5 s | 5.6 MB/s |
| pillow-jxl | 1 | 2331 MB | 437.5 s | 5.3 MB/s |
| baryon | 16 | 2331 MB | 44.7 s | 52.2 MB/s |
| imagecodecs | 16 | 2331 MB | 41.9 s | 55.6 MB/s |
| pillow-jxl | 16 | 2331 MB | 47.1 s | 49.5 MB/s |

## Reproducing

```bash
python benchmarks/bench_read.py DATASET --out results/read.json --sample 200 --threads 1 16
python benchmarks/bench_convert.py DATASET --out results/convert.json --sample 200 --threads 1 16
python benchmarks/report.py results/read.json results/convert.json
```
