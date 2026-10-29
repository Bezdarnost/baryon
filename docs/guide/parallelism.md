# Using all cores

baryon runs all decoding and processing on one shared thread pool, with the GIL released.

## Batches

The fastest way to decode many images is to hand them over together:

```python
images = baryon.open_all(paths)  # parallel reads + headers
crops = [im.crop(0, 0, 512, 512) for im in images]  # lazy, instant
arrays = baryon.to_numpy_all(crops)  # parallel decode, input order
```

`to_numpy_all` releases the GIL once for the whole batch and spreads images over every
thread. The largest images start first (their sizes are known from the headers), so a big
photo never starts last and keeps the other threads waiting; on mixed photo sizes this alone
raised 16-thread JPEG throughput by about 15%. If an image fails, the error names its
position, e.g. `DecodeError: images[17]: failed to decode jxl image: ...`.

`to_jxl_all` and `save_all` schedule the same way, and `save_all` also writes the files from
the worker threads.

## Single large images

One image is split across threads where the work allows it:

| Step | Parallel |
|---|---|
| JPEG XL decoding | yes, by tile |
| resize, crop, rotate, flip | yes, by rows (large images only) |
| JPEG, PNG decoding | no, these formats decode sequentially |

## Controlling the thread count

```python
baryon.set_num_threads(8)  # use 8 threads
baryon.set_num_threads(None)  # one per logical CPU (the default)
baryon.get_num_threads()
```

or set `BARYON_NUM_THREADS` before the first decode. The setting bounds everything baryon
runs, including JPEG XL decoding and resizing.

## Hardware

Wheels are built for the baseline of each architecture (x86-64, ARM64) and pick faster code
paths at runtime, so one wheel runs at full speed on any Intel, AMD or ARM CPU:

| Component | x86-64 | ARM64 |
|---|---|---|
| JPEG decoding (zune-jpeg) | SSE4.1, AVX2 | NEON |
| full JPEG XL decoding (jxl-rs) | SSE4.2, AVX2, AVX-512 | NEON |
| JPEG XL region decoding (jxl-oxide) | SSE4.1, AVX2 + FMA | NEON |
| JPEG XL encoding (libjxl via Highway) | SSE4, AVX2, AVX-512 | NEON |
| resizing (fast_image_resize) | SSE4.1, AVX2 | NEON |
| JPEG encoding (jpeg-encoder) | AVX2 | scalar |

A build for the exact CPU (`RUSTFLAGS="-C target-cpu=native"`) was at most 3% faster on a
Ryzen 9 9950X, so building from source for speed is not worth it.

**SMT (Hyper-Threading).** The default uses every logical CPU. On a 16-core, 32-thread Ryzen
9 9950X, JPEG XL batches decode 25% faster on 32 threads than on 16, while JPEG and PNG
batches are about 5% slower, because their decoders are limited by each core's cache and
memory bandwidth rather than by execution units. If you only decode JPEG or PNG,
`set_num_threads` with the number of physical cores is slightly faster. Intel hybrid CPUs
(P-cores and E-cores) need no tuning: idle threads steal work, so faster cores simply finish
more images.

**Scaling.** Batch throughput grows almost linearly up to about 8 threads; beyond that, memory
bandwidth and lower all-core clock speeds take over. JPEG batches reached about 6× on 8 threads
and 9× on 16 threads of a 9950X.

## Data loaders and multiprocessing

baryon works in PyTorch `DataLoader` workers and `multiprocessing` pools. Thread pools do
not survive `fork()`, so a forked worker transparently starts a fresh pool the first time it
decodes.

When every worker process decodes, limit threads per worker so the total matches your
cores, for example `BARYON_NUM_THREADS=1` with `num_workers=os.cpu_count()`, or a few
workers with several threads each.

## Your own thread pools

`Image` objects are immutable and thread-safe. Decoding from several Python threads at once
is fine; the calls share baryon's pool. The extension also supports free-threaded Python
builds.
