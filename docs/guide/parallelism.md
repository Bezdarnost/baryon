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
thread. If an image fails, the error names its position, e.g.
`DecodeError: images[17]: failed to decode jxl image: ...`.

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
