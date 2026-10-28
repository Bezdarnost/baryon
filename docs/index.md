# baryon

**Fast, lazy, region-aware image decoding for Python, written in Rust.**

baryon reads images the way data pipelines need them: it records what you want to do, works
out the cheapest way to do it, and pushes the work into the decoder, so pixels you would throw
away are never decoded. JPEG XL is its flagship format.

```python
import baryon

image = baryon.open("scan.jxl").crop(0, 0, 1024, 1024).resize(256, 256)
print(image.explain())  # the optimized plan; nothing decoded yet
pixels = image.to_numpy()  # decodes only the tiles the crop touches
```

!!! warning "Pre-alpha"
    The architecture is in place and tested, but the API may still change before 1.0.

## Why baryon

<div class="grid cards" markdown>

- **Decode less.** Crops, flips and rotations are fused and handed to the decoder. A
  512×512 crop of a large JPEG XL image decodes only the tiles it touches.
- **Use every core.** Batches decode in parallel with one GIL release, and large images split
  across threads. One setting bounds it all.
- **NumPy first.** Arrays are always `(height, width, channels)`, handed over without copying.
- **Pure Rust core.** Memory-safe decoders, no system libraries to install, wheels for every
  major platform.

</div>

## Where to next

- [Getting started](getting-started.md): install and the first five minutes.
- [Lazy pipelines](guide/lazy-pipelines.md): how the optimizer decides what to decode.
- [Using all cores](guide/parallelism.md): batches, thread control and data loaders.
- [JPEG XL](guide/jpeg-xl.md): why and how to store datasets as JPEG XL.
- [Benchmarks](benchmarks.md): measured against Pillow, OpenCV, pyvips and imagecodecs.
