# Roadmap

baryon's goal is fast, lazy, region-aware decoding of media for Python data pipelines, with
JPEG XL as the flagship image format. Images come first; video and audio follow on the same
foundation.

## Architecture

Everything is built on three layers, so new media types reuse what exists:

| Layer | Shared by all media | Images today |
|---|---|---|
| **Sources** | detect the container, read headers without decoding | JPEG, PNG, JPEG XL |
| **Lazy plans** | record operations, optimize, push work into decoders | crop, flip, rotate, resize |
| **Execution** | one thread pool, batches, zero-copy NumPy output | `to_numpy`, `to_numpy_all` |

In code, the shared runtime (errors, format detection with a media type per format, and the
thread-pool executor) is the `baryon-core` crate. Each media type is its own crate on top of
it: `baryon-image` today, `baryon-video` and `baryon-audio` later. In Python, images stay at
the top level (`baryon.open`, `baryon.Image`), and video and audio will live in
`baryon.video` and `baryon.audio`, sharing the same thread pool and conventions.

A video adds a time axis to the same model: a frame range is to a video what a crop is to an
image, so seeking to a keyframe is "region pushdown" in time. Audio is a one-dimensional
signal where time ranges and resampling play the role of crops and resizes.

## Images

**Now**

- [x] JPEG, PNG and JPEG XL decoding
- [x] lazy pipelines with region pushdown (JPEG XL tiles, PNG rows)
- [x] all cores: shared pool, batch decoding, fork-safe
- [ ] lossless JPEG → JPEG XL conversion and JPEG XL encoding
- [ ] faster full-image JPEG XL decoding
- [ ] JPEG: skip blocks outside a crop, decode at 1/2, 1/4 and 1/8 scale
- [ ] JPEG XL: region decoding for rotated files, progressive previews
- [ ] EXIF orientation and ICC colour management for JPEG and PNG

**Next**

- [ ] AVIF and WebP decoding
- [ ] resize pushdown: decode JPEG XL and JPEG directly at a reduced scale
- [ ] colour conversions (grayscale, RGB/BGR, float output) inside the plan
- [ ] PyTorch tensor output via DLPack

## Video (future)

- [ ] container demuxing (MP4, MKV, WebM) with lazy frame-range selection
- [ ] keyframe-aware seeking: decode only the frames a range needs
- [ ] the same crop/resize pipeline per frame, batched across cores
- [ ] `(frames, height, width, channels)` arrays

## Audio (future)

- [ ] decoding of common codecs (FLAC, Opus, MP3, AAC, WAV)
- [ ] lazy time ranges and channel selection
- [ ] resampling inside the plan
- [ ] `(samples, channels)` arrays
