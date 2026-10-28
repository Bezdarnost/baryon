"""Uniform decode adapters for the libraries compared in the benchmarks.

Every adapter turns encoded bytes into a NumPy array, either the whole image (``full``) or the
512x512 center crop (``crop``), using that library's fastest documented way of doing so.

Thread-pool environment variables must be set before this module is imported; see
``configure_threads``.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

CROP = 512
LIBRARIES = ("baryon", "pillow", "opencv", "pyvips", "imagecodecs", "pillow-jxl")


def configure_threads(threads: int) -> None:
    """Limit every library's internal thread pool. Call before importing any adapter."""
    for var in ("BARYON_NUM_THREADS", "RAYON_NUM_THREADS", "VIPS_CONCURRENCY", "OMP_NUM_THREADS"):
        os.environ[var] = str(threads)


def kind(data: bytes) -> str:
    """Return ``"jpeg"``, ``"png"`` or ``"jxl"`` from the file signature."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG"):
        return "png"
    if data.startswith((b"\xff\x0a", b"\x00\x00\x00\x0cJXL ")):
        return "jxl"
    msg = "unknown image format"
    raise ValueError(msg)


def _center(width: int, height: int) -> tuple[int, int, int, int]:
    w, h = min(CROP, width), min(CROP, height)
    return (width - w) // 2, (height - h) // 2, w, h


@dataclass(frozen=True)
class Adapter:
    """How one library decodes bytes to an array."""

    name: str
    full: Callable[[bytes], np.ndarray]
    crop: Callable[[bytes], np.ndarray]
    formats: frozenset[str]


def load(name: str, threads: int = 1) -> Adapter:  # noqa: C901 - one branch per library
    """Import the library and build its adapter."""
    import numpy as np  # noqa: PLC0415

    if name == "baryon":
        import baryon  # noqa: PLC0415

        baryon.set_num_threads(threads)

        def full(data: bytes) -> np.ndarray:
            return baryon.open(data).to_numpy()

        def crop(data: bytes) -> np.ndarray:
            image = baryon.open(data)
            return image.crop(*_center(image.width, image.height)).to_numpy()

        return Adapter(name, full, crop, frozenset({"jpeg", "png", "jxl"}))

    if name in {"pillow", "pillow-jxl"}:
        from PIL import Image  # noqa: PLC0415

        Image.MAX_IMAGE_PIXELS = None
        formats = {"jpeg", "png"}
        if name == "pillow-jxl":
            import pillow_jxl  # noqa: F401, PLC0415 - registers the JPEG XL plugin

            formats = {"jxl"}

        def full(data: bytes) -> np.ndarray:
            return np.asarray(Image.open(io.BytesIO(data)))

        def crop(data: bytes) -> np.ndarray:
            image = Image.open(io.BytesIO(data))
            x, y, w, h = _center(image.width, image.height)
            return np.asarray(image.crop((x, y, x + w, y + h)))

        return Adapter(name, full, crop, frozenset(formats))

    if name == "opencv":
        import cv2  # noqa: PLC0415

        cv2.setNumThreads(threads)

        def full(data: bytes) -> np.ndarray:
            # IMREAD_UNCHANGED: plain decode in BGR order, without EXIF rotation.
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)

        def crop(data: bytes) -> np.ndarray:
            a = full(data)
            x, y, w, h = _center(a.shape[1], a.shape[0])
            return a[y : y + h, x : x + w].copy()

        return Adapter(name, full, crop, frozenset({"jpeg", "png"}))

    if name == "pyvips":
        import pyvips  # noqa: PLC0415

        pyvips.cache_set_max(0)

        def full(data: bytes) -> np.ndarray:
            return pyvips.Image.new_from_buffer(data, "", access="sequential").numpy()

        def crop(data: bytes) -> np.ndarray:
            image = pyvips.Image.new_from_buffer(data, "", access="random")
            return image.crop(*_center(image.width, image.height)).numpy()

        return Adapter(name, full, crop, frozenset({"jpeg", "png"}))

    if name == "imagecodecs":
        import imagecodecs  # noqa: PLC0415

        def full(data: bytes) -> np.ndarray:
            k = kind(data)
            if k == "jpeg":
                return imagecodecs.jpeg8_decode(data)
            if k == "png":
                return imagecodecs.png_decode(data)
            return imagecodecs.jpegxl_decode(data, numthreads=threads)

        def crop(data: bytes) -> np.ndarray:
            a = full(data)
            x, y, w, h = _center(a.shape[1], a.shape[0])
            return a[y : y + h, x : x + w].copy()

        return Adapter(name, full, crop, frozenset({"jpeg", "png", "jxl"}))

    msg = f"unknown library {name!r}; expected one of {', '.join(LIBRARIES)}"
    raise ValueError(msg)
