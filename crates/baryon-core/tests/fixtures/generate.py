"""Regenerates the binary test fixtures in this directory.

The fixtures are committed so the test suites need no encoders. Run this only when adding or
changing fixtures:

    uv run --no-project --with numpy --with imagecodecs --with pypng \
        python crates/baryon-core/tests/fixtures/generate.py

JPEG XL files are written by libjxl (through imagecodecs), the reference encoder, so decoding
them exercises the same bitstream features real-world files use. Every image is defined by a
formula that the Rust and Python tests reproduce to compute expected pixels.
"""

from __future__ import annotations

from pathlib import Path

import imagecodecs
import numpy as np
import png

HERE = Path(__file__).parent


def coordinates(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` coordinate grids of shape ``(height, width)``."""
    y, x = np.mgrid[0:height, 0:width]
    return x, y


def gradient_rgb8() -> np.ndarray:
    """40x30 RGB, pixel (x, y) = [x, y, x ^ y]."""
    x, y = coordinates(40, 30)
    return np.stack([x, y, x ^ y], axis=-1).astype(np.uint8)


def gradient_rgba16() -> np.ndarray:
    """40x30 RGBA 16-bit, pixel (x, y) = [1000x, 1000y, 1000(x ^ y), 65535 - 10xy]."""
    x, y = coordinates(40, 30)
    return np.stack([x * 1000, y * 1000, (x ^ y) * 1000, 65535 - 10 * x * y], axis=-1).astype(
        np.uint16
    )


def gradient_gray8() -> np.ndarray:
    """40x30 grayscale, pixel (x, y) = 5x + y."""
    x, y = coordinates(40, 30)
    return (5 * x + y).astype(np.uint8)


def smooth_rgb8() -> np.ndarray:
    """600x450 RGB, smooth enough to survive lossy coding, spanning several 256x256 groups."""
    x, y = coordinates(600, 450)
    r = 128 + 100 * np.sin(x / 37) * np.cos(y / 23)
    g = x * 255 / 599
    b = y * 255 / 449
    return np.stack([r, g, b], axis=-1).round().astype(np.uint8)


def write_jxl(name: str, array: np.ndarray, **options: object) -> None:
    """Encode ``array`` with libjxl and write it next to this script."""
    (HERE / name).write_bytes(imagecodecs.jpegxl_encode(array, **options))


def main() -> None:
    """Write every fixture."""
    rows = gradient_rgb8().reshape(30, -1)
    with (HERE / "gradient-adam7.png").open("wb") as f:
        png.Writer(40, 30, greyscale=False, bitdepth=8, interlace=True).write(f, rows)

    write_jxl("gradient-rgb8-lossless.jxl", gradient_rgb8(), lossless=True)
    write_jxl("gradient-rgba16-lossless.jxl", gradient_rgba16(), lossless=True)
    write_jxl("gradient-gray8-lossless.jxl", gradient_gray8(), lossless=True)
    write_jxl("smooth-rgb8-lossy.jxl", smooth_rgb8(), distance=1.0, effort=7, usecontainer=True)


if __name__ == "__main__":
    main()
