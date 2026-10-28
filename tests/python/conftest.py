from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image as PILImage

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    Encoder = Callable[..., bytes]

FIXTURES = Path(__file__).parents[2] / "crates" / "baryon-image" / "tests" / "fixtures"


def gradient(width: int, height: int, channels: int = 3) -> npt.NDArray[np.uint8]:
    """Deterministic image with distinct, structured values in every channel."""
    y, x = np.mgrid[0:height, 0:width]
    planes = [x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1), (x ^ y) & 0xFF]
    planes.append(np.full_like(x, 200))
    return np.stack(planes[:channels], axis=-1).astype(np.uint8)


def _mode(array: npt.NDArray[np.generic]) -> str:
    return {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[array.shape[2]]


def _encode(array: npt.NDArray[np.uint8], fmt: str, **params: object) -> bytes:
    pil = PILImage.fromarray(array.squeeze(axis=2) if array.shape[2] == 1 else array, _mode(array))
    buf = io.BytesIO()
    pil.save(buf, format=fmt, **params)
    return buf.getvalue()


@pytest.fixture
def encode() -> Encoder:
    """Encodes an ``(H, W, C)`` uint8 array as ``"PNG"`` or ``"JPEG"`` bytes with Pillow."""
    return _encode


@pytest.fixture
def adam7_png() -> Path:
    """A 40x30 Adam7-interlaced RGB PNG whose pixel ``(x, y)`` is ``[x, y, x ^ y]``."""
    return FIXTURES / "gradient-adam7.png"
