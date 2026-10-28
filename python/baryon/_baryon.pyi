import os
from typing import Any, Literal, TypeAlias, final

import numpy as np
import numpy.typing as npt

__version__: str

Mode: TypeAlias = Literal["L", "LA", "RGB", "RGBA"]
DType: TypeAlias = Literal["uint8", "uint16"]
Format: TypeAlias = Literal["jpeg", "png", "jxl", "avif"]
Filter: TypeAlias = Literal[
    "nearest", "box", "bilinear", "hamming", "bicubic", "mitchell", "lanczos3"
]

class BaryonError(Exception):
    """Base class for every error raised by baryon."""

class DecodeError(BaryonError):
    """The image data is corrupt or could not be decoded."""

class UnsupportedFormatError(BaryonError):
    """The image format is not recognized or not supported by this build."""

def open(source: str | os.PathLike[str] | bytes | bytearray | memoryview) -> Image:
    """Open an image lazily; only the header is parsed."""

@final
class Image:
    """A lazily decoded image. Transformations return new images and do no work."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` after all pending transformations."""
    @property
    def shape(self) -> tuple[int, int, int]:
        """``(height, width, channels)`` of the array :meth:`to_numpy` returns."""
    @property
    def channels(self) -> int: ...
    @property
    def mode(self) -> Mode: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def format(self) -> Format: ...
    @property
    def source_size(self) -> tuple[int, int]:
        """``(width, height)`` of the encoded source, before any transformation."""
    @property
    def progressive(self) -> bool: ...
    def crop(self, x: int, y: int, width: int, height: int) -> Image: ...
    def resize(self, width: int, height: int, *, filter: Filter = "lanczos3") -> Image: ...
    def rotate(self, degrees: int) -> Image:
        """Rotate counter-clockwise by a multiple of 90 degrees (like ``numpy.rot90``)."""
    def flip_horizontal(self) -> Image: ...
    def flip_vertical(self) -> Image: ...
    def explain(self, *, optimized: bool = True) -> str: ...
    def to_numpy(self) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
        """Decode, apply pending transformations and return a ``(H, W, C)`` array."""
    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray[Any]: ...
