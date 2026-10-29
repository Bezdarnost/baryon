import os
from collections.abc import Sequence
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
Source: TypeAlias = str | os.PathLike[str] | bytes | bytearray | memoryview
Pixels: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]

class BaryonError(Exception):
    """Base class for every error raised by baryon."""

class DecodeError(BaryonError):
    """The image data is corrupt or could not be decoded."""

class EncodeError(BaryonError):
    """The image could not be encoded with the requested settings."""

class UnsupportedFormatError(BaryonError):
    """The image format is not recognized or not supported by this build."""

def open(source: Source) -> Image:
    """Opens an image lazily.

    Only the header is parsed; no pixels are decoded until the result is converted to an array.

    Args:
        source: A file path (``str`` or ``os.PathLike``) or the encoded image as a bytes-like
            object.

    Raises:
        UnsupportedFormatError: The format is not recognized or not supported.
        DecodeError: The header is corrupt.
        OSError: The file could not be read.
    """

def open_all(sources: Sequence[Source]) -> list[Image]:
    """Opens many images lazily, reading files and parsing headers in parallel.

    Equivalent to ``[open(s) for s in sources]``, but faster for many files.

    If any source fails, the first failure is raised as the same exception :func:`open`
    would raise, with its position prefixed to the message, e.g. ``sources[3]: ...``.

    Args:
        sources: File paths and/or bytes-like objects, as accepted by :func:`open`.
    """

def to_numpy_all(images: Sequence[Image]) -> list[Pixels]:
    """Decodes many images in parallel and returns their pixels.

    Equivalent to ``[image.to_numpy() for image in images]``, but runs on all of baryon's
    threads (see :func:`set_num_threads`) with the GIL released once for the whole batch.

    If any image fails, the first failure is raised as the same exception
    :meth:`Image.to_numpy` would raise, with its position prefixed to the message, e.g.
    ``images[3]: ...``.

    Args:
        images: Images from :func:`open` or :func:`open_all`, with any pending transformations.

    Returns:
        One ``(height, width, channels)`` array per image, in input order.
    """

def to_jxl_all(
    images: Sequence[Image],
    *,
    lossless: bool = True,
    distance: float = 1.0,
    effort: int = 7,
    transcode_jpeg: bool = True,
) -> list[bytes]:
    """Encodes many images as JPEG XL in parallel; see :meth:`Image.to_jxl` for the options.

    If any image fails, the first failure is raised as the same exception
    :meth:`Image.to_jxl` would raise, with its position prefixed to the message, e.g.
    ``images[3]: ...``.

    Returns:
        One encoded file per image, in input order.
    """

def jxl_to_jpeg(source: Source) -> bytes:
    """Restores the original JPEG file from a JPEG XL file made by lossless JPEG transcoding.

    Args:
        source: A file path or the JPEG XL file as a bytes-like object.

    Raises:
        DecodeError: The file holds no JPEG reconstruction data, e.g. because it was encoded
            from pixels.
        UnsupportedFormatError: The data is not JPEG XL.
    """

def set_num_threads(num_threads: int | None = None) -> None:
    """Sets the number of threads baryon uses for decoding and processing.

    The setting covers everything baryon runs: batch functions such as :func:`to_numpy_all`
    spread images across threads, and a single large image is split across threads where the
    format and operations allow it (JPEG XL decoding, resize, crop, rotate and flip).

    The initial value comes from the ``BARYON_NUM_THREADS`` environment variable, defaulting to
    one thread per logical CPU.

    Args:
        num_threads: Number of threads; ``None`` or ``0`` means one per logical CPU.
    """

def get_num_threads() -> int:
    """Returns the number of threads baryon uses; see :func:`set_num_threads`."""

@final
class Image:
    """A lazily decoded image.

    Transformations such as :meth:`crop` and :meth:`resize` return a new ``Image`` and do no
    work; they are optimized and executed together when pixels are requested with
    :meth:`to_numpy` or ``numpy.asarray(image)``.
    """

    @property
    def width(self) -> int:
        """Output width in pixels, after all pending transformations."""
    @property
    def height(self) -> int:
        """Output height in pixels, after all pending transformations."""
    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` after all pending transformations."""
    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the array :meth:`to_numpy` returns: ``(height, width, channels)``."""
    @property
    def channels(self) -> int:
        """Number of channels: 1 (``L``), 2 (``LA``), 3 (``RGB``) or 4 (``RGBA``)."""
    @property
    def mode(self) -> Mode:
        """Channel layout: ``"L"``, ``"LA"``, ``"RGB"`` or ``"RGBA"``."""
    @property
    def dtype(self) -> DType:
        """NumPy dtype name of the pixel data: ``"uint8"`` or ``"uint16"``."""
    @property
    def format(self) -> Format:
        """Source container format, e.g. ``"jpeg"`` or ``"png"``."""
    @property
    def source_size(self) -> tuple[int, int]:
        """``(width, height)`` of the encoded source image, before any transformation."""
    @property
    def progressive(self) -> bool:
        """Whether the source is stored in several passes (progressive JPEG, interlaced PNG)."""
    def crop(self, x: int, y: int, width: int, height: int) -> Image:
        """Returns a new image limited to the given rectangle.

        Coordinates are relative to this image, after its pending transformations. When the crop
        comes before any resize it is pushed down into the decoder.

        Raises:
            ValueError: The rectangle is empty or extends past the image.
        """
    def resize(self, width: int, height: int, *, filter: Filter = "lanczos3") -> Image:
        """Returns a new image resized to exactly ``width`` x ``height``.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.
            filter: One of ``"nearest"``, ``"box"``, ``"bilinear"``, ``"hamming"``,
                ``"bicubic"``, ``"mitchell"`` or ``"lanczos3"``.

        Raises:
            ValueError: A dimension is zero or the filter is unknown.
        """
    def rotate(self, degrees: int, *, clockwise: bool = True) -> Image:
        """Returns a new image rotated by a multiple of 90 degrees, **clockwise** by default.

        Note:
            Pillow's ``Image.rotate`` and ``numpy.rot90`` turn counter-clockwise for positive
            angles. baryon turns clockwise, like CSS and photo viewers. When porting code from
            those libraries, pass ``clockwise=False`` to keep their behaviour.

        Args:
            degrees: Angle in degrees; any multiple of 90, including negative values.
            clockwise: Rotate clockwise (default) or counter-clockwise.

        Raises:
            ValueError: ``degrees`` is not a multiple of 90.
        """
    def flip_horizontal(self) -> Image:
        """Returns a new image mirrored left to right."""
    def flip_vertical(self) -> Image:
        """Returns a new image mirrored top to bottom."""
    def explain(self, *, optimized: bool = True) -> str:
        """Describes the pipeline.

        Args:
            optimized: Show the physical plan that will run (default) instead of the
                transformations as recorded.
        """
    def to_numpy(self) -> Pixels:
        """Decodes the image, applies all pending transformations and returns the pixels.

        The array always has shape ``(height, width, channels)`` and dtype ``uint8`` or
        ``uint16``; see :attr:`shape` and :attr:`dtype`. The GIL is released while decoding, and
        large images use several threads where possible (see :func:`set_num_threads`). For many
        images, :func:`to_numpy_all` is faster.

        Raises:
            DecodeError: The image data is corrupt.
        """
    def to_jxl(
        self,
        *,
        lossless: bool = True,
        distance: float = 1.0,
        effort: int = 7,
        transcode_jpeg: bool = True,
    ) -> bytes:
        """Encodes the image, with its pending transformations, as a JPEG XL file.

        An untransformed JPEG is transcoded losslessly by default: the result is typically 20%
        smaller and the original file can be restored bit for bit with :func:`jxl_to_jpeg`.
        Everything else is encoded from its pixels.

        Args:
            lossless: Encode without any loss (default). When ``False``, ``distance`` sets
                the quality.
            distance: Butteraugli distance for lossy encoding, 0.0 to 25.0; 1.0 (default) is
                visually lossless, higher is smaller.
            effort: Encoder effort, 1 (fastest) to 10 (smallest); 7 is libjxl's default.
            transcode_jpeg: Transcode untransformed JPEGs instead of encoding their pixels
                (default). Encoding a JPEG's pixels losslessly produces much larger files.

        Raises:
            EncodeError: The encoder failed, or the JPEG cannot be transcoded losslessly (pass
                ``transcode_jpeg=False`` to encode its pixels instead).
            ValueError: ``distance`` or ``effort`` is out of range.
        """
    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray[Any]:
        """NumPy array protocol: ``numpy.asarray(image)`` decodes like :meth:`to_numpy`."""
