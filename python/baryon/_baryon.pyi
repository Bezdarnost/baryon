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
JpegFallback: TypeAlias = Literal["error", "lossless", "lossy"]

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
    would raise, with its position prefixed to the message, e.g. ``sources[3]: ...``, and the
    positions of any other failures appended. Its ``failures`` attribute lists every failure
    as a ``(position, exception)`` pair.

    Args:
        sources: File paths and/or bytes-like objects, as accepted by :func:`open`.
    """

def to_numpy_all(images: Sequence[Image]) -> list[Pixels]:
    """Decodes many images in parallel and returns their pixels.

    Equivalent to ``[image.to_numpy() for image in images]``, but runs on all of baryon's
    threads (see :func:`set_num_threads`) with the GIL released once for the whole batch.

    If any image fails, the first failure is raised as the same exception
    :meth:`Image.to_numpy` would raise, with its position prefixed to the message, e.g.
    ``images[3]: ...``, and the positions of any other failures appended. Its ``failures``
    attribute lists every failure as a ``(position, exception)`` pair.

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
    jpeg_fallback: JpegFallback = "error",
) -> list[bytes]:
    """Encodes many images as JPEG XL in parallel; see :meth:`Image.to_jxl` for the options.

    If any image fails, the first failure is raised as the same exception
    :meth:`Image.to_jxl` would raise, with its position prefixed to the message, e.g.
    ``images[3]: ...``, and the positions of any other failures appended. Its ``failures``
    attribute lists every failure as a ``(position, exception)`` pair.

    Returns:
        One encoded file per image, in input order.
    """

def save_all(
    images: Sequence[Image],
    paths: Sequence[str | os.PathLike[str]],
    *,
    format: str | None = None,
    quality: int | None = None,
    compression: Literal["fast", "balanced", "high"] | None = None,
    lossless: bool | None = None,
    distance: float | None = None,
    effort: int | None = None,
    transcode_jpeg: bool | None = None,
    jpeg_fallback: JpegFallback | None = None,
) -> list[int]:
    """Encodes many images and writes them to files, in parallel on all threads.

    This is the fastest way to convert a dataset: decoding, encoding and writing all run
    concurrently, the largest images first. Formats and options work as in
    :meth:`Image.save`; each file's format comes from its own extension unless ``format`` is
    given.

    Args:
        images: The images to write.
        paths: One destination per image.
        format: The format of every file, overriding the extensions.
        quality: JPEG quality; see :meth:`Image.save`.
        compression: PNG compression; see :meth:`Image.save`.
        lossless: JPEG XL option; see :meth:`Image.to_jxl`.
        distance: JPEG XL option; see :meth:`Image.to_jxl`.
        effort: JPEG XL option; see :meth:`Image.to_jxl`.
        transcode_jpeg: JPEG XL option; see :meth:`Image.to_jxl`.
        jpeg_fallback: JPEG XL option; see :meth:`Image.to_jxl`.

    Returns:
        The positions of the JPEGs that could not be transcoded and were encoded as
        ``jpeg_fallback`` asks instead, in increasing order; empty without a fallback.

    Raises:
        ValueError: ``images`` and ``paths`` differ in length, a format cannot be determined,
            or an option does not apply to a file's format or is out of range. Nothing is
            written in that case.

    If writing any image fails, the others are still written and the first failure is raised
    with its position prefixed to the message, e.g. ``images[3]: ...``, and the positions of
    any other failures appended. Its ``failures`` attribute lists every failure as a
    ``(position, exception)`` pair.
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
        """``(width, height)`` as stored in the file, before EXIF orientation and transforms."""
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
    def convert(self, mode: Mode) -> Image:
        """Returns a new image with another channel layout.

        ``mode`` is ``"L"``, ``"LA"``, ``"RGB"`` or ``"RGBA"``. Colour becomes gray with the
        ITU-R BT.601 luma weights (0.299, 0.587, 0.114), as in Pillow, and gray becomes colour by
        copying it into every channel. Alpha is dropped without blending, and added as fully
        opaque.

        Conversions are planned like every other operation: one that reduces the channels
        runs before rotations and flips, and a colour JPEG converted to gray is decoded from
        its luma plane alone, skipping all colour reconstruction.

        Raises:
            ValueError: ``mode`` is not one of the names above.
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
        jpeg_fallback: JpegFallback = "error",
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
            jpeg_fallback: What to do with a JPEG that libjxl cannot transcode, about 1% of
                real-world photos (mostly large progressive files): ``"error"`` (default) raises
                :class:`EncodeError`, ``"lossy"`` encodes its pixels at ``distance``, and
                ``"lossless"`` encodes its pixels exactly, typically several times larger than
                the JPEG. Applies only when transcoding.

        Raises:
            EncodeError: The encoder failed, or the JPEG cannot be transcoded losslessly and
                ``jpeg_fallback`` is ``"error"``.
            ValueError: An option is out of range, or ``jpeg_fallback`` is set while not
                transcoding (lossy or ``transcode_jpeg=False``).
        """
    def to_png(self, *, compression: Literal["fast", "balanced", "high"] = "balanced") -> bytes:
        """Encodes the image, with its pending transformations, as a PNG file.

        PNG is lossless; 16-bit images stay 16-bit.

        Args:
            compression: ``"fast"``, ``"balanced"`` (default) or ``"high"``. ``"fast"`` is
                several times faster for somewhat larger files; ``"high"`` is slower for
                slightly smaller ones.

        Raises:
            ValueError: ``compression`` is not one of the names above.
        """
    def to_jpeg(self, *, quality: int = 90) -> bytes:
        """Encodes the image, with its pending transformations, as a JPEG file.

        Args:
            quality: 1 to 100; 90 (default) is visually close to the original. Below 90 the
                colour channels are stored at half resolution, as most encoders do.

        Raises:
            ValueError: ``quality`` is out of range, or the image has an alpha channel or
                16-bit samples, which JPEG cannot store; use PNG or JPEG XL.
        """
    def save(
        self,
        path: str | os.PathLike[str],
        *,
        format: str | None = None,
        quality: int | None = None,
        compression: Literal["fast", "balanced", "high"] | None = None,
        lossless: bool | None = None,
        distance: float | None = None,
        effort: int | None = None,
        transcode_jpeg: bool | None = None,
        jpeg_fallback: JpegFallback | None = None,
    ) -> None:
        """Encodes the image, with its pending transformations, and writes it to a file.

        The format comes from the extension (``.jpg``/``.jpeg``, ``.png``, ``.jxl``) unless
        ``format`` is given. Encoder options apply to one format each, and passing one for
        another format is an error rather than being silently ignored.

        Args:
            path: The file to write.
            format: ``"jpeg"``, ``"png"`` or ``"jxl"``, overriding the extension.
            quality: JPEG quality; see :meth:`to_jpeg`.
            compression: PNG compression; see :meth:`to_png`.
            lossless: JPEG XL option; see :meth:`to_jxl`.
            distance: JPEG XL option; see :meth:`to_jxl`.
            effort: JPEG XL option; see :meth:`to_jxl`.
            transcode_jpeg: JPEG XL option; see :meth:`to_jxl`.
            jpeg_fallback: JPEG XL option; see :meth:`to_jxl`.

        Raises:
            ValueError: The format cannot be determined, an option does not apply to it or is
                out of range, or the format cannot store this image (JPEG has no alpha channel
                and no 16-bit samples).
            EncodeError: The encoder failed.
            OSError: The file could not be written.
        """
    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray[Any]:
        """NumPy array protocol: ``numpy.asarray(image)`` decodes like :meth:`to_numpy`."""
