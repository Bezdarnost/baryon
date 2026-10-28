"""Fast, lazy, region-aware image decoding and processing, powered by Rust.

Example:
    >>> import baryon
    >>> image = baryon.open("photo.png").crop(0, 0, 512, 512).resize(256, 256)
    >>> pixels = image.to_numpy()  # decoding happens here, and only for what is needed

"""

from baryon._baryon import (
    BaryonError,
    DecodeError,
    Image,
    UnsupportedFormatError,
    __version__,
    open,  # noqa: A004 - mirrors the familiar ``PIL.Image.open`` entry point
)

__all__ = [
    "BaryonError",
    "DecodeError",
    "Image",
    "UnsupportedFormatError",
    "__version__",
    "open",
]
