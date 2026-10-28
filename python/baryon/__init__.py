"""Fast, lazy, region-aware image decoding and processing, powered by Rust.

Example:
    >>> import baryon
    >>> image = baryon.open("photo.png").crop(0, 0, 512, 512).resize(256, 256)
    >>> pixels = image.to_numpy()  # decoding happens here, and only for what is needed
    >>> batch = baryon.to_numpy_all(baryon.open_all(paths))  # many images, all cores

"""

from baryon._baryon import (
    BaryonError,
    DecodeError,
    Image,
    UnsupportedFormatError,
    __version__,
    get_num_threads,
    open,  # noqa: A004 - mirrors the familiar ``PIL.Image.open`` entry point
    open_all,
    set_num_threads,
    to_numpy_all,
)

__all__ = [
    "BaryonError",
    "DecodeError",
    "Image",
    "UnsupportedFormatError",
    "__version__",
    "get_num_threads",
    "open",
    "open_all",
    "set_num_threads",
    "to_numpy_all",
]
