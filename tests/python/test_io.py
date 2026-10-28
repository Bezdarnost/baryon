from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image as PILImage

import baryon
from conftest import gradient

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import Encoder


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_png_decodes_exactly(encode: Encoder, channels: int) -> None:
    expected = gradient(37, 23, channels)
    image = baryon.open(encode(expected, "PNG"))

    assert image.format == "png"
    assert image.shape == expected.shape
    assert image.mode == {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[channels]
    np.testing.assert_array_equal(image.to_numpy(), expected)


def test_png_16_bit_is_preserved() -> None:
    expected = (np.arange(6 * 4, dtype=np.uint16) * 2731).reshape(4, 6)
    buf = io.BytesIO()
    PILImage.fromarray(expected).save(buf, format="PNG")

    image = baryon.open(buf.getvalue())

    assert image.dtype == "uint16"
    array = image.to_numpy()
    assert array.dtype == np.uint16
    np.testing.assert_array_equal(array[:, :, 0], expected)


def test_interlaced_png(adam7_png: Path) -> None:
    image = baryon.open(adam7_png)

    assert image.progressive
    array = image.to_numpy()
    y, x = np.mgrid[0:30, 0:40]
    np.testing.assert_array_equal(array, np.stack([x, y, x ^ y], axis=-1))


@pytest.mark.parametrize("channels", [1, 3])
def test_jpeg_matches_pillow_closely(encode: Encoder, channels: int) -> None:
    data = encode(gradient(64, 48, channels), "JPEG", quality=95)
    reference = np.asarray(PILImage.open(io.BytesIO(data)))
    reference = reference.reshape(48, 64, channels)

    image = baryon.open(data)

    assert image.format == "jpeg"
    assert image.shape == (48, 64, channels)
    # Decoders may differ slightly in IDCT and chroma upsampling.
    diff = np.abs(image.to_numpy().astype(np.int16) - reference.astype(np.int16))
    assert diff.max() <= 3
    assert diff.mean() < 0.5


def test_open_accepts_paths_and_buffers(tmp_path: Path, encode: Encoder) -> None:
    data = encode(gradient(8, 8), "PNG")
    path = tmp_path / "image.png"
    path.write_bytes(data)
    expected = gradient(8, 8)

    for source in (str(path), path, data, bytearray(data), memoryview(data)):
        np.testing.assert_array_equal(baryon.open(source).to_numpy(), expected)


def test_metadata_needs_no_decode(encode: Encoder) -> None:
    data = encode(gradient(300, 200), "PNG")
    # Drop the second half of the pixel data: the header is intact, so opening still works.
    broken = data[: len(data) // 2]

    image = baryon.open(broken)

    assert image.size == (300, 200)
    assert image.source_size == (300, 200)
    with pytest.raises(baryon.DecodeError):
        image.to_numpy()


def test_unrecognized_data_raises() -> None:
    with pytest.raises(baryon.UnsupportedFormatError, match="unrecognized"):
        baryon.open(b"definitely not an image")


def test_recognized_but_unsupported_format() -> None:
    avif_header = b"\x00\x00\x00\x14ftypavif\x00\x00\x00\x00mif1"
    with pytest.raises(baryon.UnsupportedFormatError, match="not implemented yet"):
        baryon.open(avif_header)


def test_corrupt_header_raises_decode_error(encode: Encoder) -> None:
    data = encode(gradient(16, 16), "JPEG")
    with pytest.raises(baryon.DecodeError):
        baryon.open(data[:20])


def test_missing_file_raises_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        baryon.open(tmp_path / "missing.png")


def test_rejects_unsupported_source_types() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        baryon.open(123)  # type: ignore[arg-type]


def test_errors_share_a_base_class() -> None:
    assert issubclass(baryon.DecodeError, baryon.BaryonError)
    assert issubclass(baryon.UnsupportedFormatError, baryon.BaryonError)


def test_version_is_exposed() -> None:
    assert baryon.__version__.count(".") == 2
