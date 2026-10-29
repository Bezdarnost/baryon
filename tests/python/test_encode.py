"""JPEG XL encoding with the vendored libjxl, and JPEG reconstruction."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageOps

import baryon
from conftest import FIXTURES, gradient

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import Encoder


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_lossless_png_round_trips_exactly(encode: Encoder, channels: int) -> None:
    pixels = gradient(83, 61, channels)
    jxl = baryon.open(encode(pixels, "PNG")).to_jxl()

    assert jxl[:2] == b"\xff\x0a"
    np.testing.assert_array_equal(baryon.open(jxl).to_numpy(), pixels)


def test_sixteen_bit_round_trips_exactly() -> None:
    image = baryon.open(FIXTURES / "gradient-rgba16-lossless.jxl")
    again = baryon.open(image.to_jxl())

    assert again.dtype == "uint16"
    np.testing.assert_array_equal(again.to_numpy(), image.to_numpy())


def test_jpeg_transcoding_is_bit_exact_and_reversible(encode: Encoder, tmp_path: Path) -> None:
    original = encode(gradient(256, 192), "JPEG", quality=90)
    jxl = baryon.open(original).to_jxl()

    assert baryon.jxl_to_jpeg(jxl) == original
    path = tmp_path / "photo.jxl"
    path.write_bytes(jxl)
    assert baryon.jxl_to_jpeg(path) == original
    # Decoding the JPEG XL and decoding the JPEG run different IDCT and chroma upsampling
    # code; like any two JPEG decoders, they agree to within a few levels.
    np.testing.assert_allclose(
        baryon.open(jxl).to_numpy(), baryon.open(original).to_numpy(), atol=4, rtol=0
    )


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_survives_transcoding(orientation: int) -> None:
    exif = PILImage.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    PILImage.fromarray(gradient(96, 64)).save(buf, "JPEG", quality=95, exif=exif)
    original = buf.getvalue()
    upright = np.asarray(ImageOps.exif_transpose(PILImage.open(io.BytesIO(original))))

    image = baryon.open(original)
    assert (image.width, image.height) == (upright.shape[1], upright.shape[0])
    assert image.source_size == (96, 64)
    np.testing.assert_allclose(image.to_numpy(), upright, atol=4, rtol=0)
    # libjxl moves the EXIF orientation into the JPEG XL header when transcoding, so the
    # JPEG XL file displays the same way.
    np.testing.assert_allclose(
        baryon.open(image.to_jxl()).to_numpy(), image.to_numpy(), atol=4, rtol=0
    )


def test_jpegs_with_comment_segments_reconstruct_exactly() -> None:
    # Some JPEG XL decoders reconstruct comment (COM) segments incorrectly; libjxl does not.
    buf = io.BytesIO()
    PILImage.fromarray(gradient(200, 120)).save(buf, "JPEG", comment=b"CREATOR: baryon test")
    original = buf.getvalue()
    assert b"\xff\xfe" in original[:64]

    assert baryon.jxl_to_jpeg(baryon.open(original).to_jxl()) == original


def test_transformed_jpegs_are_encoded_from_pixels(encode: Encoder) -> None:
    image = baryon.open(encode(gradient(256, 192), "JPEG")).crop(10, 20, 100, 50).rotate(90)
    jxl = image.to_jxl()

    np.testing.assert_array_equal(baryon.open(jxl).to_numpy(), image.to_numpy())
    with pytest.raises(baryon.DecodeError, match="no reconstructable JPEG"):
        baryon.jxl_to_jpeg(jxl)


def test_jpeg_pixels_can_be_encoded_instead_of_transcoded(encode: Encoder) -> None:
    image = baryon.open(encode(gradient(64, 48), "JPEG"))
    jxl = image.to_jxl(transcode_jpeg=False)

    np.testing.assert_array_equal(baryon.open(jxl).to_numpy(), image.to_numpy())
    with pytest.raises(baryon.DecodeError):
        baryon.jxl_to_jpeg(jxl)


def test_lossy_encoding_is_close_and_smaller() -> None:
    image = baryon.open(FIXTURES / "smooth-rgb8-lossy.jxl")
    lossless = image.to_jxl(effort=3)
    lossy = image.to_jxl(lossless=False, distance=2.0, effort=3)

    assert len(lossy) < len(lossless)
    error = np.abs(baryon.open(lossy).to_numpy().astype(int) - image.to_numpy()).mean()
    assert error < 3


def test_to_jxl_all_matches_one_by_one(encode: Encoder) -> None:
    images = [baryon.open(encode(gradient(40, 30, c), "PNG")).flip_horizontal() for c in (1, 3, 4)]
    assert baryon.to_jxl_all(images, effort=3) == [im.to_jxl(effort=3) for im in images]


def test_to_jxl_all_reports_the_failing_position(encode: Encoder) -> None:
    data = encode(gradient(300, 200), "PNG")
    ok, truncated = baryon.open(data), baryon.open(data[: len(data) // 2])
    with pytest.raises(baryon.DecodeError, match=r"^images\[1\]: failed to decode png"):
        baryon.to_jxl_all([ok, truncated, ok], effort=3)


def test_rejects_out_of_range_settings(encode: Encoder) -> None:
    image = baryon.open(encode(gradient(16, 16), "PNG"))
    with pytest.raises(ValueError, match="effort"):
        image.to_jxl(effort=0)
    with pytest.raises(ValueError, match="distance"):
        image.to_jxl(lossless=False, distance=26.0)
    with pytest.raises(OverflowError):
        image.to_jxl(effort=1000)


def test_jxl_to_jpeg_rejects_other_formats(encode: Encoder) -> None:
    with pytest.raises(baryon.UnsupportedFormatError):
        baryon.jxl_to_jpeg(encode(gradient(8, 8), "PNG"))


def test_encode_error_is_a_baryon_error() -> None:
    assert issubclass(baryon.EncodeError, baryon.BaryonError)
