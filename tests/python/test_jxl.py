"""JPEG XL decoding, using fixtures written by libjxl (see crates/baryon-core/tests/fixtures)."""

from __future__ import annotations

import random

import numpy as np
import pytest

import baryon
from conftest import FIXTURES


def coordinates(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:height, 0:width]
    return x, y


def test_lossless_rgb8_is_exact() -> None:
    image = baryon.open(FIXTURES / "gradient-rgb8-lossless.jxl")

    assert image.format == "jxl"
    assert (image.shape, image.mode, image.dtype) == ((30, 40, 3), "RGB", "uint8")
    x, y = coordinates(40, 30)
    np.testing.assert_array_equal(image.to_numpy(), np.stack([x, y, x ^ y], axis=-1))


def test_lossless_rgba16_is_exact() -> None:
    image = baryon.open(FIXTURES / "gradient-rgba16-lossless.jxl")

    assert (image.mode, image.dtype) == ("RGBA", "uint16")
    x, y = coordinates(40, 30)
    expected = np.stack([x * 1000, y * 1000, (x ^ y) * 1000, 65535 - 10 * x * y], axis=-1)
    array = image.to_numpy()
    assert array.dtype == np.uint16
    np.testing.assert_array_equal(array, expected)


def test_lossless_gray8_is_exact() -> None:
    image = baryon.open(FIXTURES / "gradient-gray8-lossless.jxl")

    assert image.shape == (30, 40, 1)
    x, y = coordinates(40, 30)
    np.testing.assert_array_equal(image.to_numpy()[:, :, 0], 5 * x + y)


@pytest.fixture(scope="module")
def lossy() -> tuple[baryon.Image, np.ndarray]:
    image = baryon.open(FIXTURES / "smooth-rgb8-lossy.jxl")
    return image, image.to_numpy()


def test_lossy_is_close_to_the_source(lossy: tuple[baryon.Image, np.ndarray]) -> None:
    _, decoded = lossy
    x, y = coordinates(600, 450)
    source = np.stack(
        [128 + 100 * np.sin(x / 37) * np.cos(y / 23), x * 255 / 599, y * 255 / 449], axis=-1
    ).round()

    assert decoded.shape == (450, 600, 3)
    assert np.abs(decoded - source).mean() < 1.5


def test_crop_is_decoded_tile_by_tile(lossy: tuple[baryon.Image, np.ndarray]) -> None:
    image, full = lossy
    cropped = image.crop(200, 200, 150, 100)

    assert "tile decoding" in cropped.explain()
    np.testing.assert_array_equal(cropped.to_numpy(), full[200:300, 200:350])


def test_random_pipelines_match_full_decode(lossy: tuple[baryon.Image, np.ndarray]) -> None:
    image, full = lossy
    rng = random.Random(42)
    for _ in range(40):
        img, ref = image, full
        for _ in range(rng.randint(1, 4)):
            choice = rng.randrange(4)
            if choice == 0:
                w, h = rng.randint(1, img.width), rng.randint(1, img.height)
                x, y = rng.randint(0, img.width - w), rng.randint(0, img.height - h)
                img, ref = img.crop(x, y, w, h), ref[y : y + h, x : x + w]
            elif choice == 1:
                img, ref = img.rotate(90), np.rot90(ref, k=-1)
            elif choice == 2:
                img, ref = img.flip_horizontal(), ref[:, ::-1]
            else:
                img, ref = img.flip_vertical(), ref[::-1]
        np.testing.assert_array_equal(img.to_numpy(), ref, err_msg=img.explain())


def test_truncated_file_opens_but_fails_to_decode() -> None:
    data = (FIXTURES / "smooth-rgb8-lossy.jxl").read_bytes()
    image = baryon.open(data[: len(data) // 2])

    assert image.size == (600, 450)
    with pytest.raises(baryon.DecodeError):
        image.to_numpy()
