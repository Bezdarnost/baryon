from __future__ import annotations

import io
import random
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image as PILImage

import baryon
from conftest import gradient

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from conftest import Encoder

    Array = npt.NDArray[np.uint8]


@pytest.fixture
def source(encode: Encoder) -> tuple[baryon.Image, Array]:
    expected = gradient(53, 41, 4)
    return baryon.open(encode(expected, "PNG")), expected


def test_crop_matches_slicing(source: tuple[baryon.Image, Array]) -> None:
    image, expected = source
    cropped = image.crop(7, 3, 20, 30)

    assert cropped.size == (20, 30)
    np.testing.assert_array_equal(cropped.to_numpy(), expected[3:33, 7:27])


@pytest.mark.parametrize("degrees", [-270, -180, -90, 0, 90, 180, 270, 360, 450])
def test_rotate_is_clockwise_by_default(source: tuple[baryon.Image, Array], degrees: int) -> None:
    image, expected = source
    # numpy.rot90 turns counter-clockwise, so a clockwise turn is a negative k.
    np.testing.assert_array_equal(
        image.rotate(degrees).to_numpy(), np.rot90(expected, k=-degrees // 90)
    )


@pytest.mark.parametrize("degrees", [-270, -90, 90, 180, 270])
def test_rotate_counter_clockwise_matches_numpy(
    source: tuple[baryon.Image, Array], degrees: int
) -> None:
    image, expected = source
    np.testing.assert_array_equal(
        image.rotate(degrees, clockwise=False).to_numpy(), np.rot90(expected, k=degrees // 90)
    )


def test_rotate_top_left_moves_to_top_right(encode: Encoder) -> None:
    corner = np.zeros((2, 3, 1), dtype=np.uint8)
    corner[0, 0] = 255
    rotated = baryon.open(encode(corner, "PNG")).rotate(90).to_numpy()
    assert rotated.shape == (3, 2, 1)
    assert rotated[0, 1, 0] == 255


def test_flips_match_numpy(source: tuple[baryon.Image, Array]) -> None:
    image, expected = source
    np.testing.assert_array_equal(image.flip_horizontal().to_numpy(), expected[:, ::-1])
    np.testing.assert_array_equal(image.flip_vertical().to_numpy(), expected[::-1])


def test_random_pipelines_match_numpy(source: tuple[baryon.Image, Array]) -> None:
    image, expected = source
    rng = random.Random(1234)

    def random_crop(img: baryon.Image, ref: Array) -> tuple[baryon.Image, Array]:
        w = rng.randint(1, img.width)
        h = rng.randint(1, img.height)
        x = rng.randint(0, img.width - w)
        y = rng.randint(0, img.height - h)
        return img.crop(x, y, w, h), ref[y : y + h, x : x + w]

    steps: list[Callable[[baryon.Image, Array], tuple[baryon.Image, Array]]] = [
        random_crop,
        lambda img, ref: (img.rotate(90), np.rot90(ref, k=-1)),
        lambda img, ref: (img.rotate(90, clockwise=False), np.rot90(ref)),
        lambda img, ref: (img.rotate(180), np.rot90(ref, k=2)),
        lambda img, ref: (img.flip_horizontal(), ref[:, ::-1]),
        lambda img, ref: (img.flip_vertical(), ref[::-1]),
    ]
    for _ in range(200):
        img, ref = image, expected
        for _ in range(rng.randint(1, 6)):
            img, ref = rng.choice(steps)(img, ref)
        assert img.shape == ref.shape
        np.testing.assert_array_equal(img.to_numpy(), ref, err_msg=img.explain())


@pytest.mark.parametrize("filt", ["bilinear", "bicubic", "lanczos3"])
def test_resize_is_close_to_pillow(encode: Encoder, filt: str) -> None:
    expected = gradient(120, 90)
    pil_filter = {
        "bilinear": PILImage.Resampling.BILINEAR,
        "bicubic": PILImage.Resampling.BICUBIC,
        "lanczos3": PILImage.Resampling.LANCZOS,
    }[filt]
    reference = np.asarray(PILImage.fromarray(expected).resize((50, 70), pil_filter))

    resized = baryon.open(encode(expected, "PNG")).resize(50, 70, filter=filt)  # type: ignore[arg-type]

    assert resized.shape == (70, 50, 3)
    diff = np.abs(resized.to_numpy().astype(np.int16) - reference.astype(np.int16))
    # Rounding and ringing near the high-frequency XOR channel differ in rare pixels only.
    assert diff.mean() < 0.15
    assert (diff > 1).mean() < 0.005


def test_resize_rejects_bad_arguments(source: tuple[baryon.Image, Array]) -> None:
    image, _ = source
    with pytest.raises(ValueError, match="non-empty"):
        image.resize(0, 10)
    with pytest.raises(ValueError, match="unknown resize filter"):
        image.resize(10, 10, filter="sinc")  # type: ignore[arg-type]


def test_crop_rejects_out_of_bounds(source: tuple[baryon.Image, Array]) -> None:
    image, _ = source
    with pytest.raises(ValueError, match="exceeds image bounds"):
        image.crop(50, 0, 10, 10)
    with pytest.raises(ValueError, match="empty"):
        image.crop(0, 0, 0, 10)
    with pytest.raises(OverflowError):
        image.crop(-1, 0, 10, 10)


def test_rotate_rejects_non_right_angles(source: tuple[baryon.Image, Array]) -> None:
    image, _ = source
    with pytest.raises(ValueError, match="multiple of 90"):
        image.rotate(45)


def test_transforms_are_immutable(source: tuple[baryon.Image, Array]) -> None:
    image, expected = source
    image.crop(0, 0, 5, 5)
    image.rotate(90)
    assert image.size == (53, 41)
    np.testing.assert_array_equal(image.to_numpy(), expected)


def test_explain_shows_pushdown(source: tuple[baryon.Image, Array]) -> None:
    image = source[0].rotate(90).crop(0, 0, 10, 20).resize(5, 5)

    logical = image.explain(optimized=False)
    physical = image.explain()

    assert "rotate 90° clockwise" in logical
    assert "region 20x10+0+31 (row streaming, stops after 41 rows)" in physical
    assert "resize 5x5 (lanczos3)" in physical
    assert repr(image) == "<baryon.Image png 5x5 RGBA uint8, 3 pending ops>"


def test_numpy_array_protocol(source: tuple[baryon.Image, Array]) -> None:
    image, expected = source
    np.testing.assert_array_equal(np.asarray(image), expected)

    as_float = np.asarray(image, dtype=np.float32)
    assert as_float.dtype == np.float32
    np.testing.assert_array_equal(as_float, expected.astype(np.float32))


def test_returned_arrays_are_owned_and_writable(source: tuple[baryon.Image, Array]) -> None:
    image, _ = source
    first = image.to_numpy()
    first[:] = 0
    assert first.flags.writeable
    assert image.to_numpy().any(), "each call must return an independent array"


def test_concurrent_decoding_is_consistent(encode: Encoder) -> None:
    expected = gradient(256, 256)
    image = baryon.open(encode(expected, "PNG")).crop(16, 16, 128, 128).rotate(90)
    reference = np.rot90(expected[16:144, 16:144], k=-1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: image.to_numpy(), range(32)))

    for result in results:
        np.testing.assert_array_equal(result, reference)


def test_image_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError):
        baryon.Image()


def test_jpeg_pipeline(encode: Encoder) -> None:
    data = encode(gradient(64, 64), "JPEG", quality=95)
    full = np.asarray(PILImage.open(io.BytesIO(data)))

    out = baryon.open(data).crop(8, 8, 32, 16).flip_horizontal().to_numpy()

    diff = np.abs(out.astype(np.int16) - full[8:24, 8:40][:, ::-1].astype(np.int16))
    assert diff.max() <= 3
