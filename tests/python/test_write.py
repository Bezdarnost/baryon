"""Writing PNG, JPEG and JPEG XL files: ``to_png``, ``to_jpeg``, ``save`` and ``save_all``."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image as PILImage

import baryon
from conftest import FIXTURES, gradient

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import Encoder


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("compression", ["fast", "balanced", "high"])
def test_png_round_trips_exactly(encode: Encoder, channels: int, compression: str) -> None:
    pixels = gradient(83, 61, channels)
    png = baryon.open(encode(pixels, "PNG")).to_png(compression=compression)  # type: ignore[arg-type]

    assert png.startswith(b"\x89PNG")
    np.testing.assert_array_equal(baryon.open(png).to_numpy(), pixels)


def test_png_keeps_sixteen_bits() -> None:
    image = baryon.open(FIXTURES / "gradient-rgba16-lossless.jxl")
    again = baryon.open(image.to_png())

    assert again.dtype == "uint16"
    np.testing.assert_array_equal(again.to_numpy(), image.to_numpy())


def test_png_rejects_unknown_compression(encode: Encoder) -> None:
    image = baryon.open(encode(gradient(8, 8), "PNG"))
    with pytest.raises(ValueError, match="compression"):
        image.to_png(compression="max")  # type: ignore[arg-type]


@pytest.mark.parametrize("channels", [1, 3])
def test_jpeg_is_close_to_the_original(encode: Encoder, channels: int) -> None:
    pixels = gradient(96, 64, channels)
    jpeg = baryon.open(encode(pixels, "PNG")).to_jpeg(quality=95)

    assert jpeg.startswith(b"\xff\xd8\xff")
    decoded = baryon.open(jpeg).to_numpy()
    assert decoded.shape == pixels.shape
    assert np.abs(decoded.astype(int) - pixels).mean() < 3


def test_jpeg_rejects_what_it_cannot_store(encode: Encoder) -> None:
    with pytest.raises(ValueError, match="alpha"):
        baryon.open(encode(gradient(8, 8, 4), "PNG")).to_jpeg()
    buf = io.BytesIO()
    PILImage.fromarray(np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000).save(buf, "PNG")
    gray16 = baryon.open(buf.getvalue())
    assert (gray16.mode, gray16.dtype) == ("L", "uint16")
    with pytest.raises(ValueError, match="8-bit"):
        gray16.to_jpeg()
    image = baryon.open(encode(gradient(8, 8), "PNG"))
    for quality in (0, 101):
        with pytest.raises(ValueError, match="quality"):
            image.to_jpeg(quality=quality)


def test_save_picks_the_format_from_the_extension(encode: Encoder, tmp_path: Path) -> None:
    pixels = gradient(64, 48)
    image = baryon.open(encode(pixels, "PNG")).crop(8, 8, 32, 16)
    for name, magic in [
        ("a.png", b"\x89PNG"),
        ("b.JPG", b"\xff\xd8\xff"),
        ("c.jpeg", b"\xff\xd8\xff"),
        ("d.jxl", b"\xff\x0a"),
    ]:
        image.save(tmp_path / name)
        assert (tmp_path / name).read_bytes().startswith(magic), name
    np.testing.assert_array_equal(baryon.open(tmp_path / "a.png").to_numpy(), image.to_numpy())
    np.testing.assert_array_equal(baryon.open(tmp_path / "d.jxl").to_numpy(), image.to_numpy())

    image.save(tmp_path / "e.bin", format="png")
    assert (tmp_path / "e.bin").read_bytes().startswith(b"\x89PNG")


def test_save_transcodes_jpegs_to_jpeg_xl(encode: Encoder, tmp_path: Path) -> None:
    original = encode(gradient(128, 96), "JPEG", quality=90)
    baryon.open(original).save(tmp_path / "photo.jxl")

    assert baryon.jxl_to_jpeg(tmp_path / "photo.jxl") == original


def test_save_rejects_unknown_formats_and_misplaced_options(
    encode: Encoder, tmp_path: Path
) -> None:
    image = baryon.open(encode(gradient(16, 16), "PNG"))
    with pytest.raises(ValueError, match="extension"):
        image.save(tmp_path / "image.bmp")
    with pytest.raises(ValueError, match="cannot write format"):
        image.save(tmp_path / "image.png", format="gif")
    with pytest.raises(ValueError, match="quality applies to JPEG, not PNG"):
        image.save(tmp_path / "image.png", quality=80)
    with pytest.raises(ValueError, match="effort applies to JPEG XL, not JPEG"):
        image.save(tmp_path / "image.jpg", effort=3)
    assert list(tmp_path.iterdir()) == []

    image.save(tmp_path / "small.jxl", lossless=False, distance=2.0, effort=3)
    image.save(tmp_path / "q.jpg", quality=50)
    image.save(tmp_path / "fast.png", compression="fast")
    assert len(list(tmp_path.iterdir())) == 3


def test_save_all_writes_every_file(encode: Encoder, tmp_path: Path) -> None:
    images = [baryon.open(encode(gradient(40 + i, 30, 3), "PNG")) for i in range(12)]
    paths = [tmp_path / f"{i}.{('png', 'jxl', 'jpg')[i % 3]}" for i in range(12)]
    baryon.save_all(images, paths)

    for image, path in zip(images, paths, strict=True):
        decoded = baryon.open(path).to_numpy()
        if path.suffix == ".jpg":
            assert decoded.shape == (image.height, image.width, 3)
        else:
            np.testing.assert_array_equal(decoded, image.to_numpy())


def test_save_all_reports_failures_by_position(encode: Encoder, tmp_path: Path) -> None:
    rgb = baryon.open(encode(gradient(16, 16), "PNG"))
    rgba = baryon.open(encode(gradient(16, 16, 4), "PNG"))
    paths = [tmp_path / f"{i}.jpg" for i in range(14)]
    images = [rgba if i % 2 else rgb for i in range(14)]

    with pytest.raises(ValueError, match=r"^images\[1\]: .*alpha") as raised:
        baryon.save_all(images, paths)
    assert [p.exists() for p in paths] == [i % 2 == 0 for i in range(14)]
    others = ", ".join(f"images[{i}]" for i in (3, 5, 7, 9, 11, 13))
    assert str(raised.value).endswith(f"(6 more failed: {others})")
    failures = raised.value.failures  # type: ignore[attr-defined]
    assert [index for index, _ in failures] == [1, 3, 5, 7, 9, 11, 13]
    assert all(isinstance(err, ValueError) for _, err in failures)

    with pytest.raises(ValueError, match="2 images but 1 paths"):
        baryon.save_all([rgb, rgb], [tmp_path / "x.png"])
    with pytest.raises(ValueError, match="quality applies to JPEG"):
        baryon.save_all([rgb], [tmp_path / "y.png"], quality=80)
    with pytest.raises(ValueError, match="distance"):
        baryon.save_all([rgb], [tmp_path / "y.jxl"], jpeg_fallback="lossy", distance=30.0)
    assert not (tmp_path / "y.png").exists()
    assert not (tmp_path / "y.jxl").exists()


def test_save_all_falls_back_and_reports_where(encode: Encoder, tmp_path: Path) -> None:
    jpeg = encode(gradient(64, 48), "JPEG", quality=90)
    images = [
        baryon.open(jpeg),
        baryon.open(FIXTURES / "untranscodable.jpg"),
        baryon.open(encode(gradient(32, 32), "PNG")),
        baryon.open(FIXTURES / "untranscodable.jpg"),
    ]
    paths = [tmp_path / f"{i}.jxl" for i in range(4)]

    with pytest.raises(
        baryon.EncodeError, match=r"^images\[1\]: .*\(1 more failed: images\[3\]\)$"
    ):
        baryon.save_all(images, paths)
    assert [p.exists() for p in paths] == [True, False, True, False]

    assert baryon.save_all(images, paths, jpeg_fallback="lossy") == [1, 3]
    assert baryon.jxl_to_jpeg(paths[0]) == jpeg
    for i in (1, 3):
        decoded = baryon.open(paths[i]).to_numpy().astype(int)
        assert np.abs(decoded - images[i].to_numpy()).mean() < 3
    assert baryon.save_all(images[:1], paths[:1]) == []
