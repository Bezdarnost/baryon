"""Thread pool control and batch APIs."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

import baryon
from conftest import FIXTURES, gradient

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from conftest import Encoder


@pytest.fixture(autouse=True)
def _restore_thread_count() -> Iterator[None]:
    before = baryon.get_num_threads()
    yield
    baryon.set_num_threads(before)


@pytest.fixture
def images(encode: Encoder) -> list[baryon.Image]:
    jxl = baryon.open(FIXTURES / "smooth-rgb8-lossy.jxl")
    png = baryon.open(encode(gradient(300, 200, 4), "PNG"))
    jpeg = baryon.open(encode(gradient(256, 192), "JPEG", quality=90))
    return [
        jxl,
        jxl.crop(100, 50, 300, 200).rotate(90),
        jxl.resize(150, 120),
        png.flip_horizontal(),
        jpeg,
        jpeg.crop(10, 10, 100, 50).resize(40, 20, filter="bilinear"),
    ]


def test_thread_count_can_be_changed() -> None:
    baryon.set_num_threads(3)
    assert baryon.get_num_threads() == 3
    baryon.set_num_threads(None)
    assert baryon.get_num_threads() >= 1


def test_thread_count_comes_from_environment() -> None:
    env = {**os.environ, "BARYON_NUM_THREADS": "3"}
    code = "import baryon; print(baryon.get_num_threads())"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "3"


def test_invalid_environment_value_is_reported() -> None:
    env = {**os.environ, "BARYON_NUM_THREADS": "many"}
    code = "import baryon; baryon.get_num_threads()"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert out.returncode != 0
    assert "BARYON_NUM_THREADS must be a non-negative integer" in out.stderr


def test_to_numpy_all_matches_one_by_one(images: list[baryon.Image]) -> None:
    expected = [image.to_numpy() for image in images]
    for got, want in zip(baryon.to_numpy_all(images), expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_results_do_not_depend_on_thread_count(images: list[baryon.Image]) -> None:
    baryon.set_num_threads(1)
    single = baryon.to_numpy_all(images)
    baryon.set_num_threads(4)
    for got, want in zip(baryon.to_numpy_all(images), single, strict=True):
        np.testing.assert_array_equal(got, want)


def test_to_numpy_all_accepts_empty_and_tuples(images: list[baryon.Image]) -> None:
    assert baryon.to_numpy_all([]) == []
    assert len(baryon.to_numpy_all(tuple(images))) == len(images)


def test_to_numpy_all_reports_the_failing_position(images: list[baryon.Image]) -> None:
    data = (FIXTURES / "smooth-rgb8-lossy.jxl").read_bytes()
    truncated = baryon.open(data[: len(data) // 2])
    with pytest.raises(baryon.DecodeError, match=r"^images\[2\]: failed to decode jxl"):
        baryon.to_numpy_all([images[0], images[1], truncated])


def test_to_numpy_all_rejects_non_images() -> None:
    with pytest.raises(TypeError):
        baryon.to_numpy_all(["not an image"])  # type: ignore[list-item]


def test_open_all_matches_open(tmp_path: Path, encode: Encoder) -> None:
    png = tmp_path / "a.png"
    png.write_bytes(encode(gradient(30, 20), "PNG"))
    jxl = FIXTURES / "gradient-rgb8-lossless.jxl"
    sources: list[str | Path | bytes] = [str(png), jxl, jxl.read_bytes()]

    opened = baryon.open_all(sources)

    assert [image.format for image in opened] == ["png", "jxl", "jxl"]
    for image, source in zip(opened, sources, strict=True):
        np.testing.assert_array_equal(image.to_numpy(), baryon.open(source).to_numpy())


def test_open_all_reports_the_failing_position(tmp_path: Path) -> None:
    jxl = FIXTURES / "gradient-rgb8-lossless.jxl"
    with pytest.raises(FileNotFoundError, match=r"^sources\[1\]: "):
        baryon.open_all([jxl, tmp_path / "missing.png"])
    with pytest.raises(baryon.UnsupportedFormatError, match=r"^sources\[0\]: unrecognized"):
        baryon.open_all([b"not an image", jxl])


@pytest.mark.skipif(sys.platform != "linux", reason="fork() with threads is only safe on Linux")
def test_works_in_forked_child_processes() -> None:
    # PyTorch DataLoader workers and multiprocessing fork a process whose parent already
    # started the thread pool; the child must get a working pool of its own.
    code = """
import os, sys
import baryon
image = baryon.open(sys.argv[1])
image.to_numpy()                         # parent starts its thread pool
pid = os.fork()
if pid == 0:
    ok = image.to_numpy().shape == (450, 600, 3) and len(baryon.to_numpy_all([image] * 4)) == 4
    os._exit(0 if ok else 1)
_, status = os.waitpid(pid, 0)
sys.exit(os.waitstatus_to_exitcode(status))
"""
    path = str(FIXTURES / "smooth-rgb8-lossy.jxl")
    out = subprocess.run([sys.executable, "-c", code, path], timeout=60, check=False)
    assert out.returncode == 0
