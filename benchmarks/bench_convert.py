"""Conversion to lossless JPEG XL: speed, size and correctness, per library.

Usage:
    python benchmarks/bench_convert.py DATASET_DIR --out results/convert.json [--sample 200]
        [--libs imagecodecs pillow-jxl] [--threads 1 16] [--effort 7]

* JPEG files are transcoded losslessly: the JPEG's DCT coefficients are re-encoded and the
  original file can be reconstructed bit for bit. Every result is checked by reconstructing
  the JPEG with libjxl and comparing bytes.
* PNG files are encoded pixel-lossless at the given effort, checked by decoding and comparing
  every pixel.

Each file is encoded with one encoder thread. With N threads, N files are encoded at once,
the way a dataset is converted in practice. Converted files are kept in memory only.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent))

import common

if TYPE_CHECKING:
    from collections.abc import Callable

CONVERTERS = ("baryon", "imagecodecs", "pillow-jxl")


def converter(lib: str, effort: int, threads: int) -> Callable[[Path], bytes]:
    """Return a function converting one file to JPEG XL with a single encoder thread."""
    if lib == "baryon":
        import baryon  # noqa: PLC0415

        # baryon's pool bounds all of its work, including calls from several Python threads,
        # so it must be as large as the number of files converted at once.
        baryon.set_num_threads(threads)

        def convert(path: Path) -> bytes:
            # JPEGs are transcoded losslessly, everything else is encoded from lossless pixels.
            return baryon.open(path).to_jxl(effort=effort)

        return convert

    if lib == "imagecodecs":
        import imagecodecs  # noqa: PLC0415

        def convert(path: Path) -> bytes:
            data = path.read_bytes()
            if data.startswith(b"\xff\xd8\xff"):
                return imagecodecs.jpegxl_encode_jpeg(data, numthreads=1)
            pixels = imagecodecs.png_decode(data)
            return imagecodecs.jpegxl_encode(pixels, lossless=True, effort=effort, numthreads=1)

        return convert

    if lib == "pillow-jxl":
        import pillow_jxl  # noqa: F401, PLC0415 - registers the JPEG XL plugin
        from PIL import Image  # noqa: PLC0415

        Image.MAX_IMAGE_PIXELS = None

        def convert(path: Path) -> bytes:
            out = io.BytesIO()
            with Image.open(path) as image:
                image.save(
                    out,
                    format="JXL",
                    lossless=True,
                    lossless_jpeg=True,
                    effort=effort,
                    num_threads=1,
                )
            return out.getvalue()

        return convert

    msg = f"unknown converter {lib!r}"
    raise ValueError(msg)


def verify(original: bytes, jxl: bytes) -> bool:
    """Check losslessness with libjxl, independently of the converter under test."""
    import imagecodecs  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if original.startswith(b"\xff\xd8\xff"):
        return bytes(imagecodecs.jpegxl_decode_jpeg(jxl)) == original
    expected = imagecodecs.png_decode(original)
    return bool(np.array_equal(imagecodecs.jpegxl_decode(jxl).reshape(expected.shape), expected))


def worker(lib: str, threads: int, effort: int, files: list[str], out: Path) -> None:
    """Convert every file with one library and write timings, sizes and checks to ``out``."""
    convert = converter(lib, effort, threads)
    paths = [Path(f) for f in files]
    convert(paths[0])  # warm-up

    def timed(path: Path) -> dict:
        start = time.perf_counter()
        try:
            jxl = convert(path)
        except Exception as exc:  # noqa: BLE001 - recorded per file
            return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        return {"path": str(path), "seconds": time.perf_counter() - start, "jxl": jxl}

    start = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        rows = list(pool.map(timed, paths))
    wall = time.perf_counter() - start

    results = []
    for row in rows:
        original = Path(row["path"]).read_bytes()
        jxl = row.pop("jxl", None)
        row["kind"] = "jpeg" if original.startswith(b"\xff\xd8\xff") else "png"
        row["bytes"] = len(original)
        if jxl is not None:
            row["jxl_bytes"] = len(jxl)
            row["lossless"] = verify(original, jxl)
        results.append(row)
    out.write_text(
        json.dumps(
            {"lib": lib, "threads": threads, "effort": effort, "wall_s": wall, "results": results}
        )
    )


def main() -> None:
    """Run every library and thread count in isolated processes and save the results."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=200, help="files per format")
    parser.add_argument("--libs", nargs="+", default=list(CONVERTERS))
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--effort", type=int, default=7, help="libjxl effort for PNG input")
    parser.add_argument(
        "--worker", nargs=3, metavar=("LIB", "THREADS", "FILE_LIST"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.worker:
        lib, threads, file_list = args.worker
        worker(lib, int(threads), args.effort, json.loads(Path(file_list).read_text()), args.out)
        return

    files = common.discover(args.dataset)
    chosen = [str(p) for fmt in ("jpeg", "png") for p in common.sample(files[fmt], args.sample)]
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        file_list = Path(tmp) / "files.json"
        file_list.write_text(json.dumps(chosen))
        for threads in args.threads:
            for lib in args.libs:
                part = Path(tmp) / f"{lib}-{threads}.json"
                common.run_isolated(
                    Path(__file__),
                    str(args.dataset),
                    "--out",
                    str(part),
                    "--effort",
                    str(args.effort),
                    "--worker",
                    lib,
                    str(threads),
                    str(file_list),
                )
                runs.append(json.loads(part.read_text()))
                print(f"done: {lib} with {threads} thread(s)", flush=True)
    common.save(args.out, {"benchmark": "convert", "runs": runs})


if __name__ == "__main__":
    main()
