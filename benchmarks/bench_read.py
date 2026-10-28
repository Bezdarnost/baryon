"""Image reading speed: single-image latency and multi-core batch throughput.

Usage:
    python benchmarks/bench_read.py DATASET_DIR --out results/read.json [--sample 300]
        [--libs baryon pillow ...] [--threads 1 16]

For every library and thread count, in a fresh process:

* latency: each image is decoded twice (full image, then the 512x512 center crop) and the
  faster of two runs is kept. With more than one thread, only libraries that can split one
  image across threads benefit.
* throughput: all images are decoded in batches of 64 using every thread, the way a training
  data loader would. baryon uses ``to_numpy_all``; other libraries use a thread pool, which is
  fair because all of them release the GIL while decoding.

Files are read into memory first, so disk speed is not measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import adapters
import common

BATCH = 64


def decode_batches(lib: str, adapter: adapters.Adapter, threads: int, blobs: list[bytes]) -> float:
    """Decode ``blobs`` in batches of ``BATCH`` with every thread; return the elapsed time."""
    batches = [blobs[i : i + BATCH] for i in range(0, len(blobs), BATCH)]
    start = time.perf_counter()
    if lib == "baryon":
        import baryon  # noqa: PLC0415

        for batch in batches:
            baryon.to_numpy_all(baryon.open_all(batch))
    else:
        with ThreadPoolExecutor(threads) as pool:
            for batch in batches:
                list(pool.map(adapter.full, batch))
    return time.perf_counter() - start


def worker(lib: str, threads: int, files: list[str], out: Path) -> None:
    """Measure one library at one thread count and write the raw timings to ``out``."""
    adapters.configure_threads(threads)
    adapter = adapters.load(lib, threads)
    blobs = [Path(f).read_bytes() for f in files]
    blobs = [b for b in blobs if adapters.kind(b) in adapter.formats]
    if not blobs:
        return
    for data in blobs[:3]:  # warm-up
        adapter.full(data)

    latency = []
    for data in blobs:
        row: dict = {"kind": adapters.kind(data), "bytes": len(data)}
        for name, fn in (("full", adapter.full), ("crop", adapter.crop)):
            best = float("inf")
            for _ in range(2):
                start = time.perf_counter()
                array = fn(data)
                best = min(best, time.perf_counter() - start)
            row[f"{name}_s"] = best
            if name == "full":
                row["megapixels"] = array.shape[0] * array.shape[1] / 1e6
        latency.append(row)

    throughput = {}
    for fmt in sorted({row["kind"] for row in latency}):
        subset = [b for b in blobs if adapters.kind(b) == fmt]
        throughput[fmt] = {
            "images": len(subset),
            "megapixels": sum(r["megapixels"] for r in latency if r["kind"] == fmt),
            "seconds": decode_batches(lib, adapter, threads, subset),
        }

    out.write_text(
        json.dumps(
            {
                "lib": lib,
                "threads": threads,
                "latency": latency,
                "throughput": throughput,
            }
        )
    )


def main() -> None:
    """Run every library and thread count in isolated processes and save the results."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=300, help="images per format")
    parser.add_argument("--libs", nargs="+", default=list(adapters.LIBRARIES))
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 16])
    parser.add_argument(
        "--worker", nargs=3, metavar=("LIB", "THREADS", "FILE_LIST"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.worker:
        lib, threads, file_list = args.worker
        worker(lib, int(threads), json.loads(Path(file_list).read_text()), args.out)
        return

    files = common.discover(args.dataset)
    chosen = [str(p) for fmt in files for p in common.sample(files[fmt], args.sample)]
    print({fmt: min(len(p), args.sample) for fmt, p in files.items()})
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
                    "--worker",
                    lib,
                    str(threads),
                    str(file_list),
                )
                if part.exists():
                    runs.append(json.loads(part.read_text()))
                    print(f"done: {lib} with {threads} thread(s)", flush=True)
    common.save(args.out, {"benchmark": "read", "runs": runs})


if __name__ == "__main__":
    main()
