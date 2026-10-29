"""Render benchmark results as Markdown tables.

Usage:
    python benchmarks/report.py results/read.json results/convert.json [more.json ...]

Several files may be given. Results are merged per library, thread count and image format;
when two files cover the same combination, the later file wins. This lets a re-run of one
library, or of one format, replace just those numbers.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ORDER = ["baryon", "imagecodecs", "pyvips", "opencv", "pillow", "pillow-jxl"]

Key = tuple[str, int, str]  # library, threads, format


def rank(lib: str) -> int:
    """Sort position of a library in every table."""
    return ORDER.index(lib) if lib in ORDER else len(ORDER)


def load(paths: list[Path]) -> tuple[dict[Key, dict[str, Any]], dict[Key, dict[str, Any]]]:
    """Index read and convert results by (library, threads, format); later files win."""
    reads: dict[Key, dict[str, Any]] = {}
    converts: dict[Key, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text())
        for run in data["runs"]:
            lib, threads = run["lib"], run["threads"]
            if data["benchmark"] == "read":
                rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in run["latency"]:
                    rows[row["kind"]].append(row)
                for fmt, latency in rows.items():
                    reads[lib, threads, fmt] = {
                        "latency": latency,
                        "batch": run["throughput"].get(fmt),
                    }
            else:
                rows = defaultdict(list)
                for row in run["results"]:
                    rows[row["kind"]].append(row)
                for fmt, results in rows.items():
                    converts[lib, threads, fmt] = {"results": results, "wall_s": run["wall_s"]}
    return reads, converts


def matching(results: dict[Key, dict[str, Any]], threads: int, fmt: str) -> list[tuple[str, Any]]:
    """Entries for one thread count and format, in table order."""
    found = [(lib, v) for (lib, t, f), v in results.items() if t == threads and f == fmt]
    return sorted(found, key=lambda item: rank(item[0]))


def read_tables(reads: dict[Key, dict[str, Any]]) -> None:
    """Latency at one thread and batch throughput at every other thread count, per format."""
    for fmt in sorted({f for (_, _, f) in reads}):
        print(f"\n### {fmt.upper()}: one image at a time, 1 thread\n")
        print("| Library | Full image (median) | Throughput | 512x512 crop (median) |")
        print("|---|---:|---:|---:|")
        for lib, entry in matching(reads, 1, fmt):
            rows = entry["latency"]
            mp = sum(r["megapixels"] for r in rows)
            seconds = sum(r["full_s"] for r in rows)
            full = statistics.median(r["full_s"] for r in rows) * 1e3
            crop = statistics.median(r["crop_s"] for r in rows) * 1e3
            print(f"| {lib} | {full:.1f} ms | {mp / seconds:.0f} MP/s | {crop:.1f} ms |")

        for threads in sorted({t for (_, t, f) in reads if f == fmt and t > 1}):
            print(f"\n### {fmt.upper()}: batches on {threads} threads\n")
            print("| Library | Images/s | Throughput | Speed-up vs 1 thread |")
            print("|---|---:|---:|---:|")
            for lib, entry in matching(reads, threads, fmt):
                batch = entry["batch"]
                single = reads.get((lib, 1, fmt), {}).get("batch")
                speedup = f"{single['seconds'] / batch['seconds']:.1f}x" if single else "-"
                print(
                    f"| {lib} | {batch['images'] / batch['seconds']:.1f} "
                    f"| {batch['megapixels'] / batch['seconds']:.0f} MP/s | {speedup} |"
                )


def convert_tables(converts: dict[Key, dict[str, Any]]) -> None:
    """Size, speed and verification of conversion to JPEG XL, per source format."""
    titles = {"jpeg": "JPEG to JPEG XL (lossless transcoding)", "png": "PNG to JPEG XL (lossless)"}
    for fmt in sorted({f for (_, _, f) in converts}):
        print(f"\n### {titles.get(fmt, fmt)}\n")
        print("| Library | Threads | Files | Saving | Speed per thread | Lossless | Failed |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for threads in sorted({t for (_, t, f) in converts if f == fmt}):
            for lib, entry in matching(converts, threads, fmt):
                rows = entry["results"]
                ok = [r for r in rows if "error" not in r]
                if not ok:
                    continue
                before = sum(r["bytes"] for r in ok)
                after = sum(r["jxl_bytes"] for r in ok)
                speed = before / sum(r["seconds"] for r in ok)
                verified = sum(bool(r.get("lossless")) for r in ok)
                print(
                    f"| {lib} | {threads} | {len(ok)} | {(1 - after / before) * 100:.1f}% "
                    f"| {speed / 1e6:.1f} MB/s | {verified}/{len(ok)} | {len(rows) - len(ok)} |"
                )

    print("\n### Whole conversion runs (all formats, wall clock)\n")
    print("| Library | Threads | Input | Wall time | Throughput |")
    print("|---|---:|---:|---:|---:|")
    runs: dict[tuple[str, int], tuple[int, float]] = {}
    for (lib, threads, _), entry in converts.items():
        size = sum(r["bytes"] for r in entry["results"] if "error" not in r)
        previous = runs.get((lib, threads), (0, entry["wall_s"]))
        runs[lib, threads] = (previous[0] + size, entry["wall_s"])
    for (lib, threads), (size, wall) in sorted(
        runs.items(), key=lambda i: (i[0][1], rank(i[0][0]))
    ):
        throughput = size / wall / 1e6
        print(f"| {lib} | {threads} | {size / 1e6:.0f} MB | {wall:.1f} s | {throughput:.1f} MB/s |")


def main() -> None:
    """Print every table that the given result files support."""
    reads, converts = load([Path(p) for p in sys.argv[1:]])
    if reads:
        read_tables(reads)
    if converts:
        convert_tables(converts)


if __name__ == "__main__":
    main()
