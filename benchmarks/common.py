"""Dataset discovery, sampling and result storage shared by the benchmark scripts."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

SIGNATURES = {
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG",),
    "jxl": (b"\xff\x0a", b"\x00\x00\x00\x0cJXL "),
}


def discover(root: Path) -> dict[str, list[Path]]:
    """Find images under ``root`` grouped by their real format (by signature, not extension)."""
    found: dict[str, list[Path]] = {k: [] for k in SIGNATURES}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as f:
            head = f.read(12)
        for fmt, signatures in SIGNATURES.items():
            if head.startswith(signatures):
                found[fmt].append(path)
                break
    return found


def sample(paths: list[Path], n: int | None, seed: int = 2026) -> list[Path]:
    """Deterministic random sample of at most ``n`` paths (all of them when ``n`` is None)."""
    if n is None or n >= len(paths):
        return list(paths)
    return sorted(random.Random(seed).sample(paths, n))


def machine() -> dict[str, Any]:
    """Describe the machine, so results can be compared across runs."""
    cpu = platform.processor() or platform.machine()
    if sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def versions() -> dict[str, str]:
    """Installed versions of every benchmarked package."""
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    packages = (
        "baryon",
        "pillow",
        "opencv-python-headless",
        "pyvips",
        "imagecodecs",
        "pillow-jxl-plugin",
        "numpy",
    )
    out = {}
    for package in packages:
        with contextlib.suppress(PackageNotFoundError):
            out[package] = version(package)
    return out


def save(path: Path, payload: dict[str, Any]) -> None:
    """Write results with machine and version information attached."""
    payload = {"machine": machine(), "versions": versions(), **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def run_isolated(script: Path, *args: str) -> None:
    """Run one benchmark worker in a fresh interpreter, so thread settings cannot leak."""
    subprocess.run([sys.executable, str(script), *args], check=True)  # noqa: S603
