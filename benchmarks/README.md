# Benchmarks

Reproducible comparisons of baryon with other Python image libraries on your own images.

| Script | Measures |
|---|---|
| `bench_read.py` | Decode latency (full image and a 512×512 center crop) and multi-core batch throughput, per format |
| `bench_convert.py` | Lossless conversion to JPEG XL: speed, size saving and verified losslessness, per library |
| `report.py` | Markdown tables from any number of result files (later files override matching runs) |

Every library and thread count runs in a fresh process with its thread pools limited, and
files are read into memory before timing, so disk speed is excluded.

## Setup

Use a separate environment with the competing libraries and a **release** build of baryon
(building needs the libjxl submodules; see [Contributing](../docs/contributing.md)):

```bash
uv venv .bench && source .bench/bin/activate
uv pip install maturin pillow opencv-python-headless "pyvips[binary]" imagecodecs pillow-jxl-plugin
maturin build --release -o dist && uv pip install dist/baryon-*.whl
```

## Running

```bash
python benchmarks/bench_read.py /path/to/images --out results/read.json --sample 300 --threads 1 16
python benchmarks/bench_convert.py /path/to/images --out results/convert.json --sample 200 --threads 1 16
python benchmarks/report.py results/read.json results/convert.json > results/report.md
```

Formats are detected from file signatures, not extensions. JPEG XL files in the dataset are
included in the read benchmark; `bench_convert.py` converts JPEG and PNG files.

Each result file records the CPU, Python and library versions next to the timings.
