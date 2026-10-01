# GeoWedge

This folder contains the core code needed to inspect and reproduce the
GeoWedge streaming package-existence experiments. It intentionally excludes
large raw datasets, intermediate outputs, plotting scripts, and exploratory
notebooks.

## Contents

```text
run_geowedge.py                     entry point for GeoWedge
data/                               place downloaded stream CSV files here
geowedge/
  streaming.py                      streaming loader and filtering pipeline
  state_search.py                   shared state objects and basic search
  geowedge_search.py                GeoWedge compressed frontier search
  cpp_backend.py                    Python adapter for the C++ backend
  cpp_bucket/                       C++17 pybind11 backend
baselines/
    run_baseline.py                 single entry point for all baselines
    methods.py                      baseline registry and C++ dispatch
    runner.py                       shared streaming runner
    cpp_backend/                    C++17 pybind11 backend for TopK/Greedy
    solver_ilp.py                   exact ILP evaluation (accuracy reference)
    solver_sketchrefine.py          SketchRefine solver port
    solver_progressive_shading.py   Progressive Shading solver port
    parallel_dual_simplex.py        LP helper used by Progressive Shading
```

## Data

The experiments run over six public streams of timestamped tuples. Each tuple
has the form `(src, dst, amt, time)`: a source key, a destination key, an
amount, and an arrival time. The raw CSV files are large and are not included
in this release. Download them from the original
[GitHub repository](https://github.com/IBM/AML-Data) or its
[Kaggle mirror](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

The following stream files are used:

| File                  | Series | Scale  |
|-----------------------|--------|--------|
| `LI-Small_Trans.csv`  | LI     | Small  |
| `LI-Medium_Trans.csv` | LI     | Medium |
| `LI-Large_Trans.csv`  | LI     | Large  |
| `HI-Small_Trans.csv`  | HI     | Small  |
| `HI-Medium_Trans.csv` | HI     | Medium |
| `HI-Large_Trans.csv`  | HI     | Large  |

Place the downloaded CSV files under `data/`:

```bash
mkdir -p data
# Example:
# mv /path/to/LI-Small_Trans.csv data/
```

Keep the original CSV header unchanged; the loader selects columns by their
original names.

Tuples are processed in arrival order, but the raw files are not guaranteed to
be sorted by time. Sort them once before running the experiments:

```bash
python data/prepare_data.py --data-dir data
```

The script sorts each available `*_Trans.csv` file in place by `Timestamp`.

## Environment

Run all commands below from this release directory:

```bash
cd GeoWedge
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This release can run the pure-Python GeoWedge implementation directly after
installing the requirements. To run the C++ GeoWedge backend or the
C++-accelerated TopK/Greedy baselines, build the C++ extensions first (see
below).

## Build C++ Extensions

Compiled extensions are not shipped with this repository. Two pybind11
extensions must be built locally before running the C++ code paths:

| Extension           | Source directory        | Needed for                                                            |
|---------------------|-------------------------|-----------------------------------------------------------------------|
| `wedge_bucket_cpp`  | `geowedge/cpp_bucket/`  | `--algo geowedge_cpp`                                                 |
| `baseline_cpp_core` | `baselines/cpp_backend/`| `topk_value`, `topk_ratio`, `greedy_value`, `greedy_ratio`, `greedy_fill` |

The pure-Python `geowedge` mode, SketchRefine, and Progressive Shading need no
build step.

### Prerequisites

- A C++17 compiler:
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: `g++` 7 or newer (for example, `sudo apt install build-essential`)
- The Python environment from the previous section, activated, with
  `requirements.txt` installed (this provides `pybind11` and `setuptools`).

Build with the same Python interpreter you will use to run the experiments;
an extension built for one Python version cannot be imported by another.

### Build

From the repository root:

```bash
cd geowedge/cpp_bucket
python setup.py build_ext --inplace
cd ../..
```

```bash
cd baselines/cpp_backend
python setup.py build_ext --inplace
cd ../..
```

Each command places a shared library next to its source file, for example
`geowedge/cpp_bucket/wedge_bucket_cpp.cpython-313-darwin.so` (the suffix
depends on your Python version and platform). Both setup scripts compile with
`-O3 -march=native`.

### Verify

```bash
python -c "import sys; sys.path.insert(0, 'geowedge/cpp_bucket'); import wedge_bucket_cpp; print('wedge_bucket_cpp OK')"
```

```bash
python -c "import sys; sys.path.insert(0, 'baselines/cpp_backend'); import baseline_cpp_core; print('baseline_cpp_core OK')"
```

If an extension is missing, the runner stops with an `ImportError` that names
the build command to run.

### Rebuild

Rebuild after editing a `.cpp` file or switching Python versions. Remove the
old build output first:

```bash
rm -rf geowedge/cpp_bucket/build geowedge/cpp_bucket/*.so
rm -rf baselines/cpp_backend/build baselines/cpp_backend/*.so
```

Then repeat the build commands above.

## Run GeoWedge

Quick smoke test on the first 100,000 rows with the C++ backend:

```bash
python run_geowedge.py \
  --algo geowedge_cpp \
  --data data/LI-Small_Trans.csv \
  --dataset-name LI-Small \
  --nrows 100000 \
  --reset
```

Full run:

```bash
python run_geowedge.py \
  --algo geowedge_cpp \
  --data data/LI-Small_Trans.csv \
  --dataset-name LI-Small \
  --reset
```

Pure-Python run, useful when a compiler is unavailable:

```bash
python run_geowedge.py \
  --data data/LI-Small_Trans.csv \
  --dataset-name LI-Small \
  --reset
```

The runner writes checkpoints, per-tuple records, and summary files to
`outputs/`. If a run is interrupted, rerun the same command without `--reset`
to resume from the checkpoint.

## Run Baselines

Run one baseline:

```bash
python baselines/run_baseline.py \
  --baseline progressive_shading \
  --data data/LI-Small_Trans.csv \
  --dataset-name LI-Small \
  --nrows 100000
```

Available baseline names are:

```text
ilp
topk_value
greedy_value
topk_ratio
greedy_ratio
greedy_fill
sketchrefine
progressive_shading
```

Run all selected baselines:

```bash
python baselines/run_baseline.py \
  --baseline all \
  --data data/LI-Small_Trans.csv \
  --dataset-name LI-Small
```

Baseline outputs are written to `outputs_baselines/`.

## Typical Dataset Loop

```bash
for ds in LI-Small LI-Medium LI-Large HI-Small HI-Medium HI-Large; do
  python run_geowedge.py \
    --algo geowedge_cpp \
    --data data/${ds}_Trans.csv \
    --dataset-name ${ds} \
    --reset
done
```

The same loop can be used for baselines by replacing the command with
`baselines/run_baseline.py` and setting `--baseline`:

```bash
for ds in LI-Small LI-Medium LI-Large HI-Small HI-Medium HI-Large; do
  python baselines/run_baseline.py \
    --baseline all \
    --data data/${ds}_Trans.csv \
    --dataset-name ${ds}
done
```

## Notes

- The data license is separate from this code release. Check IBM/Kaggle terms
  before downloading or redistributing the datasets.
- Use `--nrows` for local smoke tests before launching full-dataset runs.
- The default parameters are `theta=10000`, `eps=0.2`, and a `0.1` day sliding
  window, matching the main experimental setting.
