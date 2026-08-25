# GeoWedge Reviewer Release

This folder contains the core code needed to inspect and reproduce the
GeoWedge streaming package-existence experiments. It intentionally excludes
large raw datasets, intermediate outputs, plotting scripts, and exploratory
notebooks.

## Contents

```text
run_geowedge.py                     reviewer-facing entry point for GeoWedge
data/                               place downloaded IBM AML CSV files here
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
    solver_sketchrefine.py          SketchRefine solver port
    solver_progressive_shading.py   Progressive Shading solver port
    parallel_dual_simplex.py        LP helper used by Progressive Shading
```

## Data

The datasets are not included because the raw CSV files are large. Download the
IBM Transactions for Anti Money Laundering (AML) dataset from IBM's official
AML-Data page:

- IBM AML-Data repository: https://github.com/IBM/AML-Data
- Kaggle distribution linked by IBM: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

The experiments use the following transaction files:

- `LI-Small_Trans.csv`
- `LI-Medium_Trans.csv`
- `LI-Large_Trans.csv`
- `HI-Small_Trans.csv`
- `HI-Medium_Trans.csv`
- `HI-Large_Trans.csv`

Place the downloaded CSV files under `data/`:

```bash
mkdir -p data
# Example:
# mv /path/to/LI-Small_Trans.csv data/
```

Each CSV is expected to keep the original IBM AML column names, including
`Account`, `Timestamp`, `Amount Paid`, `Account.1`, and `Is Laundering`.

The original IBM AML transaction files may not be ordered by time. Sort the
downloaded CSV files before running streaming experiments:

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
C++-accelerated TopK/Greedy baselines, build the corresponding pybind11
extensions first.

## Build C++ Extensions

The default `geowedge` mode uses the pure-Python implementation in
`geowedge/geowedge_search.py`. The C++ backend is provided in
`geowedge/cpp_bucket/` and is called from Python through
`geowedge/cpp_backend.py`. Build it before running `--algo geowedge_cpp`:

```bash
cd geowedge/cpp_bucket
python setup.py build_ext --inplace
cd ../..
```

The TopK and Greedy baselines share one C++17 backend through pybind11. Build
it before running `topk_value`, `topk_ratio`, `greedy_value`, `greedy_ratio`,
or `greedy_fill`:

```bash
cd baselines/cpp_backend
python setup.py build_ext --inplace
cd ../..
```

SketchRefine and Progressive Shading are reviewer-facing Python ports that use
SciPy HiGHS for LP/ILP solving, avoiding external Gurobi or PostgreSQL setup.
They do not require a separate C++ build.

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

The runner writes checkpoints, per-transaction records, and summary files to
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
