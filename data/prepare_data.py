# -*- coding: utf-8 -*-
"""Prepare stream CSV files for streaming experiments.

The original stream files are not guaranteed to be ordered by
timestamp. GeoWedge evaluates a streaming query, so each CSV should be sorted
chronologically before running the experiments.

Example:
    python data/prepare_data.py --data-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FILES = (
    "LI-Small_Trans.csv",
    "LI-Medium_Trans.csv",
    "LI-Large_Trans.csv",
    "HI-Small_Trans.csv",
    "HI-Medium_Trans.csv",
    "HI-Large_Trans.csv",
)

# Timestamp format of the original files; `streaming.py` parses this format.
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


def sort_one_csv(path: Path, timestamp_col: str = "Timestamp") -> None:
    if not path.exists():
        print(f"[skip] {path} does not exist")
        return

    print(f"[load] {path}")
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(
            f"{path} does not contain the timestamp column {timestamp_col!r}"
        )

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="mixed")
    df = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)

    tmp_path = path.with_suffix(path.suffix + ".sorted.tmp")
    # Write timestamps back in the original format expected by the loader.
    df.to_csv(tmp_path, index=False, date_format=TIMESTAMP_FORMAT)
    tmp_path.replace(path)
    print(f"[done] sorted {path} by {timestamp_col}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort stream CSV files by Timestamp."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing *_Trans.csv stream files.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=list(DEFAULT_FILES),
        help="Specific CSV file names to sort. Defaults to all six files used "
             "in the experiments.",
    )
    parser.add_argument(
        "--timestamp-col",
        default="Timestamp",
        help="Timestamp column name in the original stream files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    for name in args.files:
        sort_one_csv(data_dir / name, timestamp_col=args.timestamp_col)


if __name__ == "__main__":
    main()
