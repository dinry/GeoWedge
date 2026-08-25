# -*- coding: utf-8 -*-
"""CLI entry: run one selected baseline on an IBM AML transaction stream.

Default parameters match the headline configuration:
    theta = 10000, eps = 0.2, Delta = 0.1 day, k = 8

Usage:
    python run_baseline.py --baseline topk_value
    python run_baseline.py --baseline sketchrefine
    python run_baseline.py --baseline progressive_shading
    python run_baseline.py --baseline all
    python run_baseline.py --baseline greedy_fill --nrows 1000000

Each baseline writes its own pickle to outputs_baselines/, so multiple
processes (--baseline X in different terminals) never collide.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from methods import ALL_BASELINES  # noqa: E402
from runner import run_one_baseline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                   choices=list(ALL_BASELINES.keys()) + ["all"],
                   help="Which baseline to run, or 'all' for all selected baselines.")
    p.add_argument(
        "--data",
        default="data/LI-Small_Trans.csv",
        help="Path to an IBM AML transaction CSV file.",
    )
    p.add_argument("--theta",       type=float, default=10000.0)
    p.add_argument("--eps",         type=float, default=0.2)
    p.add_argument("--window-days", type=float, default=0.1)
    p.add_argument("--k",           type=int,   default=8,
                   help="K for TopK baselines (8+8 = 16 candidates).")
    p.add_argument("--nrows",       type=int,   default=None,
                   help="Limit rows for debugging. Default = full dataset.")
    p.add_argument("--out",         default="outputs_baselines")
    p.add_argument("--dataset-name", default="LI-Small")
    p.add_argument("--progress-every", type=int, default=200_000)
    return p.parse_args()


def main():
    args = parse_args()
    params = {
        "theta":       args.theta,
        "eps":         args.eps,
        "window_days": args.window_days,
        "k":           args.k,
    }

    to_run = (
        list(ALL_BASELINES.values())
        if args.baseline == "all"
        else [ALL_BASELINES[args.baseline]]
    )

    for spec in to_run:
        run_one_baseline(
            detect_fn=spec.detect,
            baseline_id=spec.id,
            baseline_name=spec.name,
            params=params,
            csv_path=args.data,
            nrows=args.nrows,
            out_dir=args.out,
            dataset_name=args.dataset_name,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
