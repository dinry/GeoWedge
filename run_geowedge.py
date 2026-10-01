# -*- coding: utf-8 -*-
"""Run GeoWedge on a tuple stream.

This is a compact entry point. It loads one stream CSV,
streams it in timestamp order, applies the property filters, and calls the
GeoWedge compressed frontier-search routine for the remaining candidates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
GEOWEDGE_DIR = ROOT / "geowedge"
if str(GEOWEDGE_DIR) not in sys.path:
    sys.path.insert(0, str(GEOWEDGE_DIR))

import cpp_backend
import geowedge_search
import streaming
from streaming import Config, query_stream, load_stream_dataframe


ALGORITHMS = {
    "geowedge": (
        geowedge_search.frontier_search_wedgebucket_cascade,
        {"delta_sa": 0.1, "delta_d": 0.1, "max_states": 4000},
    ),
    "frontier_bucket": (
        geowedge_search.frontier_search,
        {"delta_sa": 0.1, "delta_d": 0.1, "max_states": 4000},
    ),
    "geowedge_cpp": (
        cpp_backend.frontier_search_cascade_cpp,
        {"delta_sa": 0.1, "delta_d": 0.1, "max_states": 4000,
         "compression_mode": "log_md"},
    ),
}


def _to_txns_nocap(in_list, out_list, anchor_amt, anchor_dir, max_candidates):
    """Use the full candidate window regardless of `max_candidates`."""
    return streaming.to_txns_balanced(in_list, out_list, anchor_amt, anchor_dir, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GeoWedge on a tuple stream.")
    parser.add_argument("--data", default="data/LI-Small_Trans.csv",
                        help="Path to a stream CSV file.")
    parser.add_argument("--dataset-name", default="LI-Small",
                        help="Name used in output files.")
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="geowedge",
                        help="GeoWedge variant to run.")
    parser.add_argument("--theta", type=float, default=10000.0,
                        help="Minimum incoming aggregate threshold.")
    parser.add_argument("--eps", type=float, default=0.2,
                        help="Allowed imbalance ratio.")
    parser.add_argument("--window-days", type=float, default=0.1,
                        help="Sliding window size in days.")
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="Candidate cap per anchor. Use 0 for the full window.")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Read only the first N rows for a quick smoke test.")
    parser.add_argument("--skip-rows", type=int, default=0,
                        help="Skip the first N data rows while preserving the CSV header.")
    parser.add_argument("--candidate-strategy",
                        choices=("balanced", "hybrid", "nocap"),
                        default="nocap",
                        help="Candidate selector used before frontier search.")
    parser.add_argument("--out", default="outputs",
                        help="Output directory.")
    parser.add_argument("--checkpoint-every", type=int, default=50000,
                        help="Checkpoint interval in rows.")
    parser.add_argument("--progress-every", type=int, default=50000,
                        help="Progress-print interval in rows.")
    parser.add_argument("--time-budget-sec", type=float, default=3600.0,
                        help="Soft wall-clock time budget for this invocation.")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore any existing checkpoint for this run.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.candidate_strategy == "hybrid":
        streaming.to_txns = streaming.to_txns_hybrid
    elif args.candidate_strategy == "nocap":
        streaming.to_txns = _to_txns_nocap
    else:
        streaming.to_txns = streaming.to_txns_balanced

    cfg = Config(
        min_in_sum=args.theta,
        ratio_high=args.eps,
        window_days=args.window_days,
        max_candidates=args.max_candidates,
        output_dir=args.out,
    )

    data_path = Path(args.data)
    df = load_stream_dataframe(str(data_path), nrows=args.nrows,
                               skip_rows=args.skip_rows)
    technique_fn, technique_kwargs = ALGORITHMS[args.algo]
    technique_name = f"{args.algo}_{args.dataset_name}"

    query_stream(
        df=df,
        cfg=cfg,
        technique_fn=technique_fn,
        technique_name=technique_name,
        technique_kwargs=technique_kwargs,
        output_dir=args.out,
        checkpoint_every=args.checkpoint_every,
        time_budget_sec=args.time_budget_sec,
        progress_print_every=args.progress_every,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
