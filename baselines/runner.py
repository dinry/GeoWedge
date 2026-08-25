# -*- coding: utf-8 -*-
"""Shared streaming pipeline for the baseline detectors.

The runner streams rows, maintains the sliding window, and invokes a selected
baseline at every transaction-anchor query. It reports query-processing
statistics rather than treating IBM's laundering flag as a classification
label.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the GeoWedge loader and stream window.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "geowedge"))
from streaming import Config, OUT_FLAG, StreamDetector, load_li_small_dataframe  # noqa: E402


# ==========================================================================
# Helper used by all four TopK baselines
# ==========================================================================
def topk_enumerate_numpy(top_in_amts, top_out_amts,
                          initial_sa, initial_sb,
                          theta, eps):
    """Vectorised full enumeration of all 2^(K_in + K_out) subsets.

    No pruning, no shortcut — just numpy-vectorised "compute every subset
    sum and check whether any (SA, SB) pair satisfies the IIO predicate".

    Returns 1 if any valid subset exists, else 0.
    """
    # 2^|top_in| subset sums
    in_sums = np.array([0.0])
    for amt in top_in_amts:
        in_sums = np.concatenate([in_sums, in_sums + amt])
    out_sums = np.array([0.0])
    for amt in top_out_amts:
        out_sums = np.concatenate([out_sums, out_sums + amt])

    SA = initial_sa + in_sums       # shape (2^|top_in|,)
    SB = initial_sb + out_sums      # shape (2^|top_out|,)

    SA_pass = SA >= theta
    if not SA_pass.any():
        return 0
    SA_valid = SA[SA_pass]

    SA_lo = SA_valid * (1.0 - eps)
    SA_hi = SA_valid * (1.0 + eps)
    # (P, 1) vs (1, Q) broadcast
    in_band = (SB[None, :] >= SA_lo[:, None]) & (SB[None, :] <= SA_hi[:, None])
    return int(bool(in_band.any()))


# ==========================================================================
# Streaming driver
# ==========================================================================
def run_one_baseline(detect_fn,
                      baseline_id: int,
                      baseline_name: str,
                      params: dict,
                      csv_path: str,
                      nrows=None,
                      skip_rows: int = 0,
                      out_dir: str = "outputs_baselines",
                      progress_every: int = 200_000,
                      dataset_name: str = "LI-Small",
                      latency_unit: str = "us"):
    """Stream an IBM AML CSV and run `detect_fn` at every (tx, anchor) pair.

    detect_fn signature:
        detect_fn(raw_in, raw_out, trigger_amt, anchor_type,
                  theta, eps, **kwargs) -> 0 or 1

      raw_in / raw_out are lists of (timestamp, amount) tuples from the
      window for the anchor account. NO P1..P5 has been applied.
    """
    theta = params["theta"]
    eps   = params["eps"]
    cfg = Config(min_in_sum=theta,
                 ratio_high=eps,
                 window_days=params["window_days"])

    print(f"[baseline {baseline_name}] loading {csv_path}"
          f" (skip_rows={skip_rows:,}, nrows={nrows})",
          flush=True)
    t_load = time.time()
    df = load_li_small_dataframe(csv_path, nrows=nrows, skip_rows=skip_rows)
    n_total = len(df)
    print(f"[baseline {baseline_name}] loaded {n_total:,} rows in "
          f"{time.time() - t_load:.1f}s", flush=True)

    ts_arr    = df["trans_date"].to_numpy(dtype=object)
    acc_arr   = df["account_num"].astype(str).values
    opp_arr   = df["opposite_account_num"].astype(str).values
    amt_arr   = df["trans_amount"].values
    flag_arr  = df["loan_flag"].values

    detector = StreamDetector(cfg)

    n_filter_tests = 0
    n_anchor_positives = 0
    positive_triggers = set()

    # ---- Per-arrival decision timing. One slot per arrival;
    # measured strictly around the `detect_fn(...)` calls at both anchors so
    # it captures only the method's decision logic, not window I/O. Saved
    # alongside positive_triggers for cross-method timing comparison.
    if latency_unit not in ("us", "ns"):
        raise ValueError(f"unsupported latency_unit={latency_unit!r}")
    per_arrival_latency = np.zeros(n_total, dtype=np.int64)
    if latency_unit == "ns":
        perf_counter = time.perf_counter_ns
        latency_scale = 1
    else:
        perf_counter = time.perf_counter
        latency_scale = 1_000_000

    print(f"[baseline {baseline_name}] streaming ...  "
          f"(runner is filter-agnostic; detect_fn owns any pre-filtering)",
          flush=True)
    t_run_start = time.time()
    t_chunk = t_run_start

    for idx in range(n_total):
        t = ts_arr[idx]
        acc = acc_arr[idx]
        opp = opp_arr[idx]
        amt = float(amt_arr[idx])
        flag = flag_arr[idx]

        detector.update_window(acc, t)
        detector.update_window(opp, t)

        anchors = (
            [(acc, "out"), (opp, "in")]
            if flag == OUT_FLAG
            else [(acc, "in"), (opp, "out")]
        )

        # Pass auxiliary params (e.g. k for TopK) but NOT theta/eps,
        # which are already supplied positionally.
        extra_kwargs = {kk: vv for kk, vv in params.items()
                        if kk not in ("theta", "eps")}

        trigger_positive = False
        _arr_latency = 0
        for account, anchor_type in anchors:
            n_filter_tests += 1
            raw_in  = list(detector.data[account]["in"])
            raw_out = list(detector.data[account]["out"])
            _t0 = perf_counter()
            d = detect_fn(raw_in, raw_out, amt, anchor_type,
                          theta, eps, **extra_kwargs)
            if latency_unit == "ns":
                _arr_latency += int(perf_counter() - _t0)
            else:
                _arr_latency += int((perf_counter() - _t0) * latency_scale)
            if d:
                n_anchor_positives += 1
                trigger_positive = True
        per_arrival_latency[idx] = _arr_latency
        if trigger_positive:
            positive_triggers.add(idx)

        if flag == OUT_FLAG:
            detector.add_tx(acc, "out", t, amt)
            detector.add_tx(opp, "in", t, amt)
        else:
            detector.add_tx(acc, "in", t, amt)
            detector.add_tx(opp, "out", t, amt)

        if (idx + 1) % progress_every == 0:
            now = time.time()
            chunk_sec = now - t_chunk
            overall_sec = now - t_run_start
            chunk_rate = progress_every / max(chunk_sec, 1e-9)
            overall_rate = (idx + 1) / max(overall_sec, 1e-9)
            remaining = n_total - (idx + 1)
            eta = remaining / max(overall_rate, 1e-9)
            print(
                f"  [{baseline_name:<14}] {idx + 1:>10,}/{n_total:,} "
                f"({100 * (idx + 1) / n_total:5.1f}%)   "
                f"chunk={chunk_rate:>10,.0f}/s   "
                f"positive_answers={len(positive_triggers):>9,}   "
                f"ETA={eta:6.0f}s",
                flush=True,
            )
            t_chunk = now

    runtime = time.time() - t_run_start
    return _save_results(
        baseline_id=baseline_id,
        baseline_name=baseline_name,
        dataset_name=dataset_name,
        params=params,
        n_total=n_total,
        n_filter_tests=n_filter_tests,
        n_anchor_positives=n_anchor_positives,
        positive_triggers=positive_triggers,
        per_arrival_latency=per_arrival_latency,
        latency_unit=latency_unit,
        runtime=runtime,
        out_dir=out_dir,
    )


def _save_results(*, baseline_id, baseline_name, dataset_name, params,
                  n_total, n_filter_tests, n_anchor_positives,
                  positive_triggers,
                  per_arrival_latency, latency_unit, runtime, out_dir):
    """Persist query-processing results for one baseline run."""
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_pkl = out_dir_p / f"{baseline_name}_{dataset_name}.pkl"

    n_pos_triggers = len(positive_triggers)

    payload = {
        "baseline_id":             int(baseline_id),
        "baseline_name":           baseline_name,
        "dataset_name":            dataset_name,
        "params":                  params,
        "rows_processed":          int(n_total),
        "positive_query_answers":  int(n_pos_triggers),
        "positive_triggers":       positive_triggers,
        "wallclock_sec":           float(runtime),
        "mean_us_per_tx":          float(runtime / max(n_total, 1) * 1e6),
        "latency_unit":            latency_unit,
    }
    if latency_unit == "ns":
        payload["per_arrival_ns"] = per_arrival_latency
    else:
        payload["per_arrival_us"] = per_arrival_latency

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(json.dumps({
        "method": baseline_name,
        "dataset": dataset_name,
        "rows_processed": int(n_total),
        "positive_query_answers": int(n_pos_triggers),
        "wallclock_sec": float(runtime),
        "mean_us_per_tx": float(runtime / max(n_total, 1) * 1e6),
        "output_file": str(out_pkl),
    }, indent=2, ensure_ascii=False))

    return payload


# ==========================================================================
# CLI shim: let each baseline module be run directly.
# Each baseline module's __main__ block does:
#     from runner import cli_main
#     cli_main(detect, ID, NAME)
# ==========================================================================
def cli_main(detect_fn, baseline_id: int, baseline_name: str):
    """Parse CLI args and run the streaming pipeline for a single baseline."""
    p = argparse.ArgumentParser(
        description=f"Run baseline {baseline_id}: {baseline_name}"
    )
    p.add_argument(
        "--data",
        default="data/LI-Small_Trans.csv",
        help="Path to an IBM AML transaction CSV file.",
    )
    p.add_argument("--theta",       type=float, default=10000.0,
                   help="P1 mass threshold (default 10000).")
    p.add_argument("--eps",         type=float, default=0.2,
                   help="Balance tolerance (default 0.2).")
    p.add_argument("--window-days", type=float, default=0.1,
                   help="Sliding window length in days (default 0.1).")
    p.add_argument("--k",           type=int,   default=8,
                   help="K for TopK baselines (default 8); ignored by greedy "
                        "and agg-only.")
    p.add_argument("--nrows",       type=int,   default=None,
                   help="Limit rows for debugging. Default = full dataset.")
    p.add_argument("--skip-rows",   type=int,   default=0,
                   help="Skip the first N data rows before reading. Useful "
                        "for cross-slice validation (e.g., --skip-rows 1000000 "
                        "--nrows 1000000 reads data rows [1M+1, 2M]).")
    p.add_argument("--out",         default="outputs_baselines",
                   help="Output directory for the result pickle.")
    p.add_argument("--dataset-name", default="LI-Small")
    p.add_argument("--progress-every", type=int, default=200_000,
                   help="Print progress every N rows (default 200,000).")
    p.add_argument("--latency-unit", choices=("us", "ns"), default="us",
                   help="Store per-arrival decision latency in integer "
                        "microseconds (default) or nanoseconds.")
    args = p.parse_args()

    params = {
        "theta":       args.theta,
        "eps":         args.eps,
        "window_days": args.window_days,
        "k":           args.k,
    }

    run_one_baseline(
        detect_fn=detect_fn,
        baseline_id=baseline_id,
        baseline_name=baseline_name,
        params=params,
        csv_path=args.data,
        nrows=args.nrows,
        skip_rows=args.skip_rows,
        out_dir=args.out,
        dataset_name=args.dataset_name,
        progress_every=args.progress_every,
        latency_unit=args.latency_unit,
    )
