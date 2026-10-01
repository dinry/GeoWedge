# -*- coding: utf-8 -*-
"""Shared streaming query-processing plumbing for GeoWedge and the baselines.

Pipeline (GeoWedge's prune-then-search structure; the search stage is one of
the frontier_search variants):

    for each transaction e arriving in time order:
        1. update sliding window for both endpoints (acc, opp)
        2. for each anchor view (acc-side, opp-side):
             a. fetch (raw_in_list, raw_out_list)
             b. apply Properties 1-5 filtering -> (pruned_in, pruned_out)
             c. if filtering empties the query instance => answer 0
             d. otherwise call the chosen frontier-search technique
        3. record per-transaction wall-clock time
        4. push e into the sliding window

The runner is RESUMABLE — it checkpoints (sliding-window state, counters,
per-tx records) to disk every `--checkpoint-every` transactions, and on the
next invocation it picks up where it left off. Each invocation runs for at
most `--time-budget` seconds, then writes a final checkpoint and exits.
"""

from __future__ import annotations
import json
import os
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from state_search import Txn, State


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    """Query parameters for the GeoWedge frontier-search pipeline.

    Necessary knobs:
        - theta   (min_in_sum)   : minimum aggregate amount of the incoming package
        - epsilon (ratio_high)   : maximum relative imbalance, feasible iff |r| <= eps
        - Delta   (window_days)  : sliding window length, 0.1 day for LI-Small
        - max_candidates         : perf safeguard on candidates per anchor

    Plus four bucket-compression knobs used ONLY by `bucket` / `adaptive`.
    """
    min_in_sum: float = 10000.0
    ratio_high: float = 0.2
    window_days: float = 0.1
    max_candidates: int = 32
    output_dir: str = "outputs"

    delta_sa: float = 0.1
    delta_d: float = 0.1
    delta_near: float = 0.02
    delta_far: float = 0.5
    tau: float = 1.5

    # Cascade ablation switch. When False, the wedge-bucket fallback inside
    # WedgeBucket-Cascade is skipped — only the Phase-1 greedy walk decides.
    # Measures how many positive answers the bucket tier contributes.
    # Default True keeps the full algorithm.
    bucket_enabled: bool = True


# -----------------------------------------------------------------------------
# Section 4.2 PointPrune: bound-based pruning
# -----------------------------------------------------------------------------
def point_prune(in_list: List[float], out_list: List[float],
                anchor_side: str, anchor_val: float,
                cfg: Config,
                sum_in: Optional[float] = None,
                sum_out: Optional[float] = None):
    """PointPrune (Algorithm 2): property-based pruning with P1..P5.

    `instance_prune` applies the instance-level checks P1-P3; the loop below
    removes tuples with P4/P5 and repeats until no tuple is removed.

    The caller can pass `sum_in` and `sum_out` (running totals maintained by
    `StreamQuery`) to avoid an O(W) `sum(...)` recomputation on every call.
    These are recomputed internally once any P4/P5 pruning iteration shrinks
    the lists.
    """
    if sum_in is None:
        sum_in = sum(in_list)
    if sum_out is None:
        sum_out = sum(out_list)

    one_plus_eps = 1.0 + cfg.ratio_high

    def instance_prune(sum_in_l: float, sum_out_l: float):
        if anchor_side == "out":
            if len(in_list) == 0: return 0 #P1
            if sum_out_l + anchor_val < (1 - cfg.ratio_high) * cfg.min_in_sum: return 0 #P1
            if sum_in_l < cfg.min_in_sum: return 0 #P1
            if min(in_list) * (1 - cfg.ratio_high) > sum_out_l + anchor_val: return 0 #P2
            # P3 (paper Eq. ①):  w / (1+ε)  >  Σ w_i ∈ H^in_u(t)
            #   ⇔  anchor_val  >  (1+ε) · sum_in_l
            if anchor_val > one_plus_eps * sum_in_l: return 0 #P3
        else:
            if len(out_list) == 0: return 0 #P1
            if sum_in_l + anchor_val < cfg.min_in_sum: return 0 #P1
            if sum_out_l < (1 - cfg.ratio_high) * cfg.min_in_sum: return 0 #P1
            if anchor_val * (1 - cfg.ratio_high) > sum_out_l: return 0 #P2
            # P3 (paper Eq. ②):  min(out_list) / (1+ε)  >  Σ w_i ∈ H^in_v(t) + w
            #   ⇔  min(out_list)  >  (1+ε) · (sum_in_l + anchor_val)
            if min(out_list) > one_plus_eps * (sum_in_l + anchor_val): return 0 #P3
        return -1

    flag = instance_prune(sum_in, sum_out)
    if flag == 0:
        return 0, in_list, out_list

    changed = True
    while changed:
        changed = False
        if anchor_side == "out":
            new_in = [x for x in in_list if (1 - cfg.ratio_high) * x <= sum_out + anchor_val]  #P4
            # P5 (paper Eq. ①):  (w_o + w) / (1+ε)  >  Σ w_i ∈ H^in_u(t)   ⇒ drop w_o
            #   keep w_o if  w_o + anchor_val  ≤  (1+ε) · sum_in
            _p5_rhs = one_plus_eps * sum_in
            new_out = [x for x in out_list if x + anchor_val <= _p5_rhs]   #P5
        else:
            new_in = [x for x in in_list if (1 - cfg.ratio_high) * x <= sum_out]  #P4
            # P5 (paper Eq. ②):  w_o / (1+ε)  >  Σ w_i ∈ H^in_v(t) + w   ⇒ drop w_o
            #   keep w_o if  w_o  ≤  (1+ε) · (sum_in + anchor_val)
            _p5_rhs = one_plus_eps * (sum_in + anchor_val)
            new_out = [x for x in out_list if x <= _p5_rhs]  #P5

        if len(new_in) != len(in_list) or len(new_out) != len(out_list):
            in_list, out_list = new_in, new_out
            sum_in = sum(in_list)
            sum_out = sum(out_list)
            changed = True

        flag = instance_prune(sum_in, sum_out)
        if flag == 0:
            return 0, in_list, out_list

    return -1, in_list, out_list


# -----------------------------------------------------------------------------
# Stream / window state — picklable for checkpointing
# -----------------------------------------------------------------------------
def _empty_window():
    """Picklable factory for defaultdict; replaces the original lambda."""
    return {"in": deque(), "out": deque()}


class StreamQuery:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.window = timedelta(days=cfg.window_days)
        self.data = defaultdict(_empty_window)
        # Running sums maintained incrementally — avoids the O(W) sum() in
        # point_prune / property-filter for every anchor.
        self.sum_in: defaultdict = defaultdict(float)
        self.sum_out: defaultdict = defaultdict(float)

    def update_window(self, acc, t):
        cutoff = t - self.window
        dq_in = self.data[acc]["in"]
        while dq_in and dq_in[0][0] < cutoff:
            _, v = dq_in.popleft()
            self.sum_in[acc] -= v
        dq_out = self.data[acc]["out"]
        while dq_out and dq_out[0][0] < cutoff:
            _, v = dq_out.popleft()
            self.sum_out[acc] -= v

    def add_tx(self, acc, direction, t, amt):
        self.data[acc][direction].append((t, amt))
        if direction == "in":
            self.sum_in[acc] += amt
        else:
            self.sum_out[acc] += amt


# -----------------------------------------------------------------------------
# Window lists -> Txn list adapter
# -----------------------------------------------------------------------------
def to_txns(in_list: List[float], out_list: List[float],
            anchor_amt: float, anchor_dir: str,
            max_candidates: int) -> Tuple[List[Txn], str]:
    if max_candidates > 0 and (len(in_list) + len(out_list)) > max_candidates:
        combined = (
            [("in", v) for v in in_list] + [("out", v) for v in out_list]
        )
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:max_candidates]
        in_list = [v for d, v in combined if d == "in"]
        out_list = [v for d, v in combined if d == "out"]

    anchor_eid = "anchor"
    txns: List[Txn] = [Txn(anchor_eid, anchor_dir, float(anchor_amt))]
    for i, v in enumerate(in_list):
        txns.append(Txn(f"in_{i}", "in", float(v)))
    for i, v in enumerate(out_list):
        txns.append(Txn(f"out_{i}", "out", float(v)))
    return txns, anchor_eid


def to_txns_balanced(in_list: List[float], out_list: List[float],
                      anchor_amt: float, anchor_dir: str,
                      max_candidates: int) -> Tuple[List[Txn], str]:
    """Balanced per-side top-K candidate selector.

    For each side (in / out) independently, take its own top-(max_candidates//2)
    by amount descending. This ensures both directions contribute candidates
    even when one side has much larger amounts than the other.

    Aligns the main algorithms with the per-side top-K selection used by
    `baselines/methods.py`,
    so the search spaces are directly comparable (only the SEARCH strategy
    differs, not the CANDIDATE selection).

    Difference from `to_txns` (union top-K):
        to_txns           : sort (in + out) together, take top max_candidates
                            → can be heavily skewed when one side has bigger amounts
        to_txns_balanced  : take top K/2 from each side independently
                            → both sides always represented (when available)

    Special case: max_candidates <= 0  →  NO CANDIDATE CAP.
        All candidates (sorted desc by amount) are returned.
        Use this for bucket / adaptive, whose state space is bounded by the
        bucket count (independent of candidate count), so no cap is required.
        Do NOT use this for enum / dom — their state space is O(2^N) and
        will explode.
    """
    if max_candidates > 0:
        k_per_side = max(1, max_candidates // 2)
        in_list = sorted(in_list, reverse=True)[:k_per_side]
        out_list = sorted(out_list, reverse=True)[:k_per_side]
    else:
        # No cap: keep all candidates, still sort desc by amount so the
        # frontier search expands largest first (helps early exit in
        # _alive_states for bucket / adaptive).
        in_list = sorted(in_list, reverse=True)
        out_list = sorted(out_list, reverse=True)

    anchor_eid = "anchor"
    txns: List[Txn] = [Txn(anchor_eid, anchor_dir, float(anchor_amt))]
    for i, v in enumerate(in_list):
        txns.append(Txn(f"in_{i}", "in", float(v)))
    for i, v in enumerate(out_list):
        txns.append(Txn(f"out_{i}", "out", float(v)))
    return txns, anchor_eid


def to_txns_hybrid(in_list: List[float], out_list: List[float],
                    anchor_amt: float, anchor_dir: str,
                    max_candidates: int) -> Tuple[List[Txn], str]:
    """Hybrid candidate selector — top-K by amount AND top-K by closeness to
    the trigger amount, unioned per side.

    Motivation
    ----------
    `to_txns_balanced` (top-K by amount per side) misses any feasible
    package pair whose members are not among the K largest. For triggers
    where the matching package consists of MEDIUM-sized candidates near the
    trigger amount (very common when the pass-through flow "mirrors" the
    trigger), pure top-K-by-amount has zero chance of finding them.

    The hybrid version splits the per-side budget in half:
      * half by amount descending  (captures the "biggest contributors")
      * half by |amt − anchor_amt| ascending  (captures the "best matches"
        — these directly help close the SA ≈ SB balance)

    Total candidate count is still ≤ max_candidates (slightly less when
    the two halves overlap on the same candidate, which is fine).

    Special case: max_candidates ≤ 0  →  NO CAP, same as
    `to_txns_balanced(... 0)`.

    Use case
    --------
    Drop-in replacement for `to_txns_balanced` when you want bucket / adaptive
    variants to consider both
    biggest-by-mass and closest-by-value candidates without raising K.
    """
    if max_candidates <= 0:
        in_list = sorted(in_list, reverse=True)
        out_list = sorted(out_list, reverse=True)
    else:
        k_per_side = max(1, max_candidates // 2)
        k_amount = max(1, k_per_side // 2)
        k_close  = max(1, k_per_side - k_amount)

        # Half by amount descending (mass)
        in_amount  = sorted(in_list,  reverse=True)[:k_amount]
        out_amount = sorted(out_list, reverse=True)[:k_amount]

        # Half by closeness to anchor amount (balance-relevant)
        in_close   = sorted(in_list,  key=lambda v: abs(v - anchor_amt))[:k_close]
        out_close  = sorted(out_list, key=lambda v: abs(v - anchor_amt))[:k_close]

        # Union + dedup; keep largest first for early-termination friendliness.
        in_list  = sorted(set(in_amount)  | set(in_close),  reverse=True)
        out_list = sorted(set(out_amount) | set(out_close), reverse=True)

    anchor_eid = "anchor"
    txns: List[Txn] = [Txn(anchor_eid, anchor_dir, float(anchor_amt))]
    for i, v in enumerate(in_list):
        txns.append(Txn(f"in_{i}", "in", float(v)))
    for i, v in enumerate(out_list):
        txns.append(Txn(f"out_{i}", "out", float(v)))
    return txns, anchor_eid


# -----------------------------------------------------------------------------
# Stream loader
# -----------------------------------------------------------------------------
DEFAULT_STREAM_PATH = "data/LI-Small_Trans.csv"
OUT_FLAG = "out"


def load_stream_dataframe(path: str = DEFAULT_STREAM_PATH,
                          nrows: Optional[int] = None,
                          skip_rows: int = 0) -> pd.DataFrame:
    """Memory-efficient loader: keep only the 4 columns we use, dtype-tight.

    Parameters
    ----------
    path : str
        CSV file to read.
    nrows : int, optional
        Maximum number of DATA rows to read (after skipping). None = read all.
    skip_rows : int, default 0
        Skip the first `skip_rows` DATA rows (header is preserved). Useful for
        cross-slice validation: e.g., `skip_rows=1_000_000, nrows=1_000_000`
        loads data rows 1,000,001..2,000,000 with proper header.
    """
    read_kwargs = dict(
        nrows=nrows,
        usecols=["Account", "Timestamp", "Amount Paid", "Account.1"],
        dtype={
            "Account": "category",
            "Account.1": "category",
            "Amount Paid": "float32",
        },
    )
    if skip_rows > 0:
        # skiprows=range(1, N+1) skips DATA rows 1..N while KEEPING the
        # header at file row 0. Using range() is faster than a callable.
        read_kwargs["skiprows"] = range(1, skip_rows + 1)
    df = pd.read_csv(path, **read_kwargs)
    df["direction_flag"] = OUT_FLAG
    df = df[["Account", "Timestamp", "direction_flag",
             "Amount Paid", "Account.1"]]
    df.columns = ["account_num", "trans_date", "direction_flag",
                  "trans_amount", "opposite_account_num"]
    df["trans_date"] = pd.to_datetime(df["trans_date"], format="%Y/%m/%d %H:%M")
    # Sort by time; convert account categories to plain strings only when
    # consumed in the loop (saves memory while sorting).
    df = df.sort_values("trans_date").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Checkpoint state structure (everything needed to resume)
# -----------------------------------------------------------------------------
@dataclass
class CheckpointState:
    next_idx: int = 0
    stream_query: StreamQuery = None
    n_stage2_calls: int = 0
    n_total_anchor_cases: int = 0
    n_pred_positives: int = 0
    cumulative_wallclock: float = 0.0
    done: bool = False


def _save_checkpoint(state_path: Path,
                     ckpt: CheckpointState):
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, state_path)


def _load_checkpoint(state_path: Path) -> Optional[CheckpointState]:
    if not state_path.exists():
        return None
    with open(state_path, "rb") as f:
        return pickle.load(f)


def _write_progress(out_dir: Path, technique: str,
                    ckpt: CheckpointState, total_rows: int,
                    last_chunk_seconds: float,
                    last_chunk_rows: int):
    progress = {
        "technique": technique,
        "rows_processed": ckpt.next_idx,
        "rows_total": total_rows,
        "pct_done": round(100 * ckpt.next_idx / max(total_rows, 1), 3),
        "done": ckpt.done,
        "positive_query_answers": ckpt.n_pred_positives,
        "cumulative_wallclock_sec": ckpt.cumulative_wallclock,
        "mean_ms_per_tx": float(ckpt.cumulative_wallclock /
                                  max(ckpt.next_idx, 1) * 1000),
        "last_chunk_rows": last_chunk_rows,
        "last_chunk_seconds": last_chunk_seconds,
        "last_chunk_rows_per_sec": float(last_chunk_rows /
                                          max(last_chunk_seconds, 1e-9)),
    }
    with open(out_dir / f"{technique}_progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    return progress


# -----------------------------------------------------------------------------
# Resumable streaming query loop
# -----------------------------------------------------------------------------
def query_stream(df: pd.DataFrame,
                  cfg: Config,
                  technique_fn: Callable,
                  technique_name: str,
                  technique_kwargs: Optional[dict] = None,
                  output_dir: Optional[str] = None,
                  checkpoint_every: int = 50_000,
                  time_budget_sec: float = 35.0,
                  progress_print_every: int = 50_000,
                  reset: bool = False):
    """Resumable filter-then-frontier-search runner.

    Saves checkpoint to `<output_dir>/<technique_name>_state.pkl` every
    `checkpoint_every` rows. Writes incremental progress to
    `<output_dir>/<technique_name>_progress.json`. Stops gracefully when the
    wallclock budget is exhausted.

    Per-transaction timing + records are appended to
    `<technique_name>_timings.bin` and `<technique_name>_records.jsonl`
    (NOT held in memory — avoids OOM on the 6.9M LI-Small stream).
    """
    if output_dir is None:
        output_dir = cfg.output_dir
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    technique_kwargs = technique_kwargs or {}

    state_path = out_dir / f"{technique_name}_state.pkl"
    timings_path = out_dir / f"{technique_name}_timings.bin"
    records_path = out_dir / f"{technique_name}_records.jsonl"
    summary_path = out_dir / f"{technique_name}_summary.json"
    progress_path = out_dir / f"{technique_name}_progress.json"

    if reset:
        for p in [state_path, timings_path, records_path,
                  summary_path, progress_path]:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    ckpt = _load_checkpoint(state_path)
    if ckpt is None or ckpt.stream_query is None:
        ckpt = CheckpointState(stream_query=StreamQuery(cfg))

    if ckpt.done:
        print(f"[{technique_name}] already done; rows={ckpt.next_idx}")
        return _make_summary(technique_name, cfg, ckpt, len(df))

    stream_query = ckpt.stream_query
    n_total = len(df)

    # Open append-mode streams for per-tx timings (binary, 8 bytes/double) and
    # records (jsonl). They're append-only so resuming just continues writing.
    timings_f = open(timings_path, "ab")
    records_f = open(records_path, "a", encoding="utf-8")

    import struct
    DOUBLE = struct.Struct("<d")
    chunk_start = time.time()
    chunk_start_idx = ckpt.next_idx

    # Pre-extract column arrays — iterating `df.iloc[idx]` per row is ~30x
    # slower than indexing flat numpy arrays. Cost paid once per invocation.
    ts_arr     = df["trans_date"].to_numpy(dtype=object)
    acc_arr    = df["account_num"].astype(str).values
    opp_arr    = df["opposite_account_num"].astype(str).values
    amt_arr    = df["trans_amount"].values
    flag_arr   = df["direction_flag"].values
    try:
        for idx in range(ckpt.next_idx, n_total):
            t = ts_arr[idx]
            acc = acc_arr[idx]
            opp = opp_arr[idx]
            amt = float(amt_arr[idx])
            flag = flag_arr[idx]
            t_tx0 = time.time()
            stream_query.update_window(acc, t)
            stream_query.update_window(opp, t)

            anchors = (
                [(acc, "out"), (opp, "in")]
                if flag == OUT_FLAG
                else [(acc, "in"), (opp, "out")]
            )

            rule_flags = [0, 0]
            n_cands = [0, 0]
            pred_any = 0

            for k, (account, anchor_type) in enumerate(anchors):
                ckpt.n_total_anchor_cases += 1
                raw_in = [v for _, v in stream_query.data[account]["in"]]
                raw_out = [v for _, v in stream_query.data[account]["out"]]
                rf, pi, po = point_prune(
                    raw_in, raw_out, anchor_type, amt, cfg,
                )
                rule_flags[k] = rf
                if rf == 0:
                    continue
                txns, trigger_eid = to_txns(pi, po, amt, anchor_type, cfg.max_candidates)
                n_cands[k] = len(txns) - 1
                ckpt.n_stage2_calls += 1
                found = technique_fn(txns, trigger_eid, cfg.min_in_sum, cfg.ratio_high,
                                       **technique_kwargs)
                if found is not None:
                    pred_any = 1

            dt = time.time() - t_tx0

            if pred_any == 1:
                ckpt.n_pred_positives += 1

            timings_f.write(DOUBLE.pack(dt))
            records_f.write(json.dumps({
                "idx": idx, "query_answer": pred_any,
                "rule_flag_acc": rule_flags[0], "rule_flag_opp": rule_flags[1],
                "n_cands_acc": n_cands[0], "n_cands_opp": n_cands[1],
                "ms": dt * 1000,
            }) + "\n")

            ckpt.next_idx = idx + 1
            ckpt.cumulative_wallclock += dt

            # Push trigger into the sliding window AFTER query evaluation.
            if flag == OUT_FLAG:
                stream_query.add_tx(acc, "out", t, amt)
                stream_query.add_tx(opp, "in", t, amt)
            else:
                stream_query.add_tx(acc, "in", t, amt)
                stream_query.add_tx(opp, "out", t, amt)

            # ---- periodic checkpoint + progress ------------------------------
            if (idx + 1) % checkpoint_every == 0:
                timings_f.flush()
                records_f.flush()
                _save_checkpoint(state_path, ckpt)
                chunk_sec = time.time() - chunk_start
                chunk_rows = (idx + 1) - chunk_start_idx
                prog = _write_progress(out_dir, technique_name, ckpt,
                                         n_total, chunk_sec, chunk_rows)
                if (idx + 1) % progress_print_every == 0:
                    print(f"[{technique_name}] {prog['rows_processed']}/{n_total} "
                          f"({prog['pct_done']:.2f}%)  "
                          f"positive_answers={prog['positive_query_answers']}  "
                          f"avg={prog['mean_ms_per_tx']:.3f}ms/tx  "
                          f"chunk_rps={prog['last_chunk_rows_per_sec']:.0f}",
                          flush=True)

            # ---- soft time budget --------------------------------------------
            if time.time() - chunk_start >= time_budget_sec:
                print(f"[{technique_name}] time budget {time_budget_sec:.0f}s hit "
                      f"at idx={idx+1}; checkpointing and exiting", flush=True)
                break

        else:
            ckpt.done = True

    finally:
        timings_f.flush(); timings_f.close()
        records_f.flush(); records_f.close()
        _save_checkpoint(state_path, ckpt)
        chunk_sec = time.time() - chunk_start
        chunk_rows = ckpt.next_idx - chunk_start_idx
        _write_progress(out_dir, technique_name, ckpt,
                        n_total, chunk_sec, chunk_rows)

    summary = _make_summary(technique_name, cfg, ckpt, n_total)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "technique": technique_name,
        "done": ckpt.done,
        "rows_processed": ckpt.next_idx,
        "rows_total": n_total,
        "positive_query_answers": ckpt.n_pred_positives,
        "wallclock_sec": ckpt.cumulative_wallclock,
        "mean_ms_per_tx": summary["mean_per_tx_ms"],
    }, indent=2, ensure_ascii=False))
    return summary


def _make_summary(technique_name, cfg, ckpt, n_total):
    return {
        "technique": technique_name,
        "config": {
            "min_in_sum": cfg.min_in_sum,
            "ratio_high": cfg.ratio_high,
            "window_days": cfg.window_days,
            "max_candidates": cfg.max_candidates,
        },
        "done": bool(ckpt.done),
        "rows_processed": int(ckpt.next_idx),
        "rows_total": int(n_total),
        "positive_query_answers": int(ckpt.n_pred_positives),
        "total_wallclock_seconds": float(ckpt.cumulative_wallclock),
        "mean_per_tx_ms": float(ckpt.cumulative_wallclock /
                                  max(ckpt.next_idx, 1) * 1000),
    }
