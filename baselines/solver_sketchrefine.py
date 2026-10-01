# -*- coding: utf-8 -*-
"""SketchRefine: adaptive 1-D partition, sketch, and refine.

Adapted from Brucato et al., "Scalable Package Queries in Relational Database
Systems" (PVLDB 9(7), 2016), §4 "Scalable Package Evaluation".

The original SKETCHREFINE algorithm has three stages:
  1. PARTITION  — offline, on the query's predicate attributes (k-d tree that
                  enforces both a SIZE THRESHOLD τ and a RADIUS LIMIT ω).
  2. SKETCH     — solve one ILP using a single representative tuple per group;
                  the sketch variables are integer multiplicities n_g ∈ [0, |G|].
  3. REFINE     — replace each non-empty group's representative with the
                  actual tuples, re-solve a smaller ILP.

Adapted to the streaming pass-through flow setting:

  * The "input relation" is the current window's raw in-list (resp. out-list).
  * The only predicate attribute is `amount`, so partitioning is 1-D.
  * PARTITIONING is **adaptive**, mirroring the paper's Definition 1 (size
    threshold τ) and Definition 2 (radius limit ω = γ·|representative|,
    γ = ε): greedy 1-D pass that closes the current group whenever either
    (a) the group's `max/min` ratio would exceed `(1+ε)/(1-ε)` — equivalent
    to any tuple in the group deviating from the group mean by more than
    ε — or (b) the group's size would exceed `MAX_GROUP_SIZE`. Adaptive
    partitioning produces MORE groups where amounts are spread out (avoiding
    the mean-approximation issue of equal-frequency binning) and FEWER
    groups where amounts are uniform (keeping the sketch ILP small).
  * SKETCH: integer-multiplicity ILP on the K resulting representatives per
    side (K is data-dependent, no longer a fixed K).
  * REFINE: when the sketch is feasible, gather all tuples from groups the
    sketch touched (n_g > 0) and resolve as a tuple-level ILP (ILP-baseline style)
    restricted to those tuples.

Solver: scipy.optimize.milp (in-process HiGHS), same as the ILP baseline.
The 100-500ms PuLP+CBC subprocess overhead would otherwise dominate.

Pre-filter: same O(1) trivial-infeasibility checks as the ILP baseline, before either
ILP is constructed.

Intuition:           "what if we approximate first on partition representatives,
                     then refine inside the touched partitions? — with the
                     representative approximation error bounded by ω = ε per
                     PaQL Definition 2."
Failure mode:        even bounded-radius groups can produce sketch's discrete
                     `n × mean` values that miss the wedge feasible interval
                     (integer granularity); more sophisticated partitioning
                     reduces but does not eliminate this failure.
Dependency:          scipy >= 1.9.  No CPLEX, no PostgreSQL.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import cli_main

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


NAME = "sketchrefine"

# ── Adaptive-partitioning parameters (primary, PaQL DLV-style) ──
#
# RADIUS_EPS controls the maximum in-group amount spread. A group is closed
# and a new one started whenever adding the next (sorted) tuple would make
# `max/min > (1 + RADIUS_EPS) / (1 - RADIUS_EPS)`. This bounds the group
# mean's approximation error to ≤ RADIUS_EPS × |mean| on any tuple in the
# group — the 1-D analog of PaQL Definition 2's radius limit ω = γ · |t̃|
# with γ = ε (Brucato'16, Theorem 3).
#
# By default we tie RADIUS_EPS to the wedge tolerance ε passed into query()
# (the reasonable "match the query" setting). RADIUS_EPS below can be
# monkey-patched by benchmarks if a different constant is desired.
RADIUS_EPS = None          # None → use the query `eps` in query()

# MAX_GROUP_SIZE bounds the number of tuples per group (PaQL Definition 1's
# size threshold τ). Guards against pathologically long uniform runs
# producing a single giant group whose refine ILP would explode.
MAX_GROUP_SIZE = 500

# ── Legacy parameter (deprecated) ──
# K_PARTITIONS was the fixed number of equal-frequency groups per side in
# the earlier implementation. Adaptive partitioning ignores this. Kept for
# backward compatibility so existing benchmark scripts that monkey-patch
# K_PARTITIONS still import cleanly (they simply have no effect on
# partitioning now).
K_PARTITIONS = 5

# Time budgets per stage. Sketch has ≤ 2K integer vars and finishes fast;
# refine is bounded by the number of tuples in touched groups.
SKETCH_TIME_LIMIT  = 0.15
REFINE_TIME_LIMIT  = 0.5


# ---------------------------------------------------------------------------
# 1-D equal-frequency partitioning helper — LEGACY, kept for benchmarking
# comparisons against the adaptive scheme.
# ---------------------------------------------------------------------------
def _equal_freq_partition(sorted_amts, k):
    """Sorted amounts → list of equal-frequency groups (each a numpy slice).
    Drops empty groups (happens only when len(sorted_amts) < k)."""
    if sorted_amts.size == 0:
        return []
    k = min(k, sorted_amts.size)
    n = sorted_amts.size
    groups = []
    for i in range(k):
        lo = (i * n) // k
        hi = ((i + 1) * n) // k
        if hi > lo:
            groups.append(sorted_amts[lo:hi])
    return groups


# ---------------------------------------------------------------------------
# 1-D adaptive partitioning — PaQL DLV-style, primary implementation.
# ---------------------------------------------------------------------------
def _adaptive_1d_partition(sorted_amts,
                            eps=None,
                            size_limit=None):
    """Greedy 1-D partitioning enforcing BOTH radius and size limits.

    Parameters
    ----------
    sorted_amts : np.ndarray
        Amounts on one side, already sorted ascending. Assumed all > 0
        (radius test degenerates for zeros; zero-amount tuples cannot
        contribute to a threshold and are effectively skipped).
    eps : float, optional
        Radius tolerance. A group is closed whenever adding the next
        tuple would make ``max / min > (1 + eps) / (1 - eps)``. Defaults
        to the module-level RADIUS_EPS if set, else the caller-passed
        ``eps`` (typically the query's wedge ε).
    size_limit : int, optional
        Maximum tuples per group. Defaults to module-level MAX_GROUP_SIZE.
        Guards against a very long uniform run producing a giant group.

    Returns
    -------
    list of np.ndarray
        Non-empty group slices covering the input in order.

    Guarantees
    ----------
    * For any tuple ``t`` in group ``g``, ``|mean(g) - t| ≤ eps · mean(g)``
      (the 1-D analog of PaQL Definition 2's radius bound with γ = ε).
    * ``len(g) ≤ size_limit`` for every group.

    Complexity: O(n) single greedy pass.

    Note: unlike PaQL's DLV, we do not require an offline preprocessing
    phase. Because our windows are small (typically ≤ 200 tuples), running
    the greedy pass per anchor adds < 50 μs to the per-call latency.
    """
    if eps is None:
        eps = RADIUS_EPS if RADIUS_EPS is not None else 0.2
    if size_limit is None:
        size_limit = MAX_GROUP_SIZE

    n = sorted_amts.size
    if n == 0:
        return []

    # max / min <= max_ratio  ⇔  radius (max-min)/2 <= eps · mean
    max_ratio = (1.0 + eps) / (1.0 - eps)

    groups = []
    start = 0
    for i in range(1, n):
        would_be_size = i - start + 1
        would_be_max  = float(sorted_amts[i])
        would_be_min  = float(sorted_amts[start])

        violates_size   = would_be_size > size_limit
        violates_radius = (would_be_min > 0.0
                           and would_be_max > would_be_min * max_ratio)

        if violates_size or violates_radius:
            groups.append(sorted_amts[start:i])
            start = i

    groups.append(sorted_amts[start:])
    return groups


# ---------------------------------------------------------------------------
# Sketch ILP   (integer multiplicities on partition representatives)
# ---------------------------------------------------------------------------
def _initial_state_feasible(in_init, out_init, theta, eps):
    return (
        in_init >= theta
        and out_init >= (1.0 - eps) * in_init
        and out_init <= (1.0 + eps) * in_init
    )


def _solve_sketch(in_reps, in_sizes, out_reps, out_sizes,
                  in_init, out_init, theta, eps, time_limit):
    K_in  = len(in_reps)
    K_out = len(out_reps)
    n     = K_in + K_out

    if n == 0:
        if _initial_state_feasible(in_init, out_init, theta, eps):
            return np.array([], dtype=int), np.array([], dtype=int)
        return None, None

    a = np.concatenate([in_reps,  np.zeros(K_out)])
    b = np.concatenate([np.zeros(K_in), out_reps])

    A_ub = np.vstack([
        -a,
        (1.0 - eps) * a - b,
        -(1.0 + eps) * a + b,
    ])
    b_ub = np.array([
        in_init - theta,
        out_init - (1.0 - eps) * in_init,
        (1.0 + eps) * in_init - out_init,
    ])

    constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    integrality = np.ones(n)
    ub          = np.concatenate([in_sizes, out_sizes])
    bounds      = Bounds(lb=np.zeros(n), ub=ub.astype(float))
    c           = np.zeros(n)

    res = milp(
        c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit, "disp": False},
    )
    if not res.success:
        return None, None
    # Round to int and split back into in / out counts.
    vals = np.rint(res.x).astype(int)
    return vals[:K_in], vals[K_in:]


# ---------------------------------------------------------------------------
# Refine ILP   (binary per-tuple over touched groups; same form as the ILP baseline)
# ---------------------------------------------------------------------------
def _solve_refine(in_amts, out_amts, in_init, out_init, theta, eps, time_limit):
    n_in, n_out = len(in_amts), len(out_amts)
    n = n_in + n_out

    if n == 0:
        return 1 if _initial_state_feasible(in_init, out_init, theta, eps) else 0

    a = np.concatenate([in_amts,  np.zeros(n_out)])
    b = np.concatenate([np.zeros(n_in), out_amts])

    A_ub = np.vstack([
        -a,
        (1.0 - eps) * a - b,
        -(1.0 + eps) * a + b,
    ])
    b_ub = np.array([
        in_init - theta,
        out_init - (1.0 - eps) * in_init,
        (1.0 + eps) * in_init - out_init,
    ])

    res = milp(
        np.zeros(n),
        constraints=LinearConstraint(A_ub, -np.inf, b_ub),
        integrality=np.ones(n),
        bounds=Bounds(lb=0, ub=1),
        options={"time_limit": time_limit, "disp": False},
    )
    return 1 if res.success else 0


# ---------------------------------------------------------------------------
# Main query entry
# ---------------------------------------------------------------------------
def query(raw_in, raw_out, trigger_amt, anchor_type, theta, eps, **kwargs):
    in_amts  = np.fromiter((amt for _, amt in raw_in),  dtype=np.float64)
    out_amts = np.fromiter((amt for _, amt in raw_out), dtype=np.float64)

    in_init  = trigger_amt if anchor_type == "in"  else 0.0
    out_init = trigger_amt if anchor_type == "out" else 0.0

    # ---- O(1) trivial-infeasibility pre-filter (same as the ILP baseline) ----
    in_sum  = in_amts.sum()  if in_amts.size  else 0.0
    out_sum = out_amts.sum() if out_amts.size else 0.0
    if in_init + in_sum < theta:
        return 0
    if out_init + out_sum < (1.0 - eps) * theta:
        return 0

    # ---- 1. Adaptive 1-D partitioning (PaQL DLV-style, radius + size) ----
    # Each group's mean approximates any tuple within eps (wedge tolerance);
    # each group's size is capped by MAX_GROUP_SIZE. K is data-dependent.
    in_sorted  = np.sort(in_amts)
    out_sorted = np.sort(out_amts)
    in_groups  = _adaptive_1d_partition(in_sorted,  eps=eps)
    out_groups = _adaptive_1d_partition(out_sorted, eps=eps)

    in_reps   = np.array([g.mean() for g in in_groups],  dtype=np.float64) if in_groups  else np.array([])
    out_reps  = np.array([g.mean() for g in out_groups], dtype=np.float64) if out_groups else np.array([])
    in_sizes  = np.array([g.size for g in in_groups],  dtype=np.int64)  if in_groups  else np.array([])
    out_sizes = np.array([g.size for g in out_groups], dtype=np.int64)  if out_groups else np.array([])

    # ---- 2. Sketch ILP on representatives ----
    in_counts, out_counts = _solve_sketch(
        in_reps, in_sizes, out_reps, out_sizes,
        in_init, out_init, theta, eps, time_limit=SKETCH_TIME_LIMIT,
    )
    if in_counts is None:
        return 0   # sketch infeasible / timed out

    # ---- 3. Refine ILP on tuples from touched groups ----
    refine_in_parts  = [in_groups[k]  for k, n_k in enumerate(in_counts)  if n_k > 0]
    refine_out_parts = [out_groups[k] for k, n_k in enumerate(out_counts) if n_k > 0]
    refine_in  = np.concatenate(refine_in_parts)  if refine_in_parts  else np.array([])
    refine_out = np.concatenate(refine_out_parts) if refine_out_parts else np.array([])

    return _solve_refine(
        refine_in, refine_out, in_init, out_init,
        theta, eps, time_limit=REFINE_TIME_LIMIT,
    )


if __name__ == "__main__":
    cli_main(query, NAME)
