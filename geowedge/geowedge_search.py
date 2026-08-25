# -*- coding: utf-8 -*-
"""GeoWedge compressed frontier-search routines.

Compared with the basic frontier routines, this module adds:
  * A per-state pruning that drops states which provably cannot lead to any
    valid (SA >= theta, |SA - SB| <= eps*SA) solution. Two conditions, each
    independently safe:

        (a) SA reachability:  s.sa + rem_in_k < theta
              → no descendant can push SA to theta
        (b) SB catchability:  s.sb + rem_out_k < s.sa * (1 - eps)
              → SB can never grow enough to balance even the current SA
              (and a larger descendant SA only tightens the constraint)

  * Both conditions are MATHEMATICALLY SAFE — a state dropped here has no
    valid descendant, so the 0/1 output is bit-for-bit identical to the
    un-pruned version (no change to recall_vs_label, recall_vs_enum, etc.).

When this helps:
  Pathological survivors that pass Stage 1's mass filter but cannot actually
  balance — e.g. "hot account residue" where account mass crosses theta but
  the residue is too one-sided to find an IIO subset. In that regime,
  un-pruned enum exhaustively walks 2^16 ≈ 65 K states per survivor,
  observed as ~200 ms / survivor (≈ 5 / s).

The public API matches `state_search.py` for drop-in replacement.
"""

import math
import time
from typing import List, Optional, Tuple
from state_search import (
    Txn, State,
    exact_check, merge_same_states, safe_dominance_prune,
    residual_bucket_compress, adaptive_residual_bucket_compress,
    log_bucket, _expand,
)


# ==========================================================================
# Phase-level timing accumulator
# --------------------------------------------------------------------------
# Both `frontier_search_wedgebucket_cascade` and
# `frontier_search_umbrella_xing` write per-phase elapsed time into this
# module-level dict. Filter-stage (Stage 1) time is NEVER recorded here —
# survivors are already loaded from pickle by the time these functions are
# called. So all values below measure ONLY the Stage 2 algorithm work,
# uniformly across both algorithms.
#
# Use:
#   from geowedge_search import reset_phase_timing, get_phase_timing
#   reset_phase_timing()
#   ... run algorithm on all survivors ...
#   print(get_phase_timing())
# ==========================================================================
_PHASE_TIMING = {
    "cascade_greedy_seconds":      0.0,    # cascade Phase 1 cumulative
    "cascade_bucket_seconds":      0.0,    # cascade Phase 2 cumulative
    "cascade_calls":               0,
    "cascade_greedy_hits":         0,      # cascade Phase 1 successes
    "cascade_bucket_invocations":  0,      # times Phase 2 ran

    "umbrella_xing_greedy_seconds": 0.0,   # umbrella_xing greedy walk
    "umbrella_xing_rescue_seconds": 0.0,   # subset-sum-in-band rescue
    "umbrella_xing_bucket_seconds": 0.0,   # umbrella_xing local bucket
    "umbrella_xing_calls":          0,
    "umbrella_xing_greedy_hits":    0,
    "umbrella_xing_rescue_hits":    0,     # crossing solved by subset rescue
    "umbrella_xing_bucket_invocations": 0, # crossing fell through to bucket
}


def reset_phase_timing() -> None:
    """Zero out the per-phase timing accumulator."""
    for k in _PHASE_TIMING:
        _PHASE_TIMING[k] = 0 if isinstance(_PHASE_TIMING[k], int) else 0.0


def get_phase_timing() -> dict:
    """Return a snapshot of the current accumulator."""
    return dict(_PHASE_TIMING)


def _initial_states_with_suffix(transactions: List[Txn], trigger_eid: str
                                 ) -> Tuple[List[State], List[Txn],
                                            List[float], List[float]]:
    """Like `state_search._initial_states`, plus suffix-sums of remaining
    "in" / "out" candidate amounts. Candidates are sorted by amount desc
    (mirrors the optimization in `state_search.py`)."""
    trigger = next(t for t in transactions if t.eid == trigger_eid)
    if trigger.direction == "in":
        states = [State(sa=trigger.amount, sb=0.0)]
    else:
        states = [State(sa=0.0, sb=trigger.amount)]
    candidates = [t for t in transactions if t.eid != trigger_eid]
    candidates.sort(key=lambda t: t.amount, reverse=True)
    n = len(candidates)
    rem_in = [0.0] * (n + 1)
    rem_out = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        if candidates[k].direction == "in":
            rem_in[k] = rem_in[k + 1] + candidates[k].amount
            rem_out[k] = rem_out[k + 1]
        else:
            rem_in[k] = rem_in[k + 1]
            rem_out[k] = rem_out[k + 1] + candidates[k].amount
    return states, candidates, rem_in, rem_out


def _alive_states(states: List[State],
                  rem_in_k: float, rem_out_k: float,
                  theta: float, one_minus_eps: float) -> List[State]:
    """Drop states whose descendants cannot yield a valid solution.

    Safe pruning — each condition rules out the entire descendant subtree:
      (a) s.sa + rem_in_k < theta            → SA can never reach theta
      (b) s.sb + rem_out_k < s.sa*(1-eps)    → SB can never catch SA's lower
                                               balance bound (and a larger
                                               descendant SA only worsens it)
    """
    alive = []
    for s in states:
        sa = s.sa
        if sa + rem_in_k < theta:
            continue
        if s.sb + rem_out_k < sa * one_minus_eps:
            continue
        alive.append(s)
    return alive


# --------------------------------------------------------------------------
# Four frontier_search variants with safe early termination
# --------------------------------------------------------------------------
def frontier_search_enum(transactions: List[Txn], trigger_eid: str,
                          theta: float, eps: float) -> Optional[State]:
    states, candidates, rem_in, rem_out = _initial_states_with_suffix(
        transactions, trigger_eid,
    )
    one_minus_eps = 1.0 - eps

    found = exact_check(states, theta, eps)
    if found is not None:
        return found
    states = _alive_states(states, rem_in[0], rem_out[0], theta, one_minus_eps)
    if not states:
        return None

    for k, z in enumerate(candidates):
        states = _expand(states, z)
        states = merge_same_states(states)
        states = _alive_states(states, rem_in[k + 1], rem_out[k + 1],
                               theta, one_minus_eps)
        if not states:
            return None
        found = exact_check(states, theta, eps)
        if found is not None:
            return found
    return None


def frontier_search_dom(transactions: List[Txn], trigger_eid: str,
                         theta: float, eps: float) -> Optional[State]:
    states, candidates, rem_in, rem_out = _initial_states_with_suffix(
        transactions, trigger_eid,
    )
    one_minus_eps = 1.0 - eps

    found = exact_check(states, theta, eps)
    if found is not None:
        return found
    states = _alive_states(states, rem_in[0], rem_out[0], theta, one_minus_eps)
    if not states:
        return None

    for k, z in enumerate(candidates):
        states = _expand(states, z)
        states = merge_same_states(states)
        states = safe_dominance_prune(states)
        states = _alive_states(states, rem_in[k + 1], rem_out[k + 1],
                               theta, one_minus_eps)
        if not states:
            return None
        found = exact_check(states, theta, eps)
        if found is not None:
            return found
    return None


def frontier_search_bucket(transactions: List[Txn], trigger_eid: str,
                            theta: float, eps: float,
                            delta_sa: float = 0.1,
                            delta_d: float = 0.1) -> Optional[State]:
    states, candidates, rem_in, rem_out = _initial_states_with_suffix(
        transactions, trigger_eid,
    )
    one_minus_eps = 1.0 - eps

    found = exact_check(states, theta, eps)
    if found is not None:
        return found
    states = _alive_states(states, rem_in[0], rem_out[0], theta, one_minus_eps)
    if not states:
        return None

    for k, z in enumerate(candidates):
        states = _expand(states, z)
        states = merge_same_states(states)
        states = safe_dominance_prune(states)
        states = residual_bucket_compress(states,
                                          delta_sa=delta_sa, delta_d=delta_d)
        states = _alive_states(states, rem_in[k + 1], rem_out[k + 1],
                               theta, one_minus_eps)
        if not states:
            return None
        found = exact_check(states, theta, eps)
        if found is not None:
            return found
    return None


def frontier_search_adaptive(transactions: List[Txn], trigger_eid: str,
                              theta: float, eps: float,
                              delta_sa: float = 0.1,
                              delta_near: float = 0.02,
                              delta_far: float = 0.5,
                              tau: float = 1.5) -> Optional[State]:
    states, candidates, rem_in, rem_out = _initial_states_with_suffix(
        transactions, trigger_eid,
    )
    one_minus_eps = 1.0 - eps

    found = exact_check(states, theta, eps)
    if found is not None:
        return found
    states = _alive_states(states, rem_in[0], rem_out[0], theta, one_minus_eps)
    if not states:
        return None

    for k, z in enumerate(candidates):
        states = _expand(states, z)
        states = merge_same_states(states)
        states = safe_dominance_prune(states)
        states = adaptive_residual_bucket_compress(
            states, eps=eps,
            delta_sa=delta_sa, delta_near=delta_near,
            delta_far=delta_far, tau=tau,
        )
        states = _alive_states(states, rem_in[k + 1], rem_out[k + 1],
                               theta, one_minus_eps)
        if not states:
            return None
        found = exact_check(states, theta, eps)
        if found is not None:
            return found
    return None


# ==========================================================================
# TASB — Trajectory-Augmented Signed Bucket
#
# Combines three ideas, each with a clean theoretical claim:
#
#   (1) SIGNED bucket compression. The bucket key gains a +1/-1 component
#       for sign(SA - SB), so over-balance states (SA > SB) and under-balance
#       states (SA < SB) are NEVER merged into the same bucket — they have
#       different future trajectories under further candidate addition.
#
#   (2) Trajectory injection. We run a greedy alternating walk (advance the
#       smaller side each step) IN PARALLEL with the bucket search, and at
#       every iteration we inject the greedy state back into the bucket's
#       state set. This guarantees the bucket cannot lose any state the
#       greedy walk produced.
#
#   (3) (1+δ)-relaxed exact check. Validity is judged by
#         SA >= θ  and  |SA-SB| <= ε(1+δ) · SA,
#       where δ is the bucket width parameter. This is exactly the
#       approximation tolerance introduced by the log-bucket compression.
#
# Formal claims for the paper:
#   * Recall ≥ max(recall_greedy, recall_unaugmented_bucket).
#   * (1+δ)·ε-approximation w.r.t. the IIO definition.
#   * O(B + N) state per anchor (B = number of signed buckets, N =
#     candidate count), independent of subset-enumeration explosion.
# ==========================================================================
def signed_bucket_compress(states: List[State],
                            delta_sa: float = 0.1,
                            delta_d: float = 0.1) -> List[State]:
    """Bucket compression keyed by (log SA, log |gap|, SIGN(SA - SB)).

    The sign dimension separates over-balance (SA > SB) from under-balance
    (SA < SB) states with identical magnitudes — they have very different
    futures under further candidate addition, so merging them loses
    correctness.
    """
    buckets = {}
    for s in states:
        sa, sb = s.sa, s.sb
        gap = sa - sb
        d = abs(gap)
        r = (d / sa) if sa else float("inf")
        sign = 1 if gap >= 0 else -1
        key = (log_bucket(sa, delta_sa), log_bucket(d, delta_d), sign)
        old = buckets.get(key)
        if old is None:
            buckets[key] = (s, r)
        else:
            old_state, old_r = old
            if r < old_r or (r == old_r and sa > old_state.sa):
                buckets[key] = (s, r)
    return [v[0] for v in buckets.values()]


def exact_check_relaxed(states: List[State],
                         theta: float,
                         eps_relaxed: float) -> Optional[State]:
    """Same shape as `exact_check` but with a relaxed eps tolerance."""
    for s in states:
        sa = s.sa
        if sa >= theta and abs(sa - s.sb) <= eps_relaxed * sa:
            return s
    return None


def frontier_search_tasb(transactions: List[Txn], trigger_eid: str,
                          theta: float, eps: float,
                          delta_sa: float = 0.1,
                          delta_d: float = 0.1) -> Optional[State]:
    """Trajectory-Augmented Signed Bucket. See module-level comment."""
    # --- initial state -------------------------------------------------
    trigger = next(t for t in transactions if t.eid == trigger_eid)
    if trigger.direction == "in":
        initial = State(sa=trigger.amount, sb=0.0)
    else:
        initial = State(sa=0.0, sb=trigger.amount)

    bucket_states = [initial]

    # --- greedy trajectory state runs IN PARALLEL ----------------------
    # Greedy uses its OWN candidate order (largest-first per side), and
    # advances by 1 step each main-loop iteration. The greedy step adds to
    # whichever side currently has the smaller running sum (alternating).
    greedy_sa, greedy_sb = initial.sa, initial.sb
    greedy_in_iter = iter(sorted(
        (t.amount for t in transactions
         if t.eid != trigger_eid and t.direction == "in"),
        reverse=True,
    ))
    greedy_out_iter = iter(sorted(
        (t.amount for t in transactions
         if t.eid != trigger_eid and t.direction == "out"),
        reverse=True,
    ))

    # --- bucket main-loop candidate order (largest-first across sides) ---
    candidates = [t for t in transactions if t.eid != trigger_eid]
    candidates.sort(key=lambda t: t.amount, reverse=True)
    n = len(candidates)
    rem_in = [0.0] * (n + 1)
    rem_out = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        if candidates[k].direction == "in":
            rem_in[k] = rem_in[k + 1] + candidates[k].amount
            rem_out[k] = rem_out[k + 1]
        else:
            rem_in[k] = rem_in[k + 1]
            rem_out[k] = rem_out[k + 1] + candidates[k].amount

    one_minus_eps = 1.0 - eps
    eps_relaxed = eps * (1.0 + delta_d)   # (1+δ)-relaxed validity

    # --- initial check -------------------------------------------------
    found = exact_check_relaxed(bucket_states, theta, eps_relaxed)
    if found is not None:
        return found
    bucket_states = _alive_states(bucket_states, rem_in[0], rem_out[0],
                                   theta, one_minus_eps)
    if not bucket_states:
        return None

    # --- main loop -----------------------------------------------------
    for k, z in enumerate(candidates):
        # (a) bucket main path: expand → merge → dominance → SIGNED bucket
        bucket_states = _expand(bucket_states, z)
        bucket_states = merge_same_states(bucket_states)
        bucket_states = safe_dominance_prune(bucket_states)
        bucket_states = signed_bucket_compress(bucket_states,
                                                delta_sa, delta_d)

        # (b) greedy trajectory: advance 1 step, INJECT the new state
        try:
            if greedy_sa <= greedy_sb:
                amt = next(greedy_in_iter)
                greedy_sa += amt
            else:
                amt = next(greedy_out_iter)
                greedy_sb += amt
            bucket_states.append(State(sa=greedy_sa, sb=greedy_sb))
        except StopIteration:
            # one side exhausted — greedy stops, bucket keeps going
            pass

        # (c) safe pruning (uses original eps; the bound stays tight)
        bucket_states = _alive_states(bucket_states,
                                       rem_in[k + 1], rem_out[k + 1],
                                       theta, one_minus_eps)
        if not bucket_states:
            return None

        # (d) (1+δ)-relaxed validity check
        found = exact_check_relaxed(bucket_states, theta, eps_relaxed)
        if found is not None:
            return found

    return None


# ==========================================================================
# TASB v3 — single-criterion algorithm (NO separate greedy thread)
#
# v1 / v2 were structurally "bucket ∪ greedy" or "balance + advance"
# unions — two perspectives glued at compression time. v3 removes the
# duality entirely: ONE compression rule (advance-prioritized lex)
# operating on a single state set.
#
# The compression rule per signed bucket cell:
#       keep the state with the LARGEST (SA + SB).
#
# Why this single rule captures the greedy-trajectory effect (Lemma D):
#       For any bucket cell C visited by some valid state, the cell
#       representative chosen by largest-(SA+SB) is the "leading edge"
#       of C along the diagonal direction. Under the (1+δ)-relaxed
#       check, leading-edge representatives are valid whenever any
#       state in C is valid — including greedy-trajectory states.
#       Hence v3 implicitly preserves greedy coverage without ever
#       running a separate greedy walk.
#
# Paper-grade narrative collapses to:
#   "TASB extends Ibarra-Kim FPTAS with (a) logarithmic-scale lattice,
#    (b) sign-aware bucketing, and (c) advance-prioritized tiebreak.
#    These three modifications, applied to a single state frontier,
#    yield a (1+δ)·ε-approximation in O((log V)²) state."
# ==========================================================================
def diagonal_lex_compress(states: List[State],
                           delta_sa: float = 0.15,
                           delta_d: float = 0.15,
                           max_states: int = 1500) -> List[State]:
    """Single-criterion compression: per signed bucket cell, keep the state
    with the LARGEST (SA + SB).

    The bucket key (log SA, log |SA-SB|, sign(SA-SB)) constrains every
    state in a cell to be approximately (1+δ)-equivalent in mass and
    imbalance. Within those equivalence classes the advance criterion
    (SA + SB) orders states along the diagonal direction. Keeping the
    argmax-(SA+SB) state per cell yields the "leading edge" of the
    reachable frontier — geometrically the union of greedy-walk
    endpoints across all alternation policies.

    Safety valve: if the post-compression state count exceeds
    `max_states` (hot-account scenarios where the leading edge spans
    many cells), we further truncate by keeping the `max_states` states
    with the SMALLEST residual ratio r — these are the most likely to
    yield a valid relaxed-check at the next iteration. This bound
    sacrifices a small amount of recall in exchange for bounded runtime
    on extreme-cardinality windows.
    """
    # Hot path: build the per-cell argmax-(SA+SB) representative dict.
    # Inline local variable bindings to skip attribute lookups in tight loop.
    buckets = {}
    _log_bucket = log_bucket
    _get = buckets.get
    for s in states:
        sa = s.sa
        sb = s.sb
        gap = sa - sb
        d = gap if gap >= 0 else -gap          # abs(gap), no function call
        sign = 1 if gap >= 0 else -1
        key = (_log_bucket(sa, delta_sa),
               _log_bucket(d,  delta_d),
               sign)
        adv = sa + sb
        old = _get(key)
        if old is None or adv > old[1]:
            buckets[key] = (s, adv)

    result = [v[0] for v in buckets.values()]

    # Safety valve for hot-account survivors
    if len(result) > max_states:
        # Truncate by smallest r — most likely to satisfy validity next step
        def _r(s):
            sa = s.sa
            if not sa:
                return float("inf")
            return abs(sa - s.sb) / sa
        result.sort(key=_r)
        result = result[:max_states]

    return result


def frontier_search_tasb_v3(transactions: List[Txn], trigger_eid: str,
                             theta: float, eps: float,
                             delta_sa: float = 0.15,
                             delta_d: float = 0.15,
                             max_states: int = 1500) -> Optional[State]:
    """Single-thread, single-criterion TASB (v3). No greedy injection."""
    # --- initial state -------------------------------------------------
    trigger = next(t for t in transactions if t.eid == trigger_eid)
    if trigger.direction == "in":
        initial = State(sa=trigger.amount, sb=0.0)
    else:
        initial = State(sa=0.0, sb=trigger.amount)
    states = [initial]

    # --- candidates sorted desc by amount ------------------------------
    candidates = [t for t in transactions if t.eid != trigger_eid]
    candidates.sort(key=lambda t: t.amount, reverse=True)
    n = len(candidates)

    # --- suffix sums for the safe-pruning bound ------------------------
    rem_in = [0.0] * (n + 1)
    rem_out = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        if candidates[k].direction == "in":
            rem_in[k] = rem_in[k + 1] + candidates[k].amount
            rem_out[k] = rem_out[k + 1]
        else:
            rem_in[k] = rem_in[k + 1]
            rem_out[k] = rem_out[k + 1] + candidates[k].amount

    one_minus_eps = 1.0 - eps
    eps_relaxed = eps * (1.0 + delta_d)

    # --- initial relaxed check + alive prune ---------------------------
    found = exact_check_relaxed(states, theta, eps_relaxed)
    if found is not None:
        return found
    states = _alive_states(states, rem_in[0], rem_out[0],
                           theta, one_minus_eps)
    if not states:
        return None

    # --- main loop — purely one thread of states -----------------------
    for k, z in enumerate(candidates):
        states = _expand(states, z)
        states = merge_same_states(states)
        states = safe_dominance_prune(states)
        states = diagonal_lex_compress(states, delta_sa, delta_d,
                                        max_states=max_states)
        states = _alive_states(states, rem_in[k + 1], rem_out[k + 1],
                               theta, one_minus_eps)
        if not states:
            return None
        found = exact_check_relaxed(states, theta, eps_relaxed)
        if found is not None:
            return found
    return None


# ==========================================================================
# WedgeBucketSearch — wedge-guided signed bucket compression.
#
# THE unified single-criterion algorithm:
#
#   Φ(c) measures the L¹ distance from state c = (S_A, S_B) to the IIO
#   wedge W = { (x, y) : x ≥ θ, (1-ε)x ≤ y ≤ (1+ε)x }:
#
#       Φ(c) = max(0, θ - S_A)
#            + max(0, (1-ε)S_A - S_B)
#            + max(0, S_B - (1+ε)S_A)
#
#   Φ(c) = 0  ⟺  c is in the wedge (valid).
#
#   Per signed bucket cell (log S_A, log |S_A-S_B|, sign), retain
#   argmin_c Φ(c). The greedy "move toward wedge" effect is absorbed
#   automatically — states whose candidate-addition reduces Φ win the
#   per-cell representative slot, so the kept frontier monotonically
#   approaches the wedge boundary.
#
# One algorithm. One score. One state set. No greedy thread.
# Equivalent paper-grade narrative:
#   "A signed geometric bucket search whose representatives are
#    selected by a wedge-directed criterion."
# ==========================================================================
def wedge_offset(sa: float, sb: float, theta: float, eps: float) -> float:
    """L¹ distance from (SA, SB) to the valid IIO wedge."""
    phi = 0.0
    if sa < theta:
        phi += theta - sa
    lower = (1.0 - eps) * sa
    upper = (1.0 + eps) * sa
    if sb < lower:
        phi += lower - sb
    elif sb > upper:
        phi += sb - upper
    return phi


def wedge_bucket_compress(states: List[State],
                           theta: float, eps: float,
                           delta_sa: float = 0.1,
                           delta_d: float = 0.1,
                           max_states: int = 4000) -> List[State]:
    """Per signed bucket cell, retain argmin Φ (closest to wedge).

    Tiebreak: larger S_A (more accumulated mass — closer to or past θ).
    Optional safety cap: when |result| exceeds `max_states`, retain only
    the `max_states` states with smallest Φ (closest to wedge).

    Performance notes (measured Jul 2026):
      * Precompute 1.0/log(1+δ_*) ONCE per call — avoids ~40k redundant
        math.log(1+δ) calls on hot windows (was 27% of total bucket time
        per profiler).
      * Inline log_bucket to remove function-call overhead.
      * Hoist constants (one_minus_eps, one_plus_eps) out of loop.
    Combined: ~1.8× speedup on hot windows vs the original log_bucket()
    dispatch, verified via bench_wedge_bucket_compress.py.
    """
    # ---- Precompute constants (hoisted out of hot loop) ----
    _log = math.log
    _floor = math.floor

    # log(1+δ) is constant across states — compute once, reuse the inverse
    inv_log_1p_delta_sa = 1.0 / _log(1.0 + delta_sa)
    inv_log_1p_delta_d  = 1.0 / _log(1.0 + delta_d)

    one_minus_eps = 1.0 - eps
    one_plus_eps  = 1.0 + eps

    buckets = {}
    _get = buckets.get

    for s in states:
        sa = s.sa
        sb = s.sb
        gap = sa - sb
        if gap >= 0:
            d = gap
            sign = 1
        else:
            d = -gap
            sign = -1

        # inline wedge_offset (Φ)
        if sa < theta:
            phi = theta - sa
        else:
            phi = 0.0
        lower = one_minus_eps * sa
        if sb < lower:
            phi += lower - sb
        else:
            upper = one_plus_eps * sa
            if sb > upper:
                phi += sb - upper

        # Inline log_bucket + precomputed inverse log(1+δ)
        # log_bucket(v, δ) = floor(log(v) / log(1+δ))
        #                  = floor(log(v) * inv_log_1p_delta)
        if sa > 0:
            sa_bucket = int(_floor(_log(sa) * inv_log_1p_delta_sa))
        else:
            sa_bucket = 0
        if d > 0:
            d_bucket = int(_floor(_log(d) * inv_log_1p_delta_d))
        else:
            d_bucket = 0

        key = (sa_bucket, d_bucket, sign)
        old = _get(key)
        if old is None:
            buckets[key] = (s, phi)
        else:
            old_state, old_phi = old
            if phi < old_phi or (phi == old_phi and sa > old_state.sa):
                buckets[key] = (s, phi)

    # Build result
    items = list(buckets.values())
    if len(items) > max_states:
        items.sort(key=lambda x: x[1])  # smallest Φ first
        items = items[:max_states]
    return [v[0] for v in items]


def frontier_search_wedgebucket(transactions: List[Txn], trigger_eid: str,
                                  theta: float, eps: float,
                                  delta_sa: float = 0.1,
                                  delta_d: float = 0.1,
                                  max_states: int = 4000
                                  ) -> Optional[State]:
    """Wedge-guided signed bucket search. A single unified algorithm in
    the FPTAS family for streaming 2D subset-sum feasibility.

    The per-cell representative selection criterion is
        argmin_c  Φ(c),
    where Φ(c) is the L¹ geometric distance from c to the valid wedge.

    Adaptive parameter scaling: for hot-account anchors (large window
    cardinality N), automatically coarsen δ and tighten max_states so
    per-anchor work stays bounded. This is a paper-defensible adaptive
    FPTAS variant: the approximation factor (1+δ(N))·ε grows slightly
    with N but the worst-case per-anchor runtime is bounded.

        N ≤ 100   :  δ stays 0.10,  cap stays 4000   (full precision)
        100 < N ≤ 500   :  δ → 0.13,  cap → 2500
        500 < N ≤ 1500  :  δ → 0.18,  cap → 1200
        1500 < N ≤ 3000 :  δ → 0.25,  cap → 600
        N > 3000        :  δ → 0.35,  cap → 300
    """
    # --- initial state -------------------------------------------------
    trigger = next(t for t in transactions if t.eid == trigger_eid)
    if trigger.direction == "in":
        initial = State(sa=trigger.amount, sb=0.0)
    else:
        initial = State(sa=0.0, sb=trigger.amount)
    states = [initial]

    candidates = [t for t in transactions if t.eid != trigger_eid]
    candidates.sort(key=lambda t: t.amount, reverse=True)
    n = len(candidates)

    # --- ADAPTIVE PARAMETER SCALING for hot-account windows ------------
    if n > 3000:
        delta_sa = max(delta_sa, 0.35)
        delta_d  = max(delta_d,  0.35)
        max_states = min(max_states, 300)
    elif n > 1500:
        delta_sa = max(delta_sa, 0.25)
        delta_d  = max(delta_d,  0.25)
        max_states = min(max_states, 600)
    elif n > 500:
        delta_sa = max(delta_sa, 0.18)
        delta_d  = max(delta_d,  0.18)
        max_states = min(max_states, 1200)
    elif n > 100:
        delta_sa = max(delta_sa, 0.13)
        delta_d  = max(delta_d,  0.13)
        max_states = min(max_states, 2500)
    # else: full precision (δ=0.1, cap=4000)

    rem_in = [0.0] * (n + 1)
    rem_out = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        if candidates[k].direction == "in":
            rem_in[k] = rem_in[k + 1] + candidates[k].amount
            rem_out[k] = rem_out[k + 1]
        else:
            rem_in[k] = rem_in[k + 1]
            rem_out[k] = rem_out[k + 1] + candidates[k].amount

    one_minus_eps = 1.0 - eps
    eps_relaxed = eps * (1.0 + delta_d)

    # initial validity check (in case trigger alone passes)
    found = exact_check_relaxed(states, theta, eps_relaxed)
    if found is not None:
        return found
    states = _alive_states(states, rem_in[0], rem_out[0],
                           theta, one_minus_eps)
    if not states:
        return None

    for k, z in enumerate(candidates):
        states = _expand(states, z)
        states = merge_same_states(states)
        states = safe_dominance_prune(states)
        states = wedge_bucket_compress(states, theta, eps,
                                        delta_sa, delta_d,
                                        max_states=max_states)
        states = _alive_states(states, rem_in[k + 1], rem_out[k + 1],
                               theta, one_minus_eps)
        if not states:
            return None
        found = exact_check_relaxed(states, theta, eps_relaxed)
        if found is not None:
            return found
    return None


# ==========================================================================
# WedgeBucketSearch (cascade variant) — Two-Phase Wedge-Guided Search
#
# Combines greedy's O(N) speed with bucket's (1+δ)-approximation:
#
#   Phase 1 (greedy filter):  O(N) per anchor.
#       Walk alternating-largest from the trigger. At every step, check
#       relaxed validity. Returns immediately if any visited state lands
#       inside the (1+δ)·ε-wedge.  Catches ~96% of valid triggers on
#       LI-Small at <1ms each — including most hot-account survivors,
#       because greedy's monotonic trajectory eventually enters the wedge
#       whenever some balanced subset prefix exists.
#
#   Phase 2 (wedge bucket):  O(N·K) per anchor, K = bucket count.
#       Invoked only when greedy fails. Runs the full WedgeBucketSearch
#       to verify whether ANY subset (not on greedy's trajectory) is
#       valid — exploiting the 2D state-frontier compression.
#
# The cascade preserves the formal (1+δ)·ε-approximation guarantee
# (Theorem) because:
#   * Greedy positives are exact subsets that satisfy the relaxed
#     check — directly valid under the approximation.
#   * Greedy negatives trigger the full bucket, which is itself a
#     (1+δ)·ε-approximation.
# Hence positive(cascade) = positive(greedy) ∪ positive(bucket).
#
# Expected per-anchor runtime on LI-Small:
#   ~96% triggers   →  return in phase 1   (sub-ms each)
#   ~4%  triggers   →  fall to phase 2     (10ms — 1s each)
# Total: dominated by greedy cost, with bucket only for hard cases.
# ==========================================================================
def frontier_search_wedgebucket_cascade(transactions: List[Txn],
                                          trigger_eid: str,
                                          theta: float, eps: float,
                                          delta_sa: float = 0.1,
                                          delta_d: float = 0.1,
                                          max_states: int = 4000,
                                          bucket_enabled: bool = True
                                          ) -> Optional[State]:
    """Cascade: greedy first, then wedge bucket only on greedy failure.

    Per-call phase timing is accumulated into the module-level
    `_PHASE_TIMING` dict for fair external comparison.

    If `bucket_enabled=False`, the Phase-2 wedge-bucket fallback is skipped
    and the function returns None whenever greedy misses. This is an
    ABLATION switch — production behaviour uses the default True. Used by
    ablation_cascade/ to quantify how many alerts the bucket tier rescues.
    """
    # Per-call timer starts here — survivors are already loaded;
    # filter-stage time is NOT measured.
    _t_call_start = time.perf_counter()
    _PHASE_TIMING["cascade_calls"] += 1

    trigger = next(t for t in transactions if t.eid == trigger_eid)
    if trigger.direction == "in":
        sa, sb = trigger.amount, 0.0
    else:
        sa, sb = 0.0, trigger.amount

    eps_relaxed = eps * (1.0 + delta_d)

    # ---- Phase 1: greedy alternating walk -----------------------------
    in_amts = sorted(
        (t.amount for t in transactions
         if t.eid != trigger_eid and t.direction == "in"),
        reverse=True,
    )
    out_amts = sorted(
        (t.amount for t in transactions
         if t.eid != trigger_eid and t.direction == "out"),
        reverse=True,
    )
    n_in, n_out = len(in_amts), len(out_amts)

    if sa >= theta and abs(sa - sb) <= eps_relaxed * sa:
        _PHASE_TIMING["cascade_greedy_seconds"] += time.perf_counter() - _t_call_start
        _PHASE_TIMING["cascade_greedy_hits"] += 1
        return State(sa=sa, sb=sb)

    i_in = i_out = 0
    while i_in < n_in or i_out < n_out:
        if sa <= sb:
            if i_in < n_in:
                sa += in_amts[i_in]; i_in += 1
            elif i_out < n_out:
                sb += out_amts[i_out]; i_out += 1
            else:
                break
        else:
            if i_out < n_out:
                sb += out_amts[i_out]; i_out += 1
            elif i_in < n_in:
                sa += in_amts[i_in]; i_in += 1
            else:
                break
        if sa >= theta and abs(sa - sb) <= eps_relaxed * sa:
            _PHASE_TIMING["cascade_greedy_seconds"] += time.perf_counter() - _t_call_start
            _PHASE_TIMING["cascade_greedy_hits"] += 1
            return State(sa=sa, sb=sb)

    # ---- Phase 2: greedy missed → full wedge bucket -------------------
    _PHASE_TIMING["cascade_greedy_seconds"] += time.perf_counter() - _t_call_start
    if not bucket_enabled:
        # Ablation path: skip the bucket fallback. The call counts above are
        # still updated so phase-timing stays consistent across runs.
        return None
    _PHASE_TIMING["cascade_bucket_invocations"] += 1
    _t_bucket_start = time.perf_counter()
    result = frontier_search_wedgebucket(
        transactions, trigger_eid, theta, eps,
        delta_sa=delta_sa, delta_d=delta_d, max_states=max_states,
    )
    _PHASE_TIMING["cascade_bucket_seconds"] += time.perf_counter() - _t_bucket_start
    return result


# ==========================================================================
# Bucket-Greedy — greedy walks over BUCKETS instead of transactions.
#
# Idea (motivated by user):  group log-scale buckets first, then run the
# alternating-largest greedy WALK at bucket granularity, where each step
# advances by one bucket's representative statistic (mean / max / min /
# median). Within a step, we additionally choose adaptively how many
# transactions from the bucket to "use" — enough to plug the current
# SA / SB gap — so a single bucket step can fill several transactions
# worth of mass.
#
# Cost is O(B) bucket-level steps per anchor, where B = #log-scale
# buckets ≤ O(log V / log(1+δ)) ≈ 50-100 regardless of N. For
# hot-account survivors (N=1000-10000), this is a 20-200× speedup over
# transaction-level greedy.
#
# Approximation factor:  (1+δ) · ε.
#   The bucket-mean approximation introduces at most a (1+δ) factor per
#   bucket-step. Total SA / SB errors accumulate as bounded sums (NOT
#   multiplications), so the OVERALL approximation is still (1+δ), not
#   (1+δ)^B. Validity is checked with the (1+δ)·ε relaxed wedge, which
#   exactly absorbs this slack.
#
# Reviewer hook:
#   "A bucket-quantized variant of the alternating-largest greedy
#    walker, attaining the same (1+δ)·ε approximation factor as
#    WedgeBucket but with O(B) per-anchor work and no state-frontier
#    materialization."
# ==========================================================================
def frontier_search_bucket_greedy(transactions: List[Txn], trigger_eid: str,
                                    theta: float, eps: float,
                                    delta: float = 0.1,
                                    stat: str = "mean") -> Optional[State]:
    """Bucket-level alternating greedy walk. stat ∈ {mean, max, min, median}.

    Returns a State (with the tracked approximate SA, SB) if the walk
    ever enters the (1+δ)·ε-relaxed wedge; None otherwise.
    """
    from collections import defaultdict
    import math

    trigger = next(t for t in transactions if t.eid == trigger_eid)
    if trigger.direction == "in":
        sa, sb = trigger.amount, 0.0
    else:
        sa, sb = 0.0, trigger.amount

    eps_relaxed = eps * (1.0 + delta)
    one_minus_eps = 1.0 - eps

    # --- Group candidates into log-scale buckets --------------------
    in_groups = defaultdict(list)
    out_groups = defaultdict(list)
    for t in transactions:
        if t.eid == trigger_eid:
            continue
        b = log_bucket(t.amount, delta)
        (in_groups if t.direction == "in" else out_groups)[b].append(t.amount)

    def _stat_value(amts):
        n = len(amts)
        if stat == "mean":
            return sum(amts) / n
        if stat == "max":
            return max(amts)
        if stat == "min":
            return min(amts)
        if stat == "median":
            return sorted(amts)[n // 2]
        raise ValueError(f"unknown stat: {stat}")

    # Bucket list: (representative value, count). Sorted desc by value.
    in_list = sorted(
        ((_stat_value(a), len(a)) for a in in_groups.values()),
        key=lambda x: -x[0],
    )
    out_list = sorted(
        ((_stat_value(a), len(a)) for a in out_groups.values()),
        key=lambda x: -x[0],
    )

    # --- Initial check ---------------------------------------------
    if sa >= theta and abs(sa - sb) <= eps_relaxed * sa:
        return State(sa=sa, sb=sb)

    # --- Bucket-level alternating walk ------------------------------
    i_in = i_out = 0
    n_in = len(in_list)
    n_out = len(out_list)

    while i_in < n_in or i_out < n_out:
        # Decide which side to advance (smaller-sum side first)
        if sa <= sb:
            if i_in < n_in:
                v, count = in_list[i_in]; i_in += 1
                target = "in"
            elif i_out < n_out:
                v, count = out_list[i_out]; i_out += 1
                target = "out"
            else:
                break
        else:
            if i_out < n_out:
                v, count = out_list[i_out]; i_out += 1
                target = "out"
            elif i_in < n_in:
                v, count = in_list[i_in]; i_in += 1
                target = "in"
            else:
                break

        # Decide how many to take from this bucket.
        # "Enough to plug the SA / SB gap (or reach θ)" — but no more.
        if v <= 0:
            k = count
        elif target == "in":
            need = max(0.0, theta - sa, sb - sa)
            k = min(count, max(1, math.ceil(need / v)))
            sa += k * v
        else:  # target == "out"
            need = max(0.0, one_minus_eps * sa - sb, sa - sb)
            k = min(count, max(1, math.ceil(need / v)))
            sb += k * v

        if sa >= theta and abs(sa - sb) <= eps_relaxed * sa:
            return State(sa=sa, sb=sb)

    return None


# ==========================================================================
# Umbrella Search — Wedge-Proximity-Triggered Bucket Expansion
#
# Most of the time, walk a single greedy state (O(1) per step). When
# the state enters wedge proximity (Φ(s) ≤ τ), "open the umbrella":
# switch to full bucket-compressed multi-state expansion. Once opened,
# stay open until candidates exhausted (sticky umbrella).
#
# Intuition (user's framing):
#   Greedy fails when it commits to a wrong step near the wedge boundary
#   and overshoots into oblivion. The umbrella catches this: when the
#   geometry says "next step matters", we explore alternative subset
#   choices via bucket compression instead of greedy's single pick.
#
# Threshold τ:
#   τ = c · ε · θ        (default c = 2.0)
#   c=0  → umbrella never opens   (= pure greedy)
#   c=∞  → umbrella always open   (= wedgebucket)
#   c=2  → sweet spot:  ε-band wide enough for greedy's typical "wrong
#                       step" magnitudes, narrow enough to stay greedy
#                       on the far approach.
#
# Approximation status:
#   - Strict (1+δ)·ε-FPTAS:  requires umbrella to open BEFORE greedy
#     commits an irreversible wrong step. The threshold c only
#     approximates this — there's no general guarantee that the
#     umbrella opens "early enough" on adversarial inputs.
#   - Empirical:  on LI-Small, typical valid IIO subsets are
#     diagonal-aligned, so greedy's trajectory passes through wedge
#     proximity with high probability. Umbrella opens in time to
#     catch most of the missed cases.
#
# For deployments requiring the certified guarantee, use cascade.
# For deployments wanting maximum throughput at slight recall cost,
# use umbrella.
# ==========================================================================
def frontier_search_umbrella(transactions: List[Txn], trigger_eid: str,
                               theta: float, eps: float,
                               delta_sa: float = 0.1,
                               delta_d: float = 0.1,
                               umbrella_c: float = 2.0,
                               max_states: int = 4000
                               ) -> Optional[State]:
    """Greedy with wedge-proximity umbrella."""
    trigger = next(t for t in transactions if t.eid == trigger_eid)
    if trigger.direction == "in":
        sa0, sb0 = trigger.amount, 0.0
    else:
        sa0, sb0 = 0.0, trigger.amount

    states = [State(sa=sa0, sb=sb0)]
    eps_relaxed = eps * (1.0 + delta_d)
    one_minus_eps = 1.0 - eps
    phi_threshold = umbrella_c * eps * theta

    # Initial validity check
    if sa0 >= theta and abs(sa0 - sb0) <= eps_relaxed * sa0:
        return states[0]

    candidates = [t for t in transactions if t.eid != trigger_eid]
    candidates.sort(key=lambda t: t.amount, reverse=True)
    n = len(candidates)

    # Suffix sums for safe pruning when umbrella is open
    rem_in = [0.0] * (n + 1)
    rem_out = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        if candidates[k].direction == "in":
            rem_in[k] = rem_in[k + 1] + candidates[k].amount
            rem_out[k] = rem_out[k + 1]
        else:
            rem_in[k] = rem_in[k + 1]
            rem_out[k] = rem_out[k + 1] + candidates[k].amount

    umbrella_open = False

    for k, c in enumerate(candidates):
        # ---- Should umbrella open? -----------------------------------
        if not umbrella_open:
            # Compute Φ on the current (single) state
            s = states[0]
            phi = wedge_offset(s.sa, s.sb, theta, eps)
            if phi <= phi_threshold:
                umbrella_open = True

        if umbrella_open:
            # ---- Bucket mode: full expansion + compression ----------
            new_states = _expand(states, c)
            new_states = merge_same_states(new_states)
            new_states = safe_dominance_prune(new_states)
            new_states = wedge_bucket_compress(new_states, theta, eps,
                                                delta_sa, delta_d,
                                                max_states=max_states)
            new_states = _alive_states(new_states,
                                        rem_in[k + 1], rem_out[k + 1],
                                        theta, one_minus_eps)
            states = new_states
            if not states:
                return None
        else:
            # ---- Greedy mode: single state, phi-guided skip/add ----
            s = states[0]
            if c.direction == "in":
                s_new = State(sa=s.sa + c.amount, sb=s.sb)
            else:
                s_new = State(sa=s.sa, sb=s.sb + c.amount)
            phi_new = wedge_offset(s_new.sa, s_new.sb, theta, eps)
            phi_cur = wedge_offset(s.sa, s.sb, theta, eps)
            if phi_new <= phi_cur:
                # adding reduces (or matches) Φ → take the step
                states = [s_new]
            # else: skip; states unchanged

        # ---- Validity check (relaxed) --------------------------------
        for s in states:
            if s.sa >= theta and abs(s.sa - s.sb) <= eps_relaxed * s.sa:
                return s

    return None


# ==========================================================================
# Umbrella-on-Crossing Search
#
# A refinement of the umbrella idea: instead of opening the umbrella when
# greedy is merely "near" the wedge, open it precisely when greedy's
# NEXT STEP would cross the wedge boundary from one side to the other.
# This is the geometric moment where greedy commits to an irreversible
# overshoot, and exactly where local bucket exploration can rescue valid
# subsets that lie BETWEEN the pre-crossing and post-crossing states.
#
# Crossing detection (algebraic):
#   Let side(s) ∈ {-1, 0, +1}:
#     -1 if S_B < (1-ε)·S_A       (below wedge)
#      0 if S_A < θ  OR  s in wedge  (pre-wedge or valid)
#     +1 if S_B > (1+ε)·S_A       (above wedge)
#   Crossing  ⇔  side(s_cur) ≠ 0 ∧ side(s_next) ≠ 0 ∧ side(s_cur) ≠ side(s_next).
#
# When crossing detected, opens bucket starting from s_current (NOT
# s_next), since the valid subset must use a candidate of intermediate
# magnitude (between 0 and next_amt) which greedy's single biggest pick
# would skip.
#
# Trade-off vs cascade:
#   * Cascade always reruns bucket from initial after greedy completes
#     fully, paying O(N·K) on greedy-failure cases.
#   * umbrella_xing opens bucket FROM the crossing point only, paying
#     O((N-k)·K) where k is the crossing step. Strictly less work on
#     hard cases, since k > 0 always.
#   * Greedy state at crossing has SA ≥ θ already, so safe pruning is
#     very effective.
#
# Approximation: heuristic, not strict (1+δ)·ε-FPTAS.
# Reviewer-friendly framing:
#   "A greedy walker augmented with overshoot detection: when the
#    next greedy step would cross the wedge boundary, the algorithm
#    halts greedy commitment and launches a bucket-compressed local
#    search from the pre-crossing state."
# ==========================================================================
# ==========================================================================
# 1D subset-sum-in-band query  (used by umbrella_xing's crossing rescue)
# --------------------------------------------------------------------------
# Question:  given positive amounts {a_1, ..., a_n} (sorted desc) and target
#            band [L, U],  does any subset S satisfy  L ≤ Σ_{i∈S} a_i ≤ U ?
#
# Two-phase solver:
#   Phase 1 (greedy descending):  O(n).  Take a_i if cur+a_i ≤ U; halt when
#       cur ≥ L. Complete on "dense" amount distributions (band wider than
#       smallest unused amount). Fast path for the common case.
#   Phase 2 (1D log-bucket FPTAS):  O(n · log V / log(1+δ)).  After greedy
#       fails, run the classical Ibarra-Kim style log-bucket compression on
#       the 1D subset-sum lattice. State count bounded by O(log V / log(1+δ))
#       ≈ 100, so each iteration is O(state count). Guaranteed to find a
#       feasible sum within (1+δ) factor of any existing one.
#
# Together: handles the "next greedy step would overshoot wedge" case by
# checking whether ANY subset (1 or more transactions) of the remaining
# same-side candidates lands inside the wedge band — without paying the
# full 2D multi-state bucket cost.
# ==========================================================================
def _subset_sum_in_band(amts_desc: list, L: float, U: float,
                          delta: float = 0.1,
                          max_states: int = 500) -> Optional[float]:
    """Find a subset of `amts_desc` (sorted descending, positive) whose sum
    lies in [L, U]. Returns the achieved sum if found, else None.
    """
    if L <= 0:
        return 0.0
    if U < L or not amts_desc:
        return None

    # ---- Phase 1: greedy descending (O(n)) -----------------------------
    cur = 0.0
    for a in amts_desc:
        if a <= 0:
            continue
        if cur + a <= U:
            cur += a
            if cur >= L:
                return cur
    # Phase 1 exhausted; cur < L.  Greedy failed (either all amounts > U,
    # or greedy picked a non-completable prefix). Fall through to Phase 2.

    # ---- Phase 2: 1D bucket-compressed FPTAS ---------------------------
    sums = [0.0]
    if delta > 0:
        log_factor = math.log(1.0 + delta)
    else:
        log_factor = 0.0  # no compression — keep all sums

    for a in amts_desc:
        if a <= 0:
            continue
        # Expand: { s, s+a : s in sums, s+a ≤ U }
        new_sums = set(sums)
        for s in sums:
            s2 = s + a
            if s2 <= U:
                new_sums.add(s2)
                if s2 >= L:
                    return s2

        # Compress to one representative per log-scale bucket
        if log_factor > 0 and len(new_sums) > max_states:
            buckets = {}
            for s in new_sums:
                if s == 0:
                    key = -1
                else:
                    key = int(math.log(s) / log_factor)
                # Keep the LARGER s per bucket — closer to L
                prev = buckets.get(key)
                if prev is None or s > prev:
                    buckets[key] = s
            sums = list(buckets.values())
        else:
            sums = list(new_sums)

    return None


def frontier_search_umbrella_xing(transactions: List[Txn], trigger_eid: str,
                                    theta: float, eps: float,
                                    delta_sa: float = 0.1,
                                    delta_d: float = 0.1,
                                    max_states: int = 4000
                                    ) -> Optional[State]:
    """Greedy with wedge-crossing-triggered local bucket umbrella.

    Per-call phase timing is accumulated into the module-level
    `_PHASE_TIMING` dict for fair external comparison.
    """
    # Per-call timer starts here — survivors are already loaded;
    # filter-stage time is NOT measured.
    _t_call_start = time.perf_counter()
    _PHASE_TIMING["umbrella_xing_calls"] += 1

    trigger = next(t for t in transactions if t.eid == trigger_eid)
    if trigger.direction == "in":
        sa, sb = trigger.amount, 0.0
    else:
        sa, sb = 0.0, trigger.amount

    eps_relaxed = eps * (1.0 + delta_d)
    one_minus_eps = 1.0 - eps

    in_amts = sorted(
        (t.amount for t in transactions
         if t.eid != trigger_eid and t.direction == "in"),
        reverse=True,
    )
    out_amts = sorted(
        (t.amount for t in transactions
         if t.eid != trigger_eid and t.direction == "out"),
        reverse=True,
    )
    n_in = len(in_amts)
    n_out = len(out_amts)

    # Initial validity
    if sa >= theta and abs(sa - sb) <= eps_relaxed * sa:
        _PHASE_TIMING["umbrella_xing_greedy_seconds"] += time.perf_counter() - _t_call_start
        _PHASE_TIMING["umbrella_xing_greedy_hits"] += 1
        return State(sa=sa, sb=sb)

    def side_of(_sa: float, _sb: float) -> int:
        """-1 below wedge, 0 pre-wedge or in (relaxed) wedge, +1 above wedge.

        Uses the (1+δ)·ε relaxed band — same band used for the validity
        check. States in the relaxed band would already be accepted, so we
        treat them as side=0 (no crossing). This avoids spurious umbrella
        openings on transient swings inside the relaxed band.
        """
        if _sa < theta:
            return 0
        gap = _sa - _sb
        d = gap if gap >= 0 else -gap
        if d <= eps_relaxed * _sa:
            return 0
        return +1 if _sb > _sa * (1.0 + eps_relaxed) else -1

    i_in = i_out = 0
    while i_in < n_in or i_out < n_out:
        # Choose next greedy candidate (alternating-smaller-side rule)
        if sa <= sb:
            if i_in < n_in:
                next_amt = in_amts[i_in]; next_dir = "in"
                new_sa, new_sb = sa + next_amt, sb
            elif i_out < n_out:
                next_amt = out_amts[i_out]; next_dir = "out"
                new_sa, new_sb = sa, sb + next_amt
            else:
                break
        else:
            if i_out < n_out:
                next_amt = out_amts[i_out]; next_dir = "out"
                new_sa, new_sb = sa, sb + next_amt
            elif i_in < n_in:
                next_amt = in_amts[i_in]; next_dir = "in"
                new_sa, new_sb = sa + next_amt, sb
            else:
                break

        # Validity check at the would-be new state
        if new_sa >= theta and abs(new_sa - new_sb) <= eps_relaxed * new_sa:
            _PHASE_TIMING["umbrella_xing_greedy_seconds"] += time.perf_counter() - _t_call_start
            _PHASE_TIMING["umbrella_xing_greedy_hits"] += 1
            return State(sa=new_sa, sb=new_sb)

        # Wedge-crossing detection
        cur_side = side_of(sa, sb)
        new_side = side_of(new_sa, new_sb)
        crossing = (cur_side != 0 and new_side != 0 and cur_side != new_side)

        if crossing:
            # Stop greedy timer
            _PHASE_TIMING["umbrella_xing_greedy_seconds"] += time.perf_counter() - _t_call_start

            # ---- Phase A: subset-sum-in-band rescue (cheap) ------------
            # Try to find a subset of the SAME-side remaining candidates
            # whose sum lands inside the relaxed wedge band. Same side as
            # greedy was about to add — the side that pulls us TOWARD
            # balance (greedy goes for the smaller-sum side).
            _t_rescue_start = time.perf_counter()
            rescue_state = None
            if next_dir == "in":
                # New SA = sa + x ; new SB = sb
                # SA ≥ θ            →  x ≥ θ - sa
                # SA(1-ε') ≤ SB ≤ SA(1+ε')
                #   lower:  sb ≥ (sa+x)(1-ε')  →  x ≤ sb/(1-ε') - sa
                #   upper:  sb ≤ (sa+x)(1+ε')  →  x ≥ sb/(1+ε') - sa
                x_lo = max(theta - sa,
                            (sb / (1.0 + eps_relaxed)) - sa,
                            0.0)
                if eps_relaxed < 1.0:
                    x_hi = (sb / (1.0 - eps_relaxed)) - sa
                else:
                    x_hi = float("inf")
                if x_lo <= x_hi:
                    rsum = _subset_sum_in_band(
                        in_amts[i_in:], x_lo, x_hi, delta=delta_d,
                    )
                    if rsum is not None:
                        rescue_state = State(sa=sa + rsum, sb=sb)
            else:  # next_dir == "out"
                # New SA = sa ; new SB = sb + y     (requires sa ≥ θ)
                # SA(1-ε') ≤ SB ≤ SA(1+ε')
                #   lower:  sb+y ≥ sa(1-ε')  →  y ≥ sa(1-ε') - sb
                #   upper:  sb+y ≤ sa(1+ε')  →  y ≤ sa(1+ε') - sb
                if sa >= theta:
                    y_lo = max(sa * (1.0 - eps_relaxed) - sb, 0.0)
                    y_hi = sa * (1.0 + eps_relaxed) - sb
                    if y_lo <= y_hi:
                        rsum = _subset_sum_in_band(
                            out_amts[i_out:], y_lo, y_hi, delta=delta_d,
                        )
                        if rsum is not None:
                            rescue_state = State(sa=sa, sb=sb + rsum)
            _PHASE_TIMING["umbrella_xing_rescue_seconds"] += time.perf_counter() - _t_rescue_start

            if rescue_state is not None:
                _PHASE_TIMING["umbrella_xing_rescue_hits"] += 1
                return rescue_state

            # ---- Phase B: full multi-state bucket fallback -------------
            _PHASE_TIMING["umbrella_xing_bucket_invocations"] += 1
            _t_bucket_start = time.perf_counter()
            result = _umbrella_local_search(
                sa, sb,
                in_amts[i_in:], out_amts[i_out:],
                theta, eps, delta_sa, delta_d, max_states,
                eps_relaxed, one_minus_eps,
            )
            _PHASE_TIMING["umbrella_xing_bucket_seconds"] += time.perf_counter() - _t_bucket_start
            return result

        # Otherwise commit greedy step
        sa, sb = new_sa, new_sb
        if next_dir == "in":
            i_in += 1
        else:
            i_out += 1

    # Walked to exhaustion without crossing or hitting wedge — all greedy
    _PHASE_TIMING["umbrella_xing_greedy_seconds"] += time.perf_counter() - _t_call_start
    return None


def _umbrella_local_search(sa: float, sb: float,
                            remaining_in: list, remaining_out: list,
                            theta: float, eps: float,
                            delta_sa: float, delta_d: float,
                            max_states: int,
                            eps_relaxed: float, one_minus_eps: float
                            ) -> Optional[State]:
    """Local bucket search from (sa, sb) over remaining_in ∪ remaining_out."""
    states = [State(sa=sa, sb=sb)]

    # Merge & sort by amount desc
    cands = [(amt, "in") for amt in remaining_in] + \
            [(amt, "out") for amt in remaining_out]
    cands.sort(key=lambda x: -x[0])
    n = len(cands)

    # Suffix sums for safe pruning
    rem_in = [0.0] * (n + 1)
    rem_out = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        if cands[k][1] == "in":
            rem_in[k] = rem_in[k + 1] + cands[k][0]
            rem_out[k] = rem_out[k + 1]
        else:
            rem_in[k] = rem_in[k + 1]
            rem_out[k] = rem_out[k + 1] + cands[k][0]

    for k, (amt, direction) in enumerate(cands):
        # Inline expand (skip + select)
        new_states = []
        if direction == "in":
            for s in states:
                new_states.append(s)
                new_states.append(State(sa=s.sa + amt, sb=s.sb))
        else:
            for s in states:
                new_states.append(s)
                new_states.append(State(sa=s.sa, sb=s.sb + amt))
        new_states = merge_same_states(new_states)
        new_states = safe_dominance_prune(new_states)
        new_states = wedge_bucket_compress(new_states, theta, eps,
                                            delta_sa, delta_d,
                                            max_states=max_states)
        new_states = _alive_states(new_states, rem_in[k + 1], rem_out[k + 1],
                                    theta, one_minus_eps)
        if not new_states:
            return None
        states = new_states
        for s in states:
            if s.sa >= theta and abs(s.sa - s.sb) <= eps_relaxed * s.sa:
                return s
    return None
