# -*- coding: utf-8 -*-
"""Shared state objects and exact/basic frontier-search routines.

All search functions share the same `frontier_search(transactions, trigger_eid,
theta, eps, ...) -> Optional[State]` signature, so they can be plugged into the
same query pipeline interchangeably.

Original sources:
    frontier enumeration       -> frontier_search_enum
    dominance pruning          -> frontier_search_dom
    fixed bucket compression   -> frontier_search_bucket
    adaptive bucket compression -> frontier_search_adaptive
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Txn:
    eid: str
    direction: str    # "in" or "out"
    amount: float


@dataclass
class State:
    sa: float
    sb: float
    selected: Tuple[str, ...] = field(default_factory=tuple)

    def key(self):
        return (self.sa, self.sb)

    def gap(self):
        return self.sa - self.sb

    def residual_gap(self):
        return abs(self.sa - self.sb)

    def residual_ratio(self):
        if self.sa == 0:
            return float("inf")
        return self.residual_gap() / self.sa


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------
def exact_check(states: List[State], theta: float, eps: float) -> Optional[State]:
    # Inline field reads — avoids attribute-lookup cost in the tight loop.
    for s in states:
        sa = s.sa
        if sa >= theta and abs(sa - s.sb) <= eps * sa:
            return s
    return None


def merge_same_states(states: List[State]) -> List[State]:
    # set + list is ~30% faster than dict-then-.values() at our state counts
    seen = set()
    result = []
    for s in states:
        k = (s.sa, s.sb)
        if k not in seen:
            seen.add(k)
            result.append(s)
    return result


def safe_dominance_prune(states: List[State]) -> List[State]:
    """For each signed gap z = SA - SB, keep the state with largest SA."""
    best_by_gap = {}
    for s in states:
        z = s.sa - s.sb              # inlined s.gap()
        prev = best_by_gap.get(z)
        if prev is None or s.sa > prev.sa:
            best_by_gap[z] = s
    return list(best_by_gap.values())


def log_bucket(value: float, delta: float) -> int:
    if value <= 0:
        return 0
    return int(math.floor(math.log(value) / math.log(1 + delta)))


def residual_bucket_compress(states: List[State],
                              delta_sa: float = 0.1,
                              delta_d: float = 0.1) -> List[State]:
    buckets = {}
    for s in states:
        sa, sb = s.sa, s.sb
        d = abs(sa - sb)                                  # inlined residual_gap
        r = (d / sa) if sa else float("inf")              # inlined residual_ratio
        key = (log_bucket(sa, delta_sa), log_bucket(d, delta_d))
        old = buckets.get(key)
        if old is None:
            buckets[key] = (s, r)
        else:
            old_state, old_r = old
            if r < old_r or (r == old_r and sa > old_state.sa):
                buckets[key] = (s, r)
    return [v[0] for v in buckets.values()]


def adaptive_residual_bucket_compress(states: List[State],
                                       eps: float,
                                       delta_sa: float = 0.1,
                                       delta_near: float = 0.02,
                                       delta_far: float = 0.5,
                                       tau: float = 1.5) -> List[State]:
    buckets = {}
    near_thresh = tau * eps
    for s in states:
        sa, sb = s.sa, s.sb
        d = abs(sa - sb)
        r = (d / sa) if sa else float("inf")
        if r <= near_thresh:
            delta_d = delta_near
            region = 0          # near (small int instead of string)
        else:
            delta_d = delta_far
            region = 1          # far
        key = (log_bucket(sa, delta_sa), region, log_bucket(d, delta_d))
        old = buckets.get(key)
        if old is None:
            buckets[key] = (s, r)
        else:
            old_state, old_r = old
            if r < old_r or (r == old_r and sa > old_state.sa):
                buckets[key] = (s, r)
    return [v[0] for v in buckets.values()]


# -----------------------------------------------------------------------------
# Four frontier_search variants (anchor / trigger transaction is always kept)
# -----------------------------------------------------------------------------
def _initial_states(transactions: List[Txn], trigger_eid: str
                    ) -> Tuple[List[State], List[Txn]]:
    """Build the initial state and the candidate order.

    Two perf-critical choices:
      * We do NOT populate `selected`. Downstream code only
        checks whether `frontier_search_*` returns None or a state — it never
        uses `.selected`. Dropping the tuple-concat in `_expand` saves a large
        constant factor on enum/dom.
      * Candidates are returned in DESCENDING amount order so that the largest
        ones expand first; this makes SA / SB hit the (theta, eps) feasible
        region earlier and lets `exact_check` short-circuit sooner.
    """
    trigger = next(t for t in transactions if t.eid == trigger_eid)
    if trigger.direction == "in":
        states = [State(sa=trigger.amount, sb=0.0)]
    else:
        states = [State(sa=0.0, sb=trigger.amount)]
    candidates = [t for t in transactions if t.eid != trigger_eid]
    candidates.sort(key=lambda t: t.amount, reverse=True)
    return states, candidates


def _expand(states: List[State], z: Txn) -> List[State]:
    """Branch each state into {skip z, select z}. We do not propagate
    `selected` since query-mode callers never inspect it."""
    out = []
    amt = z.amount
    if z.direction == "in":
        for s in states:
            out.append(s)
            out.append(State(sa=s.sa + amt, sb=s.sb))
    else:
        for s in states:
            out.append(s)
            out.append(State(sa=s.sa, sb=s.sb + amt))
    return out


def frontier_search_enum(transactions: List[Txn], trigger_eid: str,
                          theta: float, eps: float) -> Optional[State]:
    states, candidates = _initial_states(transactions, trigger_eid)
    found = exact_check(states, theta, eps)
    if found is not None:
        return found
    for z in candidates:
        states = _expand(states, z)
        states = merge_same_states(states)
        found = exact_check(states, theta, eps)
        if found is not None:
            return found
    return None


def frontier_search_dom(transactions: List[Txn], trigger_eid: str,
                         theta: float, eps: float) -> Optional[State]:
    states, candidates = _initial_states(transactions, trigger_eid)
    found = exact_check(states, theta, eps)
    if found is not None:
        return found
    for z in candidates:
        states = _expand(states, z)
        states = merge_same_states(states)
        states = safe_dominance_prune(states)
        found = exact_check(states, theta, eps)
        if found is not None:
            return found
    return None


def frontier_search_bucket(transactions: List[Txn], trigger_eid: str,
                            theta: float, eps: float,
                            delta_sa: float = 0.1,
                            delta_d: float = 0.1) -> Optional[State]:
    states, candidates = _initial_states(transactions, trigger_eid)
    found = exact_check(states, theta, eps)
    if found is not None:
        return found
    for z in candidates:
        states = _expand(states, z)
        states = merge_same_states(states)
        states = safe_dominance_prune(states)
        states = residual_bucket_compress(states, delta_sa=delta_sa, delta_d=delta_d)
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
    states, candidates = _initial_states(transactions, trigger_eid)
    found = exact_check(states, theta, eps)
    if found is not None:
        return found
    for z in candidates:
        states = _expand(states, z)
        states = merge_same_states(states)
        states = safe_dominance_prune(states)
        states = adaptive_residual_bucket_compress(
            states, eps=eps,
            delta_sa=delta_sa, delta_near=delta_near,
            delta_far=delta_far, tau=tau,
        )
        found = exact_check(states, theta, eps)
        if found is not None:
            return found
    return None


# Dispatch table for the basic search variants.
TECHNIQUE_REGISTRY = {
    "enum":     frontier_search_enum,
    "dom":      frontier_search_dom,
    "bucket":   frontier_search_bucket,
    "adaptive": frontier_search_adaptive,
}
