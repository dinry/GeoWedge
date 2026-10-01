# -*- coding: utf-8 -*-
"""ILP: solve the package-existence query as an integer linear program.

Adapted from Brucato et al., "Scalable Package Queries in Relational Database
Systems" (PVLDB 9(7), 2016), §3.2, which translates a PaQL package query into
an Integer Linear Program and hands it to a black-box solver (CPLEX). We keep
the same idea and adapt it to the streaming pass-through flow setting:

  * The current window's raw in-list and out-list play the role of the
    base relation;
  * At every (arriving tuple, anchor endpoint) pair, we build one ILP asking
    whether some package pair satisfies the wedge constraints;
  * The arriving tuple is treated as a fixed contribution on its anchor side
    (matching the other baselines' convention).

This is the exact evaluation of Section 2.3 and serves as the accuracy
reference for all approximate methods.

ILP formulation:

    x_i ∈ {0, 1}    for each in-side candidate i with amount a_i
    y_j ∈ {0, 1}    for each out-side candidate j with amount b_j

    S_A = trigger_amt · 1{anchor=in}  + Σ_i a_i x_i
    S_B = trigger_amt · 1{anchor=out} + Σ_j b_j y_j

    S_A                      >= θ        (threshold)
    S_B − (1 − ε) S_A        >= 0        (wedge lower)
    S_B − (1 + ε) S_A        <= 0        (wedge upper)

    Objective: feasibility (constant 0).

Solver: scipy.optimize.milp (in-process HiGHS, requires scipy >= 1.9).
We picked it over PuLP+CBC because CBC is invoked as a subprocess that costs
100-500ms per call, dominating runtime on streams with hundreds of thousands
of query instances. HiGHS through scipy runs in-process and typically
returns in single-digit milliseconds for our window sizes (≤ a few hundred
binaries).

IMPORTANT — presolve disabled. We pass `presolve=False` to milp. HiGHS 1.8
(bundled with current scipy) has a presolve bug on instances with very large
coefficients (>10^7) that incorrectly declares the problem infeasible before
branch-and-bound. See the comment in `_solve_wedge_feasibility` below.

Pre-filter: two O(1) checks rule out obviously infeasible cases before the
ILP is constructed, cutting effective ILP calls by ~80-95% on LI-Small.

Failure mode:        ILP solve time grows with window size; instances that
                     hit the time budget are returned as 0 (negative answer).
Dependency:          scipy >= 1.9 (in-process HiGHS).  No CPLEX, no PostgreSQL.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import cli_main

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


NAME = "ilp"

# Per-instance time budget for the ILP solver. Generous enough that easy
# instances always finish; tight enough that one pathological window cannot
# stall the whole stream.
TIME_LIMIT_SECONDS = 0.5


def _solve_wedge_feasibility(in_amts, out_amts, in_init, out_init,
                              theta, eps, time_limit):
    """Solve the binary wedge-feasibility ILP. Returns 1 if feasible, else 0."""
    n_in, n_out = len(in_amts), len(out_amts)
    n = n_in + n_out

    a = np.concatenate([in_amts, np.zeros(n_out)])
    b = np.concatenate([np.zeros(n_in), out_amts])

    # Constraint matrix A_ub @ z <= b_ub, where z = [x_0..x_{n_in-1}, y_0..y_{n_out-1}]
    # (1)  S_A >= theta       ⇔  -aᵀz <= in_init - theta
    # (2)  (1-eps) SA - SB <= 0  ⇔  (1-eps) aᵀz - bᵀz <= out_init - (1-eps) in_init
    # (3)  SB - (1+eps) SA <= 0  ⇔  bᵀz - (1+eps) aᵀz <= (1+eps) in_init - out_init
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
    integrality = np.ones(n)             # all variables integer
    bounds      = Bounds(lb=0, ub=1)     # binary
    c           = np.zeros(n)            # pure feasibility

    res = milp(
        c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        # NOTE: `presolve=False` is REQUIRED. HiGHS 1.8 (bundled with scipy)
        # has a presolve bug on instances with very large coefficients
        # (>10^7): its presolve sometimes incorrectly declares the problem
        # infeasible before branch-and-bound runs, even when brute-force
        # enumeration finds strict-ε feasible witnesses. Disabling presolve
        # fixes it — MILP still runs in <ms on our problem sizes so the perf
        # cost is negligible.
        options={
            "time_limit": time_limit,
            "disp":       False,
            "presolve":   False,
        },
    )
    return 1 if res.success else 0


def query(raw_in, raw_out, trigger_amt, anchor_type, theta, eps,
          time_limit=None, **kwargs):
    """Answer the package-existence query via exact ILP.

    Parameters
    ----------
    time_limit : float, optional
        Per-call solver time budget in seconds. Defaults to module-level
        ``TIME_LIMIT_SECONDS`` (0.5s), the streaming setting. Pass a larger
        value (e.g., 60.0) for correctness checks where you want to
        eliminate timeout as a variable.
    """
    in_amts  = np.fromiter((amt for _, amt in raw_in),  dtype=np.float64)
    out_amts = np.fromiter((amt for _, amt in raw_out), dtype=np.float64)

    # Trigger contributes on its anchor side; the other side starts at 0.
    in_init  = trigger_amt if anchor_type == "in"  else 0.0
    out_init = trigger_amt if anchor_type == "out" else 0.0

    # ---- O(1) trivial-infeasibility pre-filter ----
    # (a) Threshold: max attainable SA must reach theta.
    max_SA = in_init + (in_amts.sum() if in_amts.size else 0.0)
    if max_SA < theta:
        return 0
    # (b) Wedge lower bound at theta: SB ≥ (1-eps) SA ≥ (1-eps) theta.
    max_SB = out_init + (out_amts.sum() if out_amts.size else 0.0)
    if max_SB < (1.0 - eps) * theta:
        return 0
    # (c) Wedge upper bound at minimum SA (= in_init when SA ≥ theta is binding):
    #     SB ≤ (1+eps) SA. We need some attainable SB ≤ (1+eps) max_SA,
    #     and trivially SB ≥ 0 satisfies it, so this side is always OK.

    effective_time_limit = time_limit if time_limit is not None else TIME_LIMIT_SECONDS
    return _solve_wedge_feasibility(
        in_amts, out_amts, in_init, out_init,
        theta, eps, time_limit=effective_time_limit,
    )


if __name__ == "__main__":
    cli_main(query, NAME)
