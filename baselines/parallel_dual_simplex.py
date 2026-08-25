# -*- coding: utf-8 -*-
"""Parallel Dual Simplex — strict Python port of PackageQuery-master/dual.cpp.

Faithful line-by-line port of the reference implementation released with
PVLDB'24 paper "Scaling Package Queries to a Billion Tuples via Hierarchical
Partitioning and Customized Optimization" (Mai et al.).

Source file mirrored: PackageQuery-master/include/pb/core/dual.cpp
                      (632 lines, class `Dual`)

WHY THIS SOLVER
---------------
Package query LPs have a structural asymmetry: n (variables) >> m (constraints).
For n = 100M and m = 20, standard LP solvers (CPLEX, Gurobi) don't parallelize
well because their pivot cost model assumes n ~ m.  The paper's contribution
(§2.3) is a dual simplex where the O(n) parts of each pivot are parallelized
across cores.  Reported: 4.79× speedup on 80 cores, 80% work parallelized.

PYTHON ADAPTATION
-----------------
* True thread-parallelism is not viable in Python (GIL + our tiny per-anchor
  LPs, n < 1000).  We use `numpy` vectorization for the O(n) operations
  (pivot row, ratio test, reduced-cost update) — the same operations the
  C++ code parallelizes across OpenMP threads.  This preserves the ALGORITHM
  faithfully; only the parallelism mechanism differs.
* Numerical tolerances (kE_P, kE_r, kE_ap, kE_Eq, kE_Ieq) are copied verbatim.
* We do NOT include the Gurobi fallback (dual.cpp lines 611-628) that kicks
  in on severe numerical issues — instead we return `Infeasible` so the
  caller can escalate to `scipy.optimize.linprog` if needed.

FORMULATION MIRRORED FROM DUAL.CPP
----------------------------------
Maximize c^T x s.t.
    b_l <= A x <= b_u                   (m rows)
    l   <= x   <= u                     (n rows)

Introduces slack variables:
    x_slack = b - A x
    where slack i (index i+n) is bounded by [b_l[i], b_u[i]]

Non-basic variables sit at either their lower or upper bound (bounded LP).
Basic variables are indexed by `bhead` (length m).  `Binv` is m×m dense.
Steepest edge weights: `beta` (length m).  Reduced costs: `d` (length n+m).

USAGE
-----
    solver = ParallelDualSimplex(A, b_l, b_u, c, l, u)
    solver.solve()
    if solver.status == "Found":
        x = solver.sol[:solver.n]           # primal structural vars
        objective = solver.score            # c^T x
"""

import numpy as np


# ------------------------------------------------------------------
# Numerical tolerances (dual.cpp lines 11-19, verbatim)
# ------------------------------------------------------------------
K_E_P   = 1e-6      # kE_P    — primal feasibility tolerance
K_E_R   = 1e-12     # kE_r    — relative primal feasibility tolerance
K_E_AP  = 1e-6      # kE_ap   — pivot tolerance
K_E_EQ  = 1e-6      # kE_Eq   — equality tolerance (for x == bound checks)
K_E_IEQ = 1e-6      # kE_Ieq  — inequality tolerance (for alpha_r sign checks)

# From util/unumeric.h (default eps for isEqual/isGreater/isLess when no
# custom eps is passed).  Used in dual.cpp line 254: `!isEqual(delta, 0)`.
K_NUMERIC_EPS = 1e-8  # kNumericEps


def _primal_infeasibility(x, l, u):
    """dual.cpp line 21-27: `primalInfeasibilities`.

    Returns 0 if l <= x <= u (within relative tolerance), else the signed
    distance to the violated bound.
    """
    left  = l - l * K_E_R - K_E_P
    if x < left:
        return x - l
    right = u + u * K_E_R + K_E_P
    if x > right:
        return x - u
    return 0.0


def _get_slope(u, l, bu, bl, alpha_r, i, n):
    """dual.cpp line 29-36: `getSlope`.  The "slope" of a bound flip for
    non-basic var i in the ratio test — equals (u - l) * |alpha_r(i)|."""
    if i < n:
        return (u[i] - l[i]) * abs(alpha_r[i])
    idx = i - n
    return (bu[idx] - bl[idx]) * abs(alpha_r[idx])


class ParallelDualSimplex:
    """Port of the `Dual` class in dual.cpp.

    Attributes populated after `solve()`:
        status         : "Found" | "Infeasible" | "DualUnbounded"
        sol            : np.ndarray of length n+m — primal solution
        score          : c^T x   (only meaningful if status == "Found")
        bhead          : np.ndarray of length m — basis indices
        Binv           : np.ndarray (m, m) — basis inverse
        iteration_count: number of pivot iterations
    """

    def __init__(self, A, b_l, b_u, c, l, u):
        """Maximize c^T x subject to b_l <= Ax <= b_u, l <= x <= u.

        A must have shape (m, n).  Bounds may include ±inf (handled by the
        bound-strictening pass in Phase 1).
        """
        self.A = np.asarray(A, dtype=np.float64)
        self.b_l_orig = np.asarray(b_l, dtype=np.float64).copy()
        self.b_u_orig = np.asarray(b_u, dtype=np.float64).copy()
        self.c = np.asarray(c, dtype=np.float64)
        self.l = np.asarray(l, dtype=np.float64)
        self.u = np.asarray(u, dtype=np.float64)

        self.m, self.n = self.A.shape
        assert self.c.size == self.n
        assert self.l.size == self.n
        assert self.u.size == self.n
        assert self.b_l_orig.size == self.m
        assert self.b_u_orig.size == self.m

        # ------ Reflects dual.cpp state variables ------
        self.status = "NotFound"
        self.iteration_count = 0
        self.mini_iteration_count = 0
        self.score = 0.0

        # Slack bounds (mutable copies; -inf/+inf get replaced by finite equivalents)
        self.bl = self.b_l_orig.copy()
        self.bu = self.b_u_orig.copy()

        # beta — steepest-edge norms (line 97); one entry per basic slot
        self.beta = np.ones(self.m, dtype=np.float64)

        # d — reduced cost, length n+m (line 100)
        self.d = np.zeros(self.n + self.m, dtype=np.float64)

        # bhead — basis indices; initial basis = slacks (line 103-104)
        self.bhead = np.arange(self.n, self.n + self.m, dtype=np.int64)
        # inv_bhead[i] = True iff i is currently in the basis
        self.inv_bhead = np.zeros(self.n + self.m, dtype=bool)
        self.inv_bhead[self.n:self.n + self.m] = True

        # Primal solution — length n+m (line 110)
        self.sol = np.zeros(self.n + self.m, dtype=np.float64)

        # Binv — basis inverse (line 120)
        self.Binv = np.eye(self.m, dtype=np.float64)

    # ==============================================================
    # PHASE 1  (dual.cpp lines 158-216)
    # ==============================================================
    def _phase1(self):
        """Bound-strictening and initial primal solution.

        Strategy for -inf / +inf constraint bounds (line 164-193):
        if b_l[i] = -inf, replace with sum over j: (l[j] if A[i,j] > 0
        else u[j]) * A[i,j] — a valid lower bound assuming primal vars
        stay within [l, u].  Symmetric for +inf b_u.

        Then initialize primal solution:
        * Structural x[j]: at u[j] if c[j] > 0 else l[j]  (line 200-201)
        * Slack x[i+n]: A[i,:] @ x[:n]                     (line 208-213)
        """
        # -- bound strictening (lines 172-193) --
        neg_inf_mask = np.isneginf(self.b_l_orig)
        pos_inf_mask = np.isposinf(self.b_u_orig)

        if neg_inf_mask.any():
            # For rows with b_l = -inf, compute finite equivalent
            # local_bound = sum_j (l[j] if A[i,j]>0 else u[j]) * A[i,j]
            A_pos = np.where(self.A > 0, self.l, self.u)  # shape (m,n)
            local_bounds = np.einsum("ij,ij->i", self.A, A_pos)
            self.bl[neg_inf_mask] = local_bounds[neg_inf_mask]

        if pos_inf_mask.any():
            A_neg = np.where(self.A > 0, self.u, self.l)
            local_bounds = np.einsum("ij,ij->i", self.A, A_neg)
            self.bu[pos_inf_mask] = local_bounds[pos_inf_mask]

        # -- initialize d and structural sol (lines 197-202) --
        self.d[:self.n] = -self.c
        # sol[i] = u[i] if c[i] > 0 else l[i]
        self.sol[:self.n] = np.where(self.c > 0, self.u, self.l)

        # -- initialize slack sol (lines 206-214) --
        # sol[i+n] = sum_j A[i,j] * sol[j]
        self.sol[self.n:] = self.A @ self.sol[:self.n]

    # ==============================================================
    # PHASE 2  (dual.cpp lines 218-608) — main pivot loop
    # ==============================================================
    def solve(self):
        self._phase1()

        while True:
            # -------- Step 1: Pricing (lines 222-272) --------
            # For each basic slot i, compute primal infeasibility delta
            # and select r maximizing delta² / beta[i].  This is
            # steepest-edge pricing.
            r = -1
            max_dse = -np.inf
            max_delta = 0.0
            for i in range(self.m):
                idx = self.bhead[i]
                if idx < self.n:
                    delta = _primal_infeasibility(
                        self.sol[idx], self.l[idx], self.u[idx],
                    )
                else:
                    s = idx - self.n
                    delta = _primal_infeasibility(
                        self.sol[idx], self.bl[s], self.bu[s],
                    )
                # dual.cpp line 254: `!isEqual(delta, 0)` uses default
                # kNumericEps = 1e-8.  Since _primal_infeasibility already
                # snaps to 0 for violations < K_E_P = 1e-6, this test is
                # equivalent to `abs(delta) > 0`; we keep the explicit
                # tolerance for numerical safety.
                if abs(delta) > K_NUMERIC_EPS:
                    dse = delta * delta / self.beta[i]
                    if dse > max_dse:
                        max_dse = dse
                        max_delta = delta
                        r = i

            if r == -1:
                # Optimal basis found (line 264)
                self.status = "Found"
                self.score = float(self.c @ self.sol[:self.n])
                return

            sign_max_delta = 1 if max_delta > 0 else -1
            p = int(self.bhead[r])                   # leaving var index
            # BTran: rho_r = Binv.row(r)  (line 270)
            rho_r = self.Binv[r].copy()

            # -------- Step 2: Pivot row alpha_r (lines 277-288) --------
            # For i in [0, n):  alpha_r[i] = -sum_j A[j,i] * rho_r[j] * sign_max_delta
            # For i in [n, n+m): alpha_r[i] = rho_r[i-n] * sign_max_delta
            # THIS IS THE O(n*m) STEP THAT C++ PARALLELIZES — numpy vectorizes it.
            alpha_r = np.empty(self.n + self.m, dtype=np.float64)
            alpha_r[:self.n] = -(self.A.T @ rho_r) * sign_max_delta
            alpha_r[self.n:] = rho_r * sign_max_delta

            # -------- Step 3: Ratio test — collect eligible non-basic vars (lines 291-397) --------
            # A non-basic var i is eligible if:
            #   (alpha_r[i] > 0 AND sol[i] == lower bound), OR
            #   (alpha_r[i] < 0 AND sol[i] == upper bound)
            # score[i] = d[i] / alpha_r[i]  (for the BFRT ordering)
            non_basic = np.where(~self.inv_bhead)[0]

            # Vectorized eligibility test
            elig_scores = []
            elig_indices = []
            for i in non_basic:
                a = alpha_r[i]
                if i < self.n:
                    lo_i, hi_i = self.l[i], self.u[i]
                else:
                    s = i - self.n
                    lo_i, hi_i = self.bl[s], self.bu[s]
                sol_i = self.sol[i]
                if (a > K_E_IEQ and abs(sol_i - lo_i) < K_E_EQ) or \
                   (a < -K_E_IEQ and abs(sol_i - hi_i) < K_E_EQ):
                    elig_scores.append(self.d[i] / a)
                    elig_indices.append(i)

            if not elig_indices:
                self.status = "Infeasible"
                return

            # -------- Step 4: BFRT — find q by cumulative slope (lines 399-457) --------
            # Sort by score ascending (dual step is smallest first).
            elig_scores = np.array(elig_scores, dtype=np.float64)
            elig_indices = np.array(elig_indices, dtype=np.int64)
            order = np.argsort(elig_scores)
            elig_scores = elig_scores[order]
            elig_indices = elig_indices[order]

            # Compute slopes and cumulative slope sums
            slopes = np.array([
                _get_slope(self.u, self.l, self.bu, self.bl, alpha_r, i, self.n)
                for i in elig_indices
            ], dtype=np.float64)
            slope_sums = np.cumsum(slopes)

            abs_max_delta = abs(max_delta)
            if slope_sums[-1] < abs_max_delta - K_E_AP:
                # Sum of all possible slopes < |max_delta|
                # → cannot reach feasibility from this basis
                self.status = "DualUnbounded"
                return

            # Find first index where cumulative slope >= |max_delta|
            q_index = int(np.searchsorted(slope_sums, abs_max_delta, side="right"))
            q_index = min(q_index, elig_indices.size - 1)
            q = int(elig_indices[q_index])

            if q_index > 0:
                self.mini_iteration_count += q_index - 1
                max_delta = sign_max_delta * (abs_max_delta - float(slope_sums[q_index - 1]))

            dual_step = self.d[q] / alpha_r[q] * sign_max_delta
            self.d[p] = -dual_step

            # -------- Step 5: FTran — compute alpha_q and tau (lines 479-484) --------
            if q < self.n:
                alpha_q = -(self.Binv @ self.A[:, q])
            else:
                alpha_q = self.Binv[:, q - self.n].copy()
            tau = self.Binv @ rho_r

            # -------- Step 6: Update reduced costs d (lines 493-499) --------
            # d[i] -= dual_step * alpha_r[i] * sign_max_delta  for non-basic
            update_mask = ~self.inv_bhead
            self.d[update_mask] -= dual_step * alpha_r[update_mask] * sign_max_delta

            # -------- Step 7: Bound flips for indices before q_index (lines 502-529) --------
            # Vars in elig_indices[:q_index] flip from their current bound to the other,
            # accumulating tilde_a (contribution to constraint slacks).
            tilde_a = np.zeros(self.m, dtype=np.float64)
            for i in range(q_index):
                j = int(elig_indices[i])
                if j < self.n:
                    if abs(self.sol[j] - self.l[j]) < K_E_EQ:
                        tilde_a += (self.l[j] - self.u[j]) * self.A[:, j]
                        self.sol[j] = self.u[j]
                    else:
                        tilde_a += (self.u[j] - self.l[j]) * self.A[:, j]
                        self.sol[j] = self.l[j]
                else:
                    s = j - self.n
                    if abs(self.sol[j] - self.bl[s]) < K_E_EQ:
                        tilde_a[s] += (self.bu[s] - self.bl[s])
                        self.sol[j] = self.bu[s]
                    else:
                        tilde_a[s] += (self.bl[s] - self.bu[s])
                        self.sol[j] = self.bl[s]

            # -------- Step 8: Update basic sol via tilde_a (lines 537-540) --------
            delta_xB = self.Binv @ tilde_a
            self.sol[self.bhead] -= delta_xB

            # -------- Step 9: Primal step (lines 550-552) --------
            primal_step = max_delta / alpha_q[r]
            self.sol[self.bhead] -= primal_step * alpha_q
            self.sol[q] += primal_step

            # -------- Step 10: Update DSE weights beta (lines 557-564) --------
            # For i != r: beta[i] += ratio * (ratio * beta[r] - 2 * tau[i])
            # beta[r] /= alpha_q[r]²
            alpha_qr = alpha_q[r]
            for i in range(self.m):
                if i != r:
                    ratio = alpha_q[i] / alpha_qr
                    self.beta[i] += ratio * (ratio * self.beta[r] - 2 * tau[i])
            self.beta[r] /= (alpha_qr * alpha_qr)

            # -------- Step 11: Update bhead & inv_bhead (lines 567-569) --------
            self.inv_bhead[p] = False
            self.bhead[r] = q
            self.inv_bhead[q] = True

            # -------- Step 12: Update Binv (lines 591-597) --------
            # Row-elimination update, standard revised-simplex Binv maintenance
            for i in range(self.m):
                if i != r:
                    ratio = alpha_q[i] / alpha_qr
                    self.Binv[i] -= ratio * self.Binv[r]
            self.Binv[r] /= alpha_qr

            self.iteration_count += 1

            # Safety cap — avoid infinite loops on degenerate LPs
            if self.iteration_count > 50 * (self.n + self.m):
                self.status = "Infeasible"
                return


# ==============================================================
# Convenience API compatible with scipy.optimize.linprog
# ==============================================================
def solve_max(A_ub, b_ub, c, bounds):
    """Solve max c^T x s.t. A_ub x <= b_ub, bounds[i][0] <= x[i] <= bounds[i][1].

    Returns a dict compatible enough with scipy's OptimizeResult for
    Progressive Shading.
    """
    m = A_ub.shape[0]
    n = A_ub.shape[1]
    b_l = np.full(m, -np.inf)
    b_u = np.asarray(b_ub, dtype=np.float64)
    l = np.array([bnd[0] for bnd in bounds], dtype=np.float64)
    u = np.array([bnd[1] for bnd in bounds], dtype=np.float64)

    solver = ParallelDualSimplex(A_ub, b_l, b_u, c, l, u)
    solver.solve()
    return {
        "success": solver.status == "Found",
        "status_str": solver.status,
        "x": solver.sol[:n].copy(),
        "fun": -solver.score,  # scipy convention: minimize -c^T x
        "iteration_count": solver.iteration_count,
    }
