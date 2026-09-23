"""Sparse crossover engine: Phase I crash + sparse Devex Phase II simplex.

Promoted from the validated PILOT4/PILOT87 experimental pipeline
(``experiment/crossover/sparse_phase1.py`` and ``p87_phase2_v2.py``) with no
algorithmic change: the production module is the unified fallback for stalled
interior-point solves on large degenerate models.

Engine summary
--------------
* ``sparse_phase1(A, b)`` -- textbook two-phase crash: rows are normalized so
  ``b >= 0``, the augmented LP ``min sum(artificial)`` is started from the
  all-artificial basis ``B = I`` (guaranteed feasible, nonsingular), and on
  termination with ``sum(art) ~ 0`` the remaining artificial variables are
  pivoted out with a degenerate ratio-test step.  The returned basis indexes
  ORIGINAL columns and is feasible for ``A x = b, x >= 0``.  No RRQR, dense
  factorization, or external basis is required.
* ``sparse_phase2(A, b, c, basis)`` -- revised simplex with per-pivot sparse
  LU refactorization, iterative refinement, Devex pricing, Bland tie-breaking
  on the leaving variable, and condition-number-aware feasibility handling:
  basic variables infeasible only within ``kappa(B)*eps*||b||*SAFETY`` are
  soft-clamped to 0 (no objective loss), and genuinely infeasible bases are
  repaired with a Phase-I restart capped by ``REPAIR_LIMIT``.
* ``crossover_from_ipm(A, b, c)`` -- orchestrator that runs Phase I then
  Phase II end-to-end and returns a unified result dictionary.

Only ``numpy``/``scipy`` are used throughout (no dense RRQR, no external
basis, no dense production simplex path).

Fidelity note: every ``TO_*`` constant and loop decision is byte-for-byte the
logic validated on PILOT4 (m=657) and PILOT87 (m=3608) -- see
``artifacts/pilot4/p4_sparse_certificate.txt`` and
``artifacts/pilot87/p87_strict_certificate.txt``.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import splu

__all__ = [
    "CROSSOVER_MERIT_RATIO",
    "sparse_phase1",
    "sparse_phase2",
    "crossover_from_ipm",
]

# ---------------------------------------------------------------------------
# Tolerances and limits (identical to the validated experimental engine)
# ---------------------------------------------------------------------------
PIV_TOL = 1e-9          # minimum pivot ratio
TOL = 1e-7              # feasibility / reduced-cost tolerance
REFINE_TOL = 1e-9       # iterative-refinement tighten threshold
N_REFINE = 5            # refinement rounds

MAX_ITER = 25000        # Phase II pivot budget
MAX_DEGENERATE = 50     # Bland entering after this many consecutive deg. pivots
LOG_EVERY = 200
REPAIR_LIMIT = 30       # cap on full Phase-I repairs
SOFT_CLAMP_LIMIT = 100  # cap on consecutive soft clamps

SAFETY_FACTOR = 50.0    # kappa*eps*||b||*SAFETY is the effective tolerance
RECOMPUTE_EFF_TOL = 200

PHASE1_MAX_ITER = 2_000_000
PHASE1_BLAND_AFTER_STALL = 500

# Gate ratio used by the IPM crossover fallback: a crossover candidate is only
# accepted when its certified merit is within this factor of the best merit
# observed by the interior-point trajectory.
CROSSOVER_MERIT_RATIO = 1e4


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _clean_zero(v, tol=1e-9):
    x = np.asarray(v, dtype=np.float64).copy()
    x[np.abs(x) < tol] = 0.0
    return x


def _inf_norm(v):
    return float(np.max(np.abs(v))) if v.size else 0.0


def _recover_x(x_basic, basis, n):
    x = np.zeros(n, dtype=np.float64)
    x[basis] = x_basic
    return x


def _cond2_estimate(B):
    """Dense 2-norm condition estimate of a sparse basis (np.linalg.cond)."""
    try:
        return float(np.linalg.cond(B.toarray()))
    except Exception:
        return np.nan


def _try_factorize(B):
    """Try multiple factorization strategies with progressive regularization.

    Returns an LU object exposing ``solve(rhs, trans=...)`` or None.  Strategy
    order mirrors the validated experimental engine exactly: SuperLU default,
    SuperLU MMD_AT_PLUS_A, then dense LU with diagonal regularization.
    """
    try:
        return splu(B)
    except RuntimeError:
        pass
    try:
        return splu(B, permc_spec="MMD_AT_PLUS_A")
    except RuntimeError:
        pass
    Bd = B.toarray()
    for delta in [0.0, 1e-12, 1e-10, 1e-8]:
        try:
            Bp = Bd + delta * np.eye(Bd.shape[0]) if delta > 0 else Bd
            lu_d = sla.lu_factor(Bp)
            u_diag = np.diag(lu_d[0])
            if np.any(np.abs(u_diag) < 1e-14):
                continue

            class DenseLU:
                def __init__(s, lu, piv):
                    s.lu = lu
                    s.piv = piv

                def solve(s, b, trans=0):
                    return sla.lu_solve(
                        (s.lu, s.piv), b, trans=2 if trans == "T" else 0)

            return DenseLU(lu_d[0], lu_d[1])
        except Exception:
            continue
    return None


def _refine_solve(B, lu, rhs, n_refine=N_REFINE, tighten=REFINE_TOL):
    """Solve B x = rhs with iterative residual refinement."""
    x = lu.solve(rhs)
    for _ in range(n_refine):
        resid = B @ x - rhs
        rnorm = float(np.max(np.abs(resid)))
        if rnorm < tighten:
            break
        cand = None
        try:
            dc = lu.solve(resid.toarray().ravel() if hasattr(resid, "toarray")
                          else np.asarray(resid))
            cand = x + dc
        except Exception:
            break
        rc = B @ cand - rhs
        rn = float(np.max(np.abs(rc)))
        if rn < rnorm:
            x = cand
        else:
            break
    return x


def _basis_hash(basis):
    """Deterministic MD5 fingerprint of a basis (array of column indices)."""
    return int(hashlib.md5(np.array(basis, dtype=np.int32)).hexdigest()[:8], 16)


def _compute_effective_tol(B, b_norm):
    """Condition-number-aware effective feasibility tolerance.

    The backward error bound for ``B x = b`` is
        ||delta x|| <= kappa(B) * eps * ||b||.
    Any basic variable closer to zero than this bound is indistinguishable from
    numerical noise and must NOT trigger a costly Phase-I repair.
    """
    kappa = _cond2_estimate(B)
    if not np.isfinite(kappa) or kappa <= 0:
        return TOL * 1000.0, np.nan
    eff = kappa * np.finfo(np.float64).eps * b_norm * SAFETY_FACTOR
    eff = max(eff, TOL)
    return eff, kappa
# ---------------------------------------------------------------------------
# Devex (steepest-edge-like) pricing
# ---------------------------------------------------------------------------
def _devex_init_weights(A, nonbasic, lu, n_init=150):
    """Initialize Devex weights for nonbasic columns.

    A Devex weight approximates a normalized direction norm: a larger weight
    penalises columns whose entering direction ``d = B^{-1} a_j`` has large
    norm (i.e. shallow effective descent per unit step).  We initialize from
    the current feasible basis by computing ``||B^{-1} a_j||`` for the first
    columns of the nonbasic set, normalize by the median, and floor at 1.0.

    Weights are returned as a dict {column_index: weight}.  Columns that
    cannot be solved (or are not yet in the dict) default to weight 1.0 at
    selection time, i.e. standard Dantzig behaviour.  This keeps the
    initialization safe when the factorization is unstable.
    """
    n_init = min(n_init, len(nonbasic))
    weights = {}
    for jj in range(n_init):
        col = int(nonbasic[jj])
        a_col = A[:, col]
        rhs = a_col.toarray().ravel() if hasattr(a_col, "toarray") else a_col
        try:
            dj = lu.solve(rhs)
        except Exception:
            continue
        weights[col] = max(1.0, float(np.linalg.norm(dj)))
    if weights:
        med = float(np.median(list(weights.values())))
        if med > 0:
            for k in weights:
                weights[k] = max(1.0, weights[k] / med)
    return weights


def _devex_update_weights(weights, exiting_col, d_norm):
    """Update Devex weight for a column leaving the basis.

    ``_devex_init_weights`` approximates edge lengths once.  As pivots happen
    every entering column becomes a new edge and should get a weight too:
    exiting column's direction length ``||d||`` is its newly-created edge
    length, optionally normalized by the current median.  Columns with no
    recorded weight are simply dropped (default threshold 1.0 / Dantzig).
    """
    if not weights:
        return
    w = max(1.0, float(d_norm))
    med = float(np.median(list(weights.values()))) if weights else 1.0
    if med > 0:
        w = max(1.0, w / med)
    weights[int(exiting_col)] = w


def _devex_select(reduced, nonbasic, weights):
    """Devex entering-variable selection (returns index into ``nonbasic``).

    Selects the column minimizing ``reduced_cost / weight`` over the subset of
    nonbasic columns with negative reduced cost.  Columns without a recorded
    weight fall back to weight 1.0 (pure Dantzig), which is always safe.
    """
    m_ = reduced.size
    neg = reduced < -TOL
    if not np.any(neg):
        return int(np.argmin(reduced))
    scored = np.full(m_, np.inf)
    sub = np.flatnonzero(neg)
    for idx in sub:
        val = weights.get(int(nonbasic[idx]))
        w = val if (val is not None and val > 0) else 1.0
        scored[idx] = reduced[idx] / w
    return int(np.argmin(scored))
# ---------------------------------------------------------------------------
# Phase I (textbook two-phase crash, all-artificial B=I start)
# ---------------------------------------------------------------------------
def sparse_phase1(A, b, max_iter=PHASE1_MAX_ITER, verbose=2000,
                  tol=TOL, piv_tol=PIV_TOL,
                  bland_after_stall=PHASE1_BLAND_AFTER_STALL,
                  time_limit=None):
    """Compute a feasible basis for ``A x = b, x >= 0``.

    Returns ``(basis, iter, status, info)`` where ``basis`` indexes ORIGINAL
    columns (never artificials).

    ``time_limit`` (optional, seconds) bounds the wall-clock budget.  When the
    budget is exhausted the Phase I loop stops early and reports
    ``status == "time_limit"``.  ``None`` (the default) keeps the original
    iteration-only behaviour.

    Crash strategy: all-artificial ``B = I`` after row normalization so
    ``b >= 0``.  This is guaranteed feasible and nonsingular.  No external
    basis (RRQR or otherwise) is used or needed.

    Termination (textbook two-phase): Phase I is at optimum when NO nonbasic
    column has negative reduced cost.  At that point if the artificial sum is
    ~0 the original LP is feasible and each remaining basic artificial (value
    ~0) is pivoted OUT with a proper ratio-test step (theta = xB[r]/alpha[r],
    leaving variable = r).  Only then is the returned basis certified feasible
    for the original problem.
    """
    if not sp.issparse(A):
        A = sp.csc_matrix(A)
    A = A.tocsc()
    b = np.asarray(b, float)
    m, n = A.shape
    tol_b = tol

    # ---- normalize rows so b >= 0 for a feasible all-artificial start ----
    flip = b < 0.0
    if flip.any():
        D = sp.diags(np.where(flip, -1.0, 1.0))
        A = (D @ A).tocsc()
        b = np.abs(b)

    # ---- augmented problem: orig cols 0..n-1, artificials n..n+m-1 ----
    nart = m
    A_aug = sp.hstack([A, sp.identity(m, format="csc")], format="csc")

    # cost: 0 on original, +1 on artificials (minimize sum of art)
    c_aug = np.zeros(n + nart, dtype=np.float64)
    c_aug[n:] = 1.0

    # ---- starting basis: ALL artificials (I) -> x_B = b >= 0 feasible ----
    basis = list(range(n, n + m))

    nb_set = set(range(n + nart)) - set(basis)
    art_set = set(range(n, n + nart))

    t_lu = t_solve = t_d = t_alpha = 0.0
    stall = 0
    last_art_sum = None
    # Bound before the loop so a zero pivot budget reports honestly instead of
    # raising UnboundLocalError on the final iteration-limit return.
    art_sum = None
    _t_budget = time.perf_counter()

    for it in range(max_iter):
        if time_limit is not None and (time.perf_counter() - _t_budget) >= time_limit:
            return basis, it, "time_limit", {
                "err": "time limit", "nart": nart, "art_sum": last_art_sum,
                "t_lu": t_lu, "t_solve": t_solve, "t_d": t_d, "t_alpha": t_alpha,
            }
        t0 = time.perf_counter()
        try:
            B = A_aug[:, basis].tocsc()
            lu = splu(B)
        except Exception as e:
            return basis, it, "numerical_failure", {
                "err": f"splu: {e}", "nart": nart, "art_sum": None,
                "t_lu": t_lu, "t_solve": t_solve, "t_d": t_d, "t_alpha": t_alpha,
            }
        t1 = time.perf_counter()
        xb = lu.solve(b)
        t2 = time.perf_counter()
        t_lu += t1 - t0
        t_solve += t2 - t1

        xb = np.asarray(xb, float)
        art_vals = np.array([xb[i] for i in range(m) if basis[i] in art_set],
                            dtype=float)
        art_sum = float(art_vals.sum()) if art_vals.size else 0.0
        if verbose and it % verbose == 0:
            log(f"  it {it}: art_sum={art_sum:,.4f} n_art={int(art_vals.size)}")

        # termination: all artificials out of the basis => xB >= 0 by the
        # ratio-test invariant, so the original basis is feasible.
        if art_vals.size == 0:
            Bf = A[:, basis].tocsc()
            try:
                xbf = splu(Bf).solve(b)
            except Exception:
                return basis, it, "numerical_failure", {
                    "nart": nart, "err": "feas-verify splu"}
            if np.all(xbf >= -tol_b):
                return basis, it, "feasible", {
                    "nart": nart, "art_sum": 0.0, "iter": it,
                    "t_lu": t_lu, "t_solve": t_solve, "t_d": t_d, "t_alpha": t_alpha,
                    "min_xB": float(xbf.min()), "neg": int((xbf < -tol_b).sum()),
                }
            # else keep pivoting (numerical corner; rare)

        # reduced costs: c_B, y = B^{-T} c_B, d = c_{nb} - A_nb^T y
        cB = np.array([c_aug[bb] for bb in basis], dtype=float)
        t0 = time.perf_counter()
        y = lu.solve(cB, trans="T")
        nb_list = sorted(nb_set)
        Anb = A_aug[:, nb_list]
        d = np.asarray((Anb.T @ y)).ravel()
        d = c_aug[nb_list] - d
        t1 = time.perf_counter()
        t_d += t1 - t0

        cand = np.where(d < -1e-9)[0]
        if cand.size == 0:
            # true Phase-I optimum reached.  If art_sum ~ 0 the LP is
            # feasible; pivot the last basic arts out at value ~0 with a
            # proper degeneracy step, then certify the original basis.
            if art_sum > max(1.0, float(np.sum(np.abs(b)))) * 1e-6:
                return basis, it, "infeasible", {
                    "nart": nart, "art_sum": art_sum, "iter": it,
                    "t_lu": t_lu, "t_solve": t_solve, "t_d": t_d, "t_alpha": t_alpha,
                }
            for clean in range(nart):
                art_pos = [i for i in range(m) if basis[i] in art_set]
                if not art_pos:
                    break
                r = art_pos[0]
                # leave art r via entering original column with alpha[r]>0
                # (theta = xB[r]/alpha[r] ~ 0 keeps feasibility)
                orig_cand = []
                for q in nb_set:
                    if q >= n:
                        continue
                    aq = A[:, q].toarray().ravel()
                    alpha_q = lu.solve(aq)
                    a_max = float(np.max(np.abs(alpha_q)))
                    if a_max > piv_tol and alpha_q[r] > piv_tol * max(1.0, a_max):
                        orig_cand.append((alpha_q, q))
                if not orig_cand:
                    # no alpha[r]>0; any alpha[r]!=0 also works for a ~0 leave
                    for q in nb_set:
                        if q >= n:
                            continue
                        aq = A[:, q].toarray().ravel()
                        alpha_q = lu.solve(aq)
                        a_max = float(np.max(np.abs(alpha_q)))
                        if a_max > piv_tol and abs(alpha_q[r]) > piv_tol * max(1.0, a_max):
                            orig_cand.append((alpha_q, q))
                if not orig_cand:
                    return basis, it, "cleanout_failed", {
                        "err": "no original col for art leave", "nart": nart,
                        "art_sum": art_sum, "t_lu": t_lu, "t_solve": t_solve,
                        "t_d": t_d, "t_alpha": t_alpha,
                    }
                # pick the candidate with best pivot scale; force leave at the
                # artificial row r (theta = xB[r]/alpha_q[r] ~ 0), which is
                # the only leave index that removes the artificial from basis.
                best = max(orig_cand, key=lambda pair: pair[0][r])
                alpha_q, q = best
                theta = abs(xb[r] / alpha_q[r])
                leave = r
                old = basis[leave]
                basis[leave] = q
                nb_set.remove(q)
                nb_set.add(old)
                xb = xb - theta * alpha_q
                xb[leave] = theta * alpha_q[leave]
                try:
                    B = A_aug[:, basis].tocsc()
                    lu = splu(B)
                except Exception:
                    return basis, it, "numerical_failure", {
                        "err": "cleanout splu", "nart": nart, "art_sum": art_sum,
                    }
            # certify final all-original basis
            Bf = A[:, basis].tocsc()
            try:
                xbf = splu(Bf).solve(b)
            except Exception:
                return basis, it, "numerical_failure", {"err": "cand-empty verify splu"}
            if np.all(xbf >= -tol_b):
                return basis, it, "feasible", {
                    "nart": nart, "art_sum": 0.0, "iter": it,
                    "t_lu": t_lu, "t_solve": t_solve, "t_d": t_d, "t_alpha": t_alpha,
                    "min_xB": float(xbf.min()), "neg": int((xbf < -tol_b).sum()),
                }
            return basis, it, "near_feasible", {
                "nart": nart, "art_sum": art_sum, "iter": it,
                "min_xB": float(xbf.min()), "neg": int((xbf < -tol_b).sum()),
                "t_lu": t_lu, "t_solve": t_solve, "t_d": t_d, "t_alpha": t_alpha,
            }

        # anti-cycling: Bland enter (smallest index) after stall counter
        if stall >= bland_after_stall:
            order = cand[np.argsort([nb_list[int(k)] for k in cand], kind="stable")]
            stall = 0
            log(f"  it {it}: BLAND anti-cycling entry")
        else:
            order = cand[np.argsort(d[cand], kind="stable")]  # Dantzig most neg
        picked = False
        for k in order:
            q = nb_list[int(k)]
            aq = A_aug[:, q]
            t0 = time.perf_counter()
            alpha = lu.solve(aq.toarray().ravel())
            t1 = time.perf_counter()
            t_alpha += t1 - t0
            amax = float(np.max(np.abs(alpha))) if alpha.size else 0.0
            if amax < piv_tol:
                continue
            mask = alpha > piv_tol * max(1.0, amax)
            if not mask.any():
                continue
            ratios = np.full(m, np.inf)
            ratios[mask] = xb[mask] / alpha[mask]
            r = int(np.argmin(ratios))
            if alpha[r] <= piv_tol * max(1.0, amax):
                continue
            old = basis[r]
            basis[r] = q
            nb_set.remove(q)
            nb_set.add(old)
            picked = True
            # stall tracking
            if last_art_sum is not None and abs(art_sum - last_art_sum) < 1e-12:
                stall += 1
            else:
                stall = 0
            last_art_sum = art_sum
            break
        if not picked:
            return basis, it, "stalled", {
                "err": "no pivot accepted", "nart": nart, "art_sum": art_sum,
                "t_lu": t_lu, "t_solve": t_solve, "t_d": t_d, "t_alpha": t_alpha,
            }

    return basis, max_iter, "max_iterations", {
        "err": "iter limit", "nart": nart, "art_sum": art_sum,
        "t_lu": t_lu, "t_solve": t_solve, "t_d": t_d, "t_alpha": t_alpha,
    }
# ---------------------------------------------------------------------------
# Phase II (revised simplex, Devex pricing, condition-aware repair)
# ---------------------------------------------------------------------------
def sparse_phase2(A, b, c, basis, *, max_iter=MAX_ITER,
                  pricing="devex", verbose=0, time_limit=None):
    """Run sparse revised simplex Phase II from a feasible basis.

    Returns a result dict with keys: status, objective, iterations,
    x_basic, basis, nonbasic, y, reduced, rel_primal, rel_dual,
    rel_gap, primal_residual, repairs, soft_clamps, total_repair_pivots,
    solve_time.

    ``time_limit`` (optional, seconds) bounds the wall-clock budget; when it
    is exhausted the loop stops early and reports ``status == "time_limit"``.
    A shared budget is also passed into any internal full-repair Phase I.
    ``None`` (the default) keeps the original iteration-only behaviour.
    """
    if not sp.issparse(A):
        A = sp.csc_matrix(A)
    A = A.tocsc()
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    basis = list(basis)
    m, n = A.shape
    b_norm = float(np.max(np.abs(b)))
    t0 = time.perf_counter()

    B = A[:, basis].tocsc()
    lu = _try_factorize(B)
    if lu is None:
        return _phase2_fail("numerical_failure", "initial factorization failed",
                            A, b, c, basis, m, n, t0)
    x_basic = _refine_solve(B, lu, b)
    objective = float(c[basis] @ x_basic)
    basis_set = set(basis)
    nonbasic = np.array([j for j in range(n) if j not in basis_set], dtype=np.intp)
    eff_tol, kappa = _compute_effective_tol(B, b_norm)
    devex_weights = {}
    if pricing == "devex":
        devex_weights = _devex_init_weights(A, nonbasic, lu)
    repairs = 0; soft_clamps = 0; consecutive_soft = 0
    total_repair_pivots = 0; prev_obj = objective
    degenerate_run = 0; iters_since_eff_tol = 0
    seen_bases = {}; devex_count = 0; devex_degen = 0
    # Sentinel for the time-limit exit below (a fresh Phase-II loop has not
    # yet computed a dual); ``_phase2_ok`` substitutes zeros for None.
    y = None
    reduced = None
    _t_budget = time.perf_counter()
    for iteration in range(max_iter):
        if time_limit is not None and (time.perf_counter() - _t_budget) >= time_limit:
            log(f"time limit {time_limit:.3g}s reached at it={iteration}")
            # Report the dual/reduced costs actually implied by the current
            # basis (never fabricate them); fall back to None only if the
            # factorisation cannot supply them.
            if y is None:
                try:
                    y = lu.solve(np.asarray(c[basis], float), trans="T")
                    reduced = _clean_zero(
                        c[nonbasic] - A[:, nonbasic].T @ y, TOL)
                except Exception:
                    y = None
                    reduced = None
            return _phase2_ok(
                A, b, c, basis, x_basic, y, reduced, iteration,
                repairs, soft_clamps, total_repair_pivots, t0,
                status="time_limit",
                message=f"time limit {time_limit:.3g}s reached at iteration {iteration}")
        iters_since_eff_tol += 1

        # ---- FEASIBILITY CHECK (condition-number-aware) ----
        neg_count = int((x_basic < -TOL).sum())
        if neg_count > 0:
            if iters_since_eff_tol >= RECOMPUTE_EFF_TOL:
                eff_tol, kappa = _compute_effective_tol(
                    A[:, basis].tocsc(), b_norm)
                iters_since_eff_tol = 0
            min_xB = float(x_basic.min())
            if min_xB >= -eff_tol:
                # soft repair: infeasibility within numerical noise
                x_basic[x_basic < 0] = 0.0
                soft_clamps += 1
                consecutive_soft += 1
                if consecutive_soft >= SOFT_CLAMP_LIMIT:
                    log(f"it={iteration}: {consecutive_soft} consecutive "
                        f"soft clamps -- basis may be degenerating")
            else:
                # full repair: genuine infeasibility -> Phase-I restart
                obj_before = objective
                basis_p, r_its, r_status, _ = sparse_phase1(
                    A, b, max_iter=PHASE1_MAX_ITER, verbose=1 << 30,
                    time_limit=(None if time_limit is None else max(
                        0.0, time_limit - (time.perf_counter() - _t_budget))))
                Bpost = A[:, basis_p].tocsc()
                lu = _try_factorize(Bpost)
                if lu is None:
                    log(f"it={iteration}: post-repair factorization FAILED")
                    return _phase2_fail(
                        "numerical_failure", "post-repair factorization failed",
                        A, b, c, basis, m, n, t0)
                x_basic = _refine_solve(Bpost, lu, b)
                basis = list(basis_p)
                basis_set = set(basis)
                nonbasic = np.array(
                    [j for j in range(n) if j not in basis_set], dtype=np.intp)
                objective = float(c[basis] @ x_basic)
                repairs += 1
                total_repair_pivots += int(r_its)
                consecutive_soft = 0
                degenerate_run = 0
                eff_tol, kappa = _compute_effective_tol(
                    A[:, basis].tocsc(), b_norm)
                iters_since_eff_tol = 0
                log(f"  it={iteration}: REPAIR #{repairs} "
                    f"obj={obj_before:.6f} -> {objective:.6f} "
                    f"(PhaseI={r_its} {r_status}) eff_tol={eff_tol:.3e}")
                if repairs >= REPAIR_LIMIT:
                    log(f"repair limit at it={iteration}")
                    break
        else:
            consecutive_soft = 0

        # ---- DUAL SOLVE ----
        x_basic = _clean_zero(x_basic, TOL)
        c_basic = c[basis]
        try:
            y = lu.solve(c_basic, trans="T")
        except Exception:
            y = np.linalg.solve(B.T.toarray(), c_basic)

        A_nb = A[:, nonbasic]
        reduced = c[nonbasic] - A_nb.T @ y
        reduced = _clean_zero(reduced, TOL)
        objective = float(c_basic @ x_basic)
        obj_delta = objective - prev_obj
        Bcurr = A[:, basis].tocsc()
        xB_resid = float(np.max(np.abs(Bcurr @ x_basic - b)))

        neg_mask = reduced < -TOL
        neg_rc = int(np.sum(neg_mask))
        if neg_rc == 0:
            log(f"OPTIMAL at it={iteration}: obj={objective:.9f} "
                f"resid={xB_resid:.3e} repairs={repairs} "
                f"soft_clamps={soft_clamps}")
            y_final = lu.solve(np.asarray(c_basic, float), trans="T")
            return _phase2_ok(
                A, b, c, basis, x_basic, y_final, reduced, iteration,
                repairs, soft_clamps, total_repair_pivots, t0,
                status="optimal", message=f"optimal at iteration {iteration}")
        min_rc = float(reduced.min())
        min_xB = float(x_basic.min())
        bh = _basis_hash(basis)

        # ---- ENTERING VARIABLE (Devex / Dantzig / Bland) ----
        bland = degenerate_run >= MAX_DEGENERATE
        if bland:
            ni_arr = np.flatnonzero(neg_mask)
            e_idx = int(ni_arr[np.argmin(nonbasic[ni_arr])])
        elif pricing == "devex":
            e_idx = _devex_select(reduced, nonbasic, devex_weights)
            devex_count += 1
        else:
            e_idx = int(np.argmin(reduced))
        entering = int(nonbasic[e_idx])

        # ---- DIRECTION d = B^{-1} a_enter ----
        a_enter = A[:, entering]
        try:
            d = lu.solve(a_enter.toarray().ravel())
        except Exception:
            d = np.linalg.solve(B.toarray(), a_enter.toarray().ravel())
        d_norm = float(np.linalg.norm(d))

        # ---- RATIO TEST ----
        positive = d > PIV_TOL
        if not np.any(positive):
            log(f"it={iteration}: UNBOUNDED col {entering}")
            return _phase2_ok(
                A, b, c, basis, x_basic, y, reduced, iteration,
                repairs, soft_clamps, total_repair_pivots, t0,
                status="unbounded",
                message=f"unbounded in column {entering} at iteration {iteration}")
        ratios = np.full(m, np.inf)
        ratios[positive] = x_basic[positive] / d[positive]
        theta = float(np.min(ratios))
        lc = np.flatnonzero(
            np.abs(ratios - theta) <= PIV_TOL * max(1.0, abs(theta)))
        lidx = int(lc[np.argmin([basis[i] for i in lc])])
        is_degen = 1 if theta <= PIV_TOL else 0

        if verbose and (iteration % LOG_EVERY == 0):
            log(f"LIVE it={iteration:6d} obj={objective:.6f} "
                f"obj_delta={obj_delta:+.3e} neg_rc={neg_rc} "
                f"min_rc={min_rc:.3e} theta={theta:.3e} "
                f"degen={is_degen} repairs={repairs} soft={soft_clamps}")

        # ---- PIVOT UPDATE ----
        exiting_col = basis[lidx]
        basis[lidx] = entering
        basis_set = set(basis)
        nonbasic = np.array([j for j in range(n) if j not in basis_set],
                            dtype=np.intp)
        x_basic = x_basic - theta * d
        x_basic[lidx] = theta

        # ---- PER-PIVOT REFACTORIZATION ----
        B = A[:, basis].tocsc()
        lu = _try_factorize(B)
        if lu is None:
            log(f"it={iteration}: factorization failed after pivot")
            return _phase2_ok(
                A, b, c, basis, x_basic, y, reduced, iteration,
                repairs, soft_clamps, total_repair_pivots, t0,
                status="numerical_failure",
                message=f"factorization failed after pivot {iteration}")
        x_basic = _refine_solve(B, lu, b)
        if np.any(~np.isfinite(x_basic)):
            log(f"it={iteration}: non-finite x_basic after refactor")
            return _phase2_ok(
                A, b, c, basis, x_basic, y, reduced, iteration,
                repairs, soft_clamps, total_repair_pivots, t0,
                status="numerical_failure",
                message=f"non-finite x_basic after pivot {iteration}")

        # ---- DEVEX WEIGHT UPDATE (exiting col -> nonbasic) ----
        if pricing == "devex" and not bland:
            _devex_update_weights(devex_weights, exiting_col, d_norm)
            if is_degen:
                devex_degen += 1

        # ---- PERIODIC EFF_TOL UPDATE (lazy: only while gated) ----
        if iters_since_eff_tol >= RECOMPUTE_EFF_TOL and neg_count > 0:
            eff_tol, kappa = _compute_effective_tol(B, b_norm)
            iters_since_eff_tol = 0

        if is_degen:
            degenerate_run += 1
        else:
            degenerate_run = 0
            if iteration % 8 == 0:
                bh_n = _basis_hash(basis)
                if bh_n in seen_bases:
                    log(f"  it={iteration}: REPEATED BASIS "
                        f"(prev={seen_bases[bh_n]})")
                seen_bases[bh_n] = iteration
        prev_obj = objective

    log(f"terminated at iteration limit {max_iter}")
    return _phase2_ok(
        A, b, c, basis, x_basic, y, reduced, max_iter,
        repairs, soft_clamps, total_repair_pivots, t0,
        status="max_iterations",
        message=f"iteration limit {max_iter} reached without optimality")
# ---------------------------------------------------------------------------
# Phase II result assembly
# ---------------------------------------------------------------------------
def _phase2_ok(A, b, c, basis, x_basic, y, reduced, iterations,
               repairs, soft_clamps, total_repair_pivots, t0,
               status, message):
    """Assemble a Phase-II result dictionary.

    All entries are in the same (scaled standard-form) coordinate system that
    ``sparse_phase2`` received.  ``x`` is the full-length nonnegative vector,
    ``y`` the dual, ``reduced`` the nonbasic reduced costs, and every residual
    is re-derived from those arrays (nothing is reported by faith).
    """
    basis = list(basis)
    m, n = A.shape
    x_basic = np.asarray(x_basic, dtype=np.float64)
    x_full = _recover_x(x_basic, basis, n)
    norm_b = _inf_norm(b)
    norm_c = _inf_norm(c)
    resid_p = _inf_norm(A @ x_full - b)
    nb = np.array([j for j in range(n) if j not in set(basis)], dtype=np.intp)
    # An early-exit path (time / iteration limit before the first dual solve)
    # may have no dual yet.  Substitute a correctly sized zero vector so the
    # reported residuals stay well defined instead of raising; never leave the
    # caller with a silently wrong length.
    if y is None:
        y = np.zeros(m, dtype=np.float64)
    else:
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (m,):
            y = np.zeros(m, dtype=np.float64)
    if reduced is None:
        reduced = np.zeros(nb.shape[0], dtype=np.float64)
    else:
        reduced = np.asarray(reduced, dtype=np.float64)
        if reduced.shape != (nb.shape[0],):
            reduced = np.zeros(nb.shape[0], dtype=np.float64)
    z = np.zeros(n)
    z[nb] = reduced
    resid_d = _inf_norm(A.T @ y + z - c)
    cx = float(c @ x_full)
    by = float(b @ y)
    gap = abs(cx - by) / (1.0 + abs(cx) + abs(by))
    return {
        "status": status,
        "message": message,
        "objective": float(cx),
        "x": x_full,
        "x_basic": x_basic,
        "basis": np.asarray(basis, dtype=np.intp),
        "nonbasic": nb,
        "y": y,
        "reduced": reduced,
        "iterations": int(iterations),
        "phase2_iterations": int(iterations),
        "repairs": int(repairs),
        "soft_clamps": int(soft_clamps),
        "total_repair_pivots": int(total_repair_pivots),
        "rel_primal": float(resid_p / (1.0 + norm_b)),
        "rel_dual": float(resid_d / (1.0 + norm_c)),
        "rel_gap": float(gap),
        "primal_residual": float(resid_p),
        "solve_time": float(time.perf_counter() - t0),
    }


def _phase2_fail(status, message, A, b, c, basis, m, n, t0):
    """Assemble a Phase-II failure result when no iterate is available."""
    basis = list(basis)
    nb = np.array([j for j in range(n) if j not in set(basis)], dtype=np.intp)
    return {
        "status": status,
        "message": message,
        "objective": float("nan"),
        "x": np.zeros(n),
        "x_basic": np.zeros(m),
        "basis": np.asarray(basis, dtype=np.intp),
        "nonbasic": nb,
        "y": np.zeros(m),
        "reduced": np.zeros(0),
        "iterations": 0,
        "phase2_iterations": 0,
        "repairs": 0,
        "soft_clamps": 0,
        "total_repair_pivots": 0,
        "rel_primal": float("inf"),
        "rel_dual": float("inf"),
        "rel_gap": float("inf"),
        "primal_residual": float("inf"),
        "solve_time": float(time.perf_counter() - t0),
    }


# ---------------------------------------------------------------------------
# Crossover orchestrator (IPM fallback entry point)
# ---------------------------------------------------------------------------
def crossover_from_ipm(A, b, c, *, max_iter=MAX_ITER, pricing="devex",
                       verbose=0, phase1_max_iter=PHASE1_MAX_ITER,
                       time_limit=None):
    """Run the full sparse crossover on a standard-form LP.

    Parameters
    ----------
    A, b, c : the standard-form equality system ``A x = b, x >= 0``.  ``A``
        may be dense or sparse; it is converted to CSC internally.  The caller
        is responsible for any row/column scaling (the IPM already equilibrates
        before calling), and the returned primal/dual vectors are in exactly
        these coordinates.
    max_iter : Phase-II pivot budget.
    pricing : ``"devex"`` (default) or ``"dantzig"``.
    verbose : progress verbosity (0 = quiet, N > 0 = progress every N-iters
        block via the Phase-II logger; Phase I always logs periodic summaries).
    phase1_max_iter : Phase-I pivot budget (defaults to the module constant).
    time_limit : optional wall-clock budget (seconds) shared by Phase I and
        Phase II; when exhausted the current phase stops early and reports
        ``phase1_time_limit`` / ``time_limit`` respectively.  ``None`` keeps
        the original iteration-only behaviour.

    Returns a dict with the Phase-II result (see ``sparse_phase2``) augmented
    with ``"phase1"``: ({basis, iterations, status, info}, elapsed) so callers
    can attribute cost and correctness.

    Algorithm: ``sparse_phase1`` crashes from the all-artificial basis
    ``B = I`` (guaranteed feasible/nonsingular, no RRQR or external basis),
    then ``sparse_phase2`` runs Devex revised simplex with per-pivot LU
    refactorization, condition-number-aware soft clamps, and capped Phase-I
    repairs.  The result is optimal only when ``sparse_phase2`` reports
    ``status == "optimal"``.
    """
    A = sp.csc_matrix(A) if not sp.issparse(A) else A.tocsc()
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    t1 = time.perf_counter()
    basis, p1_iters, p1_status, p1_info = sparse_phase1(
        A, b, max_iter=phase1_max_iter, time_limit=time_limit)
    t_phase1 = time.perf_counter() - t1
    if verbose:
        log(f"crossover: Phase I iters={p1_iters} status={p1_status} "
            f"t={t_phase1:.1f}s")

    phase1 = {
        "basis": np.asarray(basis, dtype=np.intp),
        "iterations": int(p1_iters),
        "status": p1_status,
        "info": p1_info,
        "elapsed": float(t_phase1),
    }

    if p1_status != "feasible":
        return {
            "status": f"phase1_{p1_status}",
            "message": f"Phase I failed: {p1_status} "
                       f"{p1_info.get('err', '')}" if p1_info else
                       f"Phase I failed: {p1_status}",
            "phase1": phase1,
            "phase2": None,
        }

    result = sparse_phase2(A, b, c, list(basis), max_iter=max_iter,
                           pricing=pricing, verbose=verbose,
                           time_limit=(None if time_limit is None else max(
                               0.0, time_limit - t_phase1)))
    result["phase1"] = phase1
    return result