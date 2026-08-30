"""STAGE 2 — RRQR basis → feasibility repair → Phase II hand-off (experimental).

Isolated under experiment/crossover/.  Production code is imported READ-ONLY:

- ``simplex._solve_basis``  : production numerical basis gate (cond <= 1e12)
- ``simplex._simplex_iterations`` : production Revised-Simplex loop; accepts an
  EXTERNAL basis and requires primal feasibility (x_B >= -tol) at entry, so it
  serves directly as the Phase II engine for a repaired basis.
- ``simplex.solve_simplex`` : production cold Phase I + Phase II (comparison arm B).

Feasibility repair (Phase B/C): composite DUAL Phase-I on the sum of primal
infeasibilities (textbook warm-start method; see STAGE2_DESIGN.md):

    Given basis B with x_B = B^-1 b and infeasibility set I = {i: x_B[i] < 0},
    attach costs c_B[i] = -1 for i in I, c_B[i] = 0 otherwise, c_N = 0.
    Leaving row  r = argmin x_B (most negative).
    Tableau row  alpha_r = e_r^T B^-1 A   (via rho = B^-T e_r).
    Entering candidates: nonbasic j with alpha_r[j] < 0.  If none exist, the
    row r equation  sum_j alpha_r[j] x_j = x_B[r] < 0  with x >= 0 is a
    PRIMAL INFEASIBILITY CERTIFICATE (all alpha_r[j] >= 0, x_j >= 0 cannot
    sum to a negative value) -> terminate with a proof, not a guess.
    Dual ratio test: q = argmin |d_j / alpha_r[j]| over candidates,
    d_j = c_j - y^T a_j with y = B^-T c_B (composite objective).
    Pivot: B[:, r] <- a_q; x_q = x_B[r]/alpha_r[q] > 0, x_B -= alpha_[:,q] * x_q.

Every accepted pivot must pass, in this order:
  1. pivot stability      |alpha_r[q]| >= 1e-7 * ||alpha_r||_inf
  2. production gate      cond2(B_new) <= 1e12 (same np.linalg.cond call as
                          production _solve_basis)
  3. solve accuracy       ||B_new x_new - b||_inf <= 1e-7 * (1 + ||b||_inf)
  4. progress             sum of infeasibilities strictly nonincreasing
                          (degenerate pivots allowed but capped, anti-cycling
                          via basis-hash detection and Bland-style fallback)

No x/z or IPM information is used anywhere in the pivot rule.
"""

from __future__ import annotations

import os
import re
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.linalg.lapack as lap

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import load_numeric_mps  # noqa: E402
from mehrotra import to_standard_form  # noqa: E402
from simplex import (  # noqa: E402
    SimplexError,
    _inf_norm,
    _simplex_iterations,
    _solve_basis,
    solve_simplex,
)
from stage1_audit_rrqr import rrqr_basis  # noqa: E402

# Production gates (identical values to src/lp/simplex.py defaults).
CONDITION_LIMIT = 1e12
SIMPLEX_TOL = 1e-8
# Repair-specific safety parameters (documented in STAGE2_DESIGN.md).
PIVOT_REL_TOL = 1e-7     # |alpha_r[q]| >= PIVOT_REL_TOL * ||alpha_r||_inf
RESID_LIMIT = 1e-7       # relative solve-residual acceptance for B_new x = b
MAX_COND_EVALS = 100     # conditioning evaluations allowed per pivot
                         # (cheap O(m^2) getrf+gecon 1-norm screen, so a
                         # larger budget is affordable than with SVD)
MAX_DEGENERATE = 50      # consecutive non-improving pivots before fallback
INFEAS_TOL = 1e-7        # x_B[i] < -INFEAS_TOL counts as infeasible

# --- cheap 1-norm condition screen ------------------------------------------
# LAPACK getrf+dgecon gives the EXACT 1-norm condition number in O(m^2) after
# the O(m^3) LU (which the candidate needs anyway for the solve), vs O(m^3)
# SVD for np.linalg.cond.  By Hoelder (||.||_1 >= ||.||_2 on both A and A^-1),
# cond_1 >= cond_2 always, so screening at the production 2-norm limit is
# CONSERVATIVE: any candidate passing this screen also passes the production
# np.linalg.cond 2-norm gate.  The true 2-norm is computed once for the
# ACCEPTED pivot only, for production-parity reporting.
_dgetrf = lap.dgetrf
_dgecon = lap.dgecon


def cond1_screen(B: np.ndarray) -> tuple[float, object]:
    """Exact 1-norm condition number via LU + dgecon (inf on failure).

    Returns ``(cond1, lu_piv)`` where ``lu_piv`` is the LU factorization
    (``scipy.linalg.lu_factor`` format) of B -- candidates need this for the
    ratio/solve anyway, so no work is duplicated.
    """
    try:
        lu1, piv1, info = _dgetrf(B)
        if info < 0 or info > 0:
            return float("inf"), None
        anorm = float(np.abs(B).sum(axis=0).max())  # 1-norm
        if anorm == 0.0:
            return float("inf"), None
        rcond, _ = _dgecon(lu1, anorm, norm="1")
        if not np.isfinite(rcond) or rcond <= 0.0:
            return float("inf"), None
        return 1.0 / float(rcond), (lu1, piv1)
    except Exception:
        return float("inf"), None



def basis_metrics(A: np.ndarray, b: np.ndarray, basis) -> dict:
    """Full Stage-1-style metrics for a basis, including the production gate."""
    basis = list(basis)
    B = A[:, basis]
    s = np.linalg.svd(B, compute_uv=False)
    sigma_max, sigma_min = float(s[0]), float(s[-1])
    rank = int(np.sum(s > 1e-10 * sigma_max))
    cond2 = sigma_max / sigma_min if sigma_min > 0 else float("inf")
    try:
        x_b = _solve_basis(B, b, condition_limit=CONDITION_LIMIT)
        gate = "OK"
        resid = _inf_norm(B @ x_b - b) / (1.0 + _inf_norm(b))
    except SimplexError as exc:
        x_b, gate, resid = None, f"REJECTED ({exc})", float("inf")
    neg = int(np.sum(x_b < -INFEAS_TOL)) if x_b is not None else -1
    return {
        "size": len(basis), "rank": rank, "sigma_min": sigma_min,
        "sigma_max": sigma_max, "cond2": cond2, "gate": gate,
        "resid": resid, "x_B": x_b, "min_xB": float(np.min(x_b)) if x_b is not None else float("nan"),
        "neg_basics": neg,
    }


def repair_feasibility(A: np.ndarray, b: np.ndarray, basis0,
                       max_pivots: int = 5000, verbose: bool = False) -> dict:
    """Composite dual Phase-I: drive x_B = B^-1 b to >= -INFEAS_TOL.

    Returns a report dict; ``status`` is one of
    ``feasible`` | ``infeasible_certificate`` | ``no_admissible_pivot`` |
    ``cycling`` | ``max_pivots`` | ``numerical_failure``.
    """
    t0 = time.perf_counter()
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m, n = A.shape
    basis = list(basis0)
    bnorm = 1.0 + _inf_norm(b)

    B = A[:, basis]
    lu = sla.lu_factor(B)
    x_b = sla.lu_solve(lu, b)
    sum_infeas = float(np.sum(np.maximum(0.0, -x_b)))

    seen_bases: set = {tuple(sorted(basis))}
    pivots: list[dict] = []
    degenerate_run = 0
    status = "max_pivots"
    message = f"pivot limit ({max_pivots}) reached"
    cycling = False
    cycle_kick_pending = False
    cycle_kicks = 0
    # Devex reference-framework weights (init 1): approximate the steepest
    # edge ||B^-1 a_j|| so the entering rule maximizes the improvement RATE
    # d_j^2 / w_j instead of raw Dantzig |d_j|, which crawls on wide LPs
    # (1428 columns) because raw reduced costs are not scale comparable.
    weights = np.ones(n)

    for p in range(max_pivots):
        infeas = np.flatnonzero(x_b < -INFEAS_TOL)
        if infeas.size == 0:
            status, message = "feasible", f"primal-feasible basis after {p} pivots"
            break

        # --- composite primal Phase-I gradient ---------------------------
        # Phase-1 cost: c_B[i] = -1 for every basic with x_B[i] < 0, 0
        # otherwise, so the phase-1 objective IS the sum of infeasibilities
        # sum(max(0, -x_B)).  With y = B^-T c_B the nonbasic reduced costs
        # are d_j = -y^T a_j (c_N = 0); for any entering j with d_j < 0 the
        # phase-1 objective decreases linearly for sufficiently small primal
        # step t > 0 -- that slope condition is what guarantees monotone
        # progress (the previous row-first dual-ratio rule had no such
        # guarantee and stalled with "all rejected (progress)").
        c_b = np.zeros(m)
        c_b[x_b < 0.0] = -1.0
        y = sla.lu_solve(lu, c_b, trans=1)                 # B^-T c_B
        d_all = -A.T @ y                                   # reduced costs, c_N = 0

        nonbasic_mask = np.ones(n, dtype=bool)
        nonbasic_mask[basis] = False
        d_scale = max(1.0, _inf_norm(d_all))
        cand = np.flatnonzero(
            nonbasic_mask & (d_all < -PIVOT_REL_TOL * d_scale))
        if cand.size == 0:
            status = "infeasible_certificate"
            message = (
                f"phase-1 reduced costs d_j >= 0 for all nonbasic j "
                f"(min {float(np.min(d_all[nonbasic_mask])):.3e}) while "
                f"sum_infeas={sum_infeas:.6e}: primal infeasibility certificate"
            )
            break

        # Entering rule: Devex-normalized Dantzig -- maximize d_j^2 / w_j
        # (steepest-edge-style improvement rate) with deterministic
        # tie-break on column index; Bland fallback (lowest index) after a
        # run of degenerate (non-improving) pivots to break cycling.
        if degenerate_run >= MAX_DEGENERATE:
            order = cand[np.argsort(cand, kind="stable")]
        else:
            score = d_all[cand] ** 2 / weights[cand]
            order = cand[np.lexsort((cand, -score))]

        accepted = False
        rejected_unsafe = 0
        rejected_progress = 0
        n_cond_evals = 0
        # Bland mode scans ALL candidates in ascending column index: its
        # finite-termination argument requires the lowest-index admissible
        # pivot, so the per-pivot conditioning-evaluation budget must not
        # truncate the scan (hard gates below are unchanged).
        if degenerate_run >= MAX_DEGENERATE:
            order_scan = order
            eval_budget = len(order)
        else:
            order_scan = order[: MAX_COND_EVALS * 4]
            eval_budget = MAX_COND_EVALS
        for q in order_scan:
            if n_cond_evals >= eval_budget:
                break
            # entering direction alpha = B^-1 a_q (one triangular solve pair)
            alpha_q = sla.lu_solve(lu, A[:, q])
            piv_abs = PIVOT_REL_TOL * max(_inf_norm(alpha_q), 1e-30)
            # --- composite primal ratio test -----------------------------
            # Feasible basics (x_B[i] >= 0) block when they hit 0 from above:
            #     x_B[i] - t*alpha[i] = 0, blocking iff alpha[i] > 0.
            # Infeasible basics (x_B[i] < 0) block when they climb to 0:
            #     blocking iff alpha[i] < 0 (they increase toward zero).
            # t* = min breakpoint; the slope d_q < 0 guarantees the phase-1
            # objective (the infeasibility sum) is non-increasing on [0, t*].
            feas = x_b >= 0.0
            block = (feas & (alpha_q > piv_abs)) | (~feas & (alpha_q < -piv_abs))
            if not np.any(block):
                # cannot happen for a bounded phase-1 objective in exact
                # arithmetic; treat as numerically unusable candidate
                rejected_unsafe += 1
                continue
            safe_alpha = np.where(alpha_q == 0.0, 1.0, alpha_q)
            t_all = np.where(block, x_b / safe_alpha, np.inf)
            t_all = np.where(t_all >= 0.0, t_all, np.inf)  # forward steps only
            t_min = float(np.min(t_all))
            if not np.isfinite(t_min):
                rejected_unsafe += 1
                continue
            # Tie-break among (near-)tied minimum ratios: LEXICOGRAPHIC
            # ratio test (cycling-proof, deterministic).  Among tied rows,
            # pick r minimizing row r of B^-1 scaled by 1/alpha_r, compared
            # lexicographically.  With this rule the vector
            # (phase-1 objective, x_B) strictly decreases lexicographically
            # at EVERY pivot -- including degenerate t = 0 pivots -- so no
            # basis can repeat, irrespective of the entering rule (this
            # holds even for the composite piecewise-linear phase-1
            # objective, because during a degenerate run the infeasible set
            # -- hence the linear objective -- is fixed).  Bland-style
            # index tie-breaks are NOT sufficient here: Bland's theorem
            # covers a fixed linear objective only, and was observed to
            # cycle on SC205/SHARE2B/BLEND.  Cost: one triangular solve
            # pair (row of B^-1) per TIED row, usually 1-3 rows.
            tied = np.flatnonzero(t_all <= t_min * (1.0 + 1e-9) + 1e-12)
            if tied.size == 1:
                r = int(tied[0])
            else:
                r = -1
                best_w = None
                for r_cand in tied:
                    a_c = float(alpha_q[r_cand])
                    if abs(a_c) <= piv_abs:
                        continue
                    e_r = np.zeros(m)
                    e_r[r_cand] = 1.0
                    z = sla.lu_solve(lu, e_r, trans=1) / a_c
                    w = np.concatenate(([t_min], z))
                    if best_w is None:
                        r, best_w = int(r_cand), w
                        continue
                    # lexicographic comparison with relative tolerance
                    diff = w - best_w
                    scale = np.maximum(
                        np.maximum(1.0, np.abs(w)), np.abs(best_w))
                    nz = np.flatnonzero(np.abs(diff) > 1e-12 * scale)
                    if nz.size and diff[nz[0]] < 0.0:
                        r, best_w = int(r_cand), w
                if r < 0:  # defensive: all tied alphas below pivot floor
                    r = int(tied[np.argmax(np.abs(alpha_q[tied]))])
            if cycle_kick_pending:
                # One-shot deterministic escape (kept as a last-resort
                # backstop; the lexicographic rule should make it dead code).
                r = int(tied[np.argmax(np.abs(alpha_q[tied]))])
                cycle_kick_pending = False
            step = float(t_all[r])
            B_new = B.copy()
            B_new[:, r] = A[:, q]
            n_cond_evals += 1
            # cheap exact 1-norm screen; conservative w.r.t. the 2-norm gate
            cond1, lu_piv = cond1_screen(B_new)
            if not np.isfinite(cond1) or cond1 > CONDITION_LIMIT:
                rejected_unsafe += 1
                continue
            # reuse the screen's LU factorization (getrf output is exactly
            # the (lu, piv) format lu_factor/lu_solve use -- no refactor)
            lu_new = lu_piv
            x_new = sla.lu_solve(lu_new, b)
            # solve accuracy
            resid = _inf_norm(B_new @ x_new - b) / bnorm
            if not np.isfinite(resid) or resid > RESID_LIMIT:
                rejected_unsafe += 1
                continue
            new_infeas = float(np.sum(np.maximum(0.0, -x_new)))
            # legitimate progress toward feasibility (monotone in exact
            # arithmetic; the slack only absorbs rounding)
            if new_infeas > sum_infeas + 1e-9 * bnorm:
                rejected_progress += 1
                continue
            # ---- accept ----
            leaving = basis[r]
            # The production 2-norm gate (np.linalg.cond via SVD) is the
            # authoritative check, but the cheap 1-norm screen (LU + dgecon)
            # is conservative: cond_1 >= cond_2 always per Hoelder's inequality.
            # If the 1-norm screen passes, the 2-norm gate should also pass.
            # We keep the 1-norm screen as the pivot-selection gate and compute
            # the true 2-norm only for final reporting below.
            if not np.isfinite(cond1) or cond1 > CONDITION_LIMIT:
                rejected_unsafe += 1
                continue
            # Compute true 2-norm condition for reporting (after the 1-norm gate passes)
            try:
                cond_new = float(np.linalg.cond(B_new))
            except np.linalg.LinAlgError:
                cond_new = cond1
            # Devex weight update (Forrest-Goldfarb reference framework):
            # with entering q and leaving row r, tableau row
            # alpha_rj = (B_old^-1 A)[r, j]; for each remaining nonbasic j
            #     w_j <- max(w_j, (w_q / alpha_rq^2) * alpha_rj^2)
            # and the leaving variable inherits w = max(w_q/alpha_rq^2, 1).
            # This keeps w_j an upper-bound approximation of ||B^-1 a_j||^2,
            # so the entering rule approximates the true steepest edge.
            piv_elem = float(alpha_q[r])
            if piv_elem != 0.0:
                e_r = np.zeros(m)
                e_r[r] = 1.0
                rho = sla.lu_solve(lu, e_r, trans=1)      # OLD basis
                row_r = A.T @ rho
                ratio = weights[q] / (piv_elem * piv_elem)
                nb_old = nonbasic_mask.copy()
                nb_old[q] = False
                upd = np.maximum(weights, ratio * row_r ** 2)
                weights[:] = np.where(nb_old, upd, weights)
                weights[leaving] = max(ratio, 1.0)
                if float(np.max(weights)) > 1e6:
                    # Standard Devex safeguard: accumulated weights drift
                    # above the true ||B^-1 a_j||^2; re-anchoring the
                    # reference framework (all weights 1) keeps the
                    # approximation valid.
                    weights[:] = 1.0
            pivots.append({
                "pivot": p, "entering": int(q), "leaving": leaving,
                "leaving_value": float(x_b[r]), "step": step,
                "d_entering": float(d_all[q]),
                "sum_infeas_old": sum_infeas,
                "sum_infeas_new": new_infeas,
                "min_xB_new": float(np.min(x_new)),
                "neg_basics_new": int(np.sum(x_new < -INFEAS_TOL)),
                "cond_new": cond_new,
                "resid": resid, "gate_ok": True,
                "rejected_unsafe": rejected_unsafe,
                "rejected_progress": rejected_progress,
                "cond_evals": n_cond_evals,
            })
            if verbose:
                print(f"  pivot {p:4d}: row {r:4d} col {leaving:5d} <- col {q:5d} "
                      f"step={step:+.4e} infeas {sum_infeas:.4e} -> "
                      f"{new_infeas:.4e} cond={cond_new:.2e}")
            basis[r] = int(q)
            improved = new_infeas < sum_infeas - 1e-12 * bnorm
            B, lu, x_b, sum_infeas = B_new, lu_new, x_new, new_infeas
            degenerate_run = 0 if improved else degenerate_run + 1
            key = tuple(sorted(basis))
            if key in seen_bases and degenerate_run > MAX_DEGENERATE:
                if cycle_kicks == 0:
                    # First revisit: one deterministic escape attempt (the
                    # strongest-|alpha| tied pivot for the next iteration)
                    # before declaring failure.
                    cycle_kicks += 1
                    cycle_kick_pending = True
                else:
                    cycling = True
            seen_bases.add(key)
            accepted = True
            break
        if accepted:
            if cycling:
                status, message = "cycling", f"basis revisited at pivot {p}"
                break
            continue
        status = "no_admissible_pivot"
        message = (
            f"pivot {p}: {cand.size} candidates, all rejected "
            f"(cond-evals {n_cond_evals}, unsafe {rejected_unsafe}, "
            f"progress {rejected_progress}); sum_infeas={sum_infeas:.6e}"
        )
        break

    infeas = np.flatnonzero(x_b < -INFEAS_TOL)
    final_cond = float(np.linalg.cond(A[:, basis]))
    final_resid = _inf_norm(A[:, basis] @ x_b - b) / bnorm
    return {
        "status": status, "message": message,
        "basis": basis, "x_B": x_b, "pivots": pivots, "n_pivots": len(pivots),
        "sum_infeas": sum_infeas, "neg_basics": int(infeas.size),
        "min_xB": float(np.min(x_b)), "final_cond": final_cond,
        "final_resid": final_resid,
        "runtime": time.perf_counter() - t0,
    }

# ---------------------------------------------------------------------------
# Phase II wrapper (experimental; production code used read-only)
# ---------------------------------------------------------------------------

def phase_two_from_basis(sf, basis, *, tol: float = SIMPLEX_TOL,
                         max_iter: int = 20000) -> dict:
    """Phase II from an externally supplied PRIMAL-FEASIBLE basis.

    Reuses the production Revised-Simplex loop ``_simplex_iterations``
    (read-only import): it enforces primal feasibility and re-validates the
    basis with the production ``_solve_basis`` gate at every iteration, so
    an externally supplied basis cannot bypass any production safeguard.

    Returns a plain dict (NOT a SimplexResult) so nothing here can be
    mistaken for a production API.
    """
    t0 = time.perf_counter()
    A, b, c = sf.A, sf.b, sf.c_min
    status, message, x2, basis_out, iters, _hist = _simplex_iterations(
        A, b, c, list(basis),
        tol=tol, max_iter=max_iter,
        condition_limit=CONDITION_LIMIT, phase=2,
    )
    primal_residual = _inf_norm(A @ x2 - b)
    rel_primal = primal_residual / max(1.0, _inf_norm(b))
    x_orig = sf.recover_original(x2)
    objective = float(sf.c_orig @ x_orig)
    return {
        "status": status, "message": message, "objective": objective,
        "x_standard": x2, "x": x_orig, "iterations": iters,
        "primal_residual": primal_residual, "rel_primal": rel_primal,
        "runtime": time.perf_counter() - t0,
    }




# ---------------------------------------------------------------------------
# Instance driver
# ---------------------------------------------------------------------------

EXPECTED = {  # from tests/run_benchmarks.py
    "AFIRO": -464.7531428571, "SC205": -52.2020612117,
    "ADLITTLE": 225494.9631623802, "SHARE2B": -415.7322407414,
    "BLEND": -30.8121498458,
}


def run_instance(name: str, path: str, *, max_pivots: int = 5000,
                 verbose_pivots: bool = False, maximize: bool = False) -> dict:
    """Full Stage-2 pipeline for one instance + cold-Phase-I comparison."""
    print("=" * 78)
    print(f"INSTANCE: {name}")
    print("=" * 78)
    out: dict = {"name": name}

    sf = to_standard_form(load_numeric_mps(path), maximize=maximize)
    A, b = sf.A, sf.b
    m, n = A.shape
    print(f"A = {m} x {n}")

    # -- Stage 1: pure RRQR basis (no IPM data) --------------------------
    t0 = time.perf_counter()
    _, basis0 = rrqr_basis(A)
    rr = basis_metrics(A, b, basis0)
    rr["qr_time"] = time.perf_counter() - t0
    print(f"RRQR:   size={rr['size']} rank={rr['rank']} "
          f"sig_min={rr['sigma_min']:.3e} sig_max={rr['sigma_max']:.3e} "
          f"cond={rr['cond2']:.3e} gate={rr['gate']} resid={rr['resid']:.2e} "
          f"min_xB={rr['min_xB']:.3e} neg={rr['neg_basics']} "
          f"({rr['qr_time']:.1f}s)")
    out["rrqr"] = rr

    # -- Phase B/C: feasibility repair ------------------------------------
    rep = repair_feasibility(A, b, basis0, max_pivots=max_pivots,
                             verbose=verbose_pivots)
    print(f"REPAIR: status={rep['status']} pivots={rep['n_pivots']} "
          f"neg={rep['neg_basics']} min_xB={rep['min_xB']:.3e} "
          f"sum_infeas={rep['sum_infeas']:.3e} "
          f"cond={rep['final_cond']:.3e} resid={rep['final_resid']:.2e} "
          f"({rep['runtime']:.1f}s)")
    if rep["message"]:
        print(f"        {rep['message']}")
    out["repair"] = rep

    # -- Phase E: Phase II from the repaired basis (only if feasible) -----
    if rep["status"] == "feasible":
        p2 = phase_two_from_basis(sf, rep["basis"])
        exp = EXPECTED.get(name)
        obj = p2["objective"]
        match = (abs(obj - exp) <= 1e-6 * max(1.0, abs(exp))) if exp is not None else None
        print(f"PHASE2: status={p2['status']} iters={p2['iterations']} "
              f"obj={obj:.9f} rel_p={p2['rel_primal']:.2e} "
              f"({p2['runtime']:.1f}s)"
              + ("" if match is None else f" expected={exp:.9f} "
                 f"{'MATCH' if match else 'MISMATCH'}"))
        out["phase2"] = p2
    else:
        print("PHASE2: SKIPPED (no primal-feasible basis)")
        out["phase2"] = None

    # -- Phase G arm B: production cold Phase I + Phase II ----------------
    t0 = time.perf_counter()
    cold = solve_simplex(sf)
    cold_t = time.perf_counter() - t0
    print(f"COLD:   status={cold.status} p1={cold.phase_one_iterations} "
          f"p2={cold.phase_two_iterations} obj={cold.objective:.9f} "
          f"rel_p={cold.rel_primal:.2e} ({cold_t:.1f}s)")
    out["cold"] = {"status": cold.status, "obj": cold.objective,
                   "p1": cold.phase_one_iterations, "p2": cold.phase_two_iterations,
                   "runtime": cold_t}
    return out


def main() -> int:
    verbose = "--verbose" in sys.argv
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = a.split("=", 1)[1].upper()
    data = os.path.join(_ROOT, "data")
    names = ["AFIRO", "SC205", "ADLITTLE", "SHARE2B", "BLEND", "PILOT4"]
    files = {nm: os.path.join(data, f"{nm.lower()}.mps") for nm in names}
    files["PILOT4"] = os.path.join(data, "pilot4_plain.mps")
    if only:
        # Filter LAST so --only also excludes the PILOT87 append below.
        wanted = [t for t in re.split(r"[,\s]+", only) if t]
        unknown = [t for t in wanted if t not in files]
        if unknown:
            raise SystemExit(f"unknown instance {unknown!r}; choose from {sorted(files)}")
        names = wanted
    elif "--no-pilot87" not in sys.argv:
        names = names + ["PILOT87"]
        files["PILOT87"] = os.path.join(data, "pilot87.mps")

    results = []
    for nm in names:
        results.append(run_instance(nm, files[nm], verbose_pivots=verbose))

    print("\n" + "=" * 78)
    print("STAGE 2 SUMMARY")
    print("=" * 78)
    for r in results:
        rr, rep, p2, cold = r["rrqr"], r["repair"], r["phase2"], r["cold"]
        line = (f"{r['name']:9s} RRQR[cond={rr['cond2']:.2e} neg={rr['neg_basics']}] "
                f"REPAIR[{rep['status']} pivots={rep['n_pivots']} "
                f"cond={rep['final_cond']:.2e}]")
        if p2 is not None:
            line += f" PH2[{p2['status']} it={p2['iterations']} obj={p2['objective']:.6f}]"
        else:
            line += " PH2[skipped]"
        line += f" COLD[{cold['status']} it={cold['p1'] + cold['p2']} obj={cold['obj']:.6f}]"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
