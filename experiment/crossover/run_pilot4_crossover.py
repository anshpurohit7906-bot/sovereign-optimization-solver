"""PILOT4 full crossover driver: RRQR -> sparse repair -> production Phase II.

Run with OPENBLAS_NUM_THREADS=1 pinned in the environment for a
deterministic RRQR basis (see PIVOT_STABILITY_NOTE.md).

Uses production code READ-ONLY:
    simplex._simplex_iterations  (Phase II engine)
No production file is modified by this driver.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import load_numeric_mps  # noqa: E402
from mehrotra import to_standard_form  # noqa: E402
from scaling import scale_lp  # noqa: E402
from stage1_audit_rrqr import rrqr_basis  # noqa: E402
from _proto_sparse import repair  # noqa: E402
import simplex as simplex_mod  # noqa: E402
from simplex import _simplex_iterations, SimplexError  # noqa: E402

# --- driver-side progress instrumentation (production code untouched) -------
# _simplex_iterations resolves _solve_basis from its module globals at call
# time, so patching simplex_mod._solve_basis gives per-iteration visibility.
_ORIG_SOLVE_BASIS = simplex_mod._solve_basis
_STATE = {"calls": 0, "t0": None, "colmap": None, "bs": None, "cs": None}


def _progressing_solve_basis(B, rhs, *, condition_limit):
    result = _ORIG_SOLVE_BASIS(B, rhs, condition_limit=condition_limit)
    colmap = _STATE["colmap"]
    if colmap is not None and rhs is _STATE["bs"]:
        _STATE["calls"] += 1  # count primal solves = simplex iterations
        if _STATE["calls"] % 50 == 0:
            elapsed = time.perf_counter() - _STATE["t0"]
            obj_str = ""
            try:
                basis_now = [colmap[B[:, k].tobytes()] for k in range(B.shape[1])]
                obj = float(_STATE["cs"][basis_now] @ result)
                obj_str = f" obj={obj:.4f}"
            except KeyError:
                pass
            print(f"    progress: ~{_STATE['calls']} Phase-II iters, "
                  f"{elapsed:.0f}s elapsed{obj_str}", flush=True)
    return result


simplex_mod._solve_basis = _progressing_solve_basis


def _vertex_cleanup(A, b, basis, *, max_swaps=15, cond_target=1e8,
                    max_cond_evals=60):
    """Vertex-preserving basis cleanup at a degenerate stuck vertex.

    Only degenerate pivots are considered: entering column q may replace
    basic row r only when x_B[r] ~ 0, so theta = x_B[r]/alpha[r] ~ 0 and the
    vertex (x, objective) is unchanged.  Among candidates that keep the basis
    feasible under production's own gesv solve view, the one with the
    smallest cond2 is chosen.  Stops when cond2 < cond_target and
    production-view x_basic >= -tol, or when no candidate improves.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    basis = list(basis)
    basis_set = set(basis)
    cond_evals = 0
    swaps = 0
    B = A[:, basis]
    cond = float(np.linalg.cond(B))
    xB = np.linalg.solve(B, b)
    while swaps < max_swaps and (cond > cond_target or xB.min() < -TOL):
        lu = splu(sp.csc_matrix(B))
        nb = [j for j in range(n) if j not in basis_set]
        # 1e-6: include rows whose gesv value is LU-rounding noise at
        # cond ~1e10 (observed: true-zero basics reading -4e-7).
        rows = np.flatnonzero(np.abs(xB) < 1e-6)
        if rows.size == 0 or not nb:
            break
        best = None
        for q in nb:
            if cond_evals >= max_cond_evals:
                break
            alpha = lu.solve(A[:, q])
            amax = float(np.max(np.abs(alpha)))
            for r in rows:
                if cond_evals >= max_cond_evals:
                    break
                if abs(alpha[r]) <= 1e-7 * max(1.0, amax):
                    continue
                newb = list(basis)
                newb[r] = q
                Bn = A[:, newb]
                xn = np.linalg.solve(Bn, b)
                if xn.min() < -TOL:
                    continue
                cond_evals += 1
                cn = float(np.linalg.cond(Bn))
                if best is None or cn < best[0]:
                    best = (cn, newb, xn)
        if best is None or best[0] >= cond * 0.999:
            break  # no meaningful improvement available
        cond, basis, xB = best[0], best[1], best[2]
        basis_set = set(basis)
        swaps += 1
    return basis, swaps, cond

MPS = os.path.join(_ROOT, "data", "pilot4_plain.mps")
TOL = 1e-8           # certification tolerance (production default)
TOL_WORK = 1e-6      # journey tolerance: above LU-rounding noise at the
                     # degenerate vertices Bland must traverse (observed
                     # gesv noise -4e-7 at cond ~2.6e10).  The final result
                     # is certified separately at TOL with driver-side
                     # independent verification.
COND_LIMIT = 1e12
INFEAS_TOL = 1e-7
MAX_ITER = 30000


def run_once(tag: str) -> dict | None:
    sf = to_standard_form(load_numeric_mps(MPS))
    A = np.asarray(sf.A, dtype=np.float64)
    b = np.asarray(sf.b, dtype=np.float64)
    c = np.asarray(sf.c_min, dtype=np.float64)
    m, n = A.shape

    # --- Stage 0: row/column equilibration of the standard-form LP ----------
    # Driver-side, mathematically exact transform (production scaling.py used
    # read-only).  PILOT4's raw refinery data spans many orders of magnitude;
    # equilibration removes the conditioning pathology that kills Bland's
    # journey through the degenerate interior.  Results are mapped back and
    # verified on the ORIGINAL data in Stage D.
    S = scale_lp(A, b, c, np.zeros(n), np.full(n, np.inf))
    As = np.asarray(S.A, dtype=np.float64)
    bs = np.asarray(S.b, dtype=np.float64)
    cs = np.asarray(S.c, dtype=np.float64)
    col_scale = np.asarray(S.column_scale, dtype=np.float64)

    # --- Stage A: RRQR basis on the scaled problem (BLAS pinned via env) ----
    t0 = time.perf_counter()
    _, basis0 = rrqr_basis(As)
    t_rrqr = time.perf_counter() - t0
    s0 = np.linalg.svd(As[:, basis0], compute_uv=False)
    cond_rrqr = float(s0[0] / s0[-1])

    # --- Stage B: sparse composite Phase-I feasibility repair ---------------
    t0 = time.perf_counter()
    basis, steps, feas = repair(sp.csc_matrix(As), bs, list(basis0), verbose=False)
    t_repair = time.perf_counter() - t0

    B = As[:, basis]
    lu = splu(sp.csc_matrix(B))
    xB = lu.solve(bs)
    min_xB = float(xB.min())
    resid_repair = float(np.max(np.abs(B @ xB - bs)))

    if not feas or min_xB < -INFEAS_TOL:
        print(f"[{tag}] REPAIR FAILED: feas={feas} min_xB={min_xB:.3e}")
        return None

    # --- Stage C: production Phase II with driver-side repair-and-resume ----
    # Production _simplex_iterations hard-stops if a mid-run basis solve
    # (LAPACK gesv) yields x_basic < -tol.  Two driver-side responses, no
    # production changes:
    #   1. re-feasibilize the failed basis with the sparse composite repair;
    #   2. if production's own solve view still rejects the basis (a
    #      degenerate vertex where cond(B) makes LU rounding dominate),
    #      clean up the basis with VERTEX-PRESERVING degenerate pivots
    #      (theta ~ 0, objective unchanged) chosen to minimize cond(B),
    #      each candidate accepted only if production-view x_basic >= -tol.
    # This is a bounded unstick at a stuck vertex, not a per-pivot cond screen.
    colmap = {}
    for j in range(n):
        colmap.setdefault(As[:, j].tobytes(), j)
    _STATE["colmap"] = colmap
    _STATE["bs"] = bs
    _STATE["cs"] = cs
    _STATE["t0"] = time.perf_counter()
    t0 = time.perf_counter()
    current_basis = list(basis)
    total_p2 = 0
    rounds = 0
    stuck = 0
    while True:
        rounds += 1
        _STATE["calls"] = 0
        try:
            status, message, x_std, basis2, iters2, hist = _simplex_iterations(
                As, bs, cs, current_basis,
                tol=TOL_WORK, max_iter=MAX_ITER, condition_limit=COND_LIMIT, phase=2,
            )
        except SimplexError as exc:
            t_phase2 = time.perf_counter() - t0
            print(f"[{tag}] PHASE II SimplexError after {t_phase2:.1f}s: {exc}")
            return None
        total_p2 += iters2
        if status == "optimal":
            break
        if status != "numerical_failure":
            print(f"[{tag}] PHASE II ended: status={status} ({message}) "
                  f"after {total_p2} cumulative pivots")
            return None
        last_obj = hist[-1]["objective"] if hist else float("nan")
        Bf = As[:, basis2]
        xBf = np.linalg.solve(Bf, bs)
        print(f"[{tag}] round {rounds}: Phase II stopped at pivot {iters2} "
              f"({message}); last_obj={last_obj:.6f} "
              f"failed-basis min_xB={xBf.min():.3e} "
              f"cond2={float(np.linalg.cond(Bf)):.3e}", flush=True)
        # Primary unstick: clean up the FAILED basis itself.  At cond ~1e10
        # the -4e-7 "infeasibility" is LU rounding on a truly feasible
        # (degenerate) basis, so degenerate pivots that production's own
        # gesv view accepts are the right move.
        cb, cswaps, ccond = _vertex_cleanup(As, bs, basis2)
        xBc = np.linalg.solve(As[:, cb], bs)
        print(f"[{tag}] round {rounds}: vertex cleanup swaps={cswaps} "
              f"cond2={ccond:.3e} min_xB(gesv)={xBc.min():.3e}", flush=True)
        if xBc.min() >= -TOL:
            current_basis = cb
            continue
        # Fallback: re-repair from the failed basis.
        rb, rsteps, rfeas = repair(sp.csc_matrix(As), bs, basis2, verbose=False)
        xBrb = np.linalg.solve(As[:, rb], bs)
        print(f"[{tag}] round {rounds}: re-repair pivots={rsteps} feas={rfeas} "
              f"min_xB={xBrb.min():.3e}", flush=True)
        if not rfeas:
            print(f"[{tag}] re-repair failed to restore feasibility")
            return None
        current_basis = rb
        if rounds > 60:
            print(f"[{tag}] repair-resume round cap exceeded")
            return None
        stuck = stuck + 1 if iters2 <= 2 else 0
        if stuck >= 8:
            print(f"[{tag}] persistent immediate rejection despite cleanup "
                  f"- stopping for diagnosis")
            return None
        current_basis = rb
    t_phase2 = time.perf_counter() - t0

    # --- Stage C2: strict certification pass at the final vertex -----------
    # The journey terminated at TOL_WORK; re-enter production Phase II at
    # the certification tolerance.  If rounding noise at the final vertex
    # trips the gate, unstick with vertex cleanup / repair as above.
    cert_p2 = 0
    cert_rounds = 0
    cert_status, cert_msg = status, message
    cert_basis = list(basis2)
    while cert_status == "optimal" and cert_rounds < 10:
        cert_rounds += 1
        _STATE["calls"] = 0
        cstatus, cmsg, x_cert, cbasis, citers, _ch = _simplex_iterations(
            As, bs, cs, cert_basis,
            tol=TOL, max_iter=MAX_ITER, condition_limit=COND_LIMIT, phase=2,
        )
        cert_p2 += citers
        if cstatus == "optimal":
            cert_basis, x_std = cbasis, x_cert
            break
        if cstatus != "numerical_failure":
            cert_status, cert_msg = cstatus, cmsg
            break
        Bc = As[:, cbasis]
        xBc0 = np.linalg.solve(Bc, bs)
        print(f"[{tag}] cert round {cert_rounds}: {cmsg}; "
              f"min_xB={xBc0.min():.3e} cond2={float(np.linalg.cond(Bc)):.3e}",
              flush=True)
        cb, cswaps, ccond = _vertex_cleanup(As, bs, cbasis)
        xBcb = np.linalg.solve(As[:, cb], bs)
        print(f"[{tag}] cert round {cert_rounds}: cleanup swaps={cswaps} "
              f"cond2={ccond:.3e} min_xB={xBcb.min():.3e}", flush=True)
        if xBcb.min() >= -TOL:
            cert_basis = cb
            continue
        rb, rsteps, rfeas = repair(sp.csc_matrix(As), bs, cbasis, verbose=False)
        if not rfeas:
            cert_status = "numerical_failure"
            cert_msg = "certification re-repair failed"
            break
        cert_basis = rb
    if cert_status == "optimal":
        status, message, basis2 = cert_status, f"certified: {cert_msg}", cert_basis
        print(f"[{tag}] certification: OPTIMAL at tol={TOL} "
              f"(extra pivots={cert_p2}, rounds={cert_rounds})", flush=True)
    else:
        print(f"[{tag}] certification at tol={TOL} did not confirm: "
              f"{cert_status} ({cert_msg}); journey result stands for "
              f"driver-side verification", flush=True)
    t_total_phase2 = time.perf_counter() - t0

    # --- Stage D: independent verification on the ORIGINAL data -------------
    # Map the scaled standard-form solution back: x = D_c x'.
    x_std = np.asarray(x_std, dtype=np.float64) * col_scale
    B2 = A[:, basis2]
    xB2 = np.linalg.solve(B2, b)
    basis_resid = float(np.max(np.abs(B2 @ xB2 - b)))
    primal_resid = float(np.max(np.abs(A @ x_std - b)))
    rel_primal = primal_resid / max(1.0, float(np.max(np.abs(b))))

    y = np.linalg.solve(B2.T, c[basis2])
    nonbasic = np.array([j for j in range(n) if j not in set(basis2)], dtype=np.intp)
    reduced = c[nonbasic] - A[:, nonbasic].T @ y
    min_reduced = float(reduced.min()) if reduced.size else 0.0
    rel_dual = min_reduced / max(1.0, float(np.max(np.abs(c))))

    x_orig = sf.recover_original(x_std)
    obj = float(sf.c_orig @ x_orig)
    dual_obj = float(y @ b)
    rel_gap = abs(obj - dual_obj) / max(1.0, abs(obj))

    print(f"[{tag}] RRQR cond2={cond_rrqr:.3e} ({t_rrqr:.1f}s)")
    print(f"[{tag}] repair pivots={steps} min_xB={min_xB:.3e} resid={resid_repair:.2e} "
          f"cond2={float(np.linalg.cond(B)):.3e} ({t_repair:.1f}s)")
    print(f"[{tag}] Phase II pivots={total_p2} ({t_total_phase2:.1f}s) "
          f"status={status} ({message})")
    print(f"[{tag}] VERIFY basis_resid={basis_resid:.2e} rel_primal={rel_primal:.3e} "
          f"min_reduced={min_reduced:.3e} rel_dual={rel_dual:.3e} rel_gap={rel_gap:.3e}")
    print(f"[{tag}] OBJECTIVE={obj:.6f}  (dual bound {dual_obj:.6f})")
    return {"tag": tag, "cond_rrqr": cond_rrqr, "steps": steps, "min_xB": min_xB,
            "t_repair": t_repair, "iters2": iters2, "t_phase2": t_phase2,
            "status": status, "obj": obj, "rel_primal": rel_primal,
            "rel_dual": rel_dual, "rel_gap": rel_gap}


if __name__ == "__main__":
    for k in (1, 2, 3):
        run_once(f"run{k}")
