"""Sparse Phase I via explicit artificial variables, crashed from RRQR.

Why this exists: the composite -1/0 Phase-I on the RRQR basis cycles on
PILOT87 (neg oscillates 812..1030, x_B blows to -4e9).  This implementa-
tion follows the textbook two-phase method instead:

  minimize  sum of artificials
  s.t.      A x + [e_i for infeasible rows] * art = b,  x >= 0, art >= 0

Crash:  keep the RRQR basic columns that are already feasible; for each
infeasible row i replace the basic column with the artificial e_i and set
art_i = -xB_i >= 0.  This is an m x m non-singular, mostly-sparse basis
with a basic feasible solution of the augmented LP by construction.

The Phase-I objective is monotone (sum of artificials can only decrease
under improved reduced costs) and Bland-style anti-cycling is available
for degenerate ties.  On termination with sum(art) ~ 0 the artificials
are all nonbasic and the remaining basis is feasible for the original LP.

Sparse throughout: A/B are CSC, every solve is 2 triangular solves from
one splu, reduced costs via one sparse matvec A_nb^T y.
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import load_numeric_mps
from mehrotra import to_standard_form
from stage1_audit_rrqr import rrqr_basis


def log(*a):
    print(*a, flush=True)
def sparse_phase1(A, b, basis0=None, max_iter=2_000_000, verbose=2000,
                  tol=1e-7, piv_tol=1e-9, bland_after_stall=500):
    """Return (basis, iter, status, info); basis indexes ORIGINAL columns.

    Crash strategy: all-artificial (B = I) after row normalization so
    b >= 0.  This is guaranteed feasible and nonsingular (the RRQR-crash
    is rank-deficient as a row/column sub-block: rank 618 < 657 on
    PILOT4).  basis0 (RRQR) is kept as the numerical-quality reference
    but is NOT used for the initial basis.

    Termination (textbook two-phase): Phase I is at optimum when NO
    nonbasic column has negative reduced cost.  At that point if the
    artificial sum is ~0 the original LP is feasible and each remaining
    basic artificial (value ~0) is pivoted OUT with a proper ratio-test
    step (theta = xB[r]/alpha[r], leaving variable = r).  Only then is
    the returned basis certified feasible for the original problem.
    """
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

    for it in range(max_iter):
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
                return basis, it, "numerical_failure", {"nart": nart, "err": "feas-verify splu"}
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


def run(name: str, max_iter: int) -> None:
    log(f"=== sparse_phase1 on {name} (max_iter={max_iter}) ===")
    t0 = time.perf_counter()
    sf = to_standard_form(load_numeric_mps(f"data/{name}.mps"))
    A0 = sp.csc_matrix(sf.A)
    b0 = np.asarray(sf.b, float)
    m, n = A0.shape
    log(f"standard form: {m} x {n}  load={time.perf_counter()-t0:.2f}s  "
        f"nnz(A)={A0.nnz:,}")

    # Row/column equilibration BEFORE Phase I (matches run_pilot4 flow).
    # Operate on sparse row scaling + dense column scaling via absmax.
    from scaling import scale_lp
    S = scale_lp(A0.toarray(), b0, np.asarray(sf.c_min, float),
                 np.zeros(n), np.full(n, np.inf))
    A = sp.csc_matrix(S.A)          # scaled standard-form constraint matrix
    b = np.asarray(S.b, float)
    col_scale = np.asarray(S.column_scale, float)
    row_scale = np.asarray(S.row_scale, float)
    log(f"scaling: col_scale[{col_scale.min():.3e}..{col_scale.max():.3e}] "
        f"row_scale[{row_scale.min():.3e}..{row_scale.max():.3e}]")

    t0 = time.perf_counter()
    piv, basis0 = rrqr_basis(A.toarray())
    log(f"RRQR basis: m={len(basis0)} t={time.perf_counter()-t0:.2f}s")

    tracemalloc.start()
    t0 = time.perf_counter()
    basis, its, status, info = sparse_phase1(A, b, basis0, max_iter=max_iter)
    dt = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # independent final verify (only valid when basis is fully original)
    if status == "feasible" and all(isinstance(bb, int) and bb < n for bb in basis):
        Bf = A[:, basis].tocsc()
        luf = splu(Bf)
        xbf = luf.solve(b)
        resid = float(np.max(np.abs(Bf @ xbf - b)))
        negf = int((xbf < -1e-7).sum())
    else:
        # report status; basis may still reference augmented col ids
        xbf = np.array([])
        resid = negf = float("nan")

    log(f"\nphase1: iter={its} status={status} t={dt:.2f}s")
    log(f"  art_sum={info.get('art_sum', '?')}")
    if xbf.size:
        log(f"  final neg={negf} min_xB={xbf.min():.3e}")
        log(f"  final basis residual (||B x_B - b||_inf)={resid:.3e}")
    else:
        log(f"  full verify skipped (status={status})")
    if "t_lu" in info:
        log(f"  time breakdown: lu={info['t_lu']:.2f}s solve={info['t_solve']:.2f}s "
            f"d={info['t_d']:.2f}s alpha={info['t_alpha']:.2f}s")
    log(f"  PEAK python memory = {peak/1e6:.1f} MB   current={cur/1e6:.1f} MB")
    if status == "feasible":
        log("  PHASE I COMPLETE: feasible basis achieved, ready for Phase II")


if __name__ == "__main__":
    import argparse
    import traceback
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--max-iter", type=int, default=2_000_000)
    args = ap.parse_args()
    try:
        run(args.name, args.max_iter)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        raise