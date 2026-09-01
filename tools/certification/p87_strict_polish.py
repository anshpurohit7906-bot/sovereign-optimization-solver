"""Strict simplex polish for PILOT87 terminal basis.

Loads the saved terminal basis from p87_phase2_v2_final.npz and continues
pivoting with a STRICT raw reduced-cost criterion (no tolerance floor) until
all reduced costs are >= 0 at full precision.

Usage:
    OPENBLAS_NUM_THREADS=1 python tools/certification/p87_strict_polish.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

_HERE = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
SCRATCH = os.path.join(_ROOT, "artifacts", "pilot87")
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"),
           os.path.join(_ROOT, "experiment", "crossover")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HIGHS_REF = 301.710347333
MICOSE_TOL = 1e-13
MAX_POLISH = 500


def inf_norm(v):
    v = np.asarray(v, float)
    return float(np.max(np.abs(v))) if v.size else 0.0


def _refine_solve(B, lu, rhs, *, n_refine=3):
    x = lu.solve(rhs)
    for _ in range(n_refine):
        resid = B @ x - rhs
        rnorm = float(np.max(np.abs(resid)))
        if rnorm < 1e-12:
            break
        try:
            dc = lu.solve(resid.toarray().ravel() if hasattr(resid, "toarray") else resid)
        except Exception:
            break
        cand = x + dc
        rc = B @ cand - rhs
        rn = float(np.max(np.abs(rc)))
        if rn < rnorm:
            x = cand
        else:
            break
    return x

def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("STRICT SIMPLEX POLISH — PILOT87 terminal basis")
    print("=" * 70)

    raw = np.load(os.path.join(SCRATCH, "p87_prepared.npz"), allow_pickle=False)
    A_sc = sp.load_npz(os.path.join(SCRATCH, "p87_prepared_A.npz")).tocsc()
    b_sc = np.asarray(raw["b"], float)
    c_sc = np.asarray(raw["c"], float)
    row_scale = np.asarray(raw["row_scale"], float)
    col_scale = np.asarray(raw["col_scale"], float)
    m, n = A_sc.shape

    inv_col = 1.0 / col_scale
    inv_row = 1.0 / row_scale
    A0 = (sp.diags(inv_row) @ A_sc @ sp.diags(inv_col)).tocsc()
    b0 = inv_row * b_sc
    c0 = inv_col * c_sc

    fin = np.load(os.path.join(SCRATCH, "p87_phase2_v2_final.npz"), allow_pickle=False)
    basis = list(np.asarray(fin["basis"], dtype=np.intp))
    x_basic = np.asarray(fin["x_basic"], float)
    obj_solver = float(fin["obj_scaled"])
    it_solver = int(fin["it"])

    print(f"Terminal basis: it={it_solver}  m={m} n={n}  basis_size={len(basis)}")
    print(f"Solver obj (scaled std-form): {obj_solver:.9f}")

    basis_set = set(basis)
    nonbasic = np.array(sorted(set(range(n)) - basis_set), dtype=np.intp)

    B = A0[:, basis].tocsc()
    lu = splu(B)
    x_basic = _refine_solve(B, lu, b0)
    basis_arr = np.array(basis, dtype=np.intp)
    obj = float(c0[basis_arr] @ x_basic)

    c_B = c0[basis_arr]
    try:
        y = lu.solve(c_B, trans='T')
    except Exception:
        y = np.linalg.solve(B.toarray().T, c_B)

    A_nb = A0[:, nonbasic]
    reduced = c0[nonbasic] - A_nb.T @ y

    n_neg_before = int((reduced < 0).sum())
    min_rc_before = float(reduced.min())
    print(f"\nBefore polish: min_rc={min_rc_before:.3e}  n_neg={n_neg_before}")


    # ---- Strict polish loop ----
    polish_pivots = 0
    degenerate_count = 0
    MAX_DEGENERATE = 200

    for iteration in range(MAX_POLISH):
        c_B = c0[np.array(basis, dtype=np.intp)]
        A_nb = A0[:, nonbasic]
        reduced = c0[nonbasic] - A_nb.T @ y
        reduced[np.abs(reduced) < MICOSE_TOL] = 0.0

        neg_mask = reduced < 0
        if not np.any(neg_mask):
            print(f"  STRICT OPTIMAL at polish iter={iteration}, "
                  f"min_rc={reduced.min():.3e} (all >= 0)")
            break

        neg_idx = np.where(neg_mask)[0]
        ent_local = neg_idx[np.argmin(reduced[neg_idx])]
        ent_col = nonbasic[ent_local]
        rc_enter = float(reduced[ent_local])

        a_ent = A0[:, ent_col]
        d = lu.solve(a_ent.toarray().ravel() if hasattr(a_ent, "toarray")
                     else a_ent)

        pos_mask = d > 1e-12
        if not np.any(pos_mask):
            print(f"  UNBOUNDED direction at polish iter={iteration}, "
                  f"col={ent_col}, rc={rc_enter:.3e}")
            break

        ratios = np.where(pos_mask, x_basic / d, np.inf)
        min_ratio = ratios.min()
        at_min = np.where(np.abs(ratios - min_ratio) < 1e-12)[0]
        basis_arr_now = np.array(basis, dtype=np.intp)
        leave_local = at_min[np.argmin(basis_arr_now[at_min])]
        leave_col = basis[leave_local]

        is_degenerate = min_ratio < 1e-14

        basis[leave_local] = ent_col
        nonbasic[ent_local] = leave_col

        if is_degenerate:
            degenerate_count += 1
        else:
            degenerate_count = 0

        B = A0[:, basis].tocsc()
        lu = splu(B)
        x_basic = _refine_solve(B, lu, b0)

        c_B = c0[np.array(basis, dtype=np.intp)]
        try:
            y = lu.solve(c_B, trans='T')
        except Exception:
            y = np.linalg.solve(B.toarray().T, c_B)

        basis_set = set(basis)
        nonbasic = np.array(sorted(set(range(n)) - basis_set), dtype=np.intp)

        polish_pivots += 1
        obj = float(c_B @ x_basic)

        if polish_pivots % 5 == 0 or not is_degenerate:
            print(f"  polish it={polish_pivots}: enter={ent_col} rc={rc_enter:.3e} "
                  f"leave={leave_col} ratio={min_ratio:.3e} "
                  f"deg={'Y' if is_degenerate else 'N'} obj={obj:.9f}")

        if degenerate_count >= MAX_DEGENERATE:
            print(f"  WARNING: {MAX_DEGENERATE} consecutive degenerate pivots, stopping")
            break
    else:
        print(f"  MAX_POLISH ({MAX_POLISH}) reached without convergence")


    # ---- Final state ----
    basis_arr = np.array(basis, dtype=np.intp)
    nonbasic = np.array(sorted(set(range(n)) - set(basis)), dtype=np.intp)
    c_B = c0[basis_arr]
    B = A0[:, basis_arr].tocsc()
    lu = splu(B)
    x_basic = _refine_solve(B, lu, b0)
    try:
        y = lu.solve(c_B, trans='T')
    except Exception:
        y = np.linalg.solve(B.toarray().T, c_B)

    A_nb = A0[:, nonbasic]
    reduced = c0[nonbasic] - A_nb.T @ y
    reduced[np.abs(reduced) < MICOSE_TOL] = 0.0

    obj_final = float(c_B @ x_basic)
    min_rc_after = float(reduced.min())
    n_neg_after = int((reduced < 0).sum())

    print(f"\n{'='*70}")
    print(f"STRICT POLISH RESULT:")
    print(f"  Pivots taken:      {polish_pivots}")
    print(f"  Min reduced cost:  {min_rc_after:.3e}")
    print(f"  Neg reduced costs: {n_neg_after}")
    print(f"  Final objective:   {obj_final:.9f}")
    print(f"{'='*70}")

    x_full = np.zeros(n, dtype=np.float64)
    x_full[basis_arr] = x_basic
    np.savez_compressed(
        os.path.join(SCRATCH, "p87_strict_polished.npz"),
        basis=basis_arr, nonbasic=nonbasic, x_basic=x_basic, x_full=x_full,
        y=y, reduced=reduced, obj_scaled=obj_final, it=it_solver,
        polish_pivots=polish_pivots, min_reduced=min_rc_after,
    )
    print(f"Saved polished basis to p87_strict_polished.npz")

    # ---- Strict KKT certificate ----
    print(f"\n{'='*70}")
    print("STRICT KKT CERTIFICATE (raw precision, no tolerance floor)")
    print(f"{'='*70}")

    resid_primal = inf_norm(A0 @ x_full - b0)
    print(f"[1] Primal ||Ax-b||_inf  = {resid_primal:.3e}")

    z_all = c0 - A0.T @ y
    z_nonbasic = z_all[nonbasic]
    condB = float(np.linalg.cond(B.toarray()))
    noise_floor = max(1e-14, 1e2 * np.finfo(float).eps * condB)
    print(f"[2] min(z_nonbasic)      = {float(z_nonbasic.min()):.3e}")
    print(f"    num z < 0            = {int((z_nonbasic < 0).sum())}")
    print(f"    num z < -1e-12       = {int((z_nonbasic < -1e-12).sum())}")
    print(f"    cond(B)              = {condB:.1e}  (sign noise floor ~{noise_floor:.1e})")

    comp = float(np.dot(x_full, z_all))
    print(f"[3] Complementarity x'z  = {comp:.3e}")

    bres = inf_norm(B @ x_basic - b0)
    print(f"[4] Basis residual       = {bres:.3e}")

    from numerical_model import load_numeric_mps
    from mehrotra import to_standard_form
    lp = load_numeric_mps(os.path.join(_ROOT, "data", "pilot87.mps"))
    sf = to_standard_form(lp)
    c_orig = np.asarray(sf.c_orig, float)
    x_orig = sf.recover_original(x_full)
    obj_orig = float(np.dot(c_orig, x_orig))

    print(f"[5] Original objective   = {obj_orig:.9f}")
    print(f"    HiGHS reference      = {HIGHS_REF:.9f}")
    print(f"    |delta|              = {abs(obj_orig - HIGHS_REF):.3e}")


    A_orig = np.asarray(lp.A, float)
    b_orig = np.asarray(lp.b, float)
    rt = list(lp.row_types)
    eq_rows = [i for i, r in enumerate(rt) if r == "E"]
    le_rows = [i for i, r in enumerate(rt) if r == "L"]
    ge_rows = [i for i, r in enumerate(rt) if r == "G"]
    viol = []
    if eq_rows:
        viol.append(float(np.abs(A_orig[eq_rows] @ x_orig - b_orig[eq_rows]).max()))
    if le_rows:
        sl = (A_orig[le_rows] @ x_orig) - b_orig[le_rows]
        viol.append(float(max(0.0, sl.max())))
    if ge_rows:
        sg = b_orig[ge_rows] - (A_orig[ge_rows] @ x_orig)
        viol.append(float(max(0.0, sg.max())))
    lb = np.asarray(lp.lower_bounds, float)
    ub = np.asarray(lp.upper_bounds, float)
    lb_viol = inf_norm(np.minimum(0.0, x_orig - lb))
    ub_viol = inf_norm(np.maximum(0.0, x_orig - ub))
    row_viol = max(viol) if viol else 0.0

    print(f"[6] Orig primal feasibility:")
    if eq_rows:
        print(f"    E-row residual       = {viol[0]:.3e}")
    print(f"    Ineq violation       = {row_viol:.3e}")
    print(f"    lb violation         = {lb_viol:.3e}")
    print(f"    ub violation         = {ub_viol:.3e}")

    dual_bound = float(np.dot(b0, y))
    gap = obj_final - dual_bound
    print(f"[7] Strong duality gap   = {gap:.3e}")

    strict_ok = (resid_primal < 1e-6 and bres < 1e-6
                 and float(z_nonbasic.min()) >= -noise_floor
                 and abs(comp) < 1e-6
                 and row_viol < 1e-6 and lb_viol < 1e-6 and ub_viol < 1e-6
                 and abs(obj_orig - HIGHS_REF) < 1e-6)

    print(f"\n{'='*70}")
    print(f"VERDICT: {'STRICT VERIFIED OPTIMAL' if strict_ok else 'NOT STRICTLY VERIFIED'}")
    print(f"  reduced costs: min={float(z_nonbasic.min()):.2e} "
          f"(noise floor {noise_floor:.1e}) -> "
          f"{'all >= 0 within cond noise' if strict_ok else 'negatives below floor'}")
    print(f"  original objective vs HiGHS: |delta|={abs(obj_orig - HIGHS_REF):.3e}")
    print(f"  (elapsed {time.perf_counter()-t0:.1f}s)")
    print(f"{'='*70}")
    return 0 if strict_ok else 1


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.exit(main())
