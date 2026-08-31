"""Independent KKT certificate for the PILOT87 Devex Phase-II optimal solution.

This script does NOT trust any objective reported by the solver.  It loads the
final persisted solution, reconstructs A/b/c in *unscaled* standard form and in
the *original* LP variables, and verifies each KKT condition independently.

Usage:
    OPENBLAS_NUM_THREADS=1 python scratch/p87_certify.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import scipy.sparse as sp

_ROOT = r"c:\Users\anshp\OneDrive\SIH26119"
SCRATCH = os.path.join(_ROOT, "artifacts", "pilot87")
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"),
           os.path.join(_ROOT, "experiment", "crossover")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import load_numeric_mps
from mehrotra import to_standard_form

HIGHS_REF = 301.710347333


def inf_norm(v):
    v = np.asarray(v, float)
    return float(np.max(np.abs(v))) if v.size else 0.0


def report(title, ok, **vals):
    print(f"[{'PASS' if ok else 'FAIL'}] {title}")
    for k, v in vals.items():
        print(f"      {k} = {v}")


def main():
    t0 = time.perf_counter()
    # ------------------------------------------------------------------ data
    raw = np.load(os.path.join(SCRATCH, "p87_prepared.npz"), allow_pickle=False)
    A_sc = sp.load_npz(os.path.join(SCRATCH, "p87_prepared_A.npz")).tocsc()
    b_sc = np.asarray(raw["b"], float)
    c_sc = np.asarray(raw["c"], float)
    row_scale = np.asarray(raw["row_scale"], float)
    col_scale = np.asarray(raw["col_scale"], float)
    m, n = A_sc.shape

    fin = np.load(os.path.join(SCRATCH, "p87_phase2_v2_final.npz"),
                  allow_pickle=False)
    basis = np.asarray(fin["basis"], dtype=np.intp)
    x_basic = np.asarray(fin["x_basic"], float)
    y_sc = np.asarray(fin["y"], float)
    x_sc = np.asarray(fin["x_full"], float)
    obj_solver = float(fin["obj_scaled"])
    it = int(fin["it"])
    print(f"final: it={it}  solver obj(scaled std-form)={obj_solver:.9f}  "
          f"basis={basis.size}")
    print(f"scaling row[{row_scale.min():.3e}..{row_scale.max():.3e}] "
          f"col[{col_scale.min():.3e}..{col_scale.max():.3e}]")

    # ================================================================ 1.
    print("\n===== [1] SCALED STANDARD-FORM PRIMAL ====")
    resid_primal = inf_norm(A_sc @ x_sc - b_sc)
    print(f"  ||A x - b||_inf (scaled) = {resid_primal:.3e}")
    xmin = float(x_sc.min()) if x_sc.size else 0.0
    print(f"  x >= 0 ?  min(x)={xmin:.3e}")

    print("\n===== [2] SCALED DUAL & REDUCED COSTS ====")
    z_all = c_sc - A_sc.T @ y_sc
    z_basic = z_all[basis]
    z_nonbasic = z_all[np.setdiff1d(np.arange(n), basis, assume_unique=True)]
    print(f"  max|z_basic| = {inf_norm(z_basic):.3e}")
    print(f"  min z over nonbasic = {float(z_nonbasic.min()):.3e}")
    dual_res = inf_norm(A_sc.T @ y_sc + z_all - c_sc)
    print(f"  dual residual ||A^T y + z - c||_inf = {dual_res:.3e}")

    print("\n===== [3] SCALED COMPLEMENTARITY ====")
    comp = float(np.dot(x_sc, z_all))
    print(f"  x'z  = {comp:.3e}   min z = {float(z_all.min()):.3e}")

    print("\n===== [4] SCALED BASIS ====")
    B = A_sc[:, basis].tocsc()
    bres = inf_norm(B @ x_basic - b_sc)
    condB = np.nan
    try:
        condB = np.linalg.cond(B.toarray())
    except Exception:
        pass
    print(f"  B x_B - b residual = {bres:.3e}   cond(B) = {condB:.3e}")

    # ================================================================ 5.
    print("\n===== [5] UNSCALED STANDARD-FORM RECONSTRUCTION ====")
    inv_col = 1.0 / col_scale
    inv_row = 1.0 / row_scale
    A0 = (sp.diags(inv_row) @ A_sc @ sp.diags(inv_col)).tocsc()
    b0 = inv_row * b_sc
    c0 = inv_col * c_sc
    x0 = col_scale * x_sc
    y0 = row_scale * y_sc          # dual: y0 = R y_sc
    z0 = inv_col * z_all           # z0 = C^-1 z_sc
    pres0 = inf_norm(A0 @ x0 - b0)
    dres0 = inf_norm(A0.T @ y0 + z0 - c0)
    comp0 = float(np.dot(x0, z0))
    obj_std = float(np.dot(c0, x0))
    dual_bound = float(np.dot(b0, y0))
    gap = obj_std - dual_bound
    print(f"  ||A0 x0 - b0||_inf = {pres0:.3e}")
    print(f"  ||A0^T y0 + z0 - c0||_inf = {dres0:.3e}")
    print(f"  x0'z0 = {comp0:.3e}   min z0 = {float(z0.min()):.3e}")
    print(f"  c0'x0 (unscaled std-form objective) = {obj_std:.9f}")
    print(f"  |obj_std - solver_obj| = {abs(obj_std - obj_solver):.3e}")
    print(f"  std-form dual bound b0'y0 = {dual_bound:.9f}   gap = {gap:.3e}")

    # ================================================================ 6.
    print("\n===== [6] ORIGINAL-LP RECONSTRUCTION ====")
    lp = load_numeric_mps(os.path.join(_ROOT, "data", "pilot87.mps"))
    sf = to_standard_form(lp)
    c_orig = np.asarray(sf.c_orig, float)
    orig_offset = np.asarray(sf.orig_offset, float)
    x_orig = sf.recover_original(x0)          # standard-form -> original vars
    obj_orig = float(np.dot(c_orig, x_orig))  # INDEPENDENT original objective
    const_off = float(np.dot(c_orig, orig_offset))
    print(f"  n_orig = {sf.n_orig}")
    print(f"  c_orig @ offset (std-form constant) = {const_off:.9f}")
    print(f"  c_orig @ x_orig (ORIGINAL objective) = {obj_orig:.9f}")
    print(f"  |obj_orig - (obj_std + const_off)| = "
          f"{abs(obj_orig - (obj_std + const_off)):.3e}")
    print(f"  HiGHS reference = {HIGHS_REF:.9f}")
    print(f"  |obj_orig - HiGHS| = {abs(obj_orig - HIGHS_REF):.3e}")

    # original primal feasibility in original variables
    A_orig = np.asarray(lp.A, float)
    b_orig = np.asarray(lp.b, float)
    rt = list(lp.row_types)
    le = [i for i, r in enumerate(rt) if r == "L"]
    ge = [i for i, r in enumerate(rt) if r == "G"]
    eq = [i for i, r in enumerate(rt) if r == "E"]
    viol = []
    if eq:
        viol.append(float(np.abs(A_orig[eq] @ x_orig - b_orig[eq]).max()))
    if le:
        sl = (A_orig[le] @ x_orig) - b_orig[le]
        viol.append(float(max(0.0, sl.max())))   # L-row: A x <= b
    if ge:
        sg = b_orig[ge] - (A_orig[ge] @ x_orig)
        viol.append(float(max(0.0, sg.max())))   # G-row: A x >= b
    lb = np.asarray(lp.lower_bounds, float)
    ub = np.asarray(lp.upper_bounds, float)
    lb_viol = inf_norm(np.minimum(0.0, x_orig - lb))
    ub_viol = inf_norm(np.maximum(0.0, x_orig - ub))
    row_viol = max(viol) if viol else 0.0
    print(f"  orig E-row residual  = {viol[0]:.3e} (must be ~0)" if eq else
          "  (no equality rows)")
    print(f"  orig ineq-row violation (L:A x<=b, G:A x>=b) = {row_viol:.3e}")
    print(f"  orig bound violations: lb={lb_viol:.3e} ub={ub_viol:.3e}")

    # ------------------------------------------------------------------
    print("\n===== [7] CERTIFICATE SUMMARY ====")
    measures = dict(
        primal=pres0, dual=dres0, comp=abs(comp0), min_z=float(z0.min()),
        basis=bres, lbv=lb_viol, ubv=ub_viol, rowv=row_viol,
        gap=abs(gap),
        dv_highs=abs(obj_orig - HIGHS_REF), obj_orig=obj_orig, obj_std=obj_std,
    )
    tolP = 1e-6
    ok = (measures["primal"] <= tolP and measures["dual"] <= 1e-7
          and measures["comp"] <= 1e-6 and measures["min_z"] >= -1e-6
          and measures["basis"] <= 1e-6
          and measures["rowv"] <= 1e-6 and measures["lbv"] <= 1e-6
          and measures["ubv"] <= 1e-6)
    report("PRIMAL FEASIBLE (||Ax-b||_inf<=1e-6)", measures["primal"] <= tolP,
           **{"||Ax-b||": measures["primal"]})
    report("ORIGINAL-LP PRIMAL FEASIBLE",
           measures["rowv"] <= 1e-6 and measures["lbv"] <= 1e-6
           and measures["ubv"] <= 1e-6,
           **{"row_viol": measures["rowv"], "lb_viol": measures["lbv"],
              "ub_viol": measures["ubv"]})
    report("DUAL FEASIBLE (A^Ty+z=c, z>=-1e-6)",
           measures["dual"] <= 1e-7 and measures["min_z"] >= -1e-6,
           **{"||A^Ty+z-c||": measures["dual"], "min_z": measures["min_z"]})
    report("COMPLEMENTARITY (x^T z ~ 0)", measures["comp"] <= 1e-6,
           **{"x^T z": measures["comp"]})
    report("BASIS OK (B x_B = b)", measures["basis"] <= 1e-6,
           **{"Bx_B-b": measures["basis"]})
    print(f"\n  ORIGINAL OBJECTIVE (independently recomputed) = {obj_orig:.9f}")
    print(f"  HiGHS reference                              = {HIGHS_REF:.9f}")
    print(f"  |Delta obj| = {abs(obj_orig - HIGHS_REF):.3e}")
    print(f"\n  -> {'VERIFIED OPTIMAL' if ok else 'NOT fully verified'} "
          f"(elapsed {time.perf_counter()-t0:.1f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.exit(main())
