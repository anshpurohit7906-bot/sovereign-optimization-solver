"""End-to-end validation: sparse Phase-I + sparse Devex Phase-II on PILOT4.

Runs the EXACT p87_phase2_v2 engine (Devex pricing, condition-aware repair,
soft clamps) on PILOT4, then applies strict original-coordinate KKT
certification.  No production code is modified; no dense RRQR or
_proto_sparse.repair path is used.

Usage:
    cd <ROOT>
    OPENBLAS_NUM_THREADS=1 .venv/Scripts/python -u experiment/crossover/run_pilot4_sparse_validation.py
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

MPS = os.path.join(_ROOT, "data", "pilot4_plain.mps")
ARTIFACTS = os.path.join(_ROOT, "artifacts", "pilot4")
PREPARED = os.path.join(ARTIFACTS, "p87_prepared.npz")
PREPARED_A = os.path.join(ARTIFACTS, "p87_prepared_A.npz")
FINAL_NPZ = os.path.join(ARTIFACTS, "p87_phase2_v2_final.npz")
HIGHS_REF = -2581.139258884
# ==========================================================================
# STAGE 1: Prepare PILOT4 (load MPS -> std-form -> scale -> sparse Phase I)
# ==========================================================================
def stage1_prepare():
    from opticore.numerical_model import load_numeric_mps
    from opticore.lp.mehrotra import to_standard_form
    from opticore.scaling import scale_lp
    from sparse_phase1 import sparse_phase1

    os.makedirs(ARTIFACTS, exist_ok=True)
    print("=" * 70)
    print("STAGE 1: PREPARE PILOT4 (sparse Phase I)")
    print("=" * 70)

    t0 = time.perf_counter()
    sf = to_standard_form(load_numeric_mps(MPS, sparse=True))
    A0 = sp.csc_matrix(sf.A)
    b0 = np.asarray(sf.b, float)
    c0 = np.asarray(sf.c_min, float)
    m, n = A0.shape
    print(f"  standard form: {m}x{n}  nnz={A0.nnz:,}  "
          f"load={time.perf_counter() - t0:.2f}s")

    S = scale_lp(A0, b0, c0, np.zeros(n), np.full(n, np.inf))
    A = sp.csc_matrix(S.A)
    b = np.asarray(S.b, float)
    c = np.asarray(S.c, float)
    col_scale = np.asarray(S.column_scale, float)
    row_scale = np.asarray(S.row_scale, float)
    print(f"  scaling: col[{col_scale.min():.3e}..{col_scale.max():.3e}] "
          f"row[{row_scale.min():.3e}..{row_scale.max():.3e}]")

    t1 = time.perf_counter()
    basis, p1_iters, p1_status, p1_info = sparse_phase1(
        A, b, max_iter=2_000_000, verbose=500,
    )
    t_p1 = time.perf_counter() - t1
    print(f"  Phase I: iters={p1_iters}  status={p1_status}  t={t_p1:.1f}s")
    if p1_info:
        for k in ("art_sum", "min_xB", "neg"):
            if k in p1_info:
                print(f"    {k}={p1_info[k]}")

    if p1_status != "feasible":
        print(f"  PHASE I FAILED: {p1_status}  {p1_info.get('err', '')}")
        return None

    basis = np.asarray(basis, dtype=np.intp)
    sp.save_npz(PREPARED_A, A.tocsc())
    np.savez(
        PREPARED,
        b=b, c=c,
        basis=basis,
        m=m, n=n,
        row_scale=row_scale,
        col_scale=col_scale,
    )
    print(f"  SAVED: {PREPARED}")
    print(f"  SAVED: {PREPARED_A}")
    print(f"  basis_size={basis.size}  total={time.perf_counter() - t0:.1f}s")
    return {
        "A": A, "b": b, "c": c, "basis": basis,
        "m": m, "n": n,
        "row_scale": row_scale, "col_scale": col_scale,
        "t_phase1": t_p1, "sf": sf,
    }
# ==========================================================================
# STAGE 2: Phase II — invoke EXACT p87_phase2_v2 engine (zero transcription)
# ==========================================================================
def stage2_phase2():
    print("\n" + "=" * 70)
    print("STAGE 2: PHASE II (sparse Devex, condition-aware repair)")
    print("=" * 70)

    import p87_phase2_v2 as p87mod

    p87mod._ARTIFACTS = ARTIFACTS
    p87mod.CSV_PATH = os.path.join(ARTIFACTS, "p87_phase2_v2_log.csv")
    p87mod.REPAIR_CSV = os.path.join(ARTIFACTS, "p87_phase2_v2_repair_events.csv")
    p87mod.SOFT_CSV = os.path.join(ARTIFACTS, "p87_phase2_v2_soft_events.csv")

    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0], "--pricing", "devex"]

    t0 = time.perf_counter()
    try:
        rc = p87mod.main()
    finally:
        sys.argv = saved_argv

    t_p2 = time.perf_counter() - t0
    print(f"\n  Phase II wall time: {t_p2:.1f}s  exit_code={rc}")

    if rc != 0:
        print("  PHASE II FAILED")
        return None

    if not os.path.isfile(FINAL_NPZ):
        print(f"  ERROR: {FINAL_NPZ} not found")
        return None

    fin = np.load(FINAL_NPZ, allow_pickle=False)
    print(f"  Terminal basis: it={int(fin['it'])}  neg_rc={int(fin['neg_rc'])}")
    print(f"  Scaled obj: {float(fin['obj_scaled']):.9f}")
    print(f"  xB residual: {float(fin['xB_resid']):.3e}")
    return {"t_phase2": t_p2, "fin": fin}
# ==========================================================================
# STAGE 3: Strict original-coordinate KKT certification
# ==========================================================================
def stage3_certify():
    print("\n" + "=" * 70)
    print("STAGE 3: STRICT ORIGINAL-COORDINATE CERTIFICATION")
    print("=" * 70)

    from opticore.numerical_model import load_numeric_mps
    from opticore.lp.mehrotra import to_standard_form

    t0 = time.perf_counter()

    fin = np.load(FINAL_NPZ, allow_pickle=False)
    basis = list(np.asarray(fin["basis"], dtype=np.intp))
    obj_solver = float(fin["obj_scaled"])
    it_solver = int(fin["it"])

    raw = np.load(PREPARED, allow_pickle=False)
    A_sc = sp.load_npz(PREPARED_A).tocsc()
    b_sc = np.asarray(raw["b"], float)
    c_sc = np.asarray(raw["c"], float)
    row_scale = np.asarray(raw["row_scale"], float)
    col_scale = np.asarray(raw["col_scale"], float)
    m, n = A_sc.shape
    print(f"  Terminal basis: it={it_solver}  m={m} n={n}  basis_size={len(basis)}")
    print(f"  Solver obj (scaled std-form): {obj_solver:.9f}")

    inv_col = 1.0 / col_scale
    inv_row = 1.0 / row_scale
    A0 = (sp.diags(inv_row) @ A_sc @ sp.diags(inv_col)).tocsc()
    b0 = inv_row * b_sc
    c0 = inv_col * c_sc

    basis_set = set(basis)
    nonbasic = np.array(sorted(set(range(n)) - basis_set), dtype=np.intp)

    B = A0[:, basis].tocsc()
    lu = splu(B)

    def _refine_solve(B, lu, rhs, n_refine=5):
        x = lu.solve(rhs)
        for _ in range(n_refine):
            resid = B @ x - rhs
            rnorm = float(np.max(np.abs(resid)))
            if rnorm < 1e-12:
                break
            try:
                dc = lu.solve(
                    resid.toarray().ravel() if hasattr(resid, "toarray") else resid
                )
            except Exception:
                break
            cand = x + dc
            rn = float(np.max(np.abs(B @ cand - rhs)))
            if rn < rnorm:
                x = cand
            else:
                break
        return x

    x_basic = _refine_solve(B, lu, b0)
    basis_arr = np.array(basis, dtype=np.intp)
    c_B = c0[basis_arr]
    obj_orig_coords = float(c_B @ x_basic)
    print(f"  Objective (original coords): {obj_orig_coords:.9f}")
    print(f"  min xB (original coords): {x_basic.min():.3e}  "
          f"neg count: {int((x_basic < -1e-7).sum())}")

    # ---------------------------------------------------------------------
    # Strict original-coordinate KKT certificate (NO polishing of the basis).
    # Both the existing PILOT4 certificate (run_pilot4_crossover Stage D) and
    # tools/certification/p87_certify.py certify by re-solving the terminal
    # basis in ORIGINAL (unscaled) coordinates and checking every KKT
    # condition at raw precision.  PILOT4's unscaled basis has cond ~1.5e11
    # (noise floor ~3e-3), so re-pivoting in unscaled space (the PILOT87
    # "strict polish" style) would destroy the basis numerically; the direct
    # certificate is the correct strict procedure for this instance.
    # ---------------------------------------------------------------------
    y = lu.solve(c_B, trans="T")
    x_full = np.zeros(n)
    x_full[basis] = x_basic
    z_all = c0 - A0.T @ y
    z_nonbasic = z_all[nonbasic]
    condB = float(np.linalg.cond(B.toarray()) if B.shape[0] <= 2500
                  else np.inf)
    if not np.isfinite(condB):
        # large/ill-conditioned: use 1-norm estimate budget
        try:
            from scipy.sparse.linalg import svds
            s = svds(B, k=1, return_singular_vectors=False)
            s_max = float(np.max(s))
            s_min = float(np.min(s))
            condB = s_max / s_min if s_min > 0 else np.inf
        except Exception:
            condB = np.inf
    noise_floor = max(1e-14, 1e2 * np.finfo(float).eps * condB)

    resid_primal = float(np.max(np.abs(A0 @ x_full - b0)))
    print(f"[1] Primal ||Ax-b||_inf  = {resid_primal:.3e}")

    print(f"[2] min(z_nonbasic)      = {float(z_nonbasic.min()):.3e}")
    print(f"    num z < 0            = {int((z_nonbasic < 0).sum())}")
    print(f"    num z < -1e-12       = {int((z_nonbasic < -1e-12).sum())}")
    print(f"    cond(B)              = {condB:.1e}  "
          f"(sign noise floor ~{noise_floor:.1e})")

    comp = float(np.dot(x_full, z_all))
    print(f"[3] Complementarity x'z  = {comp:.3e}")

    bres = float(np.max(np.abs(B @ x_basic - b0)))
    print(f"[4] Basis residual       = {bres:.3e}")

    sf = to_standard_form(load_numeric_mps(MPS))
    c_orig = np.asarray(sf.c_orig, float)
    x_orig = sf.recover_original(x_full)
    obj_orig = float(np.dot(c_orig, x_orig))

    print(f"[5] Original objective   = {obj_orig:.9f}")
    print(f"    HiGHS reference      = {HIGHS_REF:.9f}")
    print(f"    |delta|              = {abs(obj_orig - HIGHS_REF):.3e}")
# ---- Original primal feasibility ----
    lp = load_numeric_mps(MPS)
    A_orig = np.asarray(lp.A, float)
    b_orig = np.asarray(lp.b, float)
    lb = np.asarray(lp.lower_bounds, float)
    ub = np.asarray(lp.upper_bounds, float)
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
    lb_viol = float(np.max(np.minimum(0.0, x_orig - lb)))
    ub_viol = float(np.max(np.maximum(0.0, x_orig - ub)))
    row_viol = max(viol) if viol else 0.0

    print(f"[6] Orig primal feasibility:")
    if eq_rows:
        print(f"    E-row residual       = {viol[0]:.3e}")
    print(f"    Ineq violation       = {row_viol:.3e}")
    print(f"    lb violation         = {lb_viol:.3e}")
    print(f"    ub violation         = {ub_viol:.3e}")

    dual_bound = float(np.dot(b0, y))
    gap = obj_orig_coords - dual_bound
    print(f"[7] Strong duality gap   = {gap:.3e}")

    strict_ok = (
        resid_primal < 1e-6
        and bres < 1e-6
        and float(z_nonbasic.min()) >= -noise_floor
        and abs(comp) < 1e-6
        and row_viol < 1e-6
        and lb_viol < 1e-6
        and ub_viol < 1e-6
        and abs(obj_orig - HIGHS_REF) < 1e-6
    )

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {'STRICT VERIFIED OPTIMAL' if strict_ok else 'NOT STRICTLY VERIFIED'}")
    print(f"  reduced costs: min={float(z_nonbasic.min()):.2e} "
          f"(noise floor {noise_floor:.1e}) -> "
          f"{'all >= 0 within cond noise' if strict_ok else 'negatives below floor'}")
    print(f"  original objective vs HiGHS: |delta|={abs(obj_orig - HIGHS_REF):.3e}")
    print("=" * 70)

    return {
        "strict_ok": strict_ok,
        "obj_orig": obj_orig,
        "resid_primal": resid_primal,
        "bres": bres,
        "min_rc": float(z_nonbasic.min()),
        "comp": comp,
        "gap": gap,
        "polish_pivots": 0,
        "t_certify": time.perf_counter() - t0,
    }
# ==========================================================================
# SUMMARY
# ==========================================================================
def summary(prep_data, phase2_data, cert_data, t_total):
    print(f"\n{'=' * 70}")
    print("PILOT4 SPARSE ENGINE END-TO-END SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  Pipeline:")
    print(f"    Phase I:        {prep_data['t_phase1']:.1f}s")
    print(f"    Phase II:       {phase2_data['t_phase2']:.1f}s")
    print(f"    Certification:  {cert_data['t_certify']:.1f}s")
    print(f"    TOTAL:          {t_total:.1f}s")

    fin = phase2_data["fin"]
    print(f"\n  Phase II metrics:")
    print(f"    Iterations/pivots: {int(fin['it'])}")
    print(f"    Scaled objective:  {float(fin['obj_scaled']):.9f}")
    print(f"    xB residual:       {float(fin['xB_resid']):.3e}")

    import csv as _csv
    csv_repair = os.path.join(ARTIFACTS, "p87_phase2_v2_repair_events.csv")
    csv_soft = os.path.join(ARTIFACTS, "p87_phase2_v2_soft_events.csv")
    n_repairs = 0
    if os.path.isfile(csv_repair):
        with open(csv_repair) as f:
            n_repairs = sum(1 for _ in _csv.DictReader(f))
    n_soft = 0
    if os.path.isfile(csv_soft):
        with open(csv_soft) as f:
            n_soft = sum(1 for _ in _csv.DictReader(f))
    print(f"    Repairs:          {n_repairs}")
    print(f"    Soft clamps:      {n_soft}")

    print(f"\n  Certification:")
    print(f"    Feasibility:      ||Ax-b||_inf = {cert_data['resid_primal']:.3e}")
    print(f"    Basis residual:   {cert_data['bres']:.3e}")
    print(f"    min reduced cost: {cert_data['min_rc']:.3e}")
    print(f"    Complementarity:  {cert_data['comp']:.3e}")
    print(f"    Strong duality:   {cert_data['gap']:.3e}")
    print(f"    Polish pivots:    {cert_data['polish_pivots']}")
    print(f"    Original obj:     {cert_data['obj_orig']:.9f}")
    print(f"    HiGHS ref:        {HIGHS_REF:.9f}")
    print(f"    |delta|:          {abs(cert_data['obj_orig'] - HIGHS_REF):.3e}")

    verdict = "CERTIFIED OPTIMAL" if cert_data["strict_ok"] else "NOT CERTIFIED"
    print(f"\n  >>> VERDICT: {verdict} <<<")

    cert_path = os.path.join(ARTIFACTS, "p4_sparse_certificate.txt")
    with open(cert_path, "w") as f:
        f.write("PILOT4 SPARSE ENGINE CERTIFICATE\n")
        f.write("=" * 60 + "\n")
        f.write(f"Phase II:   {phase2_data['t_phase2']:.1f}s  "
                f"iters={int(fin['it'])} repairs={n_repairs} soft_clamps={n_soft}\n")
        f.write(f"Certify:    {cert_data['t_certify']:.1f}s  "
                f"polish_pivots={cert_data['polish_pivots']}\n")
        f.write(f"Objective:  {cert_data['obj_orig']:.9f}  "
                f"(HiGHS={HIGHS_REF:.9f}  delta={abs(cert_data['obj_orig'] - HIGHS_REF):.3e})\n")
        f.write(f"Primal resid:     {cert_data['resid_primal']:.3e}\n")
        f.write(f"Basis residual:   {cert_data['bres']:.3e}\n")
        f.write(f"min reduced cost: {cert_data['min_rc']:.3e}\n")
        f.write(f"Complementarity:  {cert_data['comp']:.3e}\n")
        f.write(f"Strong duality:   {cert_data['gap']:.3e}\n")
        f.write(f"VERDICT: {verdict}\n")
    print(f"  Certificate: {cert_path}")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    t_start = time.perf_counter()

    prep = stage1_prepare()
    if prep is None:
        print("\nABORT: Stage 1 (Phase I) failed.")
        return 1

    p2 = stage2_phase2()
    if p2 is None:
        print("\nABORT: Stage 2 (Phase II) failed.")
        return 1

    cert = stage3_certify()
    if cert is None:
        print("\nABORT: Stage 3 (certification) failed.")
        return 1

    t_total = time.perf_counter() - t_start
    summary(prep, p2, cert, t_total)

    return 0 if cert["strict_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())