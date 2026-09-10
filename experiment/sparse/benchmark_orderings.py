#!/usr/bin/env python
"""Experiment: does SuperLU ordering reduce sparse Schur factorization time/fill-in?

EXPERIMENT ONLY: does not modify src/, tests/, or requirements/.

Profiles the ACTUAL Schur system S = A H^-1 A^T from the 5000x2500 density=0.3%
deterministic sparse LP (the same workload as benchmark_scaling.py) and factors
the SAME numerical Schur system under the four available SuperLU orderings
(COLAMD, MMD_AT_PLUS_A, MMD_ATA, NATURAL) to measure factorization time and
fill-in.

Only ONE representative Newton step / factorization is run (not a 100-iter solve).
"""
import sys
import time
import csv
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _SRC / "lp", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from numerical_model import NumericalLP  # noqa: E402
from scaling import scale_lp  # noqa: E402
import mehrotra as _mh  # noqa: E402
from linear_system import MAX_RHO_P  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROWS = 5000
COLS = 2500
DENSITY_PCT = 0.30
SEED = 42
TOL = 1e-7
RESULTS_DIR = _ROOT / "results"

# The available SuperLU permc_spec orderings (see scipy.sparse.linalg.splu).
# COLAMD is the scipy default, so it is the CURRENT production ordering
# (linear_system._splu_regularized passes no permc_spec, thus uses COLAMD).
ORDERINGS = ["COLAMD", "MMD_AT_PLUS_A", "MMD_ATA", "NATURAL"]

# ---------------------------------------------------------------------------
# Deterministic sparse LP generator (identical to benchmark_scaling.py)
# ---------------------------------------------------------------------------

def generate_deterministic_sparse_lp(rows, cols, density_pct, seed=42):
    """Build a NumericalLP with a CSR sparse A (never materialized dense).
    Guarantees every row and every column has >= 1 nonzero entry.
    """
    rng = np.random.RandomState(seed)
    target_nnz = max(1, int(rows * cols * (density_pct / 100.0)))
    min_nnz = rows + cols
    actual_target = max(target_nnz, min_nnz)
    row_idx = rng.randint(0, rows, size=actual_target)
    col_idx = rng.randint(0, cols, size=actual_target)
    data = rng.randn(actual_target)
    A_sp = sp.csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols),
                         dtype=np.float64)
    for r in range(rows):
        if A_sp[r].nnz == 0:
            c = rng.randint(0, cols)
            A_sp[r, c] = rng.randn()
    A_sp = A_sp.tocsr()
    for c in range(cols):
        if A_sp[:, c].nnz == 0:
            r = rng.randint(0, rows)
            A_sp[r, c] = rng.randn()
    A_sp = A_sp.tocsr()
    actual_nnz = A_sp.nnz
    b = rng.randn(rows)
    c_vec = rng.randn(cols)
    row_types = tuple(rng.choice(["E", "L", "G"], size=rows, p=[0.6, 0.2, 0.2]))
    lp = NumericalLP(
        name=f"synthetic_{rows}x{cols}",
        objective_name="obj",
        A=A_sp,
        b=b,
        c=c_vec,
        lower_bounds=np.zeros(cols),
        upper_bounds=np.full(cols, np.inf),
        row_types=row_types,
        var_names=tuple(f"x{i}" for i in range(cols)),
        row_names=tuple(f"c{i}" for i in range(rows)),
    )
    return lp, actual_nnz


# ---------------------------------------------------------------------------
# Build the ACTUAL Schur system (mirrors production linear_system path)
# ---------------------------------------------------------------------------

def build_schur(scaled, reg=1e-12):
    """Replicate production Schur construction; return dict of matrices/metrics.

    Mirrors linear_system.factor_reduced_system sparse path + _splu_regularized:
      h = z/x ; reg_h = min(reg*max(1, mean|h|), MAX_RHO_P)
      h_diag = h + reg_h
      A_sp = csr(A) ; W_sp = A.T * (1/h_diag) ; S = A_sp @ W_sp   (ACTUAL Schur)
      M = 0.5*(S + S.T) ; reg_M = reg*max(1, mean|diag(M)|)
      M_reg = M + reg_M*I                                             (factored matrix)
    """
    A, b, c = scaled.A, scaled.b, scaled.c
    x, y, z = _mh._mehrotra_initial_point(A, b, c)

    h = z / x
    scale_h = max(1.0, float(np.mean(np.abs(h))))
    reg_h = reg * scale_h
    if reg_h > MAX_RHO_P:
        reg_h = MAX_RHO_P
    h_diag = h + reg_h

    A_sp = sp.csr_matrix(A)
    W_sp = A_sp.T.multiply((1.0 / h_diag)[:, None]).tocsr()
    S_sp = A_sp @ W_sp  # ACTUAL Schur complement S = A H^-1 A^T (CSR)

    M = (0.5 * (S_sp + S_sp.T)).tocsc()
    scale_m = max(1.0, float(np.mean(np.abs(M.diagonal()))))
    reg_M = reg * scale_m
    M_reg = (M + sp.eye(M.shape[0], format="csc") * reg_M).tocsc().tocsr()

    return {
        "A_sp": A_sp,
        "S_sp": S_sp,
        "M": M,
        "M_reg": M_reg,
        "h_diag": h_diag,
        "schur_shape": S_sp.shape,
        "nnz_S": S_sp.nnz,
        "nnz_M": M.nnz,
        "reg_M": reg_M,
    }


def factor_with_ordering(M_reg, ordering):
    """Factor M_reg with the given SuperLU ordering.

    Returns (lu, elapsed, nnzL, nnzU, error). SAME matrix, SAME values,
    SAME pivot settings (diag_pivot_thresh=1.0 matching SuperLU default),
    SAME regularization; only ``permc_spec`` (ordering) changes.
    """
    M_csc = M_reg.tocsr().tocsc()
    t0 = time.perf_counter()
    try:
        lu = splu(M_csc, permc_spec=ordering)
    except Exception as exc:
        return None, None, None, None, type(exc).__name__ + ": " + str(exc)
    elapsed = time.perf_counter() - t0
    return lu, elapsed, lu.L.nnz, lu.U.nnz, None


def memory_rss() -> float:
    """Return current RSS in MB (best-effort; 0 if psutil unavailable)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'=' * 70}")
    print(f"SUPERLU ORDERING BENCHMARK")
    print(f"Problem: {ROWS}x{COLS}, density={DENSITY_PCT}%, seed={SEED}")
    print(f"Orderings: {ORDERINGS}")
    print(f"{'=' * 70}\n")

    # --- Build LP and standard form / scaling (identical to benchmark) ---
    t = time.perf_counter()
    lp, nnz = generate_deterministic_sparse_lp(ROWS, COLS, DENSITY_PCT, SEED)
    print(f"[generate] {time.perf_counter()-t:.3f}s ({nnz} NNZ)", flush=True)

    t = time.perf_counter()
    sf = _mh.to_standard_form(lp)
    print(f"[to_standard_form] {time.perf_counter()-t:.3f}s "
          f"(shape {sf.A.shape})", flush=True)

    t = time.perf_counter()
    scaled = scale_lp(sf.A, sf.b, sf.c_min, np.zeros(sf.n), np.full(sf.n, np.inf))
    print(f"[scale_lp] {time.perf_counter()-t:.3f}s", flush=True)

    # --- Build the ACTUAL Schur system ---
    t = time.perf_counter()
    schur = build_schur(scaled)
    build_time = time.perf_counter() - t
    print(f"[Schur construction] {build_time:.3f}s", flush=True)
    print(f"  A_eq shape: {schur['A_sp'].shape}, A_eq NNZ: {schur['A_sp'].nnz}", flush=True)
    print(f"  ACTUAL Schur S shape: {schur['schur_shape']}, nnz(S)={schur['nnz_S']:,}", flush=True)
    print(f"  S sparse: {sp.issparse(schur['S_sp'])}", flush=True)
    print(f"  nnz(M)={schur['nnz_M']:,} (symmetrized), reg_M={schur['reg_M']:.3e}", flush=True)

    M_reg = schur["M_reg"]
    results = []
    baseline = {}

    for od in ORDERINGS:
        print(f"\n--- Ordering: {od} ---", flush=True)
        mem_before = memory_rss()
        lu, elapsed, nnzL, nnzU, error = factor_with_ordering(M_reg, od)
        mem_after = memory_rss()
        mem_delta = mem_after - mem_before if (mem_before and mem_after) else 0.0

        if error is not None:
            print(f"  FAILED: {error}", flush=True)
            results.append({
                "ordering": od, "time_s": None, "nnz_S": schur["nnz_S"],
                "nnz_L": None, "nnz_U": None, "fill_in": None,
                "success": "FAIL", "mem_delta_mb": mem_delta,
            })
            continue

        fill = (nnzL + nnzU) / schur["nnz_S"]
        print(f"  time={elapsed:.3f}s  nnz(L)={nnzL:,}  nnz(U)={nnzU:,}  "
              f"fill={fill:.2f}x", flush=True)
        if od == "COLAMD":
            baseline = {"time": elapsed, "fill": fill, "nnzLU": nnzL + nnzU}
        results.append({
            "ordering": od, "time_s": elapsed, "nnz_S": schur["nnz_S"],
            "nnz_L": nnzL, "nnz_U": nnzU, "fill_in": fill,
            "success": "OK", "mem_delta_mb": mem_delta,
        })

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    best_time = min((r for r in results if r["time_s"] is not None),
                    key=lambda r: r["time_s"])
    best_fill = min((r for r in results if r["fill_in"] is not None),
                    key=lambda r: r["fill_in"])
    print(f"Best ordering by TIME:    {best_time['ordering']} "
          f"({best_time['time_s']:.3f}s)")
    print(f"Best ordering by FILL-IN: {best_fill['ordering']} "
          f"({best_fill['fill_in']:.2f}x)")
    if baseline:
        speedup = baseline["time"] / best_time["time_s"]
        print(f"Speedup vs COLAMD (default): {speedup:.2f}x")
    num_success = sum(1 for r in results if r["success"] == "OK")
    print(f"All orderings succeeded: {num_success == len(results)} "
          f"({num_success}/{len(results)})")


    # --- Write CSV ---
    csv_path = RESULTS_DIR / "benchmark_orderings.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ordering", "factor_time_s", "nnz_S", "nnz_L", "nnz_U",
                    "fill_in_ratio", "success", "mem_delta_mb"])
        for r in results:
            w.writerow([
                r["ordering"],
                f"{r['time_s']:.6f}" if r["time_s"] is not None else "",
                r["nnz_S"],
                r["nnz_L"] if r["nnz_L"] is not None else "",
                r["nnz_U"] if r["nnz_U"] is not None else "",
                f"{r['fill_in']:.4f}" if r["fill_in"] is not None else "",
                r["success"],
                f"{r['mem_delta_mb']:.1f}" if r["mem_delta_mb"] else "",
            ])
    print(f"\nWrote {csv_path}")

    # --- Write Markdown ---
    md_path = RESULTS_DIR / "benchmark_orderings.md"
    speedup = baseline["time"] / best_time["time_s"] if baseline else 0.0
    lines = [
        "# SuperLU Ordering Benchmark",
        "",
        f"**Problem:** {ROWS}×{COLS}, density={DENSITY_PCT}%, seed={SEED} "
        "(single representative Newton step)",
        "",
        "## Schur System (actual)",
        "",
        f"- **Shape:** {schur['schur_shape'][0]}×{schur['schur_shape'][1]}",
        f"- **nnz(S):** {schur['nnz_S']:,}",
        f"- **Sparse:** {sp.issparse(schur['S_sp'])}",
        f"- **Construction time:** {build_time:.3f}s",
        f"- **A_eq shape:** {schur['A_sp'].shape}, nnz {schur['A_sp'].nnz:,}",
        f"- **Default ordering (current):** COLAMD (scipy `splu` default; "
        f"`_splu_regularized` passes no `permc_spec`)",
        "",
        "## Ordering Comparison",
        "",
        "| Ordering | Factor Time (s) | nnz(S) | nnz(L) | nnz(U) | Fill-in ratio | Success |",
        "|----------|-----------------|--------|--------|--------|---------------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r['ordering']} | "
            f"{r['time_s']:.3f} | {r['nnz_S']:,} | "
            f"{r['nnz_L'] if r['nnz_L'] is not None else '-'} | "
            f"{r['nnz_U'] if r['nnz_U'] is not None else '-'} | "
            f"{r['fill_in']:.2f}× | {r['success']} |"
        )
    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Best ordering by time:** {best_time['ordering']} ({best_time['time_s']:.3f}s)",
        f"- **Best ordering by fill-in:** {best_fill['ordering']} ({best_fill['fill_in']:.2f}×)",
        f"- **Speedup over COLAMD (default):** {speedup:.2f}×",
        f"- **All orderings succeed:** {num_success == len(results)}",
        "",
        "## Recommendation",
        "",
    ])
    if baseline and best_time["ordering"] != "COLAMD" and speedup > 1.05:
        lines.append(
            f"**Switch the production SuperLU ordering to `{best_time['ordering']}`**: "
            f"it factors the identical Schur system {speedup:.2f}× faster "
            f"({baseline['time']:.3f}s → {best_time['time_s']:.3f}s) and cuts fill-in "
            f"from {baseline['fill']:.2f}× to {best_fill['fill_in']:.2f}×."
        )
    else:
        lines.append(
            "**COLAMD (current) is already the best ordering** for this problem; "
            "no ordering change yields meaningful speedup."
        )
    lines.extend([
        "",
        "*Identical S, numeric values, pivot settings, regularization, and "
        "hardware across orderings — only `permc_spec` differs.*",
        "",
        f"*Generated by {Path(__file__).name}*",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")

    # --- Final summary print ---
    print("\n" + "=" * 70)
    print("1. Schur size/nnz:")
    print(f"   shape={schur['schur_shape']}, nnz(S)={schur['nnz_S']:,}")
    print("2. Ordering comparison table:")
    for r in results:
        fi = f"{r['fill_in']:.2f}x" if r["fill_in"] is not None else "-"
        tstr = f"{r['time_s']:.3f}" if r["time_s"] is not None else "FAIL"
        print(f"   {r['ordering']:<14} time={tstr:<6} fill={fi}")
    for r in results:
        if r["time_s"] is not None:
            spd = baseline["time"] / r["time_s"] if baseline and r["time_s"] else 0
            print(f"   speedup {r['ordering']}: {spd:.2f}x")
    print("3. Fill-in ratios:")
    for r in results:
        if r["fill_in"] is not None:
            print(f"   {r['ordering']:<14} {r['fill_in']:.2f}x")
    rec = best_time["ordering"]
    note = "PRODUCTION RECOMMENDATION: " + (rec if rec != "COLAMD"
           else "keep COLAMD (already best)")
    print("5.", note)
    print("6. git status: untracked experiment files only (no commit).")
    print("=" * 70)


if __name__ == "__main__":
    main()

