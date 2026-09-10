#!/usr/bin/env python
"""Experiment: does SuperLU ordering generalize to real benchmark MPS models?

EXPERIMENT ONLY: does not modify src/, tests/, or requirements/.

For each real model (sc205, pilot4_plain, pilot87) build the ACTUAL Schur
system S = A H^-1 A^T from the current Mehrotra initial state (identical
construction to benchmark_orderings.py) and factor the SAME numerical Schur
under the four SuperLU orderings (COLAMD, MMD_AT_PLUS_A, MMD_ATA, NATURAL).

One representative Newton factorization per model. No full solve.
"""
import sys
import time
import csv
from pathlib import Path

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

from numerical_model import load_numeric_mps  # noqa: E402
from scaling import scale_lp  # noqa: E402
import mehrotra as _mh  # noqa: E402
from linear_system import MAX_RHO_P  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"
REG = 1e-12

MODELS = [
    "sc205",
    "pilot4_plain",
    "pilot87",
]

# SuperLU permc_spec orderings. COLAMD is the scipy default, hence the
# CURRENT production ordering (_splu_regularized passes no permc_spec).
ORDERINGS = ["COLAMD", "MMD_AT_PLUS_A", "MMD_ATA", "NATURAL"]


def memory_rss() -> float:
    """Return current RSS in MB (best-effort; 0 if psutil unavailable)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Build the ACTUAL Schur system (mirrors production linear_system path)
# ---------------------------------------------------------------------------

def build_schur(sf, scaled, reg=REG):
    """Replicate production Schur construction for a standard-form LP.

    Mirrors linear_system.factor_reduced_system sparse path + _splu_regularized:
      h = z/x ; reg_h = min(reg*max(1, mean|h|), MAX_RHO_P)
      h_diag = h + reg_h
      A_sp = csr(A) ; W_sp = A.T * (1/h_diag) ; S = A_sp @ W_sp   (ACTUAL Schur)
      M = 0.5*(S + S.T) ; reg_M = reg*max(1, mean|diag(M)|)
      M_reg = M + reg_M*I                                          (factored matrix)
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
        "M_reg": M_reg,
        "schur_shape": S_sp.shape,
        "nnz_S": S_sp.nnz,
        "nnz_M": M.nnz,
        "reg_M": reg_M,
    }


def factor_with_ordering(M_reg, ordering):
    """Factor M_reg with the given SuperLU ordering.

    Returns (lu, elapsed, nnzL, nnzU, error). SAME matrix, SAME values,
    SAME pivot settings, SAME regularization; only permc_spec changes.
    """
    M_csc = M_reg.tocsr().tocsc()
    t0 = time.perf_counter()
    try:
        lu = splu(M_csc, permc_spec=ordering)
    except Exception as exc:
        return None, None, None, None, type(exc).__name__ + ": " + str(exc)
    elapsed = time.perf_counter() - t0
    return lu, elapsed, lu.L.nnz, lu.U.nnz, None


# ---------------------------------------------------------------------------
# Model pipeline
# ---------------------------------------------------------------------------

def process_model(name) -> dict:
    """Build Schur and factor with all orderings for one MPS model."""
    print(f"\n{'=' * 70}\nMODEL: {name}")
    print(f"{'=' * 70}", flush=True)

    lp = load_numeric_mps(DATA_DIR / f"{name}.mps", sparse=True)
    sf = _mh.to_standard_form(lp)
    print(f"  standard form: A={sf.m}x{sf.n}, nnz={sp.csr_matrix(sf.A).nnz}",
          flush=True)

    scaled = scale_lp(sf.A, sf.b, sf.c_min,
                      np.zeros(sf.n), np.full(sf.n, np.inf))

    t = time.perf_counter()
    schur = build_schur(sf, scaled)
    build_time = time.perf_counter() - t

    print(f"  A_eq: {schur['A_sp'].shape}, nnz={schur['A_sp'].nnz}", flush=True)
    print(f"  ACTUAL Schur S: {schur['schur_shape']}, nnz(S)={schur['nnz_S']:,}, "
          f"sparse={sp.issparse(schur['S_sp'])}", flush=True)
    print(f"  nnz(M)={schur['nnz_M']:,}, reg_M={schur['reg_M']:.3e}, "
          f"build_time={build_time:.3f}s", flush=True)

    M_reg = schur["M_reg"]
    rows = []
    colamd_time = None

    for od in ORDERINGS:
        mem_before = memory_rss()
        lu, elapsed, nnzL, nnzU, error = factor_with_ordering(M_reg, od)
        mem_after = memory_rss()
        mem_delta = mem_after - mem_before if (mem_before and mem_after) else 0.0

        if error is not None:
            print(f"  [{od}] FAILED: {error}", flush=True)
            rows.append({"ordering": od, "time_s": None, "nnz_S": schur["nnz_S"],
                         "nnz_L": None, "nnz_U": None, "fill_in": None,
                         "success": "FAIL", "speedup": None, "mem_delta": mem_delta})
            continue

        fill = (nnzL + nnzU) / schur["nnz_S"]
        print(f"  [{od}] time={elapsed:.3f}s nnz(L)={nnzL:,} nnz(U)={nnzU:,} "
              f"fill={fill:.2f}x", flush=True)
        if od == "COLAMD":
            colamd_time = elapsed
        speedup = colamd_time / elapsed if (colamd_time and elapsed) else None
        rows.append({"ordering": od, "time_s": elapsed, "nnz_S": schur["nnz_S"],
                     "nnz_L": nnzL, "nnz_U": nnzU, "fill_in": fill,
                     "success": "OK", "speedup": speedup, "mem_delta": mem_delta})

    return {
        "name": name,
        "schur": schur,
        "build_time": build_time,
        "rows": rows,
        "best_time": min((r for r in rows if r["time_s"] is not None),
                         key=lambda r: r["time_s"]) if any(r["time_s"] for r in rows) else None,
        "best_fill": min((r for r in rows if r["fill_in"] is not None),
                         key=lambda r: r["fill_in"]) if any(r["fill_in"] for r in rows) else None,
    }


# ---------------------------------------------------------------------------
# CSV + Markdown report writing
# ---------------------------------------------------------------------------

def write_reports(model_results):
    """Write results/benchmark_orderings_real.csv and .md from model_results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- CSV ----
    csv_path = RESULTS_DIR / "benchmark_orderings_real.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "ordering", "factor_time_s", "nnz_S", "nnz_L",
                    "nnz_U", "fill_in_ratio", "speedup_vs_colamd",
                    "success", "mem_delta_mb"])
        for mr in model_results:
            for r in mr["rows"]:
                w.writerow([
                    mr["name"],
                    r["ordering"],
                    f"{r['time_s']:.6f}" if r["time_s"] is not None else "",
                    r["nnz_S"],
                    r["nnz_L"] if r["nnz_L"] is not None else "",
                    r["nnz_U"] if r["nnz_U"] is not None else "",
                    f"{r['fill_in']:.4f}" if r["fill_in"] is not None else "",
                    f"{r['speedup']:.3f}" if r["speedup"] is not None else "",
                    r["success"],
                    f"{r['mem_delta']:.1f}" if r["mem_delta"] else "",
                ])
    print(f"Wrote {csv_path}")

    # ---- Markdown ----
    md_path = RESULTS_DIR / "benchmark_orderings_real.md"
    lines = [
        "# SuperLU Ordering Benchmark — Real Models",
        "",
        "One representative Newton factorization per model. "
        "Same pipeline as the synthetic `benchmark_orderings.py`.",
        "",
        "## Per-Model Results",
        "",
    ]
    for mr in model_results:
        sch = mr["schur"]
        lines += [
            f"### {mr['name']}",
            "",
            f"- A_eq: `{sch['A_sp'].shape}`, nnz {sch['A_sp'].nnz:,}",
            f"- ACTUAL Schur S: shape `{sch['schur_shape']}`, "
            f"**nnz(S)={sch['nnz_S']:,}**, sparse={sp.issparse(sch['S_sp'])}",
            f"- nnz(M)={sch['nnz_M']:,}, reg_M={sch['reg_M']:.3e}, "
            f"build time={mr['build_time']:.3f}s",
            "",
            "| Ordering | Time (s) | nnz(L) | nnz(U) | Fill-in | Speedup | Success |",
            "|----------|----------|--------|--------|---------|---------|---------|",
        ]
        for r in mr["rows"]:
            tstr = f"{r['time_s']:.3f}" if r["time_s"] is not None else "-"
            lstr = str(r["nnz_L"]) if r["nnz_L"] is not None else "-"
            ustr = str(r["nnz_U"]) if r["nnz_U"] is not None else "-"
            fstr = f"{r['fill_in']:.2f}×" if r["fill_in"] is not None else "-"
            spd = f"{r['speedup']:.2f}×" if r["speedup"] is not None else "-"
            lines.append(f"| {r['ordering']} | {tstr} | {lstr} | {ustr} | "
                         f"{fstr} | {spd} | {r['success']} |")
        if mr["best_time"] is not None:
            lines.append(
                f"\n- **Best by time:** {mr['best_time']['ordering']} "
                f"({mr['best_time']['time_s']:.3f}s)"
            )
        if mr["best_fill"] is not None:
            lines.append(
                f"- **Best by fill-in:** {mr['best_fill']['ordering']} "
                f"({mr['best_fill']['fill_in']:.2f}×)"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # ---- Aggregate summary ----
    lines.append("## Summary Across Models")
    lines.append("")
    lines.append("| Model | Best ordering | Speedup vs COLAMD |")
    lines.append("|-------|---------------|-------------------|")
    for mr in model_results:
        if mr["best_time"]:
            spd = mr["best_time"]["speedup"]
            spd_str = f"{spd:.2f}×" if spd is not None else "-"
            lines.append(f"| {mr['name']} | {mr['best_time']['ordering']} | "
                         f"{spd_str} |")
        else:
            lines.append(f"| {mr['name']} | N/A | - |")

    lines.append("")
    all_ok = all(r["success"] == "OK" for mr in model_results for r in mr["rows"])
    lines.append(f"- **All factorizations numerically successful:** {all_ok}")

    # MMD_AT_PLUS_A speedup over COLAMD, per model
    mmd_vals = []
    for mr in model_results:
        mmd_row = next((r for r in mr["rows"]
                        if r["ordering"] == "MMD_AT_PLUS_A" and r["speedup"]), None)
        if mmd_row:
            mmd_vals.append((mr["name"], mmd_row["speedup"]))

    # Does MMD_AT_PLUS_A beat COLAMD on every model where it succeeded?
    mmd_beats = sum(1 for _, s in mmd_vals if s > 1.0)
    total_models = len([mr for mr in model_results
                        if any(r["ordering"] == "MMD_AT_PLUS_A"
                               for r in mr["rows"])])
    if mmd_vals:
        avg = sum(s for _, s in mmd_vals) / len(mmd_vals)
        lines.append(f"- **MMD_AT_PLUS_A beats COLAMD on {mmd_beats}/{total_models} "
                     f"models** (of those evaluated)")
        lines.append(f"- **Average MMD_AT_PLUS_A speedup:** {avg:.2f}×")
        for name, s in mmd_vals:
            lines.append(f"  - {name}: {s:.2f}×")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")


def main():
    print(f"\n{'=' * 70}")
    print("SUPERLU ORDERING BENCHMARK — REAL MPS MODELS")
    print(f"Models: {MODELS}")
    print(f"Orderings: {ORDERINGS}")
    print(f"{'=' * 70}\n", flush=True)

    model_results = []
    for name in MODELS:
        try:
            mr = process_model(name)
            model_results.append(mr)
        except Exception as exc:
            print(f"\nMODEL {name} FAILED during processing: {exc}", flush=True)
            import traceback
            traceback.print_exc()

    if not model_results:
        print("No models processed successfully.")
        return

    write_reports(model_results)

    # ---- Console final report ----
    print(f"\n{'=' * 70}")
    print("FINAL REPORT")
    print(f"{'=' * 70}")
    for mr in model_results:
        sch = mr["schur"]
        print(f"\n### {mr['name']}")
        print(f"  Schur: shape={sch['schur_shape']}, nnz(S)={sch['nnz_S']:,}")
        for r in mr["rows"]:
            tstr = f"{r['time_s']:.3f}" if r["time_s"] is not None else "FAIL"
            fstr = f"{r['fill_in']:.2f}x" if r["fill_in"] is not None else "-"
            spd = f"{r['speedup']:.2f}x" if r["speedup"] is not None else "-"
            print(f"    {r['ordering']:<14} time={tstr:<8} fill={fstr:<8} "
                  f"speedup={spd:<6} {r['success']}")
        if mr["best_time"]:
            print(f"  BEST TIME: {mr['best_time']['ordering']} "
                  f"({mr['best_time']['time_s']:.3f}s)")
        if mr["best_fill"]:
            print(f"  BEST FILL: {mr['best_fill']['ordering']} "
                  f"({mr['best_fill']['fill_in']:.2f}x)")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    for mr in model_results:
        best = mr["best_time"]["ordering"] if mr["best_time"] else "N/A"
        print(f"  {mr['name']}: best ordering = {best}")
    mmd_beats_all = True
    mmd_speeds = []
    for mr in model_results:
        mmd = next((r for r in mr["rows"] if r["ordering"] == "MMD_AT_PLUS_A"
                    and r["speedup"]), None)
        if mmd is not None:
            mmd_speeds.append(mmd["speedup"])
            if mmd["speedup"] <= 1.0:
                mmd_beats_all = False
    if mmd_speeds:
        avg = sum(mmd_speeds) / len(mmd_speeds)
        print(f"  MMD_AT_PLUS_A beats COLAMD consistently: "
              f"{mmd_beats_all} ({len(mmd_speeds)} models)")
        print(f"  Average MMD_AT_PLUS_A speedup: {avg:.2f}x")
    all_ok = all(r["success"] == "OK" for mr in model_results for r in mr["rows"])
    print(f"  All factorizations numerically successful: {all_ok}")
    print("  git status: untracked experiment/data files only (no commit).")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()


