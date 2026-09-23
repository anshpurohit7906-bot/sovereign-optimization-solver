#!/usr/bin/env python
"""Profile SuperLU ordering across the FULL production Mehrotra solve.

EXPERIMENT ONLY: does not modify src/, tests/, or requirements/.
Uses monkeypatching/wrapping only inside this process.

Runs the real production solve_lp() on pilot4_plain.mps and pilot87.mps,
twice per model (COLAMD and MMD_AT_PLUS_A), and records per-Newton-iteration
SuperLU factorization behavior: timing, nnz(M_reg), nnz(L), nnz(U), fill-in
ratio, regularization, and h statistics.

Saves:
  results/profile_ordering_full_solve.csv
  results/profile_ordering_full_solve.md
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from opticore.lp.linear_system import (  # noqa: E402
    LinearSystemError,
    factor_reduced_system as _orig_factor_reduced_system,
    _splu_regularized as _orig_splu_regularized,
)
from opticore.numerical_model import load_numeric_mps  # noqa: E402
from opticore.lp.mehrotra import solve_lp  # noqa: E402

RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "pilot4_plain.mps",
    "pilot87.mps",
]
ORDERINGS = ["COLAMD", "MMD_AT_PLUS_A"]

# ---------------------------------------------------------------------------
# Shared mutable state for monkeypatch callbacks
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "ordering": None,
    "model": None,
    "current_iter": -1,
    "h_min": None,
    "h_max": None,
    "h_ratio": None,
    "records": [],
}
_iter_counter = [0]


# ---------------------------------------------------------------------------
# Monkeypatch: _splu_regularized  (intercepts the actual splu call)
# ---------------------------------------------------------------------------
def _wrapped_splu_regularized(S: sp.csr_matrix, base_reg: float):
    """Replaces linear_system._splu_regularized; records factorization metrics."""
    t0 = time.perf_counter()
    M = (0.5 * (S + S.T)).tocsc()
    scale = max(1.0, float(np.mean(np.abs(M.diagonal()))))
    reg = base_reg * scale
    ordering = _state["ordering"]
    for attempt in range(8):
        try:
            M_reg = M + sp.eye(M.shape[0], format="csc") * reg
            lu = splu(M_reg, permc_spec=ordering)
            dt = time.perf_counter() - t0
            nnz_M = int(M_reg.nnz)
            nnz_L = int(lu.L.nnz)
            nnz_U = int(lu.U.nnz)
            fill = (nnz_L + nnz_U) / nnz_M if nnz_M > 0 else float("nan")
            _state["records"].append({
                "model": _state["model"],
                "newton_iter": _state["current_iter"],
                "ordering": ordering,
                "factor_time": dt,
                "schur_shape_0": M.shape[0],
                "schur_shape_1": M.shape[1],
                "nnz_M_reg": nnz_M,
                "nnz_L": nnz_L,
                "nnz_U": nnz_U,
                "fill_ratio": fill,
                "reg": reg,
                "success": True,
                "h_min": _state["h_min"],
                "h_max": _state["h_max"],
                "h_ratio": _state["h_ratio"],
            })
            return lu, reg
        except RuntimeError:
            reg *= 10.0
    # All 8 escalation attempts failed
    dt = time.perf_counter() - t0
    _state["records"].append({
        "model": _state["model"], "newton_iter": _state["current_iter"],
        "ordering": ordering, "factor_time": dt,
        "schur_shape_0": M.shape[0], "schur_shape_1": M.shape[1],
        "nnz_M_reg": 0, "nnz_L": 0, "nnz_U": 0,
        "fill_ratio": float("nan"), "reg": reg, "success": False,
        "h_min": _state["h_min"], "h_max": _state["h_max"],
        "h_ratio": _state["h_ratio"],
    })
    raise LinearSystemError("Sparse factorization failed even with regularization")


# ---------------------------------------------------------------------------
# Monkeypatch: factor_reduced_system  (captures h stats per iteration)
# ---------------------------------------------------------------------------
def _wrapped_factor_reduced_system(H, A_eq, reg=1e-12):
    """Wraps the real factor_reduced_system; captures h stats then delegates."""
    if H.ndim == 1:
        diag = H
    else:
        diag = np.diag(H)
    h_arr = np.asarray(diag, dtype=np.float64)
    h_finite = h_arr[np.isfinite(h_arr) & (h_arr > 0)]
    if h_finite.size > 0:
        _state["h_min"] = float(np.min(h_finite))
        _state["h_max"] = float(np.max(h_finite))
        _state["h_ratio"] = (
            _state["h_max"] / _state["h_min"]
            if _state["h_min"] > 0 else float("inf")
        )
    else:
        _state["h_min"] = float("nan")
        _state["h_max"] = float("nan")
        _state["h_ratio"] = float("nan")
    _iter_counter[0] += 1
    _state["current_iter"] = _iter_counter[0]
    return _orig_factor_reduced_system(H, A_eq, reg)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_single(model_file: str, ordering: str) -> dict[str, Any]:
    """Run solve_lp on one model with one ordering; return summary dict."""
    model_name = Path(model_file).stem
    _state["ordering"] = ordering
    _state["model"] = model_name
    _state["records"] = []
    _iter_counter[0] = 0

    import opticore.lp.linear_system as _ls_mod
    import opticore.lp.mehrotra as _mh_mod
    import opticore.lp.linear_system as _lps_mod
    import opticore.lp.mehrotra as _lpm_mod
    # The same file may be loaded under two module names (with both src and
    # src/lp on sys.path).  Patch ALL instances to guarantee interception.
    for _mod in (_ls_mod, _lps_mod):
        _mod._splu_regularized = _wrapped_splu_regularized
        _mod.factor_reduced_system = _wrapped_factor_reduced_system
    for _mod in (_mh_mod, _lpm_mod):
        _mod.factor_reduced_system = _wrapped_factor_reduced_system

    path = str(_ROOT / "data" / model_file)
    lp = load_numeric_mps(path)

    t_solve_start = time.perf_counter()
    result = solve_lp(lp, tol=1e-7, max_iter=100)
    t_solve_end = time.perf_counter()

    for _mod in (_ls_mod, _lps_mod):
        _mod._splu_regularized = _orig_splu_regularized
        _mod.factor_reduced_system = _orig_factor_reduced_system
    for _mod in (_mh_mod, _lpm_mod):
        _mod.factor_reduced_system = _orig_factor_reduced_system

    total_solve_time = t_solve_end - t_solve_start
    records = list(_state["records"])

    ftimes = [r["factor_time"] for r in records if r["success"]]
    fills = [r["fill_ratio"] for r in records if r["success"]]
    total_fac_time = sum(ftimes)
    mean_fac = float(np.mean(ftimes)) if ftimes else 0.0
    median_fac = float(np.median(ftimes)) if ftimes else 0.0
    max_fac = max(ftimes) if ftimes else 0.0
    avg_fill = float(np.mean(fills)) if fills else 0.0
    max_fill = max(fills) if fills else 0.0
    fac_pct = (total_fac_time / total_solve_time * 100.0
               if total_solve_time > 0 else 0.0)

    return {
        "model": model_name, "ordering": ordering,
        "status": result.status, "objective": result.objective,
        "iterations": result.iterations,
        "rel_primal": result.rel_primal, "rel_dual": result.rel_dual,
        "rel_gap": result.rel_gap,
        "total_solve_time": total_solve_time,
        "total_fac_time": total_fac_time, "mean_fac_time": mean_fac,
        "median_fac_time": median_fac, "max_fac_time": max_fac,
        "fac_pct": fac_pct, "avg_fill": avg_fill, "max_fill": max_fill,
        "records": records,
    }


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------
def write_csv(all_results: list[dict], path: Path) -> None:
    fieldnames = [
        "model", "ordering", "newton_iter", "factor_time",
        "schur_shape_0", "schur_shape_1", "nnz_M_reg", "nnz_L", "nnz_U",
        "fill_ratio", "reg", "success", "h_min", "h_max", "h_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for res in all_results:
            for rec in res["records"]:
                row = {k: rec.get(k, "") for k in fieldnames}
                for fk in ("factor_time", "fill_ratio", "reg", "h_min",
                           "h_max", "h_ratio"):
                    v = row[fk]
                    if isinstance(v, float):
                        row[fk] = f"{v:.6e}"
                w.writerow(row)
    print(f"Wrote {path}")


def _fmt(v, fmt: str = ".4f") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:{fmt}}"


def write_md(all_runs: dict[str, dict], path: Path) -> None:
    lines: list[str] = []
    a = lines.append

    a("# Profile Ordering Full Solve — Experiment Report\n")
    a("## Setup\n")
    a("- **Models:** pilot4_plain.mps, pilot87.mps")
    a("- **Orderings:** COLAMD (scipy default), MMD_AT_PLUS_A")
    a("- **Solver:** production `solve_lp`, tol=1e-7, max_iter=100")
    a("- **Instrumentation:** monkeypatched `_splu_regularized` + "
      "`factor_reduced_system`; no src/ modifications\n")
    a("---\n")

    for model in MODELS:
        model_stem = Path(model).stem
        col = all_runs[f"{model_stem}__COLAMD"]
        mmd = all_runs[f"{model_stem}__MMD_AT_PLUS_A"]
        a(f"## {model_stem}\n")

        for ordering, res in [("COLAMD", col), ("MMD_AT_PLUS_A", mmd)]:
            a(f"### {ordering}\n")
            a(f"**Status:** {res['status']}  |  **Iterations:** "
              f"{res['iterations']}  |  **Objective:** {res['objective']:.6f}\n")
            a(f"rel_p={res['rel_primal']:.2e}, rel_d={res['rel_dual']:.2e}, "
              f"rel_gap={res['rel_gap']:.2e}\n")
            a("#### A. Iteration-by-iteration table\n")
            a("| iter | factor_time (s) | nnz(M_reg) | nnz(L) | nnz(U) "
              "| fill | h_min | h_max | h_max/h_min |")
            a("|------|-----------------|------------|--------|--------"
              "|------|-------|-------|-------------|")
            for rec in res["records"]:
                a(f"| {rec['newton_iter']:4d} | {rec['factor_time']:15.6f} "
                  f"| {rec['nnz_M_reg']:>10,} | {rec['nnz_L']:>6,} "
                  f"| {rec['nnz_U']:>6,} | {rec['fill_ratio']:6.2f}x "
                  f"| {_fmt(rec['h_min'], '.3e'):>9s} "
                  f"| {_fmt(rec['h_max'], '.3e'):>9s} "
                  f"| {_fmt(rec['h_ratio'], '.2e'):>11s} |")
            a("")
            a("#### B. Aggregate\n")
            a(f"- Total factorization time: **{res['total_fac_time']:.4f}s**")
            a(f"- Mean factorization time: {res['mean_fac_time']:.4f}s")
            a(f"- Median factorization time: {res['median_fac_time']:.4f}s")
            a(f"- Max factorization time: {res['max_fac_time']:.4f}s")
            a(f"- Total solve time: **{res['total_solve_time']:.4f}s**")
            a(f"- Factorization % of total solve: **{res['fac_pct']:.1f}%**")
            a(f"- Average fill-in: {res['avg_fill']:.2f}x")
            a(f"- Maximum fill-in: {res['max_fill']:.2f}x")
            a("")

        a(f"### C. COLAMD vs MMD_AT_PLUS_A\n")
        fac_speedup = (col["total_fac_time"] / mmd["total_fac_time"]
                       if mmd["total_fac_time"] > 0 else float("nan"))
        solve_speedup = (col["total_solve_time"] / mmd["total_solve_time"]
                         if mmd["total_solve_time"] > 0 else float("nan"))
        a("| Metric | COLAMD | MMD_AT_PLUS_A | Speedup (COL/MMD) |")
        a("|--------|--------|---------------|-------------------|")
        a(f"| Total factorization time | {col['total_fac_time']:.4f}s "
          f"| {mmd['total_fac_time']:.4f}s | {fac_speedup:.2f}x |")
        a(f"| Mean factorization time | {col['mean_fac_time']:.4f}s "
          f"| {mmd['mean_fac_time']:.4f}s |")
        a(f"| Max factorization time | {col['max_fac_time']:.4f}s "
          f"| {mmd['max_fac_time']:.4f}s |")
        a(f"| Total solve time | {col['total_solve_time']:.4f}s "
          f"| {mmd['total_solve_time']:.4f}s | {solve_speedup:.2f}x |")
        a(f"| Average fill-in | {col['avg_fill']:.2f}x "
          f"| {mmd['avg_fill']:.2f}x |")
        a(f"| Maximum fill-in | {col['max_fill']:.2f}x "
          f"| {mmd['max_fill']:.2f}x |")
        a(f"| Newton iterations | {col['iterations']} | {mmd['iterations']} |")
        a(f"| **Factorization-only speedup** | — | — | **{fac_speedup:.2f}x** |")
        a(f"| **Full-solve speedup** | — | — | **{solve_speedup:.2f}x** |")
        a("")

        a(f"### D. Early / mid / late iteration analysis\n")
        for ordering_name, run_data in [("COLAMD", col), ("MMD_AT_PLUS_A", mmd)]:
            recs = run_data["records"]
            n = len(recs)
            a(f"**{ordering_name}** ({n} factorizations):\n")
            a("| Phase | Iters | Mean fac time (s) | Mean fill | "
              "Mean h_max/h_min |")
            a("|-------|-------|--------------------|-----------|"
              "------------------|")
            if n > 0:
                third = max(1, n // 3)
                for label, chunk in [("Early", recs[:third]),
                                     ("Mid", recs[third:2 * third]),
                                     ("Late", recs[2 * third:])]:
                    if not chunk:
                        continue
                    success_chunk = [r for r in chunk if r["success"]]
                    mt = np.mean([r["factor_time"] for r in success_chunk]) if success_chunk else 0.0
                    mf = np.mean([r["fill_ratio"] for r in success_chunk]) if success_chunk else 0.0
                    hrs = [r["h_ratio"] for r in chunk
                           if r["h_ratio"] is not None and r["h_ratio"] == r["h_ratio"]]
                    mhr = np.mean(hrs) if hrs else float("nan")
                    a(f"| {label} | {len(chunk):4d} | {mt:18.4f} "
                      f"| {mf:9.2f}x | {_fmt(mhr, '.2e'):>16s} |")
            a("")

        a(f"### E. Final 5 factorizations\n")
        for ordering_name, run_data in [("COLAMD", col), ("MMD_AT_PLUS_A", mmd)]:
            recs = run_data["records"]
            last5 = recs[-5:] if len(recs) >= 5 else recs
            a(f"**{ordering_name}** (last {len(last5)} factorizations):\n")
            a("| iter | factor_time | nnz(M_reg) | nnz(L) | nnz(U) "
              "| fill | h_min | h_max | h_max/h_min |")
            a("|------|-------------|------------|--------|--------"
              "|------|-------|-------|-------------|")
            for rec in last5:
                a(f"| {rec['newton_iter']:4d} | {rec['factor_time']:11.6f} "
                  f"| {rec['nnz_M_reg']:>10,} | {rec['nnz_L']:>6,} "
                  f"| {rec['nnz_U']:>6,} | {rec['fill_ratio']:6.2f}x "
                  f"| {_fmt(rec['h_min'], '.3e'):>9s} "
                  f"| {_fmt(rec['h_max'], '.3e'):>9s} "
                  f"| {_fmt(rec['h_ratio'], '.2e'):>11s} |")
            a("")

        a(f"### D2. Per-iteration comparison\n")
        col_recs = col["records"]
        mmd_recs = mmd["records"]
        min_len = min(len(col_recs), len(mmd_recs))
        slower_iters = []
        faster_iters = []
        if min_len > 0:
            a("| iter | COLAMD time | MMD time | COLAMD fill | MMD fill "
              "| Speedup | MMD slower? |")
            a("|------|-------------|----------|-------------|----------"
              "|---------|-------------|")
            for i in range(min_len):
                ct = col_recs[i]["factor_time"]
                mt = mmd_recs[i]["factor_time"]
                cf = col_recs[i]["fill_ratio"]
                mf = mmd_recs[i]["fill_ratio"]
                spd = ct / mt if mt > 0 else float("nan")
                if mt > ct * 1.05:
                    slower_iters.append(col_recs[i]["newton_iter"])
                elif mt < ct * 0.95:
                    faster_iters.append(col_recs[i]["newton_iter"])
                a(f"| {col_recs[i]['newton_iter']:4d} | {ct:.6f} | {mt:.6f} "
                  f"| {cf:.2f}x | {mf:.2f}x | {spd:.2f}x "
                  f"| {'YES' if mt > ct * 1.05 else ''} |")
            a("")
        a(f"- Iterations where MMD is **slower** (>5%): "
          f"{slower_iters if slower_iters else 'none'}")
        a(f"- Iterations where MMD is **faster** (>5%): "
          f"{faster_iters if faster_iters else 'none'}")
        a("---\n")

    # ---- Conclusion ----
    a("## 6. Conclusion\n")
    total_col_fac = sum(
        all_runs[f"{Path(m).stem}__COLAMD"]["total_fac_time"] for m in MODELS
    )
    total_mmd_fac = sum(
        all_runs[f"{Path(m).stem}__MMD_AT_PLUS_A"]["total_fac_time"] for m in MODELS
    )
    total_col_solve = sum(
        all_runs[f"{Path(m).stem}__COLAMD"]["total_solve_time"] for m in MODELS
    )
    total_mmd_solve = sum(
        all_runs[f"{Path(m).stem}__MMD_AT_PLUS_A"]["total_solve_time"] for m in MODELS
    )
    overall_fac_speedup = total_col_fac / total_mmd_fac if total_mmd_fac > 0 else 0
    overall_solve_speedup = (
        total_col_solve / total_mmd_solve if total_mmd_solve > 0 else 0
    )
    a(f"**Overall factorization-only speedup:** {overall_fac_speedup:.2f}x")
    a(f"**Overall full-solve speedup:** {overall_solve_speedup:.2f}x\n")
    a("### Verdict\n")
    if overall_fac_speedup > 1.10 and overall_solve_speedup > 1.05:
        a("**GENERAL WIN** — MMD_AT_PLUS_A provides meaningful speedup "
          "across both factorization-only and full-solve metrics.")
    elif overall_fac_speedup > 1.10 and overall_solve_speedup <= 1.05:
        a("**ITERATION/WORKLOAD DEPENDENT** — MMD_AT_PLUS_A is faster "
          "in factorization but the full solve speedup is diluted by "
          "iteration count differences or non-factorization overhead.")
    elif overall_fac_speedup > 1.02:
        a("**ITERATION/WORKLOAD DEPENDENT** — MMD_AT_PLUS_A shows a marginal "
          "factorization advantage that does not consistently translate to "
          "full-solve improvement.")
    else:
        a("**NO MEANINGFUL WIN** — MMD_AT_PLUS_A does not provide a consistent "
          "advantage over COLAMD across the full Newton trajectory.")
    a("")
    a("*Do NOT infer causation beyond the measurements.*")
    a(f"\n*Generated by `{Path(__file__).name}`*")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("Profile Ordering Full Solve - Experiment")
    print("Models:", MODELS)
    print("Orderings:", ORDERINGS)
    print("=" * 78, flush=True)

    all_runs: dict[str, dict] = {}
    for model_file in MODELS:
        model_stem = Path(model_file).stem
        for ordering in ORDERINGS:
            key = f"{model_stem}__{ordering}"
            print(f"\n{'-' * 78}")
            print(f"Running {model_stem} with {ordering} ...", flush=True)
            t0 = time.perf_counter()
            res = run_single(model_file, ordering)
            dt = time.perf_counter() - t0
            print(f"  Done in {dt:.2f}s - status={res['status']}, "
                  f"iters={res['iterations']}, obj={res['objective']:.6f}")
            print(f"  Total fac time: {res['total_fac_time']:.4f}s "
                  f"({res['fac_pct']:.1f}% of solve), "
                  f"avg fill: {res['avg_fill']:.2f}x", flush=True)
            all_runs[key] = res

    csv_path = RESULTS_DIR / "profile_ordering_full_solve.csv"
    write_csv(list(all_runs.values()), csv_path)

    md_path = RESULTS_DIR / "profile_ordering_full_solve.md"
    write_md(all_runs, md_path)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for model_file in MODELS:
        model_stem = Path(model_file).stem
        col = all_runs[f"{model_stem}__COLAMD"]
        mmd = all_runs[f"{model_stem}__MMD_AT_PLUS_A"]
        print(f"\n{model_stem}:")
        print(f"  COLAMD:        fac={col['total_fac_time']:.4f}s  "
              f"solve={col['total_solve_time']:.4f}s  "
              f"fill={col['avg_fill']:.2f}x  iters={col['iterations']}  "
              f"status={col['status']}")
        print(f"  MMD_AT_PLUS_A: fac={mmd['total_fac_time']:.4f}s  "
              f"solve={mmd['total_solve_time']:.4f}s  "
              f"fill={mmd['avg_fill']:.2f}x  iters={mmd['iterations']}  "
              f"status={mmd['status']}")
        if col["total_fac_time"] > 0:
            spd = col["total_fac_time"] / mmd["total_fac_time"]
            print(f"  Factorization speedup: {spd:.2f}x")
        if col["total_solve_time"] > 0:
            spd = col["total_solve_time"] / mmd["total_solve_time"]
            print(f"  Full-solve speedup:    {spd:.2f}x")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()

