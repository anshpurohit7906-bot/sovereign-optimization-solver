"""Controlled initial-point comparison: dense vs sparse arithmetic only.

Determines whether the dense-vs-sparse differences on pilot4_plain and
pilot87 are caused by the new sparse LSQR initialization or by sparse
arithmetic in the Mehrotra iteration.

Strategy
--------
1. Solve each model through the **dense** pipeline (load_numeric_mps
   sparse=False, solve_lp).  During this solve, monkeypatch
   ``_mehrotra_initial_point`` to capture the initial (x0, y0, z0) that
   the dense path computed for the scaled standard-form problem.

2. Load the same model through the **sparse** pipeline (sparse=True),
   build the standard form and apply the same scaling, then monkeypatch
   ``_mehrotra_initial_point`` to return the **exact same** (x0, y0, z0)
   captured from step 1, and solve.

3. Compare iteration-by-iteration histories to determine whether the
   trajectories remain approximately identical when the initial point is
   forced to be the same.

This script does NOT modify any production files.  It uses only
monkeypatching within the experiment process.

Run::

    python experiment/sparse/controlled_init_comparison.py
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ── path setup ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from opticore.numerical_model import load_numeric_mps  # noqa: E402
from opticore.scaling import scale_lp  # noqa: E402
import opticore.lp.mehrotra as _mh  # noqa: E402

# ── configuration ─────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"
MODELS = ["pilot4_plain.mps", "pilot87.mps"]
TOL = 1e-7
MAX_ITER = 100


# ── helpers ───────────────────────────────────────────────────────────
# Store the original initializer ONCE at import time, before any monkeypatching.
_original_mehrotra_init = _mh._mehrotra_initial_point


@dataclass
class InitCapture:
    """Mutable container used by the monkeypatch wrapper to store the
    initial point returned by _mehrotra_initial_point."""
    x0: np.ndarray | None = None
    y0: np.ndarray | None = None
    z0: np.ndarray | None = None


def _make_capture_wrapper(capture: InitCapture):
    """Return a replacement for _mehrotra_initial_point that captures
    whatever the real function returns and stores it in *capture*."""
    def _wrapper(A, b, c, **kwargs):
        x0, y0, z0 = _original_mehrotra_init(A, b, c, **kwargs)
        capture.x0 = x0.copy()
        capture.y0 = y0.copy()
        capture.z0 = z0.copy()
        return x0, y0, z0

    return _wrapper


def _make_inject_wrapper(x0: np.ndarray, y0: np.ndarray, z0: np.ndarray):
    """Return a replacement for _mehrotra_initial_point that always
    returns the supplied (x0, y0, z0), ignoring A, b, c entirely."""
    def _wrapper(A, b, c, **kwargs):
        return x0.copy(), y0.copy(), z0.copy()
    return _wrapper


def _restore_original_init():
    """Restore the original _mehrotra_initial_point function."""
    _mh._mehrotra_initial_point = _original_mehrotra_init


# ── result container ──────────────────────────────────────────────────
@dataclass
class ModelResult:
    model: str
    # A-side (dense pipeline)
    dense_status: str
    dense_objective: float
    dense_iterations: int
    dense_primal: float
    dense_dual: float
    dense_rel_gap: float
    dense_runtime: float
    dense_history: list[dict]
    # B-side (sparse pipeline, injected dense init)
    sparse_status: str
    sparse_objective: float
    sparse_iterations: int
    sparse_primal: float
    sparse_dual: float
    sparse_rel_gap: float
    sparse_runtime: float
    sparse_history: list[dict]
    # trajectory comparison
    obj_max_abs_diff: float
    primal_max_abs_diff: float
    dual_max_abs_diff: float
    comp_max_abs_diff: float


# ── core experiment ───────────────────────────────────────────────────
def _run_dense(lp_path: Path):
    """Run the normal dense pipeline.  Returns (result, capture, elapsed)."""
    lp = load_numeric_mps(lp_path, sparse=False)
    cap = InitCapture()
    _mh._mehrotra_initial_point = _make_capture_wrapper(cap)
    try:
        t0 = time.perf_counter()
        result = _mh.solve_lp(lp, tol=TOL, max_iter=MAX_ITER)
        elapsed = time.perf_counter() - t0
        return result, cap, elapsed
    finally:
        _restore_original_init()


def _run_sparse_injected(lp_path: Path, cap: InitCapture):
    """Run the sparse pipeline with the dense initial point injected."""
    lp = load_numeric_mps(lp_path, sparse=True)
    sf = _mh.to_standard_form(lp)

    # Verify the scaled standard-form matrices match the dense path.
    lp_dense = load_numeric_mps(lp_path, sparse=False)
    sf_dense = _mh.to_standard_form(lp_dense)

    scaled_sp = scale_lp(sf.A, sf.b, sf.c_min,
                         np.zeros(sf.n), np.full(sf.n, np.inf))
    scaled_dn = scale_lp(sf_dense.A, sf_dense.b, sf_dense.c_min,
                         np.zeros(sf_dense.n), np.full(sf_dense.n, np.inf))

    if sp.issparse(scaled_sp.A):
        A_sp = scaled_sp.A.toarray()
    else:
        A_sp = np.asarray(scaled_sp.A)
    A_dn = np.asarray(scaled_dn.A)

    a_ok = np.allclose(A_sp, A_dn, atol=0.0, rtol=1e-12)
    b_ok = np.allclose(scaled_sp.b, scaled_dn.b, atol=0.0, rtol=1e-12)
    c_ok = np.allclose(scaled_sp.c, scaled_dn.c, atol=0.0, rtol=1e-12)
    print(f"  Matrix check: A={'OK' if a_ok else 'MISMATCH'}, "
          f"b={'OK' if b_ok else 'MISMATCH'}, c={'OK' if c_ok else 'MISMATCH'}")
    if not (a_ok and b_ok and c_ok):
        print("  WARNING: scaled standard-form data differs; results may be confounded.")

    # Inject the dense initial point.
    _mh._mehrotra_initial_point = _make_inject_wrapper(cap.x0, cap.y0, cap.z0)
    try:
        t0 = time.perf_counter()
        result = _mh.solve_standard_form(sf, tol=TOL, max_iter=MAX_ITER)
        elapsed = time.perf_counter() - t0
        return result, elapsed
    finally:
        _restore_original_init()


def run_model(lp_path: Path) -> ModelResult | None:
    if not lp_path.is_file():
        print(f"SKIP {lp_path.name} (file not found)")
        return None

    print(f"\n{'='*60}")
    print(f"  {lp_path.name}")
    print(f"{'='*60}")

    # ── Step A: dense pipeline ────────────────────────────────────────
    dense_res, cap, dense_time = _run_dense(lp_path)
    print(f"  [Dense]  status={dense_res.status}  iters={dense_res.iterations}  "
          f"obj={dense_res.objective:.6g}  ({dense_time:.3f}s)")
    print(f"  Captured init: x0_norm={np.linalg.norm(cap.x0):.4f}, "
          f"y0_norm={np.linalg.norm(cap.y0):.4f}, z0_norm={np.linalg.norm(cap.z0):.4f}")

    # ── Step B: sparse pipeline with injected dense init ──────────────
    sparse_res, sparse_time = _run_sparse_injected(lp_path, cap)
    print(f"  [Sparse] status={sparse_res.status}  iters={sparse_res.iterations}  "
          f"obj={sparse_res.objective:.6g}  ({sparse_time:.3f}s)")

    # ── Iterate-history comparison ─────────────────────────────────────
    dh = list(dense_res.history)
    sh = list(sparse_res.history)
    n_cmp = min(len(dh), len(sh))

    obj_diffs = [abs(dh[i]["mu"] - sh[i]["mu"]) for i in range(n_cmp)]
    primal_diffs = [abs(dh[i]["primal"] - sh[i]["primal"]) for i in range(n_cmp)]
    dual_diffs = [abs(dh[i]["dual"] - sh[i]["dual"]) for i in range(n_cmp)]
    comp_diffs = [abs(dh[i]["rel_gap"] - sh[i]["rel_gap"]) for i in range(n_cmp)]

    obj_max = max(obj_diffs) if obj_diffs else 0.0
    primal_max = max(primal_diffs) if primal_diffs else 0.0
    dual_max = max(dual_diffs) if dual_diffs else 0.0
    comp_max = max(comp_diffs) if comp_diffs else 0.0

    print(f"\n  Trajectory divergence (first {n_cmp} shared iterations):")
    print(f"    mu        max |Δ| = {obj_max:.6e}")
    print(f"    primal    max |Δ| = {primal_max:.6e}")
    print(f"    dual      max |Δ| = {dual_max:.6e}")
    print(f"    rel_gap   max |Δ| = {comp_max:.6e}")

    # Per-iteration table
    print(f"\n  {'it':>3}  {'mu(d)':>12}  {'mu(s)':>12}  {'Δmu':>12}  "
          f"{'primal(d)':>12}  {'primal(s)':>12}  {'Δprimal':>12}  "
          f"{'dual(d)':>12}  {'dual(s)':>12}  {'Δdual':>12}  "
          f"{'gap(d)':>12}  {'gap(s)':>12}  {'Δgap':>12}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*12}  "
          f"{'-'*12}  {'-'*12}  {'-'*12}  "
          f"{'-'*12}  {'-'*12}  {'-'*12}  "
          f"{'-'*12}  {'-'*12}  {'-'*12}")
    for i in range(n_cmp):
        print(f"  {dh[i]['iter']:3d}  {dh[i]['mu']:12.5e}  {sh[i]['mu']:12.5e}  "
              f"{obj_diffs[i]:12.5e}  "
              f"{dh[i]['primal']:12.5e}  {sh[i]['primal']:12.5e}  "
              f"{primal_diffs[i]:12.5e}  "
              f"{dh[i]['dual']:12.5e}  {sh[i]['dual']:12.5e}  "
              f"{dual_diffs[i]:12.5e}  "
              f"{dh[i]['rel_gap']:12.5e}  {sh[i]['rel_gap']:12.5e}  "
              f"{comp_diffs[i]:12.5e}")

    if len(dh) != len(sh):
        print(f"\n  NOTE: dense finished in {len(dh)} iters, "
              f"sparse in {len(sh)} iters.")

    return ModelResult(
        model=lp_path.stem,
        dense_status=dense_res.status,
        dense_objective=dense_res.objective,
        dense_iterations=dense_res.iterations,
        dense_primal=dense_res.primal_residual,
        dense_dual=dense_res.dual_residual,
        dense_rel_gap=dense_res.rel_gap,
        dense_runtime=dense_time,
        dense_history=dh,
        sparse_status=sparse_res.status,
        sparse_objective=sparse_res.objective,
        sparse_iterations=sparse_res.iterations,
        sparse_primal=sparse_res.primal_residual,
        sparse_dual=sparse_res.dual_residual,
        sparse_rel_gap=sparse_res.rel_gap,
        sparse_runtime=sparse_time,
        sparse_history=sh,
        obj_max_abs_diff=obj_max,
        primal_max_abs_diff=primal_max,
        dual_max_abs_diff=dual_max,
        comp_max_abs_diff=comp_max,
    )


# ── output ────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "model",
    "dense_status", "sparse_status",
    "dense_objective", "sparse_objective",
    "dense_iterations", "sparse_iterations",
    "dense_primal", "sparse_primal",
    "dense_dual", "sparse_dual",
    "dense_rel_gap", "sparse_rel_gap",
    "dense_runtime", "sparse_runtime",
    "obj_max_abs_diff", "primal_max_abs_diff",
    "dual_max_abs_diff", "comp_max_abs_diff",
    "conclusion",
]


def _conclusion(r: ModelResult) -> str:
    """One-line verdict."""
    if r.obj_max_abs_diff < 1e-6 and r.dual_max_abs_diff < 1e-6:
        return "IDENTICAL trajectories -> init was the cause"
    if r.obj_max_abs_diff > 0.01 or r.dual_max_abs_diff > 0.01:
        return "DIVERGENT trajectories -> sparse arithmetic matters"
    return "SIMILAR trajectories -> init was primary cause"


def write_csv(results: list[ModelResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({
                "model": r.model,
                "dense_status": r.dense_status,
                "sparse_status": r.sparse_status,
                "dense_objective": f"{r.dense_objective:.6g}",
                "sparse_objective": f"{r.sparse_objective:.6g}",
                "dense_iterations": r.dense_iterations,
                "sparse_iterations": r.sparse_iterations,
                "dense_primal": f"{r.dense_primal:.6e}",
                "sparse_primal": f"{r.sparse_primal:.6e}",
                "dense_dual": f"{r.dense_dual:.6e}",
                "sparse_dual": f"{r.sparse_dual:.6e}",
                "dense_rel_gap": f"{r.dense_rel_gap:.6e}",
                "sparse_rel_gap": f"{r.sparse_rel_gap:.6e}",
                "dense_runtime": f"{r.dense_runtime:.4f}",
                "sparse_runtime": f"{r.sparse_runtime:.4f}",
                "obj_max_abs_diff": f"{r.obj_max_abs_diff:.6e}",
                "primal_max_abs_diff": f"{r.primal_max_abs_diff:.6e}",
                "dual_max_abs_diff": f"{r.dual_max_abs_diff:.6e}",
                "comp_max_abs_diff": f"{r.comp_max_abs_diff:.6e}",
                "conclusion": _conclusion(r),
            })


def write_md(results: list[ModelResult], path: Path) -> None:
    lines = [
        "# Controlled Initial-Point Comparison: Dense vs Sparse Arithmetic",
        "",
        f"- tolerance `tol={TOL}`, `max_iter={MAX_ITER}`",
        "- Both pipelines use the **exact same initial point** (from the dense path).",
        "- Any trajectory difference is caused by **sparse arithmetic** in the Mehrotra"
        " Newton system, NOT by the initializer.",
        "",
    ]
    for r in results:
        lines.append(f"## {r.model}")
        lines.append("")
        lines.append("### Summary")
        lines.append("")
        lines.append("| Metric | Dense | Sparse |")
        lines.append("|---|---|---|")
        lines.append(f"| Status | {r.dense_status} | {r.sparse_status} |")
        lines.append(f"| Objective | {r.dense_objective:.6g} | {r.sparse_objective:.6g} |")
        lines.append(f"| Iterations | {r.dense_iterations} | {r.sparse_iterations} |")
        lines.append(f"| Primal residual | {r.dense_primal:.6e} | {r.sparse_primal:.6e} |")
        lines.append(f"| Dual residual | {r.dense_dual:.6e} | {r.sparse_dual:.6e} |")
        lines.append(f"| Rel gap | {r.dense_rel_gap:.6e} | {r.sparse_rel_gap:.6e} |")
        lines.append(f"| Runtime (s) | {r.dense_runtime:.4f} | {r.sparse_runtime:.4f} |")
        lines.append("")
        lines.append("### Trajectory divergence (max |difference| across iterations)")
        lines.append("")
        lines.append("| Metric | max |Δ| |")
        lines.append("|---|---|")
        lines.append(f"| Complementarity (mu) | {r.obj_max_abs_diff:.6e} |")
        lines.append(f"| Primal residual | {r.primal_max_abs_diff:.6e} |")
        lines.append(f"| Dual residual | {r.dual_max_abs_diff:.6e} |")
        lines.append(f"| Relative gap | {r.comp_max_abs_diff:.6e} |")
        lines.append("")
        lines.append(f"**Conclusion:** {_conclusion(r)}")
        lines.append("")
        # Per-iteration table
        n = min(len(r.dense_history), len(r.sparse_history))
        lines.append("### Per-iteration comparison")
        lines.append("")
        lines.append(
            "| it | mu(d) | mu(s) | Δmu | primal(d) | primal(s) | Δprimal | "
            "dual(d) | dual(s) | Δdual | gap(d) | gap(s) | Δgap |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i in range(n):
            d, s = r.dense_history[i], r.sparse_history[i]
            lines.append(
                f"| {d['iter']} "
                f"| {d['mu']:.5e} | {s['mu']:.5e} | {abs(d['mu']-s['mu']):.5e} "
                f"| {d['primal']:.5e} | {s['primal']:.5e} | "
                f"{abs(d['primal']-s['primal']):.5e} "
                f"| {d['dual']:.5e} | {s['dual']:.5e} | "
                f"{abs(d['dual']-s['dual']):.5e} "
                f"| {d['rel_gap']:.5e} | {s['rel_gap']:.5e} | "
                f"{abs(d['rel_gap']-s['rel_gap']):.5e} |"
            )
        if len(r.dense_history) != len(r.sparse_history):
            lines.append("")
            lines.append(f"> Dense finished in {len(r.dense_history)} iterations, "
                         f"sparse in {len(r.sparse_history)} iterations.")
        lines.append("")
    lines.append("## Overall verdict")
    lines.append("")
    all_identical = all("IDENTICAL" in _conclusion(r) for r in results)
    all_divergent = all("DIVERGENT" in _conclusion(r) for r in results)
    if all_identical:
        lines.append(
            "**All models show identical trajectories -> the sparse LSQR "
            "initialization was the primary cause of any previous differences.**"
        )
    elif all_divergent:
        lines.append(
            "**All models show divergent trajectories -> sparse arithmetic "
            "in the Mehrotra iteration contributes independently.**"
        )
    else:
        lines.append(
            "**Mixed results** -- some models converge identically once the "
            "initial point is shared (init-driven), while others still diverge "
            "(arithmetic-driven)."
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"tol={TOL}  max_iter={MAX_ITER}")
    print("Both pipelines use the SAME initial point (from the dense path).")
    print("Any trajectory difference is caused by sparse arithmetic, not init.\n")

    results: list[ModelResult] = []
    for name in MODELS:
        r = run_model(DATA_DIR / name)
        if r is not None:
            results.append(r)

    csv_path = RESULTS_DIR / "controlled_init_comparison.csv"
    md_path = RESULTS_DIR / "controlled_init_comparison.md"
    write_csv(results, csv_path)
    write_md(results, md_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r.model:20s} -> {_conclusion(r)}")


if __name__ == "__main__":
    main()



