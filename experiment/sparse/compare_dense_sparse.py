"""Compare the dense pipeline against the new sparse pipeline end-to-end.

For each MPS benchmark model we load the model twice via
``load_numeric_mps(path, sparse=False)`` and ``load_numeric_mps(path,
sparse=True)`` and solve both with the existing ``solve_lp()`` using the
same tolerance / iteration budget.

This is a self-comparison of our own Mehrotra solver only:

* The dense path receives a dense ``np.ndarray`` ``A``.
* The sparse path receives the *sparse* ``scipy.sparse.csr_matrix`` ``A``
  and the sparse standard-form / scaling machinery keeps it sparse.  The
  sparse model is never converted to dense except for the small diagnostic
  objective/solution checks, which are exact and cheap.

Reported columns follow the spec: input stats, per-path timing, result
fields, and PASS/FAIL correctness checks driven by numerical tolerances.

Run:
    python experiment/sparse/compare_dense_sparse.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _SRC / "lp", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Project flat-layout imports (do not import external solvers).
from numerical_model import load_numeric_mps, NumericalLP  # noqa: E402
from scaling import scale_lp  # noqa: E402
from mehrotra import MehrotraResult, solve_lp, to_standard_form  # noqa: E402

# Optional dependency: only used for process-RSS reporting when present.
try:  # pragma: no cover - environment dependent
    import psutil  # type: ignore

    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    _HAVE_PSUTIL = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"
BENCHMARKS = ["afiro.mps", "blend.mps", "sc205.mps", "pilot4_plain.mps", "pilot87.mps"]
TOL = 1e-7
MAX_ITER = 100
OBJ_RTOL = 1e-6
SOL_ATOL = 1e-6
RES_ATOL = 1e-6

@dataclass
class Run:
    """One solve path's inputs, timings, results and RSS deltas."""
    path_type: str  # "dense" | "sparse"
    rows: int
    cols: int
    nnz: int
    density: float
    std_sparse: bool
    scaled_sparse: bool
    load_time: float
    solve_time: float
    total_time: float
    rss_before: Optional[float]
    rss_after: Optional[float]
    rss_delta: Optional[float]
    status: str
    objective: float
    x: Optional[np.ndarray]
    primal_residual: float
    dual_residual: float
    rel_gap: float
    iterations: int

def _rss_mb() -> Optional[float]:
    if not _HAVE_PSUTIL:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:  # pragma: no cover
        return None



# ---------------------------------------------------------------------------
# Per-path solver
# ---------------------------------------------------------------------------
def _run_path(lp: NumericalLP, *, sparse: bool, name: str) -> Run:
    """Load is timed separately; this does the solve on the already-built
    model and returns the per-path diagnostics."""
    m, n = lp.A.shape
    nnz = lp.nnz
    density = nnz / float(m * n) if m * n else 0.0

    # Standard-form / scaled-matrix sparsity diagnostics (diagnostic only).
    std_sparse = sp.issparse(to_standard_form(lp).A)
    scaled_sparse = False
    try:
        scaled = scale_lp(lp.A, lp.b, lp.c, lp.lower_bounds, lp.upper_bounds)
        scaled_sparse = sp.issparse(scaled.A)
    except Exception:
        scaled_sparse = False

    # ---- solve timing ----
    rss_before = _rss_mb()
    t_solve = time.perf_counter()
    result = solve_lp(lp, tol=TOL, max_iter=MAX_ITER)
    solve_time = time.perf_counter() - t_solve
    rss_after = _rss_mb()

    rss_delta = (
        (rss_after - rss_before)
        if (rss_after is not None and rss_before is not None)
        else None
    )

    return Run(
        path_type="sparse" if sparse else "dense",
        rows=m,
        cols=n,
        nnz=nnz,
        density=density,
        std_sparse=std_sparse,
        scaled_sparse=scaled_sparse,
        load_time=0.0,
        solve_time=solve_time,
        total_time=solve_time,
        rss_before=rss_before,
        rss_after=rss_after,
        rss_delta=rss_delta,
        status=result.status,
        objective=float(result.objective),
        x=np.asarray(result.x),
        primal_residual=float(result.primal_residual),
        dual_residual=float(result.dual_residual),
        rel_gap=float(result.rel_gap),
        iterations=int(result.iterations),
    )


def _status_compatible(dense_status: str, sparse_status: str) -> bool:
    """Both converged, or both hit an identical non-optimal condition."""
    if dense_status == sparse_status:
        return True
    optimal = {"optimal", "numerical_tail"}
    if dense_status in optimal and sparse_status in optimal:
        return True
    return False



# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------
def run_model(path: Path, quiet: bool = False) -> Optional[dict]:
    name = path.stem
    if not path.is_file():
        return None

    if not quiet:
        print(f"\n=== {name} ===", flush=True)

    # ---- load both representations (timed API calls) ----
    t0 = time.perf_counter()
    dense_lp = load_numeric_mps(path, sparse=False)
    dense_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    sparse_lp = load_numeric_mps(path, sparse=True)
    sparse_load = time.perf_counter() - t0

    assert isinstance(dense_lp.A, np.ndarray), "dense load must yield ndarray"
    assert sp.issparse(sparse_lp.A), "sparse load must yield sparse matrix"
    assert dense_lp.A.shape == sparse_lp.A.shape

    dense_run = _run_path(dense_lp, sparse=False, name=name)
    dense_run.load_time = dense_load
    dense_run.total_time = dense_load + dense_run.solve_time
    sparse_run = _run_path(sparse_lp, sparse=True, name=name)
    sparse_run.load_time = sparse_load
    sparse_run.total_time = sparse_load + sparse_run.solve_time

    # ---- correctness checks ----
    obj_diff = abs(dense_run.objective - sparse_run.objective)
    obj_pass = obj_diff <= OBJ_RTOL * max(1.0, abs(dense_run.objective))
    res_primal_pass = (
        abs(dense_run.primal_residual - sparse_run.primal_residual) <= RES_ATOL
    )
    res_dual_pass = (
        abs(dense_run.dual_residual - sparse_run.dual_residual) <= RES_ATOL
    )
    status_pass = _status_compatible(dense_run.status, sparse_run.status)

    sol_pass = False
    if dense_run.x is not None and sparse_run.x is not None:
        if dense_run.x.shape == sparse_run.x.shape:
            sol_pass = np.allclose(
                dense_run.x, sparse_run.x, atol=SOL_ATOL, rtol=OBJ_RTOL
            )

    return {
        "model": name,
        "rows": dense_run.rows,
        "cols": dense_run.cols,
        "dense_nnz": dense_run.nnz,
        "sparse_nnz": sparse_run.nnz,
        "density": dense_run.density,
        "std_A_sparse": sparse_run.std_sparse,
        "scaled_A_sparse": sparse_run.scaled_sparse,
        "dense_load": dense_load,
        "sparse_load": sparse_load,
        "dense_solve": dense_run.solve_time,
        "sparse_solve": sparse_run.solve_time,
        "dense_total": dense_run.total_time,
        "sparse_total": sparse_run.total_time,
        "dense_rss_delta": dense_run.rss_delta,
        "sparse_rss_delta": sparse_run.rss_delta,
        "dense_status": dense_run.status,
        "sparse_status": sparse_run.status,
        "dense_objective": dense_run.objective,
        "sparse_objective": sparse_run.objective,
        "obj_diff": obj_diff,
        "dense_primal": dense_run.primal_residual,
        "sparse_primal": sparse_run.primal_residual,
        "dense_dual": dense_run.dual_residual,
        "sparse_dual": sparse_run.dual_residual,
        "dense_gap": dense_run.rel_gap,
        "sparse_gap": sparse_run.rel_gap,
        "dense_iters": dense_run.iterations,
        "sparse_iters": sparse_run.iterations,
        "obj_pass": obj_pass,
        "sol_pass": sol_pass,
        "res_primal_pass": res_primal_pass,
        "res_dual_pass": res_dual_pass,
        "status_pass": status_pass,
    }



# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(v, nd: int = 6) -> str:
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_md(rows: list[dict], missing: list[str], path: Path) -> None:
    lines = []
    lines.append("# Sparse vs Dense End-to-End Comparison")
    lines.append("")
    lines.append(f"- tolerance `tol={TOL}`, `max_iter={MAX_ITER}`")
    lines.append(f"- psutil available: {_HAVE_PSUTIL}")
    lines.append("")
    lines.append("## Available / Missing models")
    lines.append("")
    lines.append(f"**Ran ({len(rows)}):** " + ", ".join(r["model"] for r in rows))
    if missing:
        lines.append("")
        lines.append(f"**Missing ({len(missing)}):** " + ", ".join(missing))
    lines.append("")
    lines.append("## Per-model results")
    lines.append("")
    lines.append(
        "| Model | Rows | Cols | SparseNNZ | Density | DenseStatus | "
        "SparseStatus | DenseObj | SparseObj | ObjDiff | DenseSolve(s) | "
        "SparseSolve(s) | DenseIters | SparseIters | Obj | Sol | PrimalRes | "
        "DualRes | Status |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            "| {model} | {rows} | {cols} | {sparse_nnz} | {density:.5g} | "
            "{dense_status} | {sparse_status} | {dense_objective:.6g} | "
            "{sparse_objective:.6g} | {obj_diff:.3e} | {dense_solve:.3g} | "
            "{sparse_solve:.3g} | {dense_iters} | {sparse_iters} | "
            "{obj} | {sol} | {res_primal} | {res_dual} | {status} |".format(
                model=r["model"], rows=r["rows"], cols=r["cols"],
                sparse_nnz=r["sparse_nnz"], density=r["density"],
                dense_status=r["dense_status"], sparse_status=r["sparse_status"],
                dense_objective=r["dense_objective"],
                sparse_objective=r["sparse_objective"], obj_diff=r["obj_diff"],
                dense_solve=r["dense_solve"], sparse_solve=r["sparse_solve"],
                dense_iters=r["dense_iters"], sparse_iters=r["sparse_iters"],
                obj=_fmt(r["obj_pass"]), sol=_fmt(r["sol_pass"]),
                res_primal=_fmt(r["res_primal_pass"]),
                res_dual=_fmt(r["res_dual_pass"]), status=_fmt(r["status_pass"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


COLUMNS = [
    "model", "rows", "cols", "dense_nnz", "sparse_nnz", "density",
    "std_A_sparse", "scaled_A_sparse",
    "dense_load", "sparse_load", "dense_solve", "sparse_solve",
    "dense_total", "sparse_total", "dense_rss_delta", "sparse_rss_delta",
    "dense_status", "sparse_status",
    "dense_objective", "sparse_objective", "obj_diff",
    "dense_primal", "sparse_primal", "dense_dual", "sparse_dual",
    "dense_gap", "sparse_gap", "dense_iters", "sparse_iters",
    "obj_pass", "sol_pass", "res_primal_pass", "res_dual_pass", "status_pass",
]


def write_csv(rows: list[dict], missing: list[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_table(rows: list[dict]) -> None:
    print("\n=== SUMMARY TABLE ===")
    header = (
        f"{'model':12} {'rows':>5} {'cols':>5} {'nnz':>7} {'density':>8} "
        f"{'d_status':>14} {'s_status':>14} "
        f"{'d_obj':>12} {'s_obj':>12} {'d_objdiff':>10} "
        f"{'d_tot':>8} {'s_tot':>8} {'d_it':>4} {'s_it':>4} "
        f"{'obj':>5} {'sol':>5} {'res':>5} {'status':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:12} {r['rows']:5d} {r['cols']:5d} {r['sparse_nnz']:7d} "
            f"{r['density']:8.4g} {r['dense_status']:>14} {r['sparse_status']:>14} "
            f"{r['dense_objective']:12.6g} {r['sparse_objective']:12.6g} "
            f"{r['obj_diff']:10.3e} {r['dense_total']:8.3g} {r['sparse_total']:8.3g} "
            f"{r['dense_iters']:4d} {r['sparse_iters']:4d} "
            f"{_fmt(r['obj_pass']):>5} {_fmt(r['sol_pass']):>5} "
            f"{_fmt(r['res_primal_pass']):>5} {_fmt(r['status_pass']):>7}"
        )
    print()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"psutil available: {_HAVE_PSUTIL}")
    print(f"tol={TOL}  max_iter={MAX_ITER}")

    models = [DATA_DIR / b for b in BENCHMARKS]
    missing = [m.name for m in models if not m.is_file()]
    if missing:
        print("Skipping missing files:")
        for name in missing:
            print(f"  - {name}")

    rows = []
    for path in models:
        r = run_model(path, quiet=("--quiet" in sys.argv))
        if r is None:
            continue
        rows.append(r)

    print_table(rows)

    csv_path = RESULTS_DIR / "sparse_dense_comparison.csv"
    md_path = RESULTS_DIR / "sparse_dense_comparison.md"
    write_csv(rows, missing, csv_path)
    write_md(rows, missing, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

