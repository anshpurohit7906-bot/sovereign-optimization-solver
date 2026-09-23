#!/usr/bin/env python
"""Sparse scalability benchmark — measures how far the sparse end-to-end
Mehrotra pipeline scales on deterministic synthetic sparse LPs.

EXPERIMENT ONLY: does not modify src/ or tests/.
"""
import sys
import time
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from opticore.numerical_model import NumericalLP  # noqa: E402
from opticore.scaling import scale_lp  # noqa: E402
import opticore.lp.mehrotra as _mh  # noqa: E402

# Optional psutil
try:
    import psutil
    HAS_PSUTIL = True
    _PROCESS = psutil.Process()
except Exception:
    HAS_PSUTIL = False
    _PROCESS = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = _ROOT / "results"
TOL = 1e-7
MAX_ITER = 100
DENSE_MEMORY_LIMIT_MB = 512.0
TEST_CASES = [
    (500, 250, 0.5),
    (1000, 500, 0.5),
    (2000, 1000, 0.5),
    (5000, 2500, 0.3),
    (10000, 5000, 0.2),
]

SEP = "=" * 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_memory_mb():
    """Return current process RSS in MB, or None if unavailable."""
    if not HAS_PSUTIL:
        return None
    try:
        return float(_PROCESS.memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def est_dense_mem(rows, cols):
    """Estimate dense A matrix memory in MB."""
    return (rows * cols * 8) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    rows: int
    cols: int
    nnz: int
    density_pct: float


@dataclass
class BenchmarkResult:
    rows: int
    cols: int
    nnz: int
    density_pct: float
    est_dense_mb: float
    # Sparse timing
    sp_load_s: float
    sp_solve_s: float
    sp_total_s: float
    sp_iters: int
    sp_status: str
    sp_obj: float
    sp_pres: float
    sp_dres: float
    sp_gap: float
    sp_mem_delta: Optional[float]
    # Dense timing
    dn_status: str
    dn_solve_s: Optional[float] = None
    dn_iters: Optional[int] = None
    dn_obj: Optional[float] = None
    dn_pres: Optional[float] = None
    dn_dres: Optional[float] = None
    dn_gap: Optional[float] = None
    # Derived
    speedup: Optional[float] = None
    objdiff: Optional[float] = None


# ---------------------------------------------------------------------------
# Synthetic LP generation
# ---------------------------------------------------------------------------

def generate_deterministic_sparse_lp(rows, cols, density_pct, seed=42):
    """Build a NumericalLP with a CSR sparse A (never materialised dense).

    - rng fixed via seed for reproducibility
    - Guarantees every row and every column has >= 1 nonzero entry
    - row_types: 60% equality, 20% <=, 20% >=
    - b, c drawn from N(0,1)
    - lower_bounds = 0, upper_bounds = inf  (non-negative vars)
    """
    rng = np.random.RandomState(seed)
    target_nnz = max(1, int(rows * cols * (density_pct / 100.0)))
    # Ensure at least one entry per row and per column
    min_nnz = rows + cols
    actual_target = max(target_nnz, min_nnz)
    row_idx = rng.randint(0, rows, size=actual_target)
    col_idx = rng.randint(0, cols, size=actual_target)
    data = rng.randn(actual_target)
    A_sp = sp.csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols),
                         dtype=np.float64)
    # Guarantee: place one nonzero in every all-zero row and column
    for r in range(rows):
        if A_sp[r].nnz == 0:
            c = rng.randint(0, cols)
            A_sp[r, c] = rng.randn()
    A_sp = A_sp.tocsr()  # re-sort after in-place edits
    for c in range(cols):
        if A_sp[:, c].nnz == 0:
            r = rng.randint(0, rows)
            A_sp[r, c] = rng.randn()
    A_sp = A_sp.tocsr()
    actual_nnz = A_sp.nnz  # may differ from target due to duplicates
    b = rng.randn(rows)
    c = rng.randn(cols)
    row_types = tuple(rng.choice(["E", "L", "G"], size=rows, p=[0.6, 0.2, 0.2]))
    lp = NumericalLP(
        name=f"synthetic_{rows}x{cols}",
        objective_name="obj",
        A=A_sp,
        b=b,
        c=c,
        lower_bounds=np.zeros(cols),
        upper_bounds=np.full(cols, np.inf),
        row_types=row_types,
        var_names=tuple(f"x{i}" for i in range(cols)),
        row_names=tuple(f"c{i}" for i in range(rows)),
    )
    return lp, SynthesisResult(rows, cols, actual_nnz, 100.0 * actual_nnz / (rows * cols))


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------

def _run_sparse(lp):
    """Solve via sparse pipeline; return (result, wall_seconds)."""
    t0 = time.perf_counter()
    result = _mh.solve_lp(lp, tol=TOL, max_iter=MAX_ITER)
    return result, time.perf_counter() - t0


def _run_dense(lp):
    """Solve via dense pipeline (dense NumericalLP copy); return (result, wall_seconds)."""
    dense_A = np.asarray(lp.A.toarray(), dtype=np.float64)
    dense_lp = NumericalLP(
        name=lp.name,
        objective_name=lp.objective_name,
        A=dense_A,
        b=lp.b.copy(),
        c=lp.c.copy(),
        lower_bounds=lp.lower_bounds.copy(),
        upper_bounds=lp.upper_bounds.copy(),
        row_types=lp.row_types,
        var_names=lp.var_names,
        row_names=lp.row_names,
    )
    t0 = time.perf_counter()
    result = _mh.solve_lp(dense_lp, tol=TOL, max_iter=MAX_ITER)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def run_case(rows, cols, density_pct):
    """Run one scaling case: generate → sparse solve → (conditional) dense solve → verify."""
    print(f"\n{SEP} {rows}x{cols}, density={density_pct:.2f}% {SEP}")

    # --- 1. Synthesis ---
    print("  [1/4] Generating synthetic LP ...")
    t0 = time.perf_counter()
    lp, synth = generate_deterministic_sparse_lp(rows, cols, density_pct)
    load_time = time.perf_counter() - t0
    print(f"    {synth.nnz} NNZ (density {synth.density_pct:.4f}%) in {load_time:.3f}s")

    # Dense memory estimate
    dense_mem_mb = est_dense_mem(rows, cols)
    dense_safe = dense_mem_mb <= DENSE_MEMORY_LIMIT_MB
    print(f"    Est dense memory: {dense_mem_mb:.1f} MB  (safe={dense_safe})")

    # --- 2. Sparse solve ---
    print("  [2/4] Sparse solver ...")
    mem_before = get_memory_mb()
    sp_result, sp_time = _run_sparse(lp)
    mem_after = get_memory_mb()
    mem_delta = None
    if mem_before is not None and mem_after is not None:
        mem_delta = mem_after - mem_before
        print(f"    RSS delta: {mem_delta:+.1f} MB")

    if sp_result is None:
        print("    FAILED — no result returned")
        return BenchmarkResult(
            rows=rows, cols=cols, nnz=synth.nnz, density_pct=synth.density_pct,
            est_dense_mb=dense_mem_mb, sp_load_s=load_time, sp_solve_s=sp_time,
            sp_total_s=load_time + sp_time, sp_iters=0, sp_status="FAILED",
            sp_obj=0.0, sp_pres=0.0, sp_dres=0.0, sp_gap=0.0, sp_mem_delta=mem_delta,
            dn_status="NOT_RUN",
        )
    print(f"    status={sp_result.status}  iters={sp_result.iterations}  "
          f"obj={sp_result.objective:.6g}")

    # --- 3. Dense solve (conditional) ---
    dn_result = None
    dn_time = None
    if dense_safe:
        print("  [3/4] Dense solver ...")
        try:
            dn_result, dn_time = _run_dense(lp)
            print(f"    status={dn_result.status}  iters={dn_result.iterations}  "
                  f"obj={dn_result.objective:.6g}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            dn_result = None
    else:
        print(f"  [3/4] Dense SKIP (est {dense_mem_mb:.0f} MB > {DENSE_MEMORY_LIMIT_MB:.0f} MB limit)")

    # --- 4. Sparsity verification ---
    print("  [4/4] Verifying sparsity ...")
    std = _mh.to_standard_form(lp)
    scaled = scale_lp(std.A, std.b, std.c_min, np.zeros(std.n), np.full(std.n, np.inf))
    std_sparse = sp.issparse(std.A)
    scaled_sparse = sp.issparse(scaled.A)
    print(f"    standard_form.A sparse={std_sparse}  scaled.A sparse={scaled_sparse}")

    # --- Build result ---
    speedup = None
    objdiff = None
    if dn_result is not None and dn_time is not None and sp_time > 0:
        speedup = dn_time / sp_time
        objdiff = abs(dn_result.objective - sp_result.objective)
        print(f"    speedup={speedup:.2f}x  |obj_diff|={objdiff:.3e}")

    return BenchmarkResult(
        rows=rows, cols=cols, nnz=synth.nnz, density_pct=synth.density_pct,
        est_dense_mb=dense_mem_mb,
        sp_load_s=load_time, sp_solve_s=sp_time,
        sp_total_s=load_time + sp_time, sp_iters=sp_result.iterations,
        sp_status=sp_result.status,
        sp_obj=sp_result.objective,
        sp_pres=sp_result.primal_residual,
        sp_dres=sp_result.dual_residual,
        sp_gap=sp_result.rel_gap,
        sp_mem_delta=mem_delta,
        dn_status=dn_result.status if dn_result else ("SKIPPED_MEMORY" if not dense_safe else "ERROR"),
        dn_solve_s=dn_time,
        dn_iters=dn_result.iterations if dn_result else None,
        dn_obj=dn_result.objective if dn_result else None,
        dn_pres=dn_result.primal_residual if dn_result else None,
        dn_dres=dn_result.dual_residual if dn_result else None,
        dn_gap=dn_result.rel_gap if dn_result else None,
        speedup=speedup,
        objdiff=objdiff,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

CSV_COLS = [
    "rows", "cols", "nnz", "density_pct", "est_dense_mb",
    "sp_load_s", "sp_solve_s", "sp_total_s", "sp_iters", "sp_status",
    "sp_obj", "sp_pres", "sp_dres", "sp_gap", "sp_mem_delta",
    "dn_status", "dn_solve_s", "dn_iters", "dn_obj",
    "dn_pres", "dn_dres", "dn_gap", "speedup", "objdiff",
]


def _fmt(val, fmt_str):
    return format(val, fmt_str) if val is not None else "N/A"


def write_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in CSV_COLS}
            # Format floats
            for k in ("density_pct", "est_dense_mb"):
                row[k] = f"{row[k]:.4f}" if row[k] is not None else "N/A"
            for k in ("sp_load_s", "sp_solve_s", "sp_total_s", "dn_solve_s"):
                row[k] = f"{row[k]:.4f}" if row[k] is not None else "N/A"
            for k in ("sp_obj", "dn_obj"):
                row[k] = _fmt(row[k], ".6g")
            for k in ("sp_pres", "sp_dres", "sp_gap", "dn_pres", "dn_dres", "dn_gap", "objdiff"):
                row[k] = _fmt(row[k], ".6e")
            if row.get("sp_mem_delta") is not None:
                row["sp_mem_delta"] = f"{row['sp_mem_delta']:.2f}"
            if row.get("speedup") is not None:
                row["speedup"] = f"{row['speedup']:.2f}"
            for k in ("sp_iters", "dn_iters"):
                if row.get(k) is not None:
                    row[k] = int(row[k])
            writer.writerow(row)
    print(f"Wrote {path}")


def write_md(results, path):
    lines = [
        "# Sparse Scalability Benchmark",
        "",
        f"- `tol={TOL}`",
        f"- `max_iter={MAX_ITER}`",
        f"- dense memory limit: {DENSE_MEMORY_LIMIT_MB:.0f} MB",
        f"- psutil available: {HAS_PSUTIL}",
        "",
        "## Summary",
        "",
    ]
    # Dense crossover
    skip = next((r for r in results if "SKIPPED" in r.dn_status), None)
    if skip:
        lines.append(
            f"- **Dense crossover:** {skip.rows}x{skip.cols} "
            f"(est {skip.est_dense_mb:.0f} MB > {DENSE_MEMORY_LIMIT_MB:.0f} MB limit)"
        )
    # Largest success
    successes = [r for r in results if r.sp_status in ("optimal", "stalled", "numerical_tail")]
    if successes:
        best = max(successes, key=lambda r: r.rows)
        lines.append(
            f"- **Largest sparse case:** {best.rows}x{best.cols} "
            f"({best.nnz} NNZ, {best.sp_total_s:.2f}s, {best.sp_iters} iters)"
        )
    # Speedup
    speedups = [r.speedup for r in results if r.speedup is not None]
    if speedups:
        lines.append(f"- **Dense-vs-sparse speedup:** {np.mean(speedups):.2f}x avg "
                     f"(min={min(speedups):.2f}x, max={max(speedups):.2f}x)")
    # Bottleneck
    stalled = [r for r in results if r.sp_status == "stalled"]
    if stalled:
        s = stalled[-1]
        lines.append(f"- **Iteration bottleneck:** {s.rows}x{s.cols} stalled at "
                     f"{s.sp_iters}/{MAX_ITER} iters, gap={s.sp_gap:.3e}")
    lines.extend(["", "## Per-case Results", ""])

    # Table header
    hdr = "| # | Rows | Cols | NNZ | Dens% | SpIters | SpStat | SpTime(s) | "
    hdr += "DnStat | DnTime(s) | Speedup | ObjDiff |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.extend([hdr, sep])
    for i, r in enumerate(results, 1):
        dn_time_s = f"{r.dn_solve_s:.3f}" if r.dn_solve_s is not None else "—"
        spdup = f"{r.speedup:.2f}x" if r.speedup is not None else "—"
        od = f"{r.objdiff:.2e}" if r.objdiff is not None else "—"
        lines.append(
            f"| {i} | {r.rows} | {r.cols} | {r.nnz} | {r.density_pct:.2f} "
            f"| {r.sp_iters} | {r.sp_status} | {r.sp_total_s:.3f} "
            f"| {r.dn_status} | {dn_time_s} | {spdup} | {od} |"
        )
    lines.extend(["", "---", f"*Generated by `{Path(__file__).name}`*"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#' * 70}")
    print(f"#  SPARSE SCALABILITY BENCHMARK")
    print(f"#  tol={TOL}  max_iter={MAX_ITER}  dense_limit={DENSE_MEMORY_LIMIT_MB:.0f}MB")
    print(f"{'#' * 70}")

    results = []
    for rows, cols, den in TEST_CASES:
        try:
            res = run_case(rows, cols, den)
            results.append(res)
        except Exception as exc:
            print(f"  *** EXCEPTION in {rows}x{cols}: {exc} ***")
            import traceback; traceback.print_exc()
            results.append(BenchmarkResult(
                rows=rows, cols=cols, nnz=0, density_pct=den,
                est_dense_mb=est_dense_mem(rows, cols),
                sp_load_s=0, sp_solve_s=0, sp_total_s=0,
                sp_iters=0, sp_status="EXCEPTION",
                sp_obj=0, sp_pres=0, sp_dres=0, sp_gap=0,
                sp_mem_delta=None, dn_status="NOT_RUN",
            ))

    csv_path = RESULTS_DIR / "sparse_scaling_benchmark.csv"
    md_path = RESULTS_DIR / "sparse_scaling_benchmark.md"
    write_csv(results, csv_path)
    write_md(results, md_path)

    print(f"\n{'#' * 70}")
    print(f"#  DONE — {len(results)} cases")
    ok = sum(1 for r in results if r.sp_status in ("optimal", "stalled"))
    print(f"#  Sparse success: {ok}/{len(results)}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()

