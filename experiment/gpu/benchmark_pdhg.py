#!/usr/bin/env python3
"""Milestone 2 - GPU track: PDHG CPU-vs-GPU benchmark.

Compares the CPU PDHG solver (``experiment/pdhg/pdhg_mixed.py``) against the
CuPy GPU mirror (``experiment/gpu/pdhg_gpu.py``) on deterministic, synthetic
sparse LPs of increasing size.

Comparison metrics for each case:
  * iterations to convergence
  * final objective
  * KKT residuals (equality, inequality, dual feasibility, complementarity)
  * wall-clock runtime (solver-only and end-to-end)

Timing modes:
  * solver-only : isolates the Chambolle-Pock iteration loop
                  (GPU timed with CUDA events, CPU with time.perf_counter)
  * end-to-end  : includes data transfer / setup on both platforms

Only NumPy, SciPy, CuPy and the two PDHG modules are used.  No real-world
problem files (e.g. PILOT87) are loaded or referenced.

Usage:
    python experiment/gpu/benchmark_pdhg.py
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

# Import CPU reference solver (pdhg_mixed imports `numerical_model` from src/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "experiment" / "pdhg"))  # pdhg_mixed
from opticore.numerical_model import NumericalLP  # noqa: E402
from pdhg_mixed import pdhg_mixed  # noqa: E402

# Import GPU solver + its internals for solver-only timing.
GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR))
from pdhg_gpu import (  # noqa: E402
    _CUPY_AVAILABLE,
    _gpu_iteration_loop,
    _spectral_norm_sparse,
    _csr_to_gpu,
    pdhg_gpu,
)

if _CUPY_AVAILABLE:
    import cupy as cp

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
RNG_SEED = 1234

# Defaults; can be overridden via CLI/env (see `_parse_args()`).
MAX_ITER = 50_000
TOL = 1e-7
CHECK_EVERY = 250

# (rows, cols, density) - deterministic, LP-shaped sparse workloads.
# Keep moderate so GPU has room within its 4 GB budget.
CASES: list[tuple[int, int, float]] = [
    (200, 100, 0.05),
    (500, 250, 0.04),
    (1_000, 500, 0.03),
    (2_000, 1_000, 0.02),
    (4_000, 2_000, 0.015),
]

SOLVER_ONLY_REPS = 3   # solver-only timing repetitions
END_TO_END_REPS = 3    # end-to-end timing repetitions


def _parse_args() -> dict[str, Any]:
    """Parse CLI/env overrides for the benchmark configuration.

    Supports: ``--max-iter``, ``--tol``, ``--check-every``,
    ``--solver-reps``, ``--e2e-reps``, ``--out DIR``, ``--no-save``.
    Environment variables (``BENCH_MAX_ITER``, ``BENCH_TOL``, ...) are used
    as fallbacks when the corresponding flag is not passed, so a fully
    non-interactive run can be configured in a single command line.
    """
    import argparse
    import os

    p = argparse.ArgumentParser(description="PDHG CPU vs GPU benchmark")
    p.add_argument("--max-iter", type=int,
                   default=os.environ.get("BENCH_MAX_ITER", None))
    p.add_argument("--tol", type=float,
                   default=os.environ.get("BENCH_TOL", None))
    p.add_argument("--check-every", type=int,
                   default=os.environ.get("BENCH_CHECK_EVERY", None))
    p.add_argument("--solver-reps", type=int,
                   default=os.environ.get("BENCH_SOLVER_REPS", None))
    p.add_argument("--e2e-reps", type=int,
                   default=os.environ.get("BENCH_E2E_REPS", None))
    p.add_argument("--out", default=os.environ.get("BENCH_OUT_DIR", None),
                   help="Directory to write results CSV/MD (default: results/).")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write results files.")
    args = p.parse_args()
    return vars(args)


def _apply_config(cfg: dict[str, Any]) -> None:
    """Fold CLI/env overrides into the module-level benchmark settings.

    Mutates the module globals so the rest of the code path is unchanged.
    """
    global MAX_ITER, TOL, CHECK_EVERY, SOLVER_ONLY_REPS, END_TO_END_REPS
    if cfg.get("max_iter") is not None:
        MAX_ITER = cfg["max_iter"]
    if cfg.get("tol") is not None:
        TOL = cfg["tol"]
    if cfg.get("check_every") is not None:
        CHECK_EVERY = cfg["check_every"]
    if cfg.get("solver_reps") is not None:
        SOLVER_ONLY_REPS = cfg["solver_reps"]
    if cfg.get("e2e_reps") is not None:
        END_TO_END_REPS = cfg["e2e_reps"]


# ---------------------------------------------------------------------------
# Problem generation (deterministic feasible LPs)
# ---------------------------------------------------------------------------
@dataclass
class GeneratedLP:
    """A deterministic sparse LP, stored densely for easy CPU use."""

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    row_types: tuple[str, ...]
    name: str
    m: int
    n: int
    nnz: int

    def to_numerical_lp(self) -> NumericalLP:
        """Build a NumericalLP for the CPU reference solver."""
        return NumericalLP(
            name=self.name,
            objective_name="bench",
            A=self.A,
            b=self.b,
            c=self.c,
            lower_bounds=self.lower,
            upper_bounds=self.upper,
            row_types=self.row_types,
            var_names=tuple(f"x{i}" for i in range(self.n)),
            row_names=tuple(f"r{i}" for i in range(self.m)),
        )


def _generate_lp(rows: int, cols: int, density: float, rng: np.random.Generator,
                 name: str) -> GeneratedLP:
    """Generate a deterministic, feasible sparse LP with E and L rows.

    A random sparse matrix is drawn, then an interior feasible point is used
    to build a consistent right-hand side so the LP is guaranteed feasible
    (and bounded by the box constraints).
    """
    # Sparse matrix with roughly `density * rows * cols` nonzeros.
    nnz = max(1, int(density * rows * cols))
    n_draws = int(nnz * 1.5) + 1
    row_idx = rng.integers(0, rows, size=n_draws)
    col_idx = rng.integers(0, cols, size=n_draws)
    data = rng.standard_normal(n_draws).astype(np.float64)

    A_csr = sp.coo_matrix((data, (row_idx, col_idx)), shape=(rows, cols)).tocsr()
    A = A_csr.toarray()

    # First third of rows are equalities, rest are <=.
    n_eq = max(1, rows // 3)
    n_ub = rows - n_eq
    row_types = ("E",) * n_eq + ("L",) * n_ub

    # A feasible interior point strictly inside the box.
    x_feas = rng.uniform(0.5, 1.5, size=cols).astype(np.float64)

    # Equality rows: b = A x_feas (exactly consistent).
    # Inequality rows: b >= A x_feas + margin (strictly feasible).
    A_eq = A[:n_eq]
    A_ub = A[n_eq:]
    b_eq = A_eq @ x_feas
    b_ub = A_ub @ x_feas + rng.uniform(0.5, 2.0, size=n_ub).astype(np.float64)
    b = np.concatenate([b_eq, b_ub])

    c = rng.standard_normal(cols).astype(np.float64)
    lower = np.full(cols, -1.0)
    upper = np.full(cols, 2.5)

    return GeneratedLP(
        A=A, b=b, c=c, lower=lower, upper=upper,
        row_types=row_types, name=name,
        m=rows, n=cols, nnz=A_csr.nnz,
    )


# ---------------------------------------------------------------------------
# CPU solver timing (uses pdhg_mixed)
# ---------------------------------------------------------------------------
def _run_cpu(gp: GeneratedLP, *, end_to_end: bool) -> tuple[float, Any]:
    """Run the CPU reference solver; returns (seconds, result)."""
    lp = gp.to_numerical_lp()
    t0 = time.perf_counter()
    result = pdhg_mixed(
        lp, max_iter=MAX_ITER, tol=TOL, check_every=CHECK_EVERY
    )
    elapsed = time.perf_counter() - t0
    # CPU end-to-end == solver-only (no separate transfer); both are same.
    return elapsed, result


# ---------------------------------------------------------------------------
# GPU solver timing (solver-only via _gpu_iteration_loop, plus end-to-end)
# ---------------------------------------------------------------------------
def _prepare_gpu(gp: GeneratedLP):
    """Build GPU-resident arrays + matrices once for reuse in timing reps."""
    m, n = gp.m, gp.n
    n_eq = sum(1 for t in gp.row_types if t == "E")

    A_csr = sp.csr_matrix(gp.A)
    norm_A = _spectral_norm_sparse(A_csr)
    tau = 1.0 if norm_A == 0.0 else 0.9 / norm_A
    sigma = tau

    A_eq_csr = sp.csr_matrix(gp.A[:n_eq])
    A_ub_csr = sp.csr_matrix(gp.A[n_eq:])

    A_eq_gpu = _csr_to_gpu(A_eq_csr) if n_eq > 0 else None
    A_ub_gpu = _csr_to_gpu(A_ub_csr) if (m - n_eq) > 0 else None
    b_eq_gpu = cp.asarray(gp.b[:n_eq], dtype=cp.float64) if n_eq > 0 else None
    b_ub_gpu = cp.asarray(gp.b[n_eq:], dtype=cp.float64) if (m - n_eq) > 0 else None

    c_gpu = cp.asarray(gp.c, dtype=cp.float64)
    lower_gpu = cp.asarray(gp.lower, dtype=cp.float64)
    upper_gpu = cp.asarray(gp.upper, dtype=cp.float64)

    return {
        "A_eq_gpu": A_eq_gpu,
        "A_ub_gpu": A_ub_gpu,
        "b_eq_gpu": b_eq_gpu,
        "b_ub_gpu": b_ub_gpu,
        "c_gpu": c_gpu,
        "lower_gpu": lower_gpu,
        "upper_gpu": upper_gpu,
        "n": n,
        "n_eq": n_eq,
        "n_ub": m - n_eq,
        "tau": tau,
        "sigma": sigma,
        "max_iter": MAX_ITER,
        "check_every": CHECK_EVERY,
        "tol": TOL,
    }


def _run_gpu_solver_only(**kw) -> tuple[float, Any]:
    """Time the pure GPU iteration loop with CUDA events (solver-only)."""
    # Warm up
    _gpu_iteration_loop(**kw)
    cp.cuda.Stream.null.synchronize()

    times: list[float] = []
    for _ in range(SOLVER_ONLY_REPS):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        iteration, converged, diagnostics, x_gpu, yeq, yub = (
            _gpu_iteration_loop(**kw)
        )
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end) / 1000.0)

    # Convert final result to a dict-like structure for reporting.
    result = _gpu_result_to_cpu(
        iteration, converged, diagnostics, x_gpu, yeq, yub, kw
    )
    return float(np.mean(times)), result


def _gpu_result_to_cpu(iteration, converged, diagnostics, x_gpu, yeq, yub, kw):
    """Build a GPUPDHGResult-compatible object from a raw iteration run."""
    x_cpu = cp.asnumpy(x_gpu)
    yeq_cpu = cp.asnumpy(yeq) if kw["n_eq"] > 0 else np.zeros(0, dtype=np.float64)
    yub_cpu = cp.asnumpy(yub) if kw["n_ub"] > 0 else np.zeros(0, dtype=np.float64)
    objective = float(cp.asnumpy(kw["c_gpu"]) @ x_cpu)
    eq, ineq, dual, comp = diagnostics
    return GPUResultProxy(
        x=x_cpu, y_eq=yeq_cpu, y_ub=yub_cpu, iterations=iteration,
        converged=converged,
        status="optimal" if converged else "iteration_limit",
        objective=objective,
        equality_residual=eq, inequality_violation=ineq,
        dual_feasibility=dual, complementarity=comp,
    )


class GPUResultProxy:
    """Lightweight result container mirroring GPUPDHGResult fields."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    @property
    def primal_feasibility(self):
        return max(self.equality_residual, self.inequality_violation)


def _run_gpu_end_to_end(gp: GeneratedLP) -> tuple[float, Any]:
    """Time the full pdhg_gpu() call as wall-clock end-to-end.

    Uses ``time.perf_counter()`` around the whole ``pdhg_gpu()`` call, with an
    explicit GPU synchronize immediately before starting and immediately
    before stopping the timer, so the wall-clock window includes both the
    CPU->GPU data transfer (setup) and the iteration work.  The solver-only
    timing (``_run_gpu_solver_only``) still uses CUDA events and is unchanged.
    """
    A, b, c, lower, upper, row_types = (
        gp.A, gp.b, gp.c, gp.lower, gp.upper, gp.row_types
    )
    # Warm up (JIT / module init / first allocation), not timed.
    pdhg_gpu(A, b, c, lower, upper, row_types,
             max_iter=MAX_ITER, tol=TOL, check_every=CHECK_EVERY)
    cp.cuda.Stream.null.synchronize()

    times: list[float] = []
    result = None
    for _ in range(END_TO_END_REPS):
        # Drain any queued work so the timer starts from a quiescent GPU.
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        result = pdhg_gpu(A, b, c, lower, upper, row_types,
                          max_iter=MAX_ITER, tol=TOL, check_every=CHECK_EVERY)
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return float(np.mean(times)), result


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}"


def _speedup(cpu_s: float, gpu_s: Optional[float]) -> str:
    if gpu_s is None or gpu_s <= 0:
        return "ERR"
    return f"{cpu_s / gpu_s:.2f}x"


def _gpu_info() -> dict[str, Any]:
    """Return basic GPU device info, or an empty dict if unavailable."""
    if not _CUPY_AVAILABLE:
        return {}
    try:
        attrs = cp.cuda.runtime.getDeviceProperties(0)
        mem = cp.cuda.Device(0).mem_info
        return {
            "name": attrs["name"].decode("utf-8", errors="replace"),
            "mem_total_gb": mem[1] / (1024 ** 3),
            "mem_free_gb": mem[0] / (1024 ** 3),
        }
    except Exception:
        return {}


def _default_out_dir() -> Path:
    """Return the default results directory (repo-root / results)."""
    return _REPO_ROOT / "results"


def _save_results(rows: list[dict[str, Any]], gpu_info: dict[str, Any],
                  out_dir: Optional[Path]) -> Optional[Path]:
    """Persist the benchmark table + residual comparison to CSV and Markdown.

    Returns the directory the files were written to, or None if saving was
    skipped (out_dir is None).  Filenames are timestamped so repeated runs
    do not clobber each other.
    """
    if out_dir is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    gpu_tag = (gpu_info.get("name", "gpu").replace(" ", "_")
               if gpu_info else "nocuda")

    # ---- CSV: one row per case ----
    csv_path = out_dir / f"bench_{gpu_tag}_{stamp}.csv"
    with open(csv_path, "w", newline="") as fh:
        import csv
        w = csv.writer(fh)
        w.writerow([
            "case", "rows", "cols", "nnz",
            "cpu_iters", "gpu_iters",
            "cpu_s", "gpu_solver_s", "gpu_e2e_s",
            "speedup_solver_only", "speedup_e2e",
            "conv_match",
            "cpu_objective", "gpu_objective",
            "cpu_eq", "gpu_eq", "cpu_ineq", "gpu_ineq",
            "cpu_dual", "gpu_dual", "cpu_comp", "gpu_comp",
        ])
        for r in rows:
            speed = (lambda v: (f"{v:.2f}x" if v else "ERR"))(
                r["cpu_s"] / r["gpu_solver_s"]
                if r.get("gpu_solver_s") else None)
            speed_e2e = (lambda v: (f"{v:.2f}x" if v else "ERR"))(
                r["cpu_s"] / r["gpu_e2e_s"]
                if r.get("gpu_e2e_s") else None)
            cstats = r.get("cpu_stats") or (None, None, None, None)
            gstats = r.get("gpu_stats") or (None, None, None, None)
            w.writerow([
                r["idx"], r["m"], r["n"], r["nnz"],
                r.get("cpu_iters"), r.get("gpu_iters"),
                f"{r.get('cpu_s'):.4f}" if r.get("cpu_s") else "",
                f"{r.get('gpu_solver_s'):.4f}" if r.get("gpu_solver_s") else "",
                f"{r.get('gpu_e2e_s'):.4f}" if r.get("gpu_e2e_s") else "",
                speed, speed_e2e,
                r.get("conv_match"),
                r.get("cpu_objective"), r.get("gpu_objective"),
                *[f"{v:.3e}" if isinstance(v, float) else "" for v in cstats],
                *[f"{v:.3e}" if isinstance(v, float) else "" for v in gstats],
            ])

    # ---- Markdown: human-readable full report ----
    md_path = out_dir / f"bench_{gpu_tag}_{stamp}.md"
    lines: list[str] = []
    lines.append("# PDHG CPU vs GPU Benchmark")
    lines.append("")
    lines.append(f"- **GPU**: {gpu_info.get('name', 'n/a')} "
                 f"({gpu_info.get('mem_total_gb', 0):.2f} GB) "
                 if gpu_info else "- **GPU**: n/a")
    lines.append(f"- **Config**: MAX_ITER={MAX_ITER}, TOL={TOL:.0e}, "
                 f"CHECK_EVERY={CHECK_EVERY}, "
                 f"solver_reps={SOLVER_ONLY_REPS}, e2e_reps={END_TO_END_REPS}")
    lines.append(f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("| # | rows | cols | nnz | cpu_iters | gpu_iters | cpu_s | "
                 "gpu_solver_s | gpu_e2e_s | str_only | str_e2e | conv_match |")
    lines.append("|---|-----:|-----:|----:|----------:|----------:|------:"
                 "|------------:|----------:|--------:|--------:|:----------:|")
    for r in rows:
        speed = (f"{r['cpu_s'] / r['gpu_solver_s']:.2f}x"
                 if r.get("gpu_solver_s") else "ERR")
        speed_e2e = (f"{r['cpu_s'] / r['gpu_e2e_s']:.2f}x"
                     if r.get("gpu_e2e_s") else "ERR")
        lines.append(
            f"| {r['idx']} | {r['m']:,} | {r['n']:,} | {r['nnz']:,} | "
            f"{r.get('cpu_iters'):,} | {r.get('gpu_iters'):,} | "
            f"{r.get('cpu_s'):.2f} | {r.get('gpu_solver_s'):.2f} | "
            f"{r.get('gpu_e2e_s'):.2f} | {speed} | {speed_e2e} | "
            f"{r.get('conv_match')} |")
    lines.append("")
    lines.append("### KKT residual / objective comparison")
    lines.append("")
    lines.append("| # | metric | CPU | GPU | match |")
    lines.append("|---|--------|----:|----:|:-----:|")
    for r in rows:
        if r.get("gpu_stats") is None:
            lines.append(f"| {r['idx']} | {r['name']} | GPU did not run | | |")
            continue
        cstats = r["cpu_stats"]
        gstats = r["gpu_stats"]
        for label, cv, gv in [
            ("objective", r["cpu_objective"], r["gpu_objective"]),
            ("eq_res", cstats[0], gstats[0]),
            ("ineq_viol", cstats[1], gstats[1]),
            ("dual_feas", cstats[2], gstats[2]),
            ("complement", cstats[3], gstats[3]),
        ]:
            fmt = f"{cv:14.9g} | {gv:14.9g}"
            match = abs(cv - gv) <= 1e-6 * max(1.0, abs(cv)) \
                if label == "objective" else abs(cv - gv) <= 1e-6
            lines.append(f"| {r['idx']} | {label} | {fmt} | {match} |")
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return out_dir


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark(cfg: Optional[dict[str, Any]] = None) -> None:
    sep = "-" * 120
    if cfg:
        _apply_config(cfg)
    print("=" * 120)
    print("  Milestone 2 - GPU track: PDHG CPU vs GPU benchmark")
    print("=" * 120)

    if not _CUPY_AVAILABLE:
        print("  CuPy is NOT available - running CPU-only comparison.")
        print("  Install CuPy in a CUDA environment for GPU benchmarks.")
        print("=" * 120)
        return

    gpu_info = _gpu_info()
    if gpu_info:
        print(f"  GPU: {gpu_info['name']} "
              f"({gpu_info['mem_total_gb']:.2f} GB)")
    else:
        print("  GPU info unavailable.")

    print(f"  MAX_ITER={MAX_ITER}, TOL={TOL:.0e}, "
          f"CHECK_EVERY={CHECK_EVERY}")
    print(sep, flush=True)

    # ---- Table header ----
    print(
        f"{'#':>3} | {'rows':>7} | {'cols':>7} | {'nnz':>10} | "
        f"{'iters_cpu':>9} | {'iters_gpu':>9} | "
        f"{'cpu_s':>9} | {'gpu_solver_s':>13} | {'gpu_e2e_s':>10} | "
        f"{'str_only':>9} | {'str_e2e':>9} | {'conv_match':>11}"
    )
    print(sep)

    rows: list[dict[str, Any]] = []

    for idx, (rows_c, cols_c, density) in enumerate(CASES, start=1):
        name = f"case{idx}"
        print(f"[{name}] start rows={rows_c} cols={cols_c} "
              f"density={density:.3f} ...", flush=True)
        rng = np.random.default_rng(RNG_SEED + idx)
        gp = _generate_lp(rows_c, cols_c, density, rng, name)
        m, n, nnz = gp.m, gp.n, gp.nnz

        # ---- CPU ----
        try:
            cpu_s, cpu_result = _run_cpu(gp, end_to_end=False)
        except Exception as exc:
            print(f"[{name}] CPU failed: {exc}")
            continue

        # ---- GPU ----
        gpu_solver_s: Optional[float] = None
        gpu_e2e_s: Optional[float] = None
        gpu_result = None
        gpu_err = ""
        try:
            prep = _prepare_gpu(gp)
            gpu_solver_s, gpu_result = _run_gpu_solver_only(**prep)
            # Only the timing matters here; the returned result is unused.
            gpu_e2e_s, _ = _run_gpu_end_to_end(gp)
        except Exception as exc:
            gpu_err = str(exc)[:160]
            gpu_solver_s = None
            gpu_e2e_s = None

        # ---- Compare convergence ----
        if gpu_result is not None:
            iters_gpu = gpu_result.iterations
            conv_match = (cpu_result.converged == gpu_result.converged)
            str_match = (cpu_result.status == gpu_result.status)
        else:
            iters_gpu = -1
            conv_match = False
            str_match = False

        c_iters = cpu_result.iterations
        cpu_s_ms = _fmt_ms(cpu_s)
        gpu_solver_ms = (_fmt_ms(gpu_solver_s) if gpu_solver_s is not None
                         else "MEM_ERR" if gpu_err else "ERR")
        gpu_e2e_ms = (_fmt_ms(gpu_e2e_s) if gpu_e2e_s is not None
                      else "MEM_ERR" if gpu_err else "ERR")
        str_only = _speedup(cpu_s, gpu_solver_s) if gpu_solver_s is not None else "ERR"
        str_e2e = _speedup(cpu_s, gpu_e2e_s) if gpu_e2e_s is not None else "ERR"

        print(
            f"{idx:>3} | {m:>7,} | {n:>7,} | {nnz:>10,} | "
            f"{c_iters:>9,} | {iters_gpu:>9,} | "
            f"{cpu_s_ms:>9} | {gpu_solver_ms:>13} | {gpu_e2e_ms:>10} | "
            f"{str_only:>9} | {str_e2e:>9} | {str(conv_match):>11}",
            flush=True,
        )
        if gpu_err:
            print(f"      [GPU warning] {gpu_err}", flush=True)
        print(f"[{name}] done (cpu_iters={c_iters}, "
              f"gpu_iters={iters_gpu}, str_only={str_only}, "
              f"str_e2e={str_e2e})", flush=True)

        rows.append({
            "idx": idx, "name": name, "m": m, "n": n, "nnz": nnz,
            "cpu_iters": c_iters, "gpu_iters": iters_gpu,
            "cpu_s": cpu_s, "gpu_solver_s": gpu_solver_s,
            "gpu_e2e_s": gpu_e2e_s,
            "cpu_converged": cpu_result.converged,
            "gpu_converged": (gpu_result.converged if gpu_result else None),
            "cpu_objective": cpu_result.objective,
            "gpu_objective": (gpu_result.objective if gpu_result else None),
            "cpu_stats": (cpu_result.equality_residual,
                          cpu_result.inequality_violation,
                          cpu_result.dual_feasibility,
                          cpu_result.complementarity),
            "gpu_stats": ((gpu_result.equality_residual,
                           gpu_result.inequality_violation,
                           gpu_result.dual_feasibility,
                           gpu_result.complementarity)
                          if gpu_result else None),
            "conv_match": conv_match, "str_match": str_match,
            "status": "MEM_ERR" if gpu_err else "OK",
        })

        del gp
        gc.collect()
        if _CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

    print(sep)
    print()

    # ---- Detailed comparison of objectives & residuals ----
    print("Convergence & residual comparison (inf-norm KKT residuals):")
    print(f"{'#':>3} | {'metric':<14} | {'CPU':>14} | {'GPU':>14} | "
          f"{'match':>8}")
    print(sep[:90])
    for r in rows:
        if r["gpu_stats"] is None:
            print(f"{r['idx']:>3} | {r['name']:<14} | GPU did not run")
            continue
        metrics = [
            ("objective", r["cpu_objective"], r["gpu_objective"]),
            ("eq_res", r["cpu_stats"][0], r["gpu_stats"][0]),
            ("ineq_viol", r["cpu_stats"][1], r["gpu_stats"][1]),
            ("dual_feas", r["cpu_stats"][2], r["gpu_stats"][2]),
            ("complement", r["cpu_stats"][3], r["gpu_stats"][3]),
        ]
        for label, cv, gv in metrics:
            if label == "objective":
                match = abs(cv - gv) <= 1e-6 * max(1.0, abs(cv))
                print(f"{r['idx']:>3} | {label:<14} | {cv:14.9g} | "
                      f"{gv:14.9g} | {str(match):>8}")
            else:
                match = abs(cv - gv) <= 1e-6
                print(f"{r['idx']:>3} | {label:<14} | {cv:14.2e} | "
                      f"{gv:14.2e} | {str(match):>8}")
    print()

    # ---- Conclusion ----
    print("=" * 120)
    print("  Conclusion")
    print("=" * 120)
    ok = [r for r in rows if r["status"] == "OK" and r["gpu_solver_s"]]
    if not ok:
        print("  No GPU case completed (all OOM or errors).")
    else:
        fast_solver = [
            r for r in ok
            if r["gpu_solver_s"] and r["gpu_solver_s"] > 0
            and r["cpu_s"] / r["gpu_solver_s"] > 1.0
        ]
        fast_e2e = [
            r for r in ok
            if r["gpu_e2e_s"] and r["gpu_e2e_s"] > 0
            and r["cpu_s"] / r["gpu_e2e_s"] > 1.0
        ]
        conv_matches = [r for r in ok if r["conv_match"]]
        print(f"  Largest successful GPU case: #{ok[-1]['idx']} "
              f"({ok[-1]['m']:,} x {ok[-1]['n']:,}, nnz={ok[-1]['nnz']:,}).")
        print(f"  Convergence match (CPU==GPU): "
              f"{len(conv_matches)}/{len(ok)} cases.")
        print(f"  GPU faster (solver-only): "
              f"{'YES' if fast_solver else 'NO'} "
              f"({len(fast_solver)}/{len(ok)} cases).")
        print(f"  GPU faster (end-to-end): "
              f"{'YES' if fast_e2e else 'NO'} "
              f"({len(fast_e2e)}/{len(ok)} cases).")
    print("=" * 120, flush=True)

    # Persist results (CSV + Markdown).
    if not (cfg or {}).get("no_save"):
        out_dir = (Path((cfg or {}).get("out"))
                   if (cfg or {}).get("out") else _default_out_dir())
        saved = _save_results(rows, gpu_info if isinstance(gpu_info, dict)
                              else {}, out_dir)
        if saved:
            print(f"  Results written to: {saved}", flush=True)
        else:
            print("  Results save skipped (use --out to enable).", flush=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _CFG = _parse_args()
    try:
        run_benchmark(_CFG)
    except KeyboardInterrupt:
        print("\n[BENCH] Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[BENCH] FATAL: {exc}", file=sys.stderr)
        sys.exit(2)




