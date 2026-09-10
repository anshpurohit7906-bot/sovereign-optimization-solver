#!/usr/bin/env python3
"""Milestone 1 - GPU track: SpMV benchmark (CPU SciPy vs GPU CuPy).

Compares sparse matrix-vector multiplication (A @ x and A.T @ y) between
CPU (scipy.sparse) and GPU (cupyx.scipy.sparse) for deterministic,
LP-shaped workloads of increasing size.

Only NumPy, SciPy, and CuPy are used.  No solver algorithm is introduced.
PILOT87 is never loaded or referenced.

Usage:
    python experiment/gpu/benchmark_spmv.py
"""

from __future__ import annotations

import gc
import sys
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Try importing CuPy; provide a graceful fallback so the script never crashes.
# ---------------------------------------------------------------------------
_CUPY_AVAILABLE = False
_cupy_import_error: str = ""

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp

    _CUPY_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment guard
    _cupy_import_error = str(exc)

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
RNG_SEED = 42
WARMUP_REPS = 3
TIMED_REPS = 10
DTYPE = np.float64

# (rows, cols, nnz) - deterministic, LP-like sparsity patterns
CASES: list[tuple[int, int, int]] = [
    (10_000, 5_000, 100_000),
    (50_000, 20_000, 500_000),
    (100_000, 40_000, 1_000_000),
    (200_000, 80_000, 2_000_000),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_sparse(
    rows: int, cols: int, nnz: int, rng: np.random.Generator
) -> sp.csr_matrix:
    """Create a deterministic float64 sparse matrix with *nnz* nonzeros.

    Random row/col indices are drawn uniformly; duplicate (i, j) pairs are
    collapsed by summing values, so the actual stored NNZ <= *nnz*.  Values
    are drawn from N(0, 1) to mimic LP constraint coefficients.
    """
    n_draws = int(nnz * 1.5) + 1
    row_idx = rng.integers(0, rows, size=n_draws)
    col_idx = rng.integers(0, cols, size=n_draws)
    data = rng.standard_normal(n_draws).astype(DTYPE)

    mat = sp.coo_matrix((data, (row_idx, col_idx)), shape=(rows, cols))
    mat = mat.tocsr()  # sums duplicates
    return mat


def _gpu_info() -> dict[str, Any]:
    """Return basic GPU device info, or an empty dict if unavailable."""
    if not _CUPY_AVAILABLE:
        return {}
    try:
        attrs = cp.cuda.runtime.getDeviceProperties(0)
        mem = cp.cuda.Device(0).mem_info  # (free, total) in bytes
        return {
            "name": attrs["name"].decode("utf-8", errors="replace"),
            "mem_total_gb": mem[1] / (1024 ** 3),
            "mem_free_gb": mem[0] / (1024 ** 3),
            "mem_used_gb": (mem[1] - mem[0]) / (1024 ** 3),
        }
    except Exception:
        return {}


def _estimate_matrix_bytes(rows: int, cols: int, nnz: int) -> int:
    """Rough byte estimate for a CSR matrix: data + col_idx + row_ptr."""
    return nnz * 8 + nnz * 4 + (rows + 1) * 4  # float64 data, int32 idx/ptr


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.4f}"


def _cpu_spmv(mat_sp: sp.csr_matrix, vec: np.ndarray) -> np.ndarray:
    """CPU sparse mat-vec (A @ x)."""
    return mat_sp @ vec


def _cpu_spmtv(mat_sp: sp.csr_matrix, vec: np.ndarray) -> np.ndarray:
    """CPU sparse transpose mat-vec (A.T @ y)."""
    return mat_sp.T @ vec


def _benchmark_cpu(
    mat_sp: sp.csr_matrix, vec: np.ndarray, *, transpose: bool = False
) -> tuple[float, np.ndarray]:
    """Warm up + timed repetitions on CPU.  Returns (avg_seconds, result)."""
    op = _cpu_spmtv if transpose else _cpu_spmv

    # Warm up
    for _ in range(WARMUP_REPS):
        result = op(mat_sp, vec)

    # Timed
    times: list[float] = []
    for _ in range(TIMED_REPS):
        t0 = time.perf_counter()
        result = op(mat_sp, vec)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.mean(times)), result


def _benchmark_gpu(
    mat_sp: sp.csr_matrix, vec: np.ndarray, *, transpose: bool = False
) -> tuple[float, np.ndarray]:
    """Warm up + timed repetitions on GPU with CUDA events.

    Returns (avg_seconds, cpu_result).
    """
    # Transfer to GPU
    cp_mat = cp_sp.csr_matrix(
        (
            cp.asarray(mat_sp.data),
            cp.asarray(mat_sp.indices),
            cp.asarray(mat_sp.indptr),
        ),
        shape=mat_sp.shape,
    )
    cp_vec = cp.asarray(vec)

    if transpose:
        def gpu_op() -> cp.ndarray:
            return cp_mat.T @ cp_vec
    else:
        def gpu_op() -> cp.ndarray:
            return cp_mat @ cp_vec

    # Warm up
    for _ in range(WARMUP_REPS):
        _ = gpu_op()
    cp.cuda.Stream.null.synchronize()

    # Timed with CUDA events
    times: list[float] = []
    for _ in range(TIMED_REPS):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()
        _ = gpu_op()
        end_event.record()
        end_event.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms / 1000.0)  # convert ms -> s

    # Final result back to CPU
    result_gpu = gpu_op()
    cp.cuda.Stream.null.synchronize()
    result_cpu = cp.asnumpy(result_gpu)

    # Free GPU memory
    del cp_mat, cp_vec, result_gpu
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    return float(np.mean(times)), result_cpu


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # ---- GPU / header info ----
    print("=" * 80)
    print("  SpMV Benchmark - CPU (SciPy) vs GPU (CuPy)")
    print("  Milestone 1 - GPU track")
    print("=" * 80)
    print()

    if not _CUPY_AVAILABLE:
        print(f"[WARN] CuPy is NOT available: {_cupy_import_error}")
        print("       GPU benchmarks will be SKIPPED for all cases.\n")
    else:
        info = _gpu_info()
        if info:
            print(f"  GPU           : {info.get('name', 'unknown')}")
            print(f"  GPU mem total : {info.get('mem_total_gb', 0):.2f} GB")
            print(f"  GPU mem free  : {info.get('mem_free_gb', 0):.2f} GB")
            print(f"  GPU mem used  : {info.get('mem_used_gb', 0):.2f} GB")
        else:
            print("  GPU info      : (unavailable)")
    print()

    # ---- Table header ----
    header = (
        f"{'case':>4} | {'rows':>8} | {'cols':>8} | {'nnz':>12} | "
        f"{'CPU Ax ms':>10} | {'GPU Ax ms':>10} | {'Ax speedup':>10} | "
        f"{'CPU ATy ms':>11} | {'GPU ATy ms':>11} | {'ATy speedup':>10} | "
        f"{'Ax error':>10} | {'ATy error':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_rows: list[dict[str, Any]] = []

    for idx, (rows, cols, nnz) in enumerate(CASES, start=1):
        # ---- Generate matrix and vectors (deterministic) ----
        mat_sp = _generate_sparse(rows, cols, nnz, rng)
        mat_nnz = int(mat_sp.nnz)
        x = rng.standard_normal(cols).astype(DTYPE)
        y = rng.standard_normal(rows).astype(DTYPE)

        # ---- CPU benchmarks ----
        cpu_ax_time, cpu_ax = _benchmark_cpu(mat_sp, x, transpose=False)
        cpu_aty_time, cpu_aty = _benchmark_cpu(mat_sp, y, transpose=True)

        gpu_ax_ms_str = "SKIPPED"
        gpu_aty_ms_str = "SKIPPED"
        ax_speedup_str = "N/A"
        aty_speedup_str = "N/A"
        ax_err_str = "N/A"
        aty_err_str = "N/A"
        status = "CPU_ONLY"
        gpu_ax_time = -1.0
        gpu_aty_time = -1.0

        if _CUPY_AVAILABLE:
            # ---- Check GPU memory feasibility (best effort) ----
            try:
                mem_info = cp.cuda.Device(0).mem_info
                free_bytes = mem_info[0]
                est_bytes = _estimate_matrix_bytes(rows, cols, nnz)
                # Rough need: several CSR copies + result vectors.
                needed = est_bytes * 4 + cols * 8 + rows * 8
                if needed > free_bytes:
                    status = "OOM"
                    gpu_ax_ms_str = "OOM"
                    gpu_aty_ms_str = "OOM"
                else:
                    # ---- GPU: A @ x ----
                    try:
                        gpu_ax_time, gpu_ax = _benchmark_gpu(
                            mat_sp, x, transpose=False
                        )
                        gpu_ax_ms_str = _fmt_ms(gpu_ax_time)
                        ax_err_str = f"{float(np.max(np.abs(cpu_ax - gpu_ax))):.2e}"
                        if cpu_ax_time > 0:
                            ax_speedup_str = f"{cpu_ax_time / gpu_ax_time:.2f}x"
                    except Exception as exc:
                        gpu_ax_ms_str = "ERR"
                        ax_err_str = str(exc)[:200]

                    # ---- GPU: A.T @ y ----
                    try:
                        gpu_aty_time, gpu_aty = _benchmark_gpu(
                            mat_sp, y, transpose=True
                        )
                        gpu_aty_ms_str = _fmt_ms(gpu_aty_time)
                        aty_err_str = f"{float(np.max(np.abs(cpu_aty - gpu_aty))):.2e}"
                        if cpu_aty_time > 0:
                            aty_speedup_str = f"{cpu_aty_time / gpu_aty_time:.2f}x"
                    except Exception as exc:
                        gpu_aty_ms_str = "ERR"
                        aty_err_str = str(exc)[:200]

                    if "ERR" in (gpu_ax_ms_str, gpu_aty_ms_str):
                        status = "PARTIAL"
                    else:
                        status = "OK"
            except Exception:
                status = "MEM_ERR"
                gpu_ax_ms_str = "MEM_ERR"
                gpu_aty_ms_str = "MEM_ERR"

        # ---- Print row ----
        print(
            f"{idx:>4} | {rows:>8,} | {cols:>8,} | {mat_nnz:>12,} | "
            f"{_fmt_ms(cpu_ax_time):>10} | {gpu_ax_ms_str:>10} | {ax_speedup_str:>10} | "
            f"{_fmt_ms(cpu_aty_time):>11} | {gpu_aty_ms_str:>11} | {aty_speedup_str:>10} | "
            f"{ax_err_str:>10} | {aty_err_str:>10}"
        )

        all_rows.append({
            "case": idx, "rows": rows, "cols": cols, "nnz": mat_nnz,
            "ax_err": ax_err_str, "aty_err": aty_err_str, "status": status,
            "ax_speedup": (cpu_ax_time / gpu_ax_time
                           if (gpu_ax_ms_str not in ("SKIPPED", "OOM", "ERR", "MEM_ERR")
                               and gpu_ax_time > 0)
                           else None),
            "aty_speedup": (cpu_aty_time / gpu_aty_time
                            if (gpu_aty_ms_str not in ("SKIPPED", "OOM", "ERR", "MEM_ERR")
                                and gpu_aty_time > 0)
                            else None),
        })

        del mat_sp, x, y
        gc.collect()

    print(sep)
    print()

    # ---- Numerical validation summary ----
    print("Numerical validation (infinity-norm, CPU vs GPU):")
    for r in all_rows:
        print(
            f"  Case {r['case']} ({r['rows']:,} x {r['cols']:,}, "
            f"nnz={r['nnz']:,}): Ax err = {r['ax_err']}, ATy err = {r['aty_err']}"
        )
    print()

    # ---- Conclusion ----
    print("=" * 80)
    print("  Conclusion")
    print("=" * 80)
    if not _CUPY_AVAILABLE:
        print("  CuPy was NOT available, so no GPU benchmarks ran.")
        print("  Install CuPy in a CUDA-enabled environment to compare.")
    else:
        ok_rows = [r for r in all_rows if r["status"] == "OK"]
        if not ok_rows:
            print("  No GPU case completed (all OOM or errors).")
            print("  GPU faster for A @ x   : not measured")
            print("  GPU faster for A.T @ y : not measured")
        else:
            best = ok_rows[-1]
            c = CASES[best["case"] - 1]
            print(f"  Largest successful GPU case: #{best['case']} "
                  f"({c[0]:,} x {c[1]:,}, nnz={best['nnz']:,}).")
            ax_faster = bool(best["ax_speedup"] and best["ax_speedup"] > 1.0)
            aty_faster = bool(best["aty_speedup"] and best["aty_speedup"] > 1.0)
            any_ax = any(r["ax_speedup"] for r in ok_rows)
            any_aty = any(r["aty_speedup"] for r in ok_rows)
            print("  GPU faster for A @ x   :",
                  "YES" if (ax_faster or any_ax) else "NO")
            print("  GPU faster for A.T @ y :",
                  "YES" if (aty_faster or any_aty) else "NO")
    print("=" * 80)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n[BENCH] Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[BENCH] FATAL: {exc}", file=sys.stderr)
        sys.exit(2)
