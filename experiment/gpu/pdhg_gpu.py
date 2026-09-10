#!/usr/bin/env python3
"""GPU PDHG solver using CuPy — exact mirror of pdhg_mixed.py iteration.

This implements the identical Chambolle-Pock primal-dual hybrid gradient
algorithm as ``experiment/pdhg/pdhg_mixed.py``, but executes the
iteration loop entirely on GPU using CuPy and cupyx.scipy.sparse.

Mathematical formulation (preserved exactly):

    y_eq  = y_eq  + sigma (A_eq x_bar - b_eq)
    y_ub  = max(y_ub + sigma (A_ub x_bar - b_ub), 0)
    g     = c + A_eq^T y_eq + A_ub^T y_ub
    x     = proj_{[lower,upper]}(x - tau g)
    x_bar = 2x - x_prev

Step sizes: tau = sigma = 0.9 / L where L = ||A||_2 from power iteration.

Usage:
    Can be called as a library or run standalone for a quick smoke test.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# CuPy availability guard
# ---------------------------------------------------------------------------
_CUPY_AVAILABLE = False
_cupy_import_error: str = ""

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp

    _CUPY_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _cupy_import_error = str(exc)


# ---------------------------------------------------------------------------
# Result dataclass (mirrors MixedPDHGResult from pdhg_mixed.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GPUPDHGResult:
    """Solution and KKT-style termination diagnostics from the GPU solver."""

    x: np.ndarray
    y_eq: np.ndarray
    y_ub: np.ndarray
    iterations: int
    converged: bool
    status: str
    objective: float
    equality_residual: float
    inequality_violation: float
    dual_feasibility: float
    complementarity: float

    @property
    def primal_feasibility(self) -> float:
        """Largest primal feasibility residual across both row classes."""
        return max(self.equality_residual, self.inequality_violation)


# ---------------------------------------------------------------------------
# CPU helpers (spectral norm, CSR transfer)
# ---------------------------------------------------------------------------
def _spectral_norm_sparse(
    A_sp: sp.csr_matrix, *, max_iter: int = 200, tol: float = 1e-12
) -> float:
    """Estimate ||A||_2 via power iteration on A^T A (CPU, scipy.sparse)."""
    m, n = A_sp.shape
    if A_sp.nnz == 0:
        return 0.0

    v = np.ones(n, dtype=np.float64)
    v /= np.linalg.norm(v)
    previous = 0.0

    for _ in range(max_iter):
        w = A_sp.T @ (A_sp @ v)
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            return 0.0
        v = w / norm_w
        estimate = float(np.linalg.norm(A_sp @ v))
        if abs(estimate - previous) <= tol * max(1.0, estimate):
            return estimate
        previous = estimate
    return previous


def _csr_to_gpu(A_csr: sp.csr_matrix):
    """Convert a scipy CSR matrix to cupyx.scipy.sparse.csr_matrix on GPU."""
    return cp_sp.csr_matrix(
        (
            cp.asarray(A_csr.data, dtype=cp.float64),
            cp.asarray(A_csr.indices, dtype=cp.int32),
            cp.asarray(A_csr.indptr, dtype=cp.int32),
        ),
        shape=A_csr.shape,
    )


# ---------------------------------------------------------------------------
# Core GPU iteration loop (exposed for benchmark timing)
# ---------------------------------------------------------------------------
def _gpu_iteration_loop(
    *,
    A_eq_gpu,
    A_ub_gpu,
    b_eq_gpu,
    b_ub_gpu,
    c_gpu,
    lower_gpu,
    upper_gpu,
    n: int,
    n_eq: int,
    n_ub: int,
    tau: float,
    sigma: float,
    max_iter: int,
    check_every: int,
    tol: float,
):
    """Run the PDHG iteration loop entirely on GPU.

    All arrays must already live on GPU.  Returns
    ``(iteration, converged, diagnostics, x_gpu, y_eq_gpu, y_ub_gpu)``.
    """
    tau_cp = cp.float64(tau)
    sigma_cp = cp.float64(sigma)

    # --- Initialise primal/dual variables on GPU ---
    x_gpu = cp.clip(cp.zeros(n, dtype=np.float64), lower_gpu, upper_gpu)
    x_bar_gpu = x_gpu.copy()
    y_eq_gpu = cp.zeros(n_eq, dtype=np.float64)
    y_ub_gpu = cp.zeros(n_ub, dtype=np.float64)

    diagnostics = (float("inf"),) * 4
    converged = False
    iteration = 0
    last_gradient = None  # cache gradient for diagnostics reuse

    for iteration in range(1, max_iter + 1):
        # --- Dual-first Chambolle-Pock update ---
        # Equality duals are unconstrained; <= duals belong to R_+.
        # This ordering is critical for equality rows (see pdhg_mixed.py).
        if n_eq > 0:
            y_eq_gpu = y_eq_gpu + sigma_cp * (A_eq_gpu @ x_bar_gpu - b_eq_gpu)
        if n_ub > 0:
            y_ub_gpu = cp.maximum(
                y_ub_gpu + sigma_cp * (A_ub_gpu @ x_bar_gpu - b_ub_gpu),
                0.0,
            )

        # --- Gradient of augmented Lagrangian ---
        prev_x = x_gpu
        gradient = c_gpu.copy()
        if n_eq > 0:
            gradient = gradient + A_eq_gpu.T @ y_eq_gpu
        if n_ub > 0:
            gradient = gradient + A_ub_gpu.T @ y_ub_gpu

        # --- Primal update with box projection ---
        x_gpu = cp.clip(x_gpu - tau_cp * gradient, lower_gpu, upper_gpu)

        # --- Extrapolation (Chambolle-Pock momentum) ---
        x_bar_gpu = 2.0 * x_gpu - prev_x

        # Cache gradient (identical to lagrangian_gradient in diagnostics)
        last_gradient = gradient

        # --- Periodic diagnostics (minimal CPU sync) ---
        if iteration % check_every == 0 or iteration == max_iter:
            if n_eq > 0:
                eq_res = cp.max(cp.abs(A_eq_gpu @ x_gpu - b_eq_gpu))
            else:
                eq_res = cp.float64(0.0)

            if n_ub > 0:
                ineq_slack = A_ub_gpu @ x_gpu - b_ub_gpu
                ineq_viol = cp.max(cp.maximum(ineq_slack, 0.0))
                comp = cp.max(cp.abs(y_ub_gpu * ineq_slack))
            else:
                ineq_viol = cp.float64(0.0)
                comp = cp.float64(0.0)

            # Stationarity: ||x - proj(x - g)||_inf
            stat = cp.max(
                cp.abs(
                    x_gpu - cp.clip(x_gpu - last_gradient, lower_gpu, upper_gpu)
                )
            )

            # Single batched transfer to CPU (one sync point)
            results = cp.stack([eq_res, ineq_viol, stat, comp])
            r = cp.asnumpy(results)
            diagnostics = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))

            if max(diagnostics) <= tol:
                converged = True
                break

    return iteration, converged, diagnostics, x_gpu, y_eq_gpu, y_ub_gpu


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------
def pdhg_gpu(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    row_types: tuple[str, ...],
    *,
    max_iter: int = 200_000,
    tol: float = 1e-7,
    check_every: int = 250,
) -> GPUPDHGResult:
    """GPU PDHG solver -- exact mirror of ``pdhg_mixed()``.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Dense constraint matrix.
    b : np.ndarray, shape (m,)
        Right-hand side.
    c : np.ndarray, shape (n,)
        Objective coefficients.
    lower, upper : np.ndarray, shape (n,)
        Variable bounds (may contain -inf / +inf).
    row_types : tuple of str
        ``'E'`` for equality rows, ``'L'`` for ``<=`` rows.
    max_iter : int
        Maximum number of Chambolle-Pock iterations.
    tol : float
        Convergence tolerance (inf-norm on all KKT residuals).
    check_every : int
        Diagnostic check frequency.

    Returns
    -------
    GPUPDHGResult
        Solution vectors are returned on CPU (numpy arrays).
    """
    if not _CUPY_AVAILABLE:
        raise RuntimeError(f"CuPy not available: {_cupy_import_error}")
    if max_iter <= 0 or check_every <= 0 or tol <= 0.0:
        raise ValueError("max_iter, check_every, and tol must be positive")

    # --- Validate row types ---
    rt = np.asarray(row_types)
    unsupported = set(rt).difference({"E", "L"})
    if unsupported:
        raise ValueError(
            f"Only E and L rows supported; found {sorted(unsupported)}"
        )

    # --- Partition rows ---
    eq_idx = np.flatnonzero(rt == "E")
    ub_idx = np.flatnonzero(rt == "L")

    A_eq_dense = A[eq_idx]
    A_ub_dense = A[ub_idx]
    b_eq = b[eq_idx]
    b_ub = b[ub_idx]
    n_eq = len(eq_idx)
    n_ub = len(ub_idx)

    # --- Spectral norm on CPU (sparse power iteration) ---
    A_csr = sp.csr_matrix(A)
    norm_A = _spectral_norm_sparse(A_csr)
    tau = 1.0 if norm_A == 0.0 else 0.9 / norm_A
    sigma = tau

    # --- Convert sub-matrices to CSR for GPU ---
    A_eq_csr = sp.csr_matrix(A_eq_dense)
    A_ub_csr = sp.csr_matrix(A_ub_dense)

    # --- Transfer to GPU ---
    A_eq_gpu = _csr_to_gpu(A_eq_csr) if n_eq > 0 else None
    A_ub_gpu = _csr_to_gpu(A_ub_csr) if n_ub > 0 else None
    b_eq_gpu = cp.asarray(b_eq, dtype=cp.float64) if n_eq > 0 else None
    b_ub_gpu = cp.asarray(b_ub, dtype=cp.float64) if n_ub > 0 else None

    c_gpu = cp.asarray(c, dtype=cp.float64)
    lower_gpu = cp.asarray(lower, dtype=cp.float64)
    upper_gpu = cp.asarray(upper, dtype=cp.float64)

    # --- Run iteration loop ---
    iteration, converged, diagnostics, x_gpu, y_eq_gpu, y_ub_gpu = (
        _gpu_iteration_loop(
            A_eq_gpu=A_eq_gpu,
            A_ub_gpu=A_ub_gpu,
            b_eq_gpu=b_eq_gpu,
            b_ub_gpu=b_ub_gpu,
            c_gpu=c_gpu,
            lower_gpu=lower_gpu,
            upper_gpu=upper_gpu,
            n=c.shape[0],
            n_eq=n_eq,
            n_ub=n_ub,
            tau=tau,
            sigma=sigma,
            max_iter=max_iter,
            check_every=check_every,
            tol=tol,
        )
    )

    # --- Transfer results to CPU ---
    x_cpu = cp.asnumpy(x_gpu)
    y_eq_cpu = cp.asnumpy(y_eq_gpu) if n_eq > 0 else np.zeros(0, dtype=np.float64)
    y_ub_cpu = cp.asnumpy(y_ub_gpu) if n_ub > 0 else np.zeros(0, dtype=np.float64)

    eq_res_f, ineq_viol_f, dual_f, comp_f = diagnostics
    objective = float(c @ x_cpu)

    return GPUPDHGResult(
        x=x_cpu,
        y_eq=y_eq_cpu,
        y_ub=y_ub_cpu,
        iterations=iteration,
        converged=converged,
        status="optimal" if converged else "iteration_limit",
        objective=objective,
        equality_residual=eq_res_f,
        inequality_violation=ineq_viol_f,
        dual_feasibility=dual_f,
        complementarity=comp_f,
    )


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not _CUPY_AVAILABLE:
        print(f"SKIP: CuPy not available ({_cupy_import_error})")
        sys.exit(0)

    print("=" * 72)
    print("  pdhg_gpu.py  --  standalone smoke test")
    print("=" * 72)

    # Small feasible LP with both equality and <= rows.
    n, m = 5, 8
    rng = np.random.default_rng(42)
    A = rng.standard_normal((m, n)).astype(np.float64)
    x_feas = rng.uniform(0.5, 2.0, size=n)
    b_eq = A[:3] @ x_feas
    b_ub = A[3:] @ x_feas + 0.5
    b = np.concatenate([b_eq, b_ub])
    c = rng.standard_normal(n).astype(np.float64)
    lower = np.zeros(n)
    upper = np.full(n, np.inf)
    row_types = ("E",) * 3 + ("L",) * 5

    t0 = time.perf_counter()
    result = pdhg_gpu(
        A, b, c, lower, upper, row_types,
        max_iter=10_000, tol=1e-7, check_every=100,
    )
    elapsed = time.perf_counter() - t0

    print(f"  Iterations : {result.iterations}")
    print(f"  Converged  : {result.converged}")
    print(f"  Status     : {result.status}")
    print(f"  Objective  : {result.objective:.9g}")
    print(f"  Primal feas: {result.primal_feasibility:.2e}")
    print(f"  Dual feasi : {result.dual_feasibility:.2e}")
    print(f"  Complement : {result.complementarity:.2e}")
    print(f"  Wall time  : {elapsed:.4f}s")
    print("=" * 72)
    print("  SMOKE TEST PASSED" if result.converged else "  NOT CONVERGED (check problem)")

