"""Optional CUDA/CuPy backend for the Mehrotra reduced Newton system.

This backend intentionally keeps the existing CPU/SciPy implementation intact.
When CuPy and a compatible CUDA runtime are installed, the diagonal-H Schur
system is assembled, factorized, refined, and solved on the GPU.  The current
implementation uses a dense Schur complement on the GPU for reliability and
clear numerical verification; the CPU backend remains the production fallback
for very large/sparse models until a sparse CUDA factorization backend is
introduced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # Optional dependency: the normal project must still run without it.
    import cupy as cp
    _CUPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on machine CUDA setup
    cp = None
    _CUPY_IMPORT_ERROR = exc


class GPUBackendError(RuntimeError):
    """Raised when the CUDA/CuPy backend cannot be used."""


def gpu_available() -> bool:
    """Return True only when CuPy imports and at least one CUDA device exists."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_info() -> dict:
    """Return safe runtime information for the dashboard."""
    if cp is None:
        return {
            "available": False,
            "backend": "unavailable",
            "reason": f"CuPy is not installed/importable: {_CUPY_IMPORT_ERROR}",
        }
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            return {"available": False, "backend": "cuda", "reason": "No CUDA device detected"}
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name", b"CUDA GPU")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        return {
            "available": True,
            "backend": "cupy-cuda",
            "device_id": int(dev.id),
            "device_name": str(name),
            "memory_free_mb": round(float(free_b) / 2**20, 1),
            "memory_total_mb": round(float(total_b) / 2**20, 1),
            "cupy_version": getattr(cp, "__version__", "unknown"),
        }
    except Exception as exc:  # pragma: no cover - hardware dependent
        return {"available": False, "backend": "cuda", "reason": str(exc)}


def _require_gpu() -> None:
    if not gpu_available():
        info = gpu_info()
        raise GPUBackendError(
            "GPU backend requested, but CUDA/CuPy is unavailable. "
            "Install a matching CuPy CUDA package and ensure the NVIDIA driver/CUDA runtime is available. "
            f"Details: {info.get('reason', 'no CUDA device')}"
        )


@dataclass
class GPUReducedNewtonFactorization:
    """GPU factorization for H=diag(h) and S=A H^-1 A^T."""

    A_gpu: object
    h_gpu: object
    L_gpu: object | None
    schur_reg: float
    h_reg: float

    @property
    def num_eq(self) -> int:
        return int(self.A_gpu.shape[0])


def factor_reduced_system_gpu(H: np.ndarray, A_eq: np.ndarray, reg: float = 1e-12) -> GPUReducedNewtonFactorization:
    """Build and factor the reduced Newton Schur system on CUDA."""
    _require_gpu()
    if H.ndim != 1:
        raise GPUBackendError("The GPU backend currently requires diagonal H (1-D h=z/x).")

    h = np.asarray(H, dtype=np.float64)
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise GPUBackendError("GPU backend received non-positive or non-finite H diagonal.")

    scale = max(1.0, float(np.mean(np.abs(h))))
    reg_h = min(reg * scale, 5e-8)
    h_gpu = cp.asarray(h + reg_h, dtype=cp.float64)
    A_gpu = cp.asarray(np.asarray(A_eq, dtype=np.float64), dtype=cp.float64)

    if A_gpu.shape[0] == 0:
        return GPUReducedNewtonFactorization(A_gpu, h_gpu, None, 0.0, reg_h)

    # S = A diag(1/h) A^T, assembled on GPU.
    S_gpu = (A_gpu / h_gpu[None, :]) @ A_gpu.T
    S_gpu = 0.5 * (S_gpu + S_gpu.T)
    diag_scale = max(1.0, float(cp.mean(cp.abs(cp.diag(S_gpu))).get()))
    schur_reg = reg * diag_scale

    for _ in range(8):
        try:
            L_gpu = cp.linalg.cholesky(S_gpu + schur_reg * cp.eye(S_gpu.shape[0], dtype=cp.float64))
            return GPUReducedNewtonFactorization(A_gpu, h_gpu, L_gpu, schur_reg, reg_h)
        except Exception:
            schur_reg *= 10.0

    raise GPUBackendError("GPU Schur Cholesky factorization failed even with regularization")


def _chol_solve_gpu(L, b):
    y = cp.linalg.solve(L, b)
    return cp.linalg.solve(L.T, y)


def _schur_residual_gpu(fac, b_gpu, dy_gpu):
    q = (fac.A_gpu.T @ dy_gpu) / fac.h_gpu
    # The factored system uses the symmetrized S.  A A^T construction is already
    # symmetric up to floating-point noise, so this is the exact intended form.
    r = b_gpu - fac.A_gpu @ q - fac.schur_reg * dy_gpu
    return r, float(cp.max(cp.abs(r)).get())


def solve_reduced_system_gpu(fac: GPUReducedNewtonFactorization, rhs_x: np.ndarray,
                             rhs_eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve both Newton directions on GPU; return NumPy arrays for the solver."""
    _require_gpu()
    rhs_x_gpu = cp.asarray(rhs_x, dtype=cp.float64)
    rhs_eq_gpu = cp.asarray(rhs_eq, dtype=cp.float64)
    t = rhs_x_gpu / fac.h_gpu
    if fac.num_eq == 0:
        return cp.asnumpy(t), np.zeros(0, dtype=np.float64)

    b = rhs_eq_gpu - fac.A_gpu @ t
    dy = _chol_solve_gpu(fac.L_gpu, b)

    # Up to two GPU-side iterative-refinement corrections.
    best = dy
    r, rnorm = _schur_residual_gpu(fac, b, best)
    for _ in range(2):
        if rnorm == 0.0:
            break
        corr = _chol_solve_gpu(fac.L_gpu, r)
        cand = best + corr
        rc, rn = _schur_residual_gpu(fac, b, cand)
        if rn < rnorm:
            best, r, rnorm = cand, rc, rn
        else:
            break

    dx = (rhs_x_gpu + fac.A_gpu.T @ best) / fac.h_gpu
    return cp.asnumpy(dx), cp.asnumpy(best)
