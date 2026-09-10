"""Optional CuPy GPU linear algebra backend.

The production solver remains dependency-free from CUDA: GPU use is selected
explicitly. If CuPy is unavailable, the backend reports a clear message.
"""
from __future__ import annotations
import numpy as np


def gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def solve_dense_kkt_gpu(H, A, rhs, reg=1e-10):
    try:
        import cupy as cp
    except ImportError as e:
        raise RuntimeError("GPU backend requires CuPy; install the CUDA-matched cupy package") from e
    Hc = cp.asarray(H)
    rc = cp.asarray(rhs)
    n = Hc.shape[0]
    if A is None or A.shape[0] == 0:
        sol = cp.linalg.solve(Hc + reg*cp.eye(n, dtype=Hc.dtype), rc[:n])
        return cp.asnumpy(sol), np.empty(0)
    Ac = cp.asarray(A)
    meq = Ac.shape[0]
    K = cp.block([[Hc + reg*cp.eye(n, dtype=Hc.dtype), Ac.T],
                  [Ac, cp.zeros((meq, meq), dtype=Hc.dtype)]])
    sol = cp.linalg.solve(K, rc)
    return cp.asnumpy(sol[:n]), cp.asnumpy(sol[n:])
