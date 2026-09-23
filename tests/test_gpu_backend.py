"""GPU backend (backend="gpu"/"auto") tests.

CPU-only by design: these tests prove the GPU integration fails gracefully
and that CPU behavior is untouched.  The single real GPU-vs-CPU comparison
test is skipped unless CuPy with a CUDA device is actually available --
no fake GPU results are produced without hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from opticore.lp.gpu_linear_system import (  # noqa: E402
    GPUBackendError, gpu_available, gpu_info,
)
from opticore.lp.linear_system import factor_reduced_system  # noqa: E402
from opticore.lp.mehrotra import (  # noqa: E402
    MehrotraError, NumericalLP, solve_lp,
)

_HAS_GPU = gpu_available()


def _tiny_lp() -> NumericalLP:
    """min -x1 - x2 s.t. x1+x2+x3 = 10 (E), x1+2x2 <= 8 (L), 3x1+x2 <= 9 (L).

    Optimum (2, 3, 5), objective -5 (the mehrotra.py test-harness LP).
    """
    return NumericalLP(
        name="TINY_MIN",
        objective_name="COST",
        var_names=("x1", "x2", "x3"),
        row_names=("E1", "L1", "L2"),
        A=np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 0.0], [3.0, 1.0, 0.0]]),
        b=np.array([10.0, 8.0, 9.0]),
        c=np.array([-1.0, -1.0, 0.0]),
        row_types=("E", "L", "L"),
        lower_bounds=np.zeros(3),
        upper_bounds=np.full(3, np.inf),
    )


def test_gpu_available_returns_bool_and_info_is_safe():
    assert isinstance(gpu_available(), bool)
    info = gpu_info()
    assert isinstance(info, dict) and "available" in info
    if not _HAS_GPU:
        assert info["available"] is False


@pytest.mark.skipif(_HAS_GPU, reason="GPU present; failure path not applicable")
def test_factor_reduced_system_gpu_raises_without_cuda():
    h = np.array([1.0, 2.0, 3.0])
    A = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises(GPUBackendError):
        factor_reduced_system(h, A, backend="gpu")


@pytest.mark.skipif(_HAS_GPU, reason="GPU present; failure path not applicable")
def test_solve_lp_backend_gpu_raises_gracefully_without_cuda():
    with pytest.raises(MehrotraError, match="GPU backend"):
        solve_lp(_tiny_lp(), backend="gpu")


@pytest.mark.skipif(not _HAS_GPU, reason="GPU present; auto uses it")
def test_solve_lp_backend_auto_falls_back_to_cpu_without_cuda():
    r_auto = solve_lp(_tiny_lp(), backend="auto")
    r_cpu = solve_lp(_tiny_lp(), backend="cpu")
    assert r_auto.status == r_cpu.status == "optimal"
    assert abs(r_auto.objective - r_cpu.objective) <= 1e-9


def test_solve_lp_backend_auto_without_cuda_equals_cpu():
    """backend='auto' must be behaviorally identical to 'cpu' here.

    On a CPU-only (CuPy-less) machine this proves the fallback; on a GPU
    machine the auto result must still match the CPU optimum.
    """
    r_auto = solve_lp(_tiny_lp(), backend="auto")
    r_cpu = solve_lp(_tiny_lp(), backend="cpu")
    assert r_auto.status == "optimal" and r_cpu.status == "optimal"
    assert abs(r_auto.objective - (-5.0)) <= 1e-6
    assert abs(r_cpu.objective - (-5.0)) <= 1e-6
    assert abs(r_auto.objective - r_cpu.objective) <= 1e-9


def test_solve_lp_backend_invalid_raises():
    with pytest.raises(ValueError, match="backend"):
        solve_lp(_tiny_lp(), backend="bogus")


def test_factor_reduced_system_backend_invalid_raises():
    h = np.array([1.0, 2.0, 3.0])
    A = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="backend"):
        factor_reduced_system(h, A, backend="bogus")


def test_cpu_backend_default_unchanged_tiny_lp():
    """backend='cpu' (and the no-arg default) must remain identical."""
    r_default = solve_lp(_tiny_lp())
    r_cpu = solve_lp(_tiny_lp(), backend="cpu")
    assert r_default.status == r_cpu.status == "optimal"
    assert abs(r_default.objective - r_cpu.objective) <= 1e-12


@pytest.mark.skipif(not _HAS_GPU, reason="CUDA/CuPy device not available")
def test_deterministic_lp_cpu_vs_gpu_match():
    """Real GPU-vs-CPU comparison; skipped (not faked) without hardware."""
    r_cpu = solve_lp(_tiny_lp(), backend="cpu", tol=1e-9)
    r_gpu = solve_lp(_tiny_lp(), backend="gpu", tol=1e-9)
    assert r_cpu.status == r_gpu.status
    assert abs(r_cpu.objective - r_gpu.objective) <= 1e-7
    assert abs(r_cpu.objective - (-5.0)) <= 1e-7
