from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ScaledLP:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    row_scale: np.ndarray
    column_scale: np.ndarray

def _safe_reciprocal_max(values: np.ndarray, axis: int) -> np.ndarray:
    # np.max raises on an empty reduction axis (e.g. an LP with zero rows,
    # which arises when every variable is fixed); such systems are trivially
    # equilibrated with all scales 1.
    if values.shape[axis] == 0:
        return np.ones(values.shape[1 - axis])
    magnitude = np.max(np.abs(values), axis=axis)
    return np.where(magnitude > 0.0, 1.0 / magnitude, 1.0)

def scale_lp(A, b, c, lower, upper) -> ScaledLP:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    m, n = A.shape
    if b.shape != (m,) or c.shape != (n,):
        raise ValueError("b/c shapes do not match A")
    if lower.shape != (n,) or upper.shape != (n,):
        raise ValueError("bound shapes do not match A")
    if np.any(lower > upper):
        raise ValueError("lower bound exceeds upper bound")
    row_scale = _safe_reciprocal_max(A, axis=1)
    A_row = row_scale[:, None] * A
    column_scale = _safe_reciprocal_max(A_row, axis=0)
    A_scaled = A_row * column_scale[None, :]
    b_scaled = row_scale * b
    c_scaled = column_scale * c
    lower_scaled = lower / column_scale
    upper_scaled = upper / column_scale
    return ScaledLP(A_scaled, b_scaled, c_scaled, lower_scaled, upper_scaled, row_scale, column_scale)

def unscale_solution(z, column_scale):
    z = np.asarray(z, dtype=np.float64)
    column_scale = np.asarray(column_scale, dtype=np.float64)
    if z.shape != column_scale.shape:
        raise ValueError("z and column_scale must have the same shape")
    return column_scale * z

def _self_test():
    A = np.array([[1e-6, 2.0], [500_000.0, 3.0]])
    b = np.array([2.0, 500_000.0])
    c = np.array([1.0, 2.0])
    lower = np.array([0.0, 0.0])
    upper = np.array([np.inf, np.inf])
    scaled = scale_lp(A, b, c, lower, upper)
    original_nz = np.abs(A[np.nonzero(A)])
    scaled_nz = np.abs(scaled.A[np.nonzero(scaled.A)])
    original_spread = original_nz.max() / original_nz.min()
    scaled_spread = scaled_nz.max() / scaled_nz.min()
    x = np.array([3.0, 4.0])
    z = x / scaled.column_scale
    assert np.allclose(unscale_solution(z, scaled.column_scale), x)
    assert np.all(np.isfinite(scaled.A))
    assert scaled_spread < original_spread
    print("Scaling self-test: PASS")
    print(f"Original nonzero magnitude spread: {original_spread:.3e}")
    print(f"Scaled nonzero magnitude spread:   {scaled_spread:.3e}")
    print(f"Row scales:    {scaled.row_scale}")
    print(f"Column scales: {scaled.column_scale}")

if __name__ == "__main__":
    _self_test()
