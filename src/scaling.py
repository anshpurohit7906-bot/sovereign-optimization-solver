from __future__ import annotations
from dataclasses import dataclass
from typing import Union

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class ScaledLP:
    A: Union[np.ndarray, sp.csr_matrix]
    b: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    row_scale: np.ndarray
    column_scale: np.ndarray


def _safe_reciprocal_max(values, axis: int) -> np.ndarray:
    """Compute ``1 / max(|values|, axis)`` with ``1.0`` for zero rows/cols.

    Works for both dense ``np.ndarray`` and ``scipy.sparse`` matrices.
    """
    if sp.issparse(values):
        if values.shape[axis] == 0:
            return np.ones(values.shape[1 - axis])
        # scipy sparse ``.max`` returns a sparse matrix; materialise it.
        magnitude = np.abs(values)
        magnitude = np.asarray(magnitude.max(axis=axis).toarray()).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(magnitude > 0.0, 1.0 / magnitude, 1.0)
    # Dense path — preserved exactly.
    values = np.asarray(values)
    if values.shape[axis] == 0:
        return np.ones(values.shape[1 - axis])
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = np.max(np.abs(values), axis=axis)
        return np.where(magnitude > 0.0, 1.0 / magnitude, 1.0)


def scale_lp(A, b, c, lower, upper) -> ScaledLP:
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    # ---- Sparse path ----
    if sp.issparse(A):
        A = A.astype(np.float64, copy=False).tocsr()
        m, n = A.shape
        if b.shape != (m,) or c.shape != (n,):
            raise ValueError("b/c shapes do not match A")
        if lower.shape != (n,) or upper.shape != (n,):
            raise ValueError("bound shapes do not match A")
        if np.any(lower > upper):
            raise ValueError("lower bound exceeds upper bound")

        row_scale = _safe_reciprocal_max(A, axis=1)           # (m,)

        # Row scaling: diag(row_scale) @ A — O(nnz), no dense intermediate.
        # CSR data is grouped by row, so ``indptr`` gives the slice per row.
        A_row = A.copy()
        row_idx = np.repeat(np.arange(m), np.diff(A_row.indptr))
        A_row.data *= row_scale[row_idx]

        column_scale = _safe_reciprocal_max(A_row, axis=0)    # (n,)

        # Column scaling: A_row @ diag(column_scale) — O(nnz).
        A_scaled = A_row.copy()
        A_scaled.data *= column_scale[A_scaled.indices]

        b_scaled = row_scale * b
        c_scaled = column_scale * c
        lower_scaled = lower / column_scale
        upper_scaled = upper / column_scale
        return ScaledLP(A_scaled, b_scaled, c_scaled,
                        lower_scaled, upper_scaled,
                        row_scale, column_scale)

    # ---- Dense path (unchanged) ----
    A = np.asarray(A, dtype=np.float64)
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
