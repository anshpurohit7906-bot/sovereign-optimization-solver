"""Focused tests for sparse Mehrotra initialization (Phase 3B).

Verifies that:
- Dense A still uses np.linalg.lstsq (regression).
- Sparse A uses scipy.sparse.linalg.lsqr without densifying.
- Returned x, y, z are finite, correct shape, with x > 0 and z > 0 after shifting.
- Rank-deficient and ill-conditioned matrices are handled gracefully.
- Large sparse matrices do not trigger densification.
- Fallback mechanisms work when LSQR fails.
- End-to-end sparse LP solve works.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import NumericalLP  # noqa: E402
from lp.mehrotra import (  # noqa: E402
    _mehrotra_initial_point,
    _shift_positive,
    solve_lp,
)


# ============================================================================
# Test 1: Dense-path regression
# ============================================================================
def test_dense_path_unchanged():
    """Dense A should still use np.linalg.lstsq exactly as before."""
    A_dense = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    b = np.array([1., 2., 3.])
    c = np.array([1., -1., 0.5])

    x, y, z = _mehrotra_initial_point(A_dense, b, c)

    assert x.shape == (3,)
    assert y.shape == (3,)
    assert z.shape == (3,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.all(x > 0.0)
    assert np.all(z > 0.0)

    # Cross-check against direct lstsq + shift
    x_ref = np.linalg.lstsq(A_dense, b, rcond=None)[0]
    y_ref = np.linalg.lstsq(A_dense.T, c, rcond=None)[0]
    z_ref = c - A_dense.T @ y_ref
    np.testing.assert_allclose(x, _shift_positive(x_ref), rtol=1e-14)
    np.testing.assert_allclose(y, y_ref, rtol=1e-14)
    np.testing.assert_allclose(z, _shift_positive(z_ref), rtol=1e-14)


# ============================================================================
# Test 2: Sparse LSQR basics
# ============================================================================
def test_sparse_lsqr_basic():
    """Sparse A should return finite x, y, z with correct shapes."""
    A_sparse = sp.csr_matrix(np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
    b = np.array([1., 2., 3.])
    c = np.array([1., -1., 0.5])

    x, y, z = _mehrotra_initial_point(A_sparse, b, c)

    assert x.shape == (3,)
    assert y.shape == (3,)
    assert z.shape == (3,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.all(x > 0.0)
    assert np.all(z > 0.0)


# ============================================================================
# Test 3: Strict positivity after shifting
# ============================================================================
def test_strict_positivity_enforced():
    """x > 0 and z > 0 must be enforced by _shift_positive()."""
    A_sparse = sp.csr_matrix(np.array([
        [1., -10., 0.],
        [0., 1., -10.],
        [0., 0., 1.]
    ]))
    b = np.array([1., 1., 1.])
    c = np.array([1., -100., -100.])

    x, y, z = _mehrotra_initial_point(A_sparse, b, c)
    assert np.all(x > 1e-10), f"x has values <= 1e-10: {x}"
    assert np.all(z > 1e-10), f"z has values <= 1e-10: {z}"


# ============================================================================
# Test 4: Dense vs. sparse agreement (small problem)
# ============================================================================
def test_sparse_vs_dense_agreement():
    """Small sparse and dense A should give numerically close initializers."""
    rng = np.random.default_rng(42)
    m, n = 8, 12
    A_dense = rng.standard_normal((m, n))
    A_sparse = sp.csr_matrix(A_dense)
    b = rng.standard_normal(m)
    c = rng.standard_normal(n)

    x_d, y_d, z_d = _mehrotra_initial_point(A_dense, b, c)
    x_s, y_s, z_s = _mehrotra_initial_point(A_sparse, b, c)

    np.testing.assert_allclose(x_d, x_s, rtol=1e-8)
    np.testing.assert_allclose(y_d, y_s, rtol=1e-8)
    np.testing.assert_allclose(z_d, z_s, rtol=1e-8)


# ============================================================================
# Test 5: Rank-deficient sparse matrix
# ============================================================================
def test_rank_deficient_sparse():
    """Rank-deficient sparse A should return usable finite initializer."""
    A_dense = np.array([[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]])
    A_sparse = sp.csr_matrix(A_dense)
    b = np.array([1., 2., 3.])
    c = np.array([1., 1., 1.])

    x, y, z = _mehrotra_initial_point(A_sparse, b, c)
    assert x.shape == (3,)
    assert y.shape == (3,)
    assert z.shape == (3,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.all(x > 0.0)
    assert np.all(z > 0.0)


# ============================================================================
# Test 6: Ill-conditioned sparse matrix
# ============================================================================
def test_ill_conditioned_sparse():
    """Ill-conditioned sparse A should not produce NaN/Inf."""
    n = 8
    A_dense = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    A_sparse = sp.csr_matrix(A_dense)
    b = np.ones(n)
    c = np.ones(n)
    assert np.linalg.cond(A_dense) > 1e4, "Expected ill-conditioned matrix"

    x, y, z = _mehrotra_initial_point(A_sparse, b, c)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.all(x > 0.0)
    assert np.all(z > 0.0)


# ============================================================================
# Test 7: Large sparse matrix (no densification)
# ============================================================================
def test_large_sparse_no_densification():
    """Large sparse A should not be densified."""
    rng = np.random.default_rng(7)
    m, n = 1000, 2000
    nnz = 5000
    row_idx = rng.integers(0, m, nnz)
    col_idx = rng.integers(0, n, nnz)
    data = rng.standard_normal(nnz)
    A_sparse = sp.csr_matrix((data, (row_idx, col_idx)), shape=(m, n))
    b = rng.standard_normal(m)
    c = rng.standard_normal(n)

    x, y, z = _mehrotra_initial_point(A_sparse, b, c)
    assert x.shape == (n,) and y.shape == (m,) and z.shape == (n,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.all(x > 0.0) and np.all(z > 0.0)


# ============================================================================
# Test 8: LSQR failure fallback (simulated)
# ============================================================================
def test_lsqr_failure_fallback():
    """If LSQR fails, fallback to dense or trivial initializer."""
    A_sparse = sp.csr_matrix(np.array([[1., 2., 3.], [4., 5., 6.]]))
    b = np.array([1., 2.])
    c = np.array([1., -1., 0.5])

    with patch("lp.mehrotra.lsqr") as mock_lsqr:
        mock_lsqr.return_value = (
            np.full(3, np.nan), 7, 10, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        x, y, z = _mehrotra_initial_point(A_sparse, b, c)

    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.all(x > 0.0)
    assert np.all(z > 0.0)


# ============================================================================
# Test 9: End-to-end sparse LP
# ============================================================================
def test_end_to_end_sparse_lp():
    """Small sparse LP should solve successfully."""
    # min x1 + 2*x2 s.t. x1 + x2 = 3, x >= 0  =>  optimal (0,3), obj = 6
    A_sparse = sp.csr_matrix(np.array([[1., 1.]]))
    lp = NumericalLP(
        name="SPARSE_TEST", objective_name="OBJ",
        A=A_sparse, b=np.array([3.]), c=np.array([1., 2.]),
        lower_bounds=np.zeros(2), upper_bounds=np.full(2, np.inf),
        row_types=("E",), var_names=("x1", "x2"), row_names=("eq1",),
    )
    result = solve_lp(lp, tol=1e-7, max_iter=100)
    assert result.status in ("optimal", "numerical_tail", "stalled"), result.message
    assert np.isfinite(result.objective)


# ============================================================================
# Test 10: Dense regression
# ============================================================================
def test_dense_regression():
    """Dense edge cases must produce no regression."""
    A_dense = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0]])
    b = np.array([6.0, 4.0])
    c = np.array([2.0, 3.0, 1.0])
    x, y, z = _mehrotra_initial_point(A_dense, b, c)
    assert x.shape == (3,) and y.shape == (2,) and z.shape == (3,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.all(x > 0.0) and np.all(z > 0.0)


# ============================================================================
# Test 11: Dense fallback triggered by small dense_size_limit
# ============================================================================
def test_small_sparse_uses_dense_fallback():
    """Sparse A with forced LSQR failure and dense_size_limit >= m*n
    should fall back to dense lstsq, NOT trivial initializer."""
    A_sparse = sp.csr_matrix(np.array([[1., 2., 3.], [4., 5., 6.]]))
    b = np.array([1., 2.])
    c = np.array([1., -1., 0.5])

    # A is 2×3 = 6 dense entries.  Set dense_size_limit=10 so the
    # shape-product check (m*n=6 <= 10) allows densification.
    with patch("lp.mehrotra.lsqr") as mock_lsqr:
        mock_lsqr.return_value = (
            np.full(3, np.nan), 7, 10, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        x, y, z = _mehrotra_initial_point(A_sparse, b, c,
                                           dense_size_limit=10)

    # Should match the dense-lstsq result exactly
    A_dense = np.array([[1., 2., 3.], [4., 5., 6.]])
    x_dense = np.linalg.lstsq(A_dense, b, rcond=None)[0]
    y_dense = np.linalg.lstsq(A_dense.T, c, rcond=None)[0]

    np.testing.assert_allclose(x, _shift_positive(x_dense), rtol=1e-14)
    np.testing.assert_allclose(y, y_dense, rtol=1e-14)
    np.testing.assert_allclose(z, _shift_positive(c - A_dense.T @ y_dense),
                               rtol=1e-14)


# ============================================================================
# Test 12: Sparse CSC matrix support
# ============================================================================
def test_sparse_csc_matrix():
    """Sparse A in CSC format should also work."""
    A_csr = sp.csr_matrix(np.array([[1., 2., 3.], [4., 5., 6.]]))
    A_csc = A_csr.tocsc()
    b = np.array([1., 2.])
    c = np.array([1., -1., 0.5])

    x_csr, y_csr, z_csr = _mehrotra_initial_point(A_csr, b, c)
    x_csc, y_csc, z_csc = _mehrotra_initial_point(A_csc, b, c)

    np.testing.assert_allclose(x_csr, x_csc, rtol=1e-10)
    np.testing.assert_allclose(y_csr, y_csc, rtol=1e-10)
    np.testing.assert_allclose(z_csr, z_csc, rtol=1e-10)


# ============================================================================
# Test 13: Trivial initializer fallback for large sparse
# ============================================================================
def test_trivial_initializer_fallback():
    """For large sparse with forced LSQR failure, trivial initializer works."""
    rng = np.random.default_rng(13)
    m, n = 500, 1000
    nnz = 2000
    row_idx = rng.integers(0, m, nnz)
    col_idx = rng.integers(0, n, nnz)
    data = 1e-6 * rng.standard_normal(nnz)
    A_sparse = sp.csr_matrix((data, (row_idx, col_idx)), shape=(m, n))
    b = 1e-6 * rng.standard_normal(m)
    c = 1e-6 * rng.standard_normal(n)

    with patch("lp.mehrotra.lsqr") as mock_lsqr:
        mock_lsqr.return_value = (
            np.full(n, np.nan), 7, 10, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        x, y, z = _mehrotra_initial_point(A_sparse, b, c, dense_size_limit=100)

    assert np.all(np.isfinite(x))
    assert np.all(x > 0.0) and np.all(z > 0.0)
    assert x.shape == (n,) and y.shape == (m,) and z.shape == (n,)


# ============================================================================
# Test 14: Large-sparse LSQR failure must never call toarray()
# ============================================================================
def test_large_sparse_lsqr_failure_never_densifies():
    """A 1000x2000 sparse matrix with few NNZ must NOT be densified even
    if dense_size_limit is large.  The criterion is based on shape
    (m * n), not on nnz.  This test patches ``A.toarray`` and asserts
    it is never called, and also patches LSQR to force failure."""
    rng = np.random.default_rng(42)
    m, n = 1000, 2000
    nnz = 500
    row_idx = rng.integers(0, m, nnz)
    col_idx = rng.integers(0, n, nnz)
    data = rng.standard_normal(nnz)
    A_sparse = sp.csr_matrix((data, (row_idx, col_idx)), shape=(m, n))
    b = rng.standard_normal(m)
    c = rng.standard_normal(n)

    # Force LSQR to return NaN so the fallback path is taken
    with patch("lp.mehrotra.lsqr") as mock_lsqr:
        mock_lsqr.return_value = (
            np.full(n, np.nan), 7, 10, 1.0, 1.0, 1.0, 1.0, 1.0,
        )
        with patch.object(
            A_sparse, "toarray", wraps=A_sparse.toarray
        ) as spy:
            x, y, z = _mehrotra_initial_point(A_sparse, b, c)

    # m * n = 2_000_000 >> dense_size_limit (default 5000) => trivial path
    spy.assert_not_called()
    assert x.shape == (n,) and y.shape == (m,) and z.shape == (n,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.all(x > 0.0) and np.all(z > 0.0)
    # The trivial initializer yields x = ones(n) before _shift_positive,
    # so all entries of x are identical; all y are 0.
    assert np.allclose(x, x[0])
    np.testing.assert_array_equal(y, np.zeros(m))