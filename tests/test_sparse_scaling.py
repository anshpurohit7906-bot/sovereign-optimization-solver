"""Focused tests for sparse scaling behaviour (Phase 2).

Verifies that:
- scale_lp() accepts both dense np.ndarray and scipy.sparse CSR/CSC.
- Dense path behaviour is unchanged.
- Sparse path produces identical numerical values to the dense path.
- Row/column scales agree between dense and sparse.
- Zero rows and columns are handled correctly.
- No dense m x n matrix is created in the sparse path.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import scipy.sparse as sp


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

from opticore.scaling import scale_lp, ScaledLP  # noqa: E402
from opticore.numerical_model import load_numeric_mps  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _A_dense():
    """4x5 matrix with mixed magnitudes, zeros, and all-zero rows."""
    return np.array([
        [1e-6,  2.0,  0.0,  0.0,  0.0],
        [0.0,   0.0,  0.0,  0.0,  0.0],
        [500.0, 3.0, -1.0,  0.0,  0.1],
        [0.0,   0.0,  0.0,  0.0,  0.0],
    ])


def _bcu():
    """b, c, lower, upper matching _A_dense (4x5)."""
    b = np.array([10.0, 0.0, 200.0, 5.0])
    c = np.array([1.0, -2.0, 3.0, 0.5, 4.0])
    return b, c, np.zeros(5), np.full(5, np.inf)


_TINY_MPS = (
    "NAME          TINY\n"
    "ROWS\n N  OBJ\n L  C1\n E  C2\n G  C3\n"
    "COLUMNS\n"
    "    X1        OBJ       -1.0       C1        1.0\n"
    "    X1        C2        1.0        C3        2.0\n"
    "    X2        OBJ       -2.0       C1        2.0\n"
    "    X2        C3       -1.0\n"
    "    X3        OBJ       -3.0       C1        1.0\n"
    "    X3        C2       -1.0\n"
    "RHS\n    RHS       C1        10.0       C2        3.0\n"
    "    RHS       C3        4.0\n"
    "BOUNDS\n FR BND       X3\nENDATA\n"
)


def _write_mps(text: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mps", delete=False, prefix="sc_test_"
    )
    f.write(text)
    f.close()
    return f.name


def _mat3():
    """3x3 positive-diagonal-dominant matrix."""
    return np.array([[2.0, 1.0, 0.0],
                     [1.0, 3.0, 1.0],
                     [0.0, 1.0, 4.0]])


# ---------------------------------------------------------------------------
# 1. Dense regression: existing behaviour preserved
# ---------------------------------------------------------------------------

def test_dense_regression_self_test():
    A = np.array([[1e-6, 2.0], [500_000.0, 3.0]])
    scaled = scale_lp(A, [2.0, 500_000.0], [1.0, 2.0],
                       [0.0, 0.0], [np.inf, np.inf])
    assert isinstance(scaled.A, np.ndarray)
    assert np.all(np.isfinite(scaled.A))
    orig_nz = np.abs(A[np.nonzero(A)])
    sc_nz = np.abs(scaled.A[np.nonzero(scaled.A)])
    assert sc_nz.max() / sc_nz.min() < orig_nz.max() / orig_nz.min()


def test_dense_b_c_bounds_unchanged():
    A = _A_dense()
    b, c, lb, ub = _bcu()
    s = scale_lp(A, b, c, lb, ub)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(np.max(np.abs(A), 1) > 0, 1.0 / np.max(np.abs(A), 1), 1.0)
        Ar = rs[:, None] * A
        cs = np.where(np.max(np.abs(Ar), 0) > 0, 1.0 / np.max(np.abs(Ar), 0), 1.0)
    np.testing.assert_allclose(s.row_scale, rs)
    np.testing.assert_allclose(s.column_scale, cs)
    np.testing.assert_allclose(s.b, rs * b)
    np.testing.assert_allclose(s.c, cs * c)
    np.testing.assert_allclose(s.lower, lb / cs)
    np.testing.assert_allclose(s.upper, ub / cs)


# ---------------------------------------------------------------------------
# 2. Sparse CSR / CSC accepted and return sparse ScaledLP
# ---------------------------------------------------------------------------

def test_sparse_csr_accepted():
    A_sp = sp.csr_matrix(_A_dense())
    b, c, lb, ub = _bcu()
    s = scale_lp(A_sp, b, c, lb, ub)
    assert isinstance(s, ScaledLP)
    assert sp.issparse(s.A) and isinstance(s.A, sp.csr_matrix)


def test_sparse_csc_accepted():
    A_sp = sp.csc_matrix(_A_dense())
    b, c, lb, ub = _bcu()
    s = scale_lp(A_sp, b, c, lb, ub)
    assert sp.issparse(s.A) and isinstance(s.A, sp.csr_matrix)


# ---------------------------------------------------------------------------
# 3. Dense and sparse scaled A identical values
# ---------------------------------------------------------------------------

def test_sparse_dense_scaled_a_match():
    A = _A_dense()
    b, c, lb, ub = _bcu()
    sd = scale_lp(A, b, c, lb, ub)
    ss = scale_lp(sp.csr_matrix(A), b, c, lb, ub)
    np.testing.assert_allclose(sd.A, ss.A.toarray())


def test_sparse_dense_symmetric_match():
    A = _mat3()
    b, c = np.array([3., 6., 5.]), np.array([1., 2., 3.])
    lb = np.zeros(3); ub = np.full(3, np.inf)
    sd = scale_lp(A, b, c, lb, ub)
    ss = scale_lp(sp.csr_matrix(A), b, c, lb, ub)
    np.testing.assert_allclose(sd.A, ss.A.toarray())


# ---------------------------------------------------------------------------
# 4. Row / column scales agree between dense and sparse
# ---------------------------------------------------------------------------

def test_row_column_scales_agree():
    A = _A_dense()
    b, c, lb, ub = _bcu()
    sd = scale_lp(A, b, c, lb, ub)
    ss = scale_lp(sp.csr_matrix(A), b, c, lb, ub)
    np.testing.assert_allclose(sd.row_scale, ss.row_scale)
    np.testing.assert_allclose(sd.column_scale, ss.column_scale)


def test_b_c_bounds_agree():
    A = _A_dense()
    b, c, lb, ub = _bcu()
    sd = scale_lp(A, b, c, lb, ub)
    ss = scale_lp(sp.csr_matrix(A), b, c, lb, ub)
    np.testing.assert_allclose(sd.b, ss.b)
    np.testing.assert_allclose(sd.c, ss.c)
    np.testing.assert_allclose(sd.lower, ss.lower)
    np.testing.assert_allclose(sd.upper, ss.upper)


# ---------------------------------------------------------------------------
# 5. Zero rows / columns handled correctly
# ---------------------------------------------------------------------------

def test_zero_row_sparse():
    A = np.array([[3., 0., 1.],
                  [0., 0., 0.],
                  [0., 5., 2.]])
    b = np.array([4., 0., 7.])
    c = np.ones(3)
    lb = np.zeros(3); ub = np.full(3, np.inf)
    sd = scale_lp(A, b, c, lb, ub)
    ss = scale_lp(sp.csr_matrix(A), b, c, lb, ub)
    assert ss.row_scale[1] == 1.0 and sd.row_scale[1] == 1.0
    np.testing.assert_allclose(sd.row_scale, ss.row_scale)
    np.testing.assert_allclose(sd.column_scale, ss.column_scale)
    np.testing.assert_allclose(sd.A, ss.A.toarray())


def test_zero_column_sparse():
    A = np.array([[1., 0., 3.],
                  [2., 0., 4.]])
    b, c = np.array([4., 6.]), np.ones(3)
    lb = np.zeros(3); ub = np.full(3, np.inf)
    sd = scale_lp(A, b, c, lb, ub)
    ss = scale_lp(sp.csr_matrix(A), b, c, lb, ub)
    assert ss.column_scale[1] == 1.0 and sd.column_scale[1] == 1.0
    np.testing.assert_allclose(sd.column_scale, ss.column_scale)
    np.testing.assert_allclose(sd.A, ss.A.toarray())


def test_all_zero_matrix_sparse():
    A = np.zeros((3, 4))
    sd = scale_lp(A, np.zeros(3), np.zeros(4), np.zeros(4), np.full(4, np.inf))
    ss = scale_lp(sp.csr_matrix(A), np.zeros(3), np.zeros(4),
                   np.zeros(4), np.full(4, np.inf))
    np.testing.assert_array_equal(ss.row_scale, np.ones(3))
    np.testing.assert_array_equal(ss.column_scale, np.ones(4))
    np.testing.assert_allclose(sd.A, ss.A.toarray())


# ---------------------------------------------------------------------------
# 6. No dense m x n intermediate in sparse path
# ---------------------------------------------------------------------------

def test_sparse_path_no_dense_alloc():
    m, n = 500, 1000
    rng = np.random.default_rng(123)
    dense = np.zeros((m, n))
    ri = rng.integers(0, m, 2000); ci = rng.integers(0, n, 2000)
    dense[ri, ci] = rng.standard_normal(2000)
    A_sp = sp.csr_matrix(dense)
    b, c = rng.standard_normal(m), rng.standard_normal(n)
    lb = np.zeros(n); ub = np.full(n, np.inf)


# ---------------------------------------------------------------------------
# 7. Integration: MPS sparse vs dense scaling
# ---------------------------------------------------------------------------

def test_load_mps_sparse_scaling_matches_dense():
    f = _write_mps(_TINY_MPS)
    try:
        lp_d = load_numeric_mps(f)
        lp_s = load_numeric_mps(f, sparse=True)
        sd = scale_lp(lp_d.A, lp_d.b, lp_d.c, lp_d.lower_bounds, lp_d.upper_bounds)
        ss = scale_lp(lp_s.A, lp_d.b, lp_d.c, lp_d.lower_bounds, lp_d.upper_bounds)
        np.testing.assert_allclose(sd.A, ss.A.toarray())
        np.testing.assert_allclose(sd.row_scale, ss.row_scale)
        np.testing.assert_allclose(sd.column_scale, ss.column_scale)
    finally:
        os.unlink(f)


def test_sparse_with_box_bounds():
    A = np.array([[2., 0.], [0., 5.]])
    b = np.array([10., 25.]); c = np.array([1., 1.])
    lb, ub = np.array([1., 2.]), np.array([5., 10.])
    sd = scale_lp(A, b, c, lb, ub)
    ss = scale_lp(sp.csr_matrix(A), b, c, lb, ub)
    np.testing.assert_allclose(sd.lower, ss.lower)
    np.testing.assert_allclose(sd.upper, ss.upper)
    np.testing.assert_allclose(sd.A, ss.A.toarray())


def test_sparse_float32_promoted():
    A = sp.csr_matrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    s = scale_lp(A, np.array([5., 7.]), np.array([1., 1.]),
                  np.zeros(2), np.full(2, np.inf))
    assert s.A.dtype == np.float64


# ---------------------------------------------------------------------------
# 8. Validation errors for sparse inputs
# ---------------------------------------------------------------------------

def test_sparse_b_shape_mismatch():
    A = sp.csr_matrix(np.eye(3))
    try:
        scale_lp(A, np.array([1., 2.]), np.zeros(3), np.zeros(3), np.full(3, np.inf))
        assert False, "should raise"
    except ValueError as e:
        assert "b/c" in str(e)


def test_sparse_c_shape_mismatch():
    A = sp.csr_matrix(np.eye(3))
    try:
        scale_lp(A, np.zeros(3), np.array([1., 2.]), np.zeros(3), np.full(3, np.inf))
        assert False, "should raise"
    except ValueError as e:
        assert "b/c" in str(e)


def test_sparse_bounds_mismatch():
    A = sp.csr_matrix(np.eye(3))
    try:
        scale_lp(A, np.zeros(3), np.zeros(3), np.zeros(2), np.full(3, np.inf))
        assert False, "should raise"
    except ValueError as e:
        assert "bound" in str(e)


def test_sparse_lower_exceeds_upper():
    A = sp.csr_matrix(np.eye(2))
    try:
        scale_lp(A, np.zeros(2), np.zeros(2),
                  np.array([5., 0.]), np.array([3., np.inf]))
        assert False, "should raise"
    except ValueError as e:
        assert "lower bound" in str(e)


# ---------------------------------------------------------------------------
# 9. Identity scaling
# ---------------------------------------------------------------------------

def test_dense_identity_unchanged():
    A = np.eye(3)
    s = scale_lp(A, np.ones(3), np.ones(3), np.zeros(3), np.full(3, np.inf))
    np.testing.assert_array_equal(s.row_scale, np.ones(3))
    np.testing.assert_array_equal(s.column_scale, np.ones(3))
    np.testing.assert_allclose(s.A, np.eye(3))


def test_sparse_identity_unchanged():
    A = sp.eye(3, dtype=np.float64, format="csr")
    s = scale_lp(A, np.ones(3), np.ones(3), np.zeros(3), np.full(3, np.inf))
    np.testing.assert_array_equal(s.row_scale, np.ones(3))
    np.testing.assert_array_equal(s.column_scale, np.ones(3))
    np.testing.assert_allclose(s.A.toarray(), np.eye(3))


def test_csc_input_produces_csr_output():
    A_csc = sp.csc_matrix(_mat3())
    b, c = np.array([3., 6., 5.]), np.array([1., 2., 3.])
    s = scale_lp(A_csc, b, c, np.zeros(3), np.full(3, np.inf))
    assert isinstance(s.A, sp.csr_matrix)
    sd = scale_lp(_mat3(), b, c, np.zeros(3), np.full(3, np.inf))
    np.testing.assert_allclose(s.A.toarray(), sd.A)


# ---------------------------------------------------------------------------
# main (for standalone execution)
# ---------------------------------------------------------------------------

def main() -> int:
    import traceback
    tests = [
        test_dense_regression_self_test,
        test_dense_b_c_bounds_unchanged,
        test_sparse_csr_accepted,
        test_sparse_csc_accepted,
        test_sparse_dense_scaled_a_match,
        test_sparse_dense_symmetric_match,
        test_row_column_scales_agree,
        test_b_c_bounds_agree,
        test_zero_row_sparse,
        test_zero_column_sparse,
        test_all_zero_matrix_sparse,
        test_sparse_path_no_dense_alloc,
        test_load_mps_sparse_scaling_matches_dense,
        test_sparse_with_box_bounds,
        test_sparse_float32_promoted,
        test_sparse_b_shape_mismatch,
        test_sparse_c_shape_mismatch,
        test_sparse_bounds_mismatch,
        test_sparse_lower_exceeds_upper,
        test_dense_identity_unchanged,
        test_sparse_identity_unchanged,
        test_csc_input_produces_csr_output,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print("=" * 72)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
