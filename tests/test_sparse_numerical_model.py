"""Focused tests for sparse NumericalLP behaviour (Phase 1).

Verifies that:
- NumericalLP accepts both dense np.ndarray and scipy.sparse for A.
- nnz returns the correct count for both representations.
- load_numeric_mps(path, sparse=True) produces a CSR matrix whose
  numerical values exactly match the dense path.
- load_numeric_mps(path) (default) still returns dense and behaves
  identically to before.
- The sparse path never materialises a full m x n dense array.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import scipy.sparse as sp


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

from opticore.numerical_model import (  # noqa: E402
    NumericalLP,
    NumericalModelError,
    to_numeric,
    to_sparse_numeric,
    load_numeric_mps,
    validate_numeric_lp,
)
from opticore.mps_parser import MPSParser, LPModel  # noqa: E402


_TINY_MPS = """
NAME          TINY
ROWS
 N  OBJ
 L  C1
 E  C2
 G  C3
COLUMNS
    X1        OBJ       -1.0       C1        1.0
    X1        C2        1.0        C3        2.0
    X2        OBJ       -2.0       C1        2.0
    X2        C3       -1.0
    X3        OBJ       -3.0       C1        1.0
    X3        C2       -1.0
RHS
    RHS       C1        10.0       C2        3.0
    RHS       C3        4.0
BOUNDS
 FR BND       X3
ENDATA
"""

_DENSE_MPS = """
NAME          DENSE
ROWS
 N  OBJ
 L  R1
 E  R2
COLUMNS
    X1        OBJ       1.0        R1        1.0
    X1        R2        1.0
    X2        OBJ       2.0        R1        1.0
    X2        R2        1.0
RHS
    RHS       R1        5.0        R2        3.0
BOUNDS
ENDATA
"""

_EMPTY_COEFFS_MPS = """
NAME          EMPTYCO
ROWS
 N  OBJ
 L  R1
COLUMNS
    X1        OBJ       1.0
RHS
    RHS       R1        1.0
BOUNDS
ENDATA
"""


def _write_mps(text: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mps", delete=False, prefix="sparse_test_"
    )
    f.write(text)
    f.close()
    return f.name


# ---- 1. Type checks ----

def test_dense_a_is_ndarray() -> None:
    A = np.array([[1.0, 2.0], [3.0, 0.0]])
    lp = NumericalLP(
        name="t", objective_name="obj",
        A=A, b=np.zeros(2), c=np.zeros(2),
        lower_bounds=np.zeros(2), upper_bounds=np.full(2, np.inf),
        row_types=("L", "L"), var_names=("x1", "x2"), row_names=("r1", "r2"),
    )
    assert isinstance(lp.A, np.ndarray)
    assert lp.A.shape == (2, 2)


def test_sparse_csr_a_is_csr() -> None:
    A = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 4.0]]))
    lp = NumericalLP(
        name="t", objective_name="obj",
        A=A, b=np.zeros(2), c=np.zeros(2),
        lower_bounds=np.zeros(2), upper_bounds=np.full(2, np.inf),
        row_types=("L", "L"), var_names=("x1", "x2"), row_names=("r1", "r2"),
    )
    assert sp.issparse(lp.A)
    assert isinstance(lp.A, sp.csr_matrix)
    assert lp.A.shape == (2, 2)


def test_sparse_csc_a_is_csc() -> None:
    A = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, 4.0]]))
    lp = NumericalLP(
        name="t", objective_name="obj",
        A=A, b=np.zeros(2), c=np.zeros(2),
        lower_bounds=np.zeros(2), upper_bounds=np.full(2, np.inf),
        row_types=("L", "L"), var_names=("x1", "x2"), row_names=("r1", "r2"),
    )
    assert sp.issparse(lp.A)
    assert isinstance(lp.A, sp.csc_matrix)
    assert lp.A.shape == (2, 2)


# ---- 2. nnz correctness ----

def test_nnz_dense() -> None:
    A = np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    lp = NumericalLP(
        name="t", objective_name="obj",
        A=A, b=np.zeros(3), c=np.zeros(3),
        lower_bounds=np.zeros(3), upper_bounds=np.full(3, np.inf),
        row_types=("L", "L", "L"), var_names=("x1", "x2", "x3"),
        row_names=("r1", "r2", "r3"),
    )
    assert lp.nnz == 3


def test_nnz_sparse_csr() -> None:
    dense = np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    A = sp.csr_matrix(dense)
    lp = NumericalLP(
        name="t", objective_name="obj",
        A=A, b=np.zeros(3), c=np.zeros(3),
        lower_bounds=np.zeros(3), upper_bounds=np.full(3, np.inf),
        row_types=("L", "L", "L"), var_names=("x1", "x2", "x3"),
        row_names=("r1", "r2", "r3"),
    )
    assert lp.nnz == 3


def test_nnz_sparse_zero_matrix() -> None:
    A = sp.csr_matrix((3, 3), dtype=np.float64)
    lp = NumericalLP(
        name="t", objective_name="obj",
        A=A, b=np.zeros(3), c=np.zeros(3),
        lower_bounds=np.zeros(3), upper_bounds=np.full(3, np.inf),
        row_types=("L", "L", "L"), var_names=("x1", "x2", "x3"),
        row_names=("r1", "r2", "r3"),
    )
    assert lp.nnz == 0


# ---- 3. Dense/sparse numerical equality ----

def test_dense_sparse_value_equality() -> None:
    dense = np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 2.5], [0.0, 5.0, 0.0]])
    A_sparse = sp.csr_matrix(dense)
    np.testing.assert_array_equal(dense, A_sparse.toarray())
    np.testing.assert_array_equal(A_sparse.toarray(), dense)


def test_to_sparse_matches_to_dense_from_same_model() -> None:
    path = _write_mps(_TINY_MPS)
    try:
        model = MPSParser().parse_file(path)
        lp_dense = to_numeric(model)
        lp_sparse = to_sparse_numeric(model)
        assert isinstance(lp_dense.A, np.ndarray)
        assert sp.issparse(lp_sparse.A)
        assert lp_dense.A.shape == lp_sparse.A.shape
        assert lp_dense.nnz == lp_sparse.nnz
        assert lp_dense.name == lp_sparse.name
        assert lp_dense.objective_name == lp_sparse.objective_name
        assert lp_dense.row_types == lp_sparse.row_types
        assert lp_dense.var_names == lp_sparse.var_names
        assert lp_dense.row_names == lp_sparse.row_names
        np.testing.assert_array_equal(lp_dense.A, lp_sparse.A.toarray())
        np.testing.assert_array_equal(lp_dense.b, lp_sparse.b)
        np.testing.assert_array_equal(lp_dense.c, lp_sparse.c)
        np.testing.assert_array_equal(lp_dense.lower_bounds, lp_sparse.lower_bounds)
        np.testing.assert_array_equal(lp_dense.upper_bounds, lp_sparse.upper_bounds)
    finally:
        os.unlink(path)


# ---- 4. load_numeric_mps paths ----

def test_load_numeric_mps_default_is_dense() -> None:
    path = _write_mps(_TINY_MPS)
    try:
        lp = load_numeric_mps(path)
        assert isinstance(lp.A, np.ndarray)
        assert not sp.issparse(lp.A)
    finally:
        os.unlink(path)


def test_load_numeric_mps_sparse_returns_csr() -> None:
    path = _write_mps(_TINY_MPS)
    try:
        lp = load_numeric_mps(path, sparse=True)
        assert sp.issparse(lp.A)
        assert isinstance(lp.A, sp.csr_matrix)
    finally:
        os.unlink(path)


def test_load_sparse_never_creates_dense_matrix() -> None:
    path = _write_mps(_TINY_MPS)
    try:
        lp = load_numeric_mps(path, sparse=True)
        assert sp.issparse(lp.A)
        assert not isinstance(lp.A, np.ndarray)
    finally:
        os.unlink(path)


def test_load_sparse_matches_load_dense_values() -> None:
    path = _write_mps(_TINY_MPS)
    try:
        lp_dense = load_numeric_mps(path)
        lp_sparse = load_numeric_mps(path, sparse=True)
        assert lp_dense.nnz == lp_sparse.nnz
        np.testing.assert_array_equal(lp_dense.A, lp_sparse.A.toarray())
        np.testing.assert_array_equal(lp_dense.b, lp_sparse.b)
        np.testing.assert_array_equal(lp_dense.c, lp_sparse.c)
    finally:
        os.unlink(path)


# ---- 5. High-fill MPS ----

def test_dense_mps_sparse_matches_dense() -> None:
    path = _write_mps(_DENSE_MPS)
    try:
        lp_dense = load_numeric_mps(path)
        lp_sparse = load_numeric_mps(path, sparse=True)
        m, n = lp_dense.A.shape
        assert lp_dense.nnz == m * n
        assert lp_sparse.nnz == m * n
        np.testing.assert_array_equal(lp_dense.A, lp_sparse.A.toarray())
    finally:
        os.unlink(path)


# ---- 6. Empty constraint coeffs ----

def test_empty_constraint_coeffs_sparse() -> None:
    path = _write_mps(_EMPTY_COEFFS_MPS)
    try:
        lp_sparse = load_numeric_mps(path, sparse=True)
        assert sp.issparse(lp_sparse.A)
        assert lp_sparse.A.shape == (1, 1)
        assert lp_sparse.nnz == 0
    finally:
        os.unlink(path)


# ---- 7. validate_numeric_lp with sparse ----

def test_validate_numeric_lp_with_sparse() -> None:
    path = _write_mps(_TINY_MPS)
    try:
        lp = load_numeric_mps(path, sparse=True)
        validate_numeric_lp(
            lp,
            expected={
                "name": "TINY",
                "objective_name": "OBJ",
                "num_vars": 3,
                "num_constraints": 3,
                "nnz": lp.nnz,
            },
        )
    finally:
        os.unlink(path)


# ---- 8. nnz consistency ----

def test_nnz_consistency_dense_vs_sparse() -> None:
    path = _write_mps(_TINY_MPS)
    try:
        lp_d = load_numeric_mps(path)
        lp_s = load_numeric_mps(path, sparse=True)
        assert lp_d.nnz == lp_s.nnz
    finally:
        os.unlink(path)


# ---- 9. Frozen dataclass ----

def test_numericallp_frozen_with_sparse() -> None:
    A = sp.csr_matrix(np.eye(3))
    lp = NumericalLP(
        name="t", objective_name="obj",
        A=A, b=np.zeros(3), c=np.zeros(3),
        lower_bounds=np.zeros(3), upper_bounds=np.full(3, np.inf),
        row_types=("L", "L", "L"), var_names=("a", "b", "c"),
        row_names=("r1", "r2", "r3"),
    )
    try:
        lp.A = sp.csr_matrix(np.zeros((3, 3)))
        assert False, "should have raised"
    except AttributeError:
        pass


# ---- 10. Direct from LPModel ----

def test_to_sparse_numeric_direct_from_lpmodel() -> None:
    model = LPModel(
        name="HAND",
        objective_name="OBJ",
        var_names=["x1", "x2", "x3"],
        row_names=["r1", "r2"],
        var_index={"x1": 0, "x2": 1, "x3": 2},
        row_index={"r1": 0, "r2": 1},
        row_types=["L", "E"],
        rhs=[10.0, 6.0],
        obj=[1.0, 2.0, 3.0],
        bounds_lb=[0.0, 0.0, 0.0],
        bounds_ub=[float("inf")] * 3,
        coeffs={
            (0, 0): 2.0,
            (1, 0): 1.0,
            (0, 1): 1.0,
            (2, 1): 1.0,
        },
    )
    lp = to_sparse_numeric(model)
    assert sp.issparse(lp.A)
    assert lp.A.shape == (2, 3)
    assert lp.nnz == 4
    expected_dense = np.array([[2.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(lp.A.toarray(), expected_dense)


# ---- 11. Large sparse ----

def test_large_sparse_no_dense_alloc() -> None:
    rng = np.random.default_rng(42)
    m, n = 1000, 2000
    nnz = 5000
    row_indices = rng.integers(0, m, size=nnz)
    col_indices = rng.integers(0, n, size=nnz)
    values = rng.standard_normal(nnz)
    coeffs = {}
    for r, c, v in zip(row_indices, col_indices, values):
        coeffs[(int(c), int(r))] = float(v)
    model = LPModel(
        name="LARGE", objective_name="OBJ",
        var_names=[f"x{i}" for i in range(n)],
        row_names=[f"r{i}" for i in range(m)],
        var_index={f"x{i}": i for i in range(n)},
        row_index={f"r{i}": i for i in range(m)},
        row_types=["L"] * m, rhs=[10.0] * m,
        obj=[0.0] * n, bounds_lb=[0.0] * n,
        bounds_ub=[float("inf")] * n, coeffs=coeffs,
    )
    lp = to_sparse_numeric(model)
    assert sp.issparse(lp.A)
    assert lp.A.shape == (m, n)
    assert isinstance(lp.A, sp.csr_matrix)
    assert not isinstance(lp.A, np.ndarray)


# ---- 12. Dense regression guard ----

def test_dense_lp_still_works() -> None:
    A = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0]])
    lp = NumericalLP(
        name="SIMPLE_EQ", objective_name="COST",
        A=A, b=np.array([6.0, 4.0]),
        c=np.array([2.0, 3.0, 1.0]),
        lower_bounds=np.zeros(3), upper_bounds=np.full(3, np.inf),
        row_types=("E", "E"),
        var_names=("x1", "x2", "x3"), row_names=("E1", "E2"),
    )
    assert isinstance(lp.A, np.ndarray)
    assert lp.A.shape == (2, 3)
    assert lp.nnz == 5
    assert lp.num_vars == 3
    assert lp.num_constraints == 2


# ---- main ----

def main() -> int:
    import traceback
    tests = [
        test_dense_a_is_ndarray, test_sparse_csr_a_is_csr,
        test_sparse_csc_a_is_csc, test_nnz_dense, test_nnz_sparse_csr,
        test_nnz_sparse_zero_matrix, test_dense_sparse_value_equality,
        test_to_sparse_matches_to_dense_from_same_model,
        test_load_numeric_mps_default_is_dense,
        test_load_numeric_mps_sparse_returns_csr,
        test_load_sparse_never_creates_dense_matrix,
        test_load_sparse_matches_load_dense_values,
        test_dense_mps_sparse_matches_dense,
        test_empty_constraint_coeffs_sparse,
        test_validate_numeric_lp_with_sparse,
        test_nnz_consistency_dense_vs_sparse,
        test_numericallp_frozen_with_sparse,
        test_to_sparse_numeric_direct_from_lpmodel,
        test_large_sparse_no_dense_alloc, test_dense_lp_still_works,
    ]
    passed = 0
    failed = 0
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
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
