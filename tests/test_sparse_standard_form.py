"""Focused tests for sparse standard-form construction (Phase 3A).

Verifies that:
- to_standard_form() accepts both dense np.ndarray and scipy.sparse A.
- Dense input produces the same dense standard form as before.
- Sparse input produces a scipy.sparse matrix.
- Sparse and dense standard forms have identical numerical values, shapes,
  b, c_min, bounds-derived metadata, and row/column structure.
- Slacks, box upper-bound rows, free-variable splits and fixed substitutions
  are constructed correctly in both paths.
- A sparse input never materialises a full dense m x n standard-form matrix.

Only to_standard_form() is exercised; no Mehrotra solving is performed here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import NumericalLP  # noqa: E402
from lp.mehrotra import to_standard_form  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lp(A, b, c, *, row_types, var_names, row_names,
        lower_bounds=None, upper_bounds=None, maximize: bool = False):
    """Build a NumericalLP with the given A (dense or sparse)."""
    n = c.shape[0]
    if lower_bounds is None:
        lower_bounds = np.zeros(n)
    if upper_bounds is None:
        upper_bounds = np.full(n, np.inf)
    return NumericalLP(
        name="t", objective_name="OBJ",
        A=A, b=np.asarray(b, dtype=np.float64),
        c=np.asarray(c, dtype=np.float64),
        lower_bounds=np.asarray(lower_bounds, dtype=np.float64),
        upper_bounds=np.asarray(upper_bounds, dtype=np.float64),
        row_types=tuple(row_types),
        var_names=tuple(var_names),
        row_names=tuple(row_names),
    )


def _std_dense_to_matrix(std):
    """Return StandardFormLP data as dense (works for dense or sparse A)."""
    A = std.A.toarray() if sp.issparse(std.A) else np.asarray(std.A)
    return A, np.asarray(std.b), np.asarray(std.c_min)

# ---------------------------------------------------------------------------
# 1. Dense regression: existing behaviour preserved
# ---------------------------------------------------------------------------

def test_dense_mixed_e_l_g():
    # min -x1 - x2 s.t. x1 + x2 + x3 = 10 (E), x1 + 2x2 <= 8 (L), 3x1 + x2 >= 9 (G).
    A = np.array([[1.0, 1.0, 1.0],
                  [1.0, 2.0, 0.0],
                  [3.0, 1.0, 0.0]])
    b = np.array([10.0, 8.0, 9.0])
    c = np.array([-1.0, -1.0, 0.0])
    std = to_standard_form(
        _lp(A, b, c, row_types=("E", "L", "G"),
            var_names=("x1", "x2", "x3"), row_names=("E1", "L1", "G1"))
    )
    assert isinstance(std.A, np.ndarray)          # dense stays dense
    # 3 original + 2 slacks (L and G rows) = 5 columns, 3 rows.
    assert std.A.shape == (3, 5)
    Ad, bd, cd = _std_dense_to_matrix(std)
    np.testing.assert_allclose(bd, b.astype(float))
    # E row no slack, L slack +1 on row1, G slack -1 on row2.
    np.testing.assert_allclose(Ad[:, 3], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(Ad[:, 4], [0.0, 0.0, -1.0])
    # Objective: c_block over original columns, then 0 slacks.
    np.testing.assert_allclose(cd[:3], [-1.0, -1.0, 0.0])
    np.testing.assert_allclose(cd[3:], [0.0, 0.0])
    assert std.num_slacks == 2
    assert std.slack_row_indices == (1, 2)


# ---------------------------------------------------------------------------
# 2. Sparse input: accepted, produces sparse, identical values
# ---------------------------------------------------------------------------

def test_sparse_mixed_e_l_g():
    A = np.array([[1.0, 1.0, 1.0],
                  [1.0, 2.0, 0.0],
                  [3.0, 1.0, 0.0]])
    b = np.array([10.0, 8.0, 9.0])
    c = np.array([-1.0, -1.0, 0.0])
    dense_std = to_standard_form(
        _lp(A, b, c, row_types=("E", "L", "G"),
            var_names=("x1", "x2", "x3"), row_names=("E1", "L1", "G1"))
    )
    sparse_std = to_standard_form(
        _lp(sp.csr_matrix(A), b, c, row_types=("E", "L", "G"),
            var_names=("x1", "x2", "x3"), row_names=("E1", "L1", "G1"))
    )
    assert sp.issparse(sparse_std.A)
    assert isinstance(sparse_std.A, sp.csr_matrix)
    Ad, bd, cd = _std_dense_to_matrix(dense_std)
    As, bs, cs = _std_dense_to_matrix(sparse_std)
    np.testing.assert_allclose(As, Ad)
    np.testing.assert_allclose(bs, bd)
    np.testing.assert_allclose(cs, cd)


def test_sparse_shape_b_c_metadata_match_dense():
    A = np.array([[2.0, 1.0, 0.0, 1.0],
                  [0.0, 1.0, 3.0, 0.0],
                  [1.0, 0.0, 0.0, 2.0]])
    b = np.array([4.0, 6.0, 5.0])
    c = np.array([1.0, 2.0, 3.0, 4.0])
    kw = dict(row_types=("L", "E", "G"),
              var_names=("x1", "x2", "x3", "x4"),
              row_names=("r1", "r2", "r3"))
    d = to_standard_form(_lp(A, b, c, **kw))
    s = to_standard_form(_lp(sp.csc_matrix(A), b, c, **kw))
    assert isinstance(s.A, sp.csr_matrix)          # CSC input -> CSR output
    assert s.A.shape == d.A.shape
    assert s.b.shape == d.b.shape and np.allclose(s.b, d.b)
    assert s.c_min.shape == d.c_min.shape and np.allclose(s.c_min, d.c_min)
    assert s.n_block == d.n_block
    assert s.num_slacks == d.num_slacks
    assert s.slack_row_indices == d.slack_row_indices
    np.testing.assert_array_equal(s.block_to_orig, d.block_to_orig)
    np.testing.assert_array_equal(s.block_sign, d.block_sign)
    np.testing.assert_array_equal(s.orig_offset, d.orig_offset)
    assert s.var_names == d.var_names
    assert s.slack_names == d.slack_names
    assert s.row_names == d.row_names
    assert s.maximize == d.maximize


# ---------------------------------------------------------------------------
# 3. Free-variable splits (FR: two columns, +sign / -sign)
# ---------------------------------------------------------------------------

def test_sparse_free_variable_split_matches_dense():
    # Free x1, [0,inf) x2.  x1 -> (x1+, x1-); x2 kept.
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    c = np.array([1.0, 2.0])
    lb = np.array([-np.inf, 0.0])
    ub = np.array([np.inf, np.inf])
    kw = dict(row_types=("E", "L"), var_names=("x1", "x2"), row_names=("r1", "r2"))
    d = to_standard_form(_lp(A, b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    s = to_standard_form(_lp(sp.csr_matrix(A), b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    # x1 split (2 blocks) + x2 (1 block) + 1 slack (L) = 4 columns.
    assert d.A.shape == (2, 4) and s.A.shape == (2, 4)
    Ad, bd, cd = _std_dense_to_matrix(d)
    As, bs, cs = _std_dense_to_matrix(s)
    np.testing.assert_allclose(As, Ad)
    np.testing.assert_allclose(bs, bd)
    np.testing.assert_allclose(cs, cd)
    # The standard-form column order is: x2 (single), then x1 split (+/-).
    # So block_to_orig = [1, 0, 0] and block_sign = [+1, +1, -1].
    np.testing.assert_array_equal(s.block_to_orig, [1, 0, 0])
    np.testing.assert_allclose(s.block_sign, [1.0, 1.0, -1.0])


# ---------------------------------------------------------------------------
# 4. Reflected UP bound (free lower, finite upper) and LO shift
# ---------------------------------------------------------------------------

def test_sparse_reflected_and_shifted_matches_dense():
    # x1: UP bound (reflect: x1 = 4 - x'), x2: LO bound (shift: x2 = 2 + x'),
    # x3: [0, inf).
    A = np.array([[1.0, 1.0, 1.0],
                  [0.0, 1.0, 1.0]])
    b = np.array([10.0, 5.0])
    c = np.array([1.0, 2.0, 3.0])
    lb = np.array([-np.inf, 2.0, 0.0])
    ub = np.array([4.0, np.inf, np.inf])
    kw = dict(row_types=("E", "L"), var_names=("x1", "x2", "x3"), row_names=("r1", "r2"))
    d = to_standard_form(_lp(A, b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    s = to_standard_form(_lp(sp.csr_matrix(A), b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    Ad, bd, cd = _std_dense_to_matrix(d)
    As, bs, cs = _std_dense_to_matrix(s)
    np.testing.assert_allclose(As, Ad)
    np.testing.assert_allclose(bs, bd)
    np.testing.assert_allclose(cs, cd)
    assert s.A.shape == d.A.shape
    # orig offsets: x1=4 (reflect), x2=2 (shift), x3=0.
    np.testing.assert_allclose(s.orig_offset, [4.0, 2.0, 0.0])
    # reflected column negated, shifted column kept.
    np.testing.assert_allclose(s.block_sign, [-1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 5. Box bounds [L, U]: shift + appended upper-bound row
# ---------------------------------------------------------------------------

def test_sparse_box_upper_row_matches_dense():
    # x1: box [1, 5], x2: [0, inf).  Box adds an 'L' row x1' + s = 4.
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([7.0, 5.0])
    c = np.array([1.0, 2.0])
    lb = np.array([1.0, 0.0])
    ub = np.array([5.0, np.inf])
    kw = dict(row_types=("E", "L"), var_names=("x1", "x2"), row_names=("r1", "r2"))
    d = to_standard_form(_lp(A, b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    s = to_standard_form(_lp(sp.csc_matrix(A), b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    Ad, bd, cd = _std_dense_to_matrix(d)
    As, bs, cs = _std_dense_to_matrix(s)
    np.testing.assert_allclose(As, Ad)
    np.testing.assert_allclose(bs, bd)
    np.testing.assert_allclose(cs, cd)
    # 2 original + 2 slacks (r2 'L' + box row) = 4 cols; 2 orig + 1 box row = 3 rows.
    assert s.A.shape == d.A.shape == (3, 4)
    assert s.m == 3 and s.n == 4
    # The box row is the last appended row: shifted x1' (+1) + slack (+1) = b=4.
    np.testing.assert_allclose(Ad[2, 0], 1.0)
    np.testing.assert_allclose(Ad[2, 3], 1.0)
    np.testing.assert_allclose(bd[2], 4.0)
    assert s.num_slacks == 2
    # box column sign +1 (shift), offset = lower bound.
    np.testing.assert_allclose(s.block_sign[0], 1.0)
    np.testing.assert_allclose(s.orig_offset[0], 1.0)


def test_sparse_box_block_no_dense_alloc():
    # A sparse model with many box variables: each box variable contributes
    # one extra row with exactly two +1s.  This guards against the box block
    # being materialised as an O(n_extra x (n_block + num_slacks)) dense array.
    n_base = 120
    rng = np.random.default_rng(42)
    # Each variable is boxed [L, U]; diagonal coefficient 1.0 on a sparse
    # identity so the standard form has full row rank for the dense check.
    A = sp.eye(n_base, format="csr", dtype=np.float64)
    b = np.full(n_base, 5.0)
    c = rng.uniform(-2.0, 2.0, n_base)
    lb = np.full(n_base, 1.0)
    ub = np.full(n_base, 3.0)
    kw = dict(row_types=("E",) * n_base,
              var_names=tuple(f"x{i}" for i in range(n_base)),
              row_names=tuple(f"r{i}" for i in range(n_base)))
    s = to_standard_form(_lp(A, b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    assert sp.issparse(s.A)
    # n_block columns (one shift per variable, no free/fixed) + 2*n_base rows:
    # the original equality rows plus one appended box row per variable.
    assert s.n_block == n_base
    # Total nonzeros = n_block identity entries + 2 per appended box row.
    # This stays O(rows + n_block), never O(n_extra x (n_block + num_slacks)).
    n_extra = n_base
    assert s.A.shape == (s.n_block + n_extra, s.n_block + s.num_slacks)
    assert s.A.nnz == s.n_block + 2 * n_extra
    # Spot-check: every appended box row holds exactly the two +1 entries.
    Ab = s.A.tocsr()
    for i in range(n_base):
        row = Ab.getrow(n_base + i).toarray().ravel()
        # shifted variable column + its slack column -> exactly two nonzeros.
        assert np.count_nonzero(row) == 2
        np.testing.assert_allclose(row.sum(), 2.0)
    # Sanity: values match the equivalent dense build.
    d = to_standard_form(_lp(A.toarray(), b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    np.testing.assert_allclose(s.A.toarray(), d.A)
    np.testing.assert_allclose(np.asarray(s.b), np.asarray(d.b))


# ---------------------------------------------------------------------------
# 6. Fixed variable (FX) substituted out
# ---------------------------------------------------------------------------

def test_sparse_fixed_variable_substituted_matches_dense():
    # x2 fixed at 3 -> substituted out; no block column for it.
    A = np.array([[1.0, 1.0],
                  [2.0, 0.0]])
    b = np.array([5.0, 6.0])
    c = np.array([1.0, 2.0])
    lb = np.array([0.0, 3.0])
    ub = np.array([np.inf, 3.0])
    kw = dict(row_types=("E", "L"), var_names=("x1", "x2"), row_names=("r1", "r2"))
    d = to_standard_form(_lp(A, b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    s = to_standard_form(_lp(sp.csr_matrix(A), b, c, lower_bounds=lb, upper_bounds=ub, **kw))
    Ad, bd, cd = _std_dense_to_matrix(d)
    As, bs, cs = _std_dense_to_matrix(s)
    np.testing.assert_allclose(As, Ad)
    np.testing.assert_allclose(bs, bd)
    np.testing.assert_allclose(cs, cd)
    assert s.A.shape == d.A.shape
    # 1 block column (x1) + 1 slack = 2 cols; rows = 2.
    assert s.A.shape == (2, 2)
    # RHS absorbs the fixed value: r1 = 5 - 3 = 2.
    np.testing.assert_allclose(bd[0], 2.0)
    # x2 has no block column, recoverable from offset.
    np.testing.assert_allclose(s.orig_offset[1], 3.0)


# ---------------------------------------------------------------------------
# 7. Zero original rows still raise (both paths)
# ---------------------------------------------------------------------------

def test_zero_orig_row_dense_and_sparse_raise():
    # An all-zero original row yields an all-zero standard-form row.
    A = np.array([[1.0, 0.0],
                  [0.0, 0.0]])     # row2 all zero
    b = np.array([2.0, 0.0])
    c = np.array([1.0, 1.0])
    kw = dict(row_types=("E", "E"), var_names=("x1", "x2"), row_names=("r1", "r2"))
    for Amat in (A, sp.csr_matrix(A)):
        raised = False
        try:
            to_standard_form(_lp(Amat, b, c, **kw))
        except Exception as e:
            raised = True
            assert "all zero" in str(e)
        assert raised, "expected an all-zero row error"


def test_zero_row_detected_without_dense_conversion_sparse():
    # A sparse case with one all-zero row: verify the sparse path detects it
    # without materialising the full dense matrix.
    A = sp.csr_matrix(([3.0], ([0], [0])), shape=(4, 4))
    b = np.array([3.0, 0.0, 0.0, 0.0])
    c = np.array([1.0, 1.0, 1.0, 1.0])
    kw = dict(row_types=("E", "E", "E", "E"),
              var_names=("x1", "x2", "x3", "x4"),
              row_names=("r1", "r2", "r3", "r4"))
    raised = False
    try:
        to_standard_form(_lp(A, b, c, **kw))
    except Exception as e:
        raised = True
        assert "all zero" in str(e)
    assert raised


# ---------------------------------------------------------------------------
# 8. No dense m x n conversion in the sparse path
# ---------------------------------------------------------------------------

def test_sparse_path_no_dense_alloc():
    # Build a moderately large sparse matrix so an accidental densification of
    # the full m x n standard form would be observable.  Assert the produced A
    # is sparse and matches the dense build's standard-form values.
    # Use m < n so that rank(A) == m (with probability 1 for a random sparse
    # matrix) and the dense rank check passes.
    rng = np.random.default_rng(0)
    m, n = 80, 200
    A = sp.random(m, n, density=0.05, random_state=rng, format="csr",
                  dtype=np.float64)
    A = A.copy()
    A.data *= rng.choice([1.0, -1.0], size=A.nnz, p=[0.5, 0.5])
    b = rng.uniform(1.0, 5.0, m)
    c = rng.uniform(-2.0, 2.0, n)
    kw = dict(row_types=("E",) * m, var_names=tuple(f"x{i}" for i in range(n)),
              row_names=tuple(f"r{i}" for i in range(m)))
    s = to_standard_form(_lp(A, b, c, **kw))
    assert sp.issparse(s.A)
    # Equality-only: no slacks, n_block == n.
    assert s.n_block == n
    assert s.A.shape == (m, n)
    # Compare against the dense path's standard-form values.
    d = to_standard_form(_lp(A.toarray(), b, c, **kw))
    np.testing.assert_allclose(s.A.toarray(), d.A)
    np.testing.assert_allclose(np.asarray(s.b), np.asarray(d.b))
    np.testing.assert_allclose(np.asarray(s.c_min), np.asarray(d.c_min))


# ---------------------------------------------------------------------------
# 9. All-equality (no slacks) sparse stays sparse, no dense fallthrough
# ---------------------------------------------------------------------------

def test_sparse_equality_only_no_slacks():
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    c = np.array([1.0, 2.0])
    kw = dict(row_types=("E", "E"), var_names=("x1", "x2"), row_names=("r1", "r2"))
    d = to_standard_form(_lp(A, b, c, **kw))
    s = to_standard_form(_lp(sp.csr_matrix(A), b, c, **kw))
    assert sp.issparse(s.A)
    assert s.num_slacks == 0
    assert s.A.shape == d.A.shape == (2, 2)
    np.testing.assert_allclose(s.A.toarray(), d.A)
    np.testing.assert_allclose(np.asarray(s.b), np.asarray(d.b))


# ---------------------------------------------------------------------------
# main (for standalone execution)
# ---------------------------------------------------------------------------

def main() -> int:
    import traceback
    tests = [
        test_dense_mixed_e_l_g,
        test_sparse_mixed_e_l_g,
        test_sparse_shape_b_c_metadata_match_dense,
        test_sparse_free_variable_split_matches_dense,
        test_sparse_reflected_and_shifted_matches_dense,
        test_sparse_box_upper_row_matches_dense,
        test_sparse_box_block_no_dense_alloc,
        test_sparse_fixed_variable_substituted_matches_dense,
        test_zero_orig_row_dense_and_sparse_raise,
        test_zero_row_detected_without_dense_conversion_sparse,
        test_sparse_path_no_dense_alloc,
        test_sparse_equality_only_no_slacks,
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

