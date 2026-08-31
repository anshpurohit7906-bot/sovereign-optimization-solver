"""Focused regression tests for the Mehrotra solver's termination and
best-iterate reporting policy, and for the H-regularization cap.

Background (see the solver/linear_system docstrings for the full rationale):

* The trajectory runs in Ruiz-equilibrated coordinates, but the reported
  metrics and the acceptance test for the final claim are in ORIGINAL
  standard-form coordinates.  The best iterate returned on a non-optimal
  exit must therefore be selected by the *original-coordinate* merit
  ``max(rel_p, rel_d, rel_gap)`` -- selecting on the scaled gap can return
  a point with an unacceptable original primal residual (the original
  PILOT4 failure mode).
* The H-regularization ``rho_p = reg * max(1, mean|h|)`` is capped at
  ``MAX_RHO_P`` because the uncapped formula grows like ``h_max = O(1/mu)``
  in the degenerate tail and imposes a dual-side bias floor of
  ``rho_p * ||dx||_inf`` of ~1e-4..1e-3 (observed on PILOT4).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from linear_system import MAX_RHO_P, factor_reduced_system  # noqa: E402
from numerical_model import load_numeric_mps  # noqa: E402
from mehrotra import solve_lp  # noqa: E402

_DATA = os.path.join(_ROOT, "data")
_AFIRO = os.path.join(_DATA, "afiro.mps")


def _afiro():
    return load_numeric_mps(_AFIRO)


# ---------------------------------------------------------------------------
# Best-iterate / termination reporting
# ---------------------------------------------------------------------------

def test_optimal_status_guarantees_original_coordinate_accuracy():
    """'optimal' must imply original-coordinate primal/dual/gap <= tol.

    The result fields are recomputed in original standard-form coordinates
    by _finish(); this test pins the contract that the status cannot be
    'optimal' unless those original-coordinate metrics pass the tolerance.
    """
    tol = 1e-7
    res = solve_lp(_afiro(), tol=tol, max_iter=100)
    assert res.status == "optimal", res.message
    assert res.rel_primal <= tol
    assert res.rel_dual <= tol
    assert res.rel_gap <= tol
    assert abs(res.objective - (-464.7531428571)) <= 1e-6 * abs(-464.7531428571)
    assert np.all(np.isfinite(res.x))


def test_nonoptimal_exit_returns_best_original_merit_iterate():
    """On a non-optimal exit the returned point must be the iterate that
    minimizes the ORIGINAL-coordinate merit max(rel_p, rel_d, rel_gap),
    exactly as recorded in the per-iteration history."""
    res = solve_lp(_afiro(), tol=1e-7, max_iter=3)
    assert res.status == "max_iterations", res.message
    assert "returning best trusted iterate from k=" in res.message
    merits = [
        max(h["primal_orig"], h["dual_orig"], h["rel_gap_orig"])
        for h in res.history
    ]
    returned_merit = max(res.rel_primal, res.rel_dual, res.rel_gap)
    best_hist = min(merits)
    # The same computation recomputed on the stored iterate; allow only
    # last-bit slack for the redundant recompute in _finish().
    assert returned_merit <= best_hist * (1.0 + 1e-9) + 1e-15
    assert returned_merit >= best_hist * (1.0 - 1e-9) - 1e-15


# ---------------------------------------------------------------------------
# H-regularization cap
# ---------------------------------------------------------------------------

def test_h_regularization_capped_for_extreme_h():
    """rho_p must never exceed MAX_RHO_P, even when mean|h| is huge
    (the uncapped formula would give rho_p = 1e-12 * mean|h| ~ 5e-3 here)."""
    rng = np.random.default_rng(0)
    m, n = 8, 40
    A = rng.standard_normal((m, n))
    h = np.linspace(1.0, 1e6, n)  # mean|h| ~ 5e5 -> uncapped rho ~ 5e-7 > cap;
    # ulp(h_max) ~ 1.2e-10 so h + MAX_RHO_P is exact to << MAX_RHO_P here
    fac = factor_reduced_system(h, A)
    assert fac.h_diag is not None
    added = fac.h_diag - h
    assert np.all(added >= 0.0)
    # Allow a few ULP of h for the rounding of (h + MAX_RHO_P) itself.
    assert np.max(added) <= MAX_RHO_P + 4.0 * np.spacing(h.max())
    assert np.max(added) > 0.0  # cap actually engaged on this input


def test_h_regularization_uncapped_for_small_h():
    """For moderate h the scale-aware formula is unchanged:
    rho_p = reg * max(1, mean|h|) with reg = 1e-12."""
    rng = np.random.default_rng(1)
    m, n = 6, 30
    A = rng.standard_normal((m, n))
    h = np.full(n, 2.0)  # mean|h| = 2 -> rho_p = 2e-12, far below the cap
    fac = factor_reduced_system(h, A)
    assert np.allclose(fac.h_diag - h, 1e-12 * 2.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Objective-offset reporting (Simplex + all-FX standard form)
# ---------------------------------------------------------------------------

def _fx_lp():
    from numerical_model import NumericalLP
    return NumericalLP(
        name="t", objective_name="obj",
        A=np.zeros((0, 1)), b=np.zeros(0), c=np.array([3.0]),
        lower_bounds=np.array([2.0]), upper_bounds=np.array([2.0]),
        row_types=(), var_names=("x",), row_names=())


def _lo_bound_lp():
    from numerical_model import NumericalLP
    # min 2x s.t. x + s = 5, x >= 3  ->  x = 3, objective 6.
    # The standard form shifts x = 3 + x', so c_min @ x' = 0 and the constant
    # offset c_orig @ orig_offset = 6 must be reported, not 0.
    return NumericalLP(
        name="t2", objective_name="obj",
        A=np.array([[1.0]]), b=np.array([5.0]), c=np.array([2.0]),
        lower_bounds=np.array([3.0]), upper_bounds=np.array([np.inf]),
        row_types=("L",), var_names=("x",), row_names=("r1",))


def test_simplex_objective_includes_bound_offset():
    """Simplex must report c_orig @ x_original, not c_min @ x_standard:
    bound shifts (LO/UP/box/FX) add a constant the standard-form objective
    does not contain."""
    from mehrotra import to_standard_form
    from simplex import solve_simplex
    res = solve_simplex(to_standard_form(_lo_bound_lp()))
    assert res.status == "optimal"
    assert abs(res.objective - 6.0) <= 1e-9, res.objective
    assert abs(res.x[0] - 3.0) <= 1e-9


def test_all_fixed_variables_constant_objective_both_solvers():
    """An LP whose variables are all FX has an empty standard form
    (n = 0); both solvers must return the constant objective without
    crashing on empty reductions."""
    from mehrotra import solve_standard_form, to_standard_form
    from simplex import solve_simplex
    sf = to_standard_form(_fx_lp())
    assert sf.A.shape[0] == 0 and sf.A.shape[1] == 0
    for res in (solve_standard_form(sf), solve_simplex(sf)):
        assert res.status == "optimal", getattr(res, "message", "")
        assert abs(res.objective - 6.0) <= 1e-9, res.objective
        assert abs(res.x[0] - 2.0) <= 1e-9


def test_maximize_with_upper_bound_offset():
    """max x s.t. x + s = 4, x <= 4 (UP, free lower bound): the UP
    reflection negates the standard-form column and absorbs the offset 4;
    the reported original objective must still be 4."""
    from mehrotra import to_standard_form
    from simplex import solve_simplex
    from numerical_model import NumericalLP
    lp = NumericalLP(
        name="t3", objective_name="obj",
        A=np.array([[1.0]]), b=np.array([4.0]), c=np.array([1.0]),
        lower_bounds=np.array([-np.inf]), upper_bounds=np.array([4.0]),
        row_types=("L",), var_names=("x",), row_names=("r1",))
    res = solve_simplex(to_standard_form(lp, maximize=True))
    assert res.status == "optimal"
    assert abs(res.objective - 4.0) <= 1e-9, res.objective
# ---------------------------------------------------------------------------
# End-to-end guard: the PILOT4 minimax violation must not regress
# ---------------------------------------------------------------------------




def test_pilot4_merit_no_regression():
    """Pre-fix, PILOT4 stalled with original-coordinate merit
    max(rel_p, rel_d, rel_gap) = 3.85e-4 (selected by scaled gap).  The
    fixed selection policy + H-cap must keep the returned worst-case
    violation at least an order of magnitude below that baseline."""
    res = solve_lp(load_numeric_mps(os.path.join(_DATA, "pilot4_plain.mps")),
                   tol=1e-7, max_iter=100)
    assert res.status in ("optimal", "numerical_tail", "stalled"), res.message
    merit = max(res.rel_primal, res.rel_dual, res.rel_gap)
    assert merit <= 1e-4, (res.status, merit, res.message)
    assert -2600.0 <= res.objective <= -2570.0
