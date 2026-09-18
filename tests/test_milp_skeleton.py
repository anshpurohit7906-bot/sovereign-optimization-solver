"""MILP branch-and-bound skeleton tests.

These are the M1 gate for the MILP vertical slice: a from-scratch B&B loop that
uses the production LP core for relaxations and proves the optimal solution on
hand-verifiable integer problems.  No external solver is used.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import NumericalLP
from lp.branch_bound import solve_milp, MilpError


def _mk_lp(name, A, b, c, row_types, lb, ub, maximize=False):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    return NumericalLP(
        name=name,
        objective_name="OBJ",
        A=A,
        b=b,
        c=c,
        lower_bounds=np.asarray(lb, dtype=np.float64),
        upper_bounds=np.asarray(ub, dtype=np.float64),
        row_types=tuple(row_types),
        var_names=tuple(f"x{i}" for i in range(c.shape[0])),
        row_names=tuple(f"r{i}" for i in range(b.shape[0])),
    )


def test_knapsack_01_proven_optimal():
    """max 4x0 + 5x1 + 3x2 + 7x3 s.t. 3x0+4x1+2x2+6x3 <= 8, x in {0,1}^4.

    Optimal: choose x2 (3) and x3 (7) -> 10 (cost 2+6=8).  Any triple
    (x0,x1,x3) costs 13 > 8.
    """
    A = np.array([[3.0, 4.0, 2.0, 6.0]])
    b = np.array([8.0])
    c = np.array([4.0, 5.0, 3.0, 7.0])
    lp = _mk_lp(
        "KNAPSACK_01", A, b, c, ("L",),
        np.zeros(4), np.ones(4),
    )
    res = solve_milp(lp, integer_mask=np.ones(4, dtype=bool), maximize=True,
                     gap_tol=1e-6)
    assert res.status == "optimal", (res.status, res.message)
    assert abs(res.objective - 10.0) <= 1e-6, res.objective
    assert abs(res.x[2] - 1.0) <= 1e-5 and abs(res.x[3] - 1.0) <= 1e-5
    assert abs(res.x[0]) <= 1e-5 and abs(res.x[1]) <= 1e-5
    assert res.best_bound is not None and res.best_bound >= res.objective - 1e-6
    assert res.gap is not None and res.gap <= 1e-6


def test_assignment_2x2():
    """min 2x00+3x01+1x10+2x11 s.t. row/col sums = 1, binary.

    Optimal: x01=1, x10=1 -> objective 4.  (Choosing x11=1, x00=1 -> 5.)
    """
    A = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    b = np.ones(4)
    c = np.array([2.0, 3.0, 1.0, 2.0])
    lp = _mk_lp(
        "ASSIGN_2x2", A, b, c, ("E", "E", "E", "E"),
        np.zeros(4), np.ones(4),
    )
    res = solve_milp(lp, integer_mask=np.ones(4, dtype=bool), gap_tol=1e-6)
    assert res.status == "optimal", (res.status, res.message)
    assert abs(res.objective - 4.0) <= 1e-6, res.objective


def test_mixed_integer_bound_example():
    """min  x1 + 2x2   (x1 continuous, x2 binary)
        s.t. x1 + x2 >= 1, 0 <= x1 <= 3.

    With x2 in {0,1}: if x2=0 then x1>=1 -> obj 1; if x2=1 then x1=0 -> obj 2.
    Optimal = 1 with x1=1, x2=0.
    """
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    c = np.array([1.0, 2.0])
    lp = _mk_lp(
        "MIXED_INT", A, b, c, ("G",),
        np.array([0.0, 0.0]), np.array([3.0, 1.0]),
    )
    res = solve_milp(lp, integer_mask=np.array([False, True]), gap_tol=1e-6)
    assert res.status == "optimal", (res.status, res.message)
    assert abs(res.objective - 1.0) <= 1e-6, res.objective
    assert abs(res.x[0] - 1.0) <= 1e-5 and abs(res.x[1] - 0.0) <= 1e-5


def test_infeasible_integer_problem():
    """x binary,  x >= 2 impossible -> infeasible."""
    A = np.array([[1.0]])
    b = np.array([2.0])
    c = np.array([1.0])
    lp = _mk_lp(
        "INFEAS_INT", A, b, c, ("G",),
        np.zeros(1), np.ones(1),
    )
    res = solve_milp(lp, integer_mask=np.ones(1, dtype=bool))
    assert res.status == "infeasible", (res.status, res.message)


def test_requires_integer_mask():
    lp = _mk_lp(
        "NOMASK", [[1.0]], [1.0], [1.0], ("G",),
        np.zeros(1), np.ones(1),
    )
    try:
        solve_milp(lp)
    except MilpError:
        pass
    else:
        raise AssertionError("expected MilpError without integer_mask")


def test_unbounded_integer_lp_reported_honestly():
    """integer var with infinite upper bound -> rejected loudly (v1 scope)."""
    lp = _mk_lp(
        "FREE_INT", [[1.0]], [1.0], [1.0], ("G",),
        np.zeros(1), np.array([np.inf]),
    )
    try:
        solve_milp(lp, integer_mask=np.ones(1, dtype=bool))
    except MilpError:
        pass
    else:
        raise AssertionError("expected MilpError for unbounded integer variable")


def test_rounding_heuristic_accepts_only_feasible_points():
    """Regression: the v2 rounding heuristic must never return an infeasible
    point as feasible (the greedy one-flip path once lost the improving flip
    and reported x=[1,1,1,0] for a knapsack whose only row has RHS 8)."""
    from lp.branch_bound import _rounding_heuristic

    A = np.array([[3.0, 4.0, 2.0, 6.0]])
    b = np.array([8.0])
    c = np.array([-4.0, -5.0, -3.0, -7.0])   # min-form of the max knapsack
    x_lp = np.array([1.0, 0.75, 1.0, 0.0])
    feas, x_heur, obj = _rounding_heuristic(
        x_lp,
        np.ones(4, dtype=bool),
        np.zeros(4),
        np.ones(4),
        A, b, ("L",), c,
    )
    assert feas, "heuristic should find a feasible rounding on this knapsack"
    activity = float(np.asarray(A @ x_heur).flat[0])
    assert activity <= 8.0 + 1e-5, (
        f"heuristic returned infeasible point: A@x={activity} > 8"
    )


def _mk_mixed_integer_with_continuous():
    """min -(4x0 + 5x1 + 3x2 + x3)  s.t.  3x0 + 4x1 + 2x2 + x3 <= 8

    x0, x1, x2 binary;  x3 continuous on [0, 4].

    * Root LP relaxation optimum is fractional: x = (1, 0.75, 1, 0),
      objective = -10.75 (x1 = 3/4).
    * The nearest-rounded point (1, 1, 1, 0) is INFEASIBLE (activity 9 > 8),
      so a pure rounding heuristic gets stuck.
    * Fixing x1 = 0 and re-solving the continuous LP pushes the free variable
      x3 up to 3, giving the integer-feasible point (1, 0, 1, 3) with
      objective -10.
    * The true MIP optimum is -10 (attained, e.g., by (1, 1, 0, 1)).
    """
    A = np.array([[3.0, 4.0, 2.0, 1.0]])
    b = np.array([8.0])
    c = np.array([-4.0, -5.0, -3.0, -1.0])
    lb = np.zeros(4)
    ub = np.array([1.0, 1.0, 1.0, 4.0])
    mask = np.array([True, True, True, False])
    lp = NumericalLP(
        name="mixed_continuous",
        objective_name="OBJ",
        A=A,
        b=b,
        c=c,
        lower_bounds=lb,
        upper_bounds=ub,
        row_types=("L",),
        var_names=("x0", "x1", "x2", "x3"),
        row_names=("r0",),
    )
    return lp, mask


def test_integer_fixing_repair_reoptimizes_continuous_vars():
    """Fractional LP solution -> fix integers -> re-solve continuous LP.

    The root relaxation of the mixed model has x1 = 0.75.  Fixing the
    integer variables to the repaired assignment (1, 0, 1) and re-solving
    the continuous LP must push the free variable x3 to 3 and return a
    feasible integer-compatible point with objective -10.
    """
    from lp.branch_bound import _rounding_heuristic
    from lp.mehrotra import solve_lp

    lp, mask = _mk_mixed_integer_with_continuous()
    rel = solve_lp(lp)
    assert rel.status == "optimal"
    x_lp = np.asarray(rel.x, dtype=np.float64)
    # Sanity: the relaxation really is fractional where it matters.
    assert abs(x_lp[0] - 1.0) < 1e-4
    assert abs(x_lp[1] - 0.75) < 1e-4
    assert abs(x_lp[2] - 1.0) < 1e-4
    assert abs(rel.objective - (-10.75)) < 1e-3

    feas, x_heur, obj = _rounding_heuristic(
        x_lp, mask,
        np.asarray(lp.lower_bounds), np.asarray(lp.upper_bounds),
        np.asarray(lp.A, dtype=np.float64), np.asarray(lp.b, dtype=np.float64),
        lp.row_types, np.asarray(lp.c, dtype=np.float64), lp=lp,
    )
    assert feas, "repair must produce a feasible integer-compatible point"
    # Integer variables fixed to integral values.
    for i in np.flatnonzero(mask):
        assert abs(x_heur[i] - np.round(x_heur[i])) < 1e-6, x_heur
    # Continuous variable re-optimized: x3 = 3 (not the LP value 0).
    assert abs(x_heur[3] - 3.0) < 1e-3, x_heur
    activity = float(np.asarray(lp.A @ x_heur).flat[0])
    assert activity <= 8.0 + 1e-5, f"A@x = {activity} exceeds RHS"
    assert abs(obj - (-10.0)) < 1e-6, f"objective {obj} != -10"


def test_infeasible_fixed_integer_assignment_rejected():
    """A fixed-integer assignment whose continuous LP is infeasible is
    rejected (and never crashes the solver path)."""
    from lp.branch_bound import _repair_via_continuous_lp

    lp, mask = _mk_mixed_integer_with_continuous()
    bad = np.array([1.0, 1.0, 1.0, 0.0])
    # At its minimum allowable value x3 = 0 the activity already exceeds the
    # RHS, and x3 >= 0 can only increase it: the fixed-integer LP is
    # truly infeasible.
    assert np.asarray(lp.A @ bad).flat[0] > 8.0 + 1e-12, (
        "test setup: matrix-consistent infeasibility check"
    )
    ok, x_out, obj_out = _repair_via_continuous_lp(lp, bad, mask)
    assert not ok, "infeasible fixed assignment must be rejected"
    assert np.isnan(obj_out)
    assert np.allclose(x_out, bad)


def test_repaired_assignment_becomes_incumbent_in_solve_milp():
    """The repair heuristic's feasible point flows into solve_milp as an
    incumbent and the solve still proves the true optimum (-10)."""
    from lp.branch_bound import solve_milp

    lp, mask = _mk_mixed_integer_with_continuous()
    res = solve_milp(lp, integer_mask=mask, node_limit=40, verbose=False)
    assert res.status == "optimal", res.message
    assert abs(res.objective - (-10.0)) < 1e-4, res.objective
    assert res.x is not None
    x = np.asarray(res.x, dtype=np.float64)
    activity = float(np.asarray(lp.A @ x).flat[0])
    assert activity <= 8.0 + 1e-5, f"A@x = {activity} > 8"
    for i in np.flatnonzero(mask):
        assert abs(x[i] - np.round(x[i])) < 1e-6, x
    assert res.heuristic_attempts >= 1, "repair heuristic never called"
    assert res.heuristic_successes >= 1, "repair heuristic never succeeded"
    assert res.heuristic_lp_solves >= 1, "repair never re-solved a continuous LP"


def test_heuristic_incumbent_objective_is_reported():
    """Diagnostics: when the rounding heuristic finds an incumbent, the best
    heuristic objective and attempt counts are exposed on MilpResult and the
    heuristic value never exceeds the proven optimum (minimization)."""
    lp, mask = _mk_mixed_integer_with_continuous()
    res = solve_milp(lp, integer_mask=mask, node_limit=40, verbose=False)
    assert res.status == "optimal", res.message
    assert res.heuristic_attempts >= 1
    assert res.heuristic_successes >= 1
    assert res.heuristic_best_objective is not None
    assert abs(res.heuristic_best_objective - (-10.0)) < 1e-4, (
        res.heuristic_best_objective
    )
    # Heuristic incumbent is never strictly better than the proven optimum.
    assert res.heuristic_best_objective >= res.objective - 1e-4


def test_repair_solution_passes_independent_checks():
    """The repaired solution is verifiable directly from the original data:
    bounds, integrality, row constraints, and objective are all consistent."""
    import math

    from lp.branch_bound import _repair_via_continuous_lp

    lp, mask = _mk_mixed_integer_with_continuous()
    cand = np.array([1.0, 0.0, 1.0, 0.0])
    ok, x, obj = _repair_via_continuous_lp(lp, cand, mask)
    assert ok

    # 1. Bounds.
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    assert np.all(x >= lb - 1e-6) and np.all(x <= ub + 1e-6), x

    # 2. Integrality of integer variables.
    for i in np.flatnonzero(mask):
        assert abs(x[i] - np.round(x[i])) <= 1e-6, x

    # 3. Row constraints (independent of the solver's status).
    residual = float(np.asarray(lp.A @ x - lp.b).flat[0])
    assert residual <= 1e-6, f"row residual {residual} > 0"

    # 4. Objective consistency.
    computed = float(np.asarray(lp.c, dtype=np.float64) @ x)
    assert math.isclose(computed, obj, rel_tol=1e-9, abs_tol=1e-9)

    # 5. Deterministic nearest rounding alone would have failed here,
    #    proving repair was required (activity of (1,1,1,0) is 9 > 8).
    assert np.asarray(lp.A @ np.array([1.0, 1.0, 1.0, 0.0])).flat[0] > 8.0


def test_strong_branch_handles_numerically_failed_child():
    """Strong branching must survive a node whose child relaxation fails
    numerically (e.g. crossover returns a dict without residual keys)."""
    from lp.branch_bound import _strong_branch_select
    from dataclasses import replace as _replace

    lp = _mk_lp(
        "SB_CASE", [[3.0, 4.0, 2.0, 6.0]], [8.0],
        [4.0, 5.0, 3.0, 7.0], ("L",),
        np.zeros(4), np.ones(4),
    )
    x = np.array([1.0, 0.75, 1.0, 0.0])
    j = _strong_branch_select(
        lp, x, np.ones(4, dtype=bool),
        np.zeros(4), np.ones(4),
        k=6, int_tol=1e-6, crossover_fallback=False,
    )
    # j may be None (all children failed) or a valid index; must not raise.
    assert j is None or 0 <= int(j) < 4