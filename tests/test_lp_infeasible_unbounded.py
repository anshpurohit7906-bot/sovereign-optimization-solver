"""Honesty tests: clearly infeasible and clearly unbounded LPs on the
production Mehrotra IPM (``opticore.lp.mehrotra.solve_lp``).

What is asserted (and why):

* The solver TERMINATES on both fixtures without hanging or crashing.
  The iteration loop is bounded by ``max_iter``; the wall-clock guard below
  is a generous backstop (observed ~0.03 s per solve for these 1x1 fixtures).
* The solver NEVER reports ``"optimal"`` on either fixture.  Independently
  of the status string, the returned best iterate misses the acceptance
  tolerance by orders of magnitude (merit >> tol), so an "optimal" claim
  would be false on the solver's own metrics.

What is deliberately NOT asserted (the solver does not support it today):

* ``MehrotraResult.status`` vocabulary is
  ``"optimal" | "numerical_tail" | "max_iterations" | "stalled" |
  "numerical_failure"`` — the IPM has no infeasibility or unboundedness
  certificate and never emits ``"infeasible"``/``"unbounded"``.  No such
  status is invented or expected here.
* Exact observed behavior on BOTH fixtures (35/35 probe runs, defaults and
  varied ``tol``/``max_iter``/``crossover_fallback``): status ``"stalled"``,
  i.e. complementarity ``mu`` reached the scale-aware floor without
  practical accuracy.  The test pins that characterization; if the solver
  ever gains a real certificate, this is the line to update — nothing here
  forces a pass for behavior the solver cannot produce.
"""

from __future__ import annotations

import os
import time

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

from opticore.numerical_model import NumericalLP  # noqa: E402
from opticore.lp.mehrotra import solve_lp  # noqa: E402

_TOL = 1e-8
_MAX_ITER = 100
# Generous termination backstop: these fixtures solve in ~0.03 s locally.
# A hang would blow far past this; normal CI jitter cannot.
_WALL_CLOCK_LIMIT_S = 10.0


def _infeasible_lp() -> NumericalLP:
    """min x1 s.t. x1 <= -1 (L row), x >= 0 — no feasible point exists."""
    return NumericalLP(
        name="INFEAS_1VAR",
        objective_name="COST",
        A=np.array([[1.0]]),
        b=np.array([-1.0]),
        c=np.array([1.0]),
        lower_bounds=np.zeros(1),
        upper_bounds=np.full(1, np.inf),
        row_types=("L",),
        var_names=("x1",),
        row_names=("r1",),
    )


def _unbounded_lp() -> NumericalLP:
    """min -x1 s.t. -x1 <= 0 (redundant with x >= 0), x >= 0.

    Nothing bounds x1 from above, so the objective tends to -infinity:
    the LP is unbounded below.  The single row only keeps the standard
    form non-empty; it does not bound the feasible ray x1 -> +inf.
    """
    return NumericalLP(
        name="UNBOUNDED_1VAR",
        objective_name="COST",
        A=np.array([[-1.0]]),
        b=np.array([0.0]),
        c=np.array([-1.0]),
        lower_bounds=np.zeros(1),
        upper_bounds=np.full(1, np.inf),
        row_types=("L",),
        var_names=("x1",),
        row_names=("r1",),
    )


def test_infeasible_lp_terminates_and_never_reports_optimal():
    """The infeasible fixture must finish quickly and must not claim optimal.

    Observed production behavior (defaults): status == "stalled" — mu hits
    the scale-aware floor at iteration ~17 while the primal residual stays
    at rel_p = 0.5.  The solver cannot certify infeasibility, so "stalled"
    plus a non-optimal claim is the honest outcome it supports today.
    """
    lp = _infeasible_lp()

    t0 = time.perf_counter()
    res = solve_lp(lp, tol=_TOL, max_iter=_MAX_ITER)
    elapsed = time.perf_counter() - t0

    # Terminates without hanging.
    assert elapsed <= _WALL_CLOCK_LIMIT_S, (
        f"solve took {elapsed:.3f}s (limit {_WALL_CLOCK_LIMIT_S}s)"
    )

    # Never incorrectly reports optimal.
    assert res.status != "optimal", (res.status, res.message)

    # Exact observed behavior of the current solver on this fixture.
    assert res.status == "stalled", (res.status, res.message)

    # Independent evidence: the returned iterate is nowhere near KKT
    # acceptance, so "optimal" would be false on the solver's own metrics.
    merit = max(res.rel_primal, res.rel_dual, res.rel_gap)
    assert merit > _TOL, (res.status, merit, res.message)


def test_unbounded_lp_terminates_and_never_reports_optimal():
    """The unbounded fixture must finish quickly and must not claim optimal.

    Observed production behavior (defaults): status == "stalled" — mu hits
    the scale-aware floor at iteration ~17 while the dual residual stays at
    rel_d = 0.25.  The solver cannot certify unboundedness, so "stalled"
    plus a non-optimal claim is the honest outcome it supports today.
    """
    lp = _unbounded_lp()

    t0 = time.perf_counter()
    res = solve_lp(lp, tol=_TOL, max_iter=_MAX_ITER)
    elapsed = time.perf_counter() - t0

    # Terminates without hanging.
    assert elapsed <= _WALL_CLOCK_LIMIT_S, (
        f"solve took {elapsed:.3f}s (limit {_WALL_CLOCK_LIMIT_S}s)"
    )

    # Never incorrectly reports optimal.
    assert res.status != "optimal", (res.status, res.message)

    # Exact observed behavior of the current solver on this fixture.
    assert res.status == "stalled", (res.status, res.message)

    # Independent evidence: the returned iterate is nowhere near KKT
    # acceptance, so "optimal" would be false on the solver's own metrics.
    merit = max(res.rel_primal, res.rel_dual, res.rel_gap)
    assert merit > _TOL, (res.status, merit, res.message)
