"""MILP robustness tests: numerical-failure honesty, per-node bounds, sparsity.

These tests pin the robustness contract added on top of the v2 branch-and-bound
skeleton (``src/lp/branch_bound.py``) without touching the LP core:

1. A relaxation that FAILS numerically is never reported as ``"infeasible"``
   and never silently pruned; it is counted in ``nodes_failed`` and blocks an
   ``"optimal"`` claim.
2. Only a Phase-I infeasibility certificate maps a failed relaxation to
   ``"infeasible"`` (those nodes are counted in ``nodes_infeasible``).
3. Every per-node relaxation is bounded: pivot caps and an optional shared
   wall-clock budget are propagated into Phase I / Phase II, and an exhausted
   budget is reported as ``"time_limit"`` (never as a solution).
4. Sparse models stay sparse end to end (no whole-matrix densification) and
   give the same answers as their dense counterparts.
5. Regression: the 4-crude refinery MILP from the demo still solves to 23320
   with status ``"optimal"``.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

from opticore.numerical_model import NumericalLP  # noqa: E402
from opticore.lp import branch_bound as bb  # noqa: E402
from opticore.lp.branch_bound import (  # noqa: E402
    solve_milp, _relax, _drop_redundant_rows,
)
from opticore.lp.crossover import (  # noqa: E402
    sparse_phase1, sparse_phase2, crossover_from_ipm,
)


def _mk_lp(name, A, b, c, row_types, lb, ub, row_names=None):
    A = A if sp.issparse(A) else np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    m = b.shape[0]
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
        row_names=tuple(row_names) if row_names is not None
        else tuple(f"r{i}" for i in range(m)),
    )


def _knapsack_min():
    """min -(4x0+5x1+3x2+7x3) s.t. 3x0+4x1+2x2+6x3 <= 8, x binary.

    Root LP relaxation is fractional (x = (1, 0.75, 1, 0)); MIP optimum = -10.
    """
    return _mk_lp(
        "KNAPSACK_MIN",
        np.array([[3.0, 4.0, 2.0, 6.0]]),
        np.array([8.0]),
        np.array([-4.0, -5.0, -3.0, -7.0]),
        ("L",),
        np.zeros(4),
        np.ones(4),
    )


def _small_standard_form():
    """min -x1 - x2  s.t.  x1 + x2 + s = 3, x1 + 2 s = 4 (x, s >= 0)."""
    A = sp.csc_matrix(np.array([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 2.0],
    ]))
    b = np.array([3.0, 4.0])
    c = np.array([-1.0, -1.0, 0.0, 0.0])
    return A, b, c


# ---------------------------------------------------------------------------
# 1/2. Numerical failure is never conflated with infeasibility
# ---------------------------------------------------------------------------
def test_relax_time_budget_reported_honestly():
    """A bounded relaxation that runs out of budget reports ``time_limit``.

    Forcing the IPM to fail (``max_iter=0``) sends the relaxation into the
    bounded simplex fallback with a zero-second budget, so the fallback must
    stop at its very first iteration and say so -- not guess a solution and not
    claim infeasibility.
    """
    lp = _knapsack_min()
    rel = _relax(lp, max_iter=0, time_limit=0.0)
    assert rel.status == "time_limit", (rel.status, rel.message)
    assert "time" in rel.message.lower()
    assert rel.status not in ("optimal", "infeasible")


def test_relax_infeasible_only_from_phase1_certificate():
    """``infeasible`` requires a Phase-I certificate; failures never fake it."""
    # Genuinely infeasible LP: x >= 3 and x <= 1.
    bad = _mk_lp(
        "INFEASIBLE_LP",
        np.array([[1.0], [1.0]]),
        np.array([3.0, 1.0]),
        np.array([1.0]),
        ("G", "L"),
        np.array([0.0]),
        np.array([10.0]),
    )
    rel = _relax(bad)
    assert rel.status == "infeasible", (rel.status, rel.message)
    assert "certificate" in rel.message.lower()

    # A feasible LP whose relaxation was cut off numerically must NOT be
    # reported as infeasible by the same code path.
    cut_off = _relax(_knapsack_min(), max_iter=0, time_limit=0.0)
    assert cut_off.status != "infeasible"

def test_root_numerical_failure_not_reported_as_infeasible(monkeypatch):
    """A root relaxation that fails numerically yields ``lp_status:<status>``."""
    lp = _knapsack_min()

    def fake_relax(lp_arg, **kwargs):
        return bb._RelaxResult("numerical_failure", float("nan"), None,
                               "synthetic numerical failure")

    monkeypatch.setattr(bb, "_relax", fake_relax)
    res = solve_milp(lp, integer_mask=np.ones(4, dtype=bool), gap_tol=1e-6)
    assert res.status == "lp_status:numerical_failure", (res.status, res.message)
    assert res.status != "infeasible"
    assert res.objective is None and res.x is None


def test_exhausted_tree_with_failed_nodes_is_not_claimed_optimal(monkeypatch):
    """Dropped nodes must block an ``optimal`` claim even with an incumbent.

    Root solve and the integer-fixing repair heuristic are allowed to run for
    real (so an incumbent exists and the tree genuinely empties); every genuine
    NODE relaxation is forced to fail numerically.  The tree then exhausts with
    an incumbent and two dropped nodes -- the old skeleton reported
    ``"optimal"`` here, which the honesty contract forbids.
    """
    lp = _knapsack_min()
    lb0 = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub0 = np.asarray(lp.upper_bounds, dtype=np.float64)
    mask = np.ones(4, dtype=bool)
    real_relax = bb._relax
    intercepted: list = []

    def fake_relax(lp_arg, **kwargs):
        same_lb = np.array_equal(np.asarray(lp_arg.lower_bounds), lb0)
        same_ub = np.array_equal(np.asarray(lp_arg.upper_bounds), ub0)
        all_fixed = bool(np.all(
            np.asarray(lp_arg.lower_bounds)[mask]
            == np.asarray(lp_arg.upper_bounds)[mask]
        ))
        is_root = same_lb and same_ub
        is_repair = (not same_lb) and all_fixed
        if is_root or is_repair:
            return real_relax(lp_arg, **kwargs)
        intercepted.append(1)
        return bb._RelaxResult("numerical_failure", float("nan"), None,
                               "synthetic node failure")

    monkeypatch.setattr(bb, "_relax", fake_relax)
    res = solve_milp(lp, integer_mask=mask, gap_tol=1e-6, node_limit=10,
                     heuristic_freq=0)

    assert intercepted, "test setup failed: no node relaxation was intercepted"
    assert res.status == "stalled", (res.status, res.message)
    assert res.nodes_failed == len(intercepted) >= 1
    assert res.nodes_infeasible == 0
    assert len(res.node_failure_statuses) == res.nodes_failed
    assert set(res.node_failure_statuses) == {"numerical_failure"}
    assert "failed" in res.message.lower()
    # The incumbent is still reported honestly (it just is not proven optimal).
    assert res.objective is not None

# ---------------------------------------------------------------------------
# 2b. Infeasible nodes are pruned AND counted as infeasible, not as failures
# ---------------------------------------------------------------------------
def test_infeasible_nodes_pruned_and_counted():
    """min x0+x1 s.t. 2x0+x1>=2, x0+2x1>=2, binary -> optimum 2.

    The LP relaxation is fractional (2/3, 2/3) so some children are infeasible
    by integrality; those nodes must be pruned by a Phase-I certificate
    (counted in ``nodes_infeasible``) and must NOT be counted as failures.
    """
    lp = _mk_lp(
        "TWO_VAR_COVER",
        np.array([[2.0, 1.0], [1.0, 2.0]]),
        np.array([2.0, 2.0]),
        np.array([1.0, 1.0]),
        ("G", "G"),
        np.zeros(2),
        np.ones(2),
    )
    res = solve_milp(lp, integer_mask=np.ones(2, dtype=bool), gap_tol=1e-6)
    assert res.status == "optimal", (res.status, res.message)
    assert abs(res.objective - 2.0) <= 1e-6, res.objective
    assert res.nodes_infeasible >= 1, (res.nodes_infeasible, res.message)
    assert res.nodes_failed == 0, (res.nodes_failed, res.node_failure_statuses)


# ---------------------------------------------------------------------------
# 3. The crossover phases honour the pivot / time budgets they are given
# ---------------------------------------------------------------------------
def test_sparse_phase1_pivot_and_time_budgets():
    A, b, _ = _small_standard_form()
    # Zero Phase-I pivot budget => explicit iteration-limit report.
    _, _, status, _ = sparse_phase1(A, b, max_iter=0)
    assert status == "max_iterations"
    # Zero time budget => explicit timeout report (not a silent success).
    _, _, status, _ = sparse_phase1(A, b, max_iter=10_000, time_limit=0.0)
    assert status == "time_limit"


def test_sparse_phase2_time_budget():
    A, b, c = _small_standard_form()
    basis, _, p1_status, _ = sparse_phase1(A, b, max_iter=10_000)
    assert p1_status == "feasible"
    res = sparse_phase2(A, b, c, list(basis), max_iter=10_000, time_limit=0.0)
    assert res["status"] == "time_limit", res.get("message")
    assert res["status"] != "optimal"


def test_crossover_from_ipm_propagates_budgets():
    A, b, c = _small_standard_form()
    res = crossover_from_ipm(A, b, c, phase1_max_iter=0)
    assert res["status"] == "phase1_max_iterations", res.get("message")
    res = crossover_from_ipm(A, b, c, time_limit=0.0)
    assert res["status"] == "phase1_time_limit", res.get("message")
    # Unbounded defaults still solve the toy exactly.
    res = crossover_from_ipm(A, b, c)
    assert res["status"] == "optimal", res.get("message")
    assert abs(res["objective"] - (-3.0)) <= 1e-6
def test_solve_milp_relaxation_budgets_never_fake_a_solution():
    """``node_solve_*`` budgets propagate and exhaustion is reported honestly.

    With a zero iteration budget and a zero wall-clock budget the very first
    (root) relaxation cannot finish; ``solve_milp`` must report the timeout
    status rather than returning a fabricated solution or claiming
    infeasibility.  No monkeypatching: this is the real bounded code path.
    """
    lp = _knapsack_min()
    res = solve_milp(lp, integer_mask=np.ones(4, dtype=bool), gap_tol=1e-6,
                     node_solve_max_iter=0, node_solve_time_limit=0.0)
    assert res.status == "lp_status:time_limit", (res.status, res.message)
    assert res.status not in ("optimal", "infeasible")
    assert res.objective is None and res.x is None
    assert "time" in res.message.lower()



# ---------------------------------------------------------------------------
# 4. Sparse safety: no whole-matrix densification, no behaviour change
# ---------------------------------------------------------------------------
def _sparse_assignment_with_duplicate_row():
    """2x2 assignment (rank-deficient E rows) plus an exact duplicate row."""
    dense = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    A = np.vstack([dense, dense[0:1]])          # row 4 duplicates row 0
    b = np.ones(5)
    c = np.array([2.0, 3.0, 1.0, 2.0])
    return A, b, c


def test_drop_redundant_rows_is_sparse_safe():
    A, b, c = _sparse_assignment_with_duplicate_row()
    lp_sparse = _mk_lp("ASSIGN_SPARSE", sp.csr_matrix(A), b, c,
                       ("E",) * 5, np.zeros(4), np.ones(4))
    ded = _drop_redundant_rows(lp_sparse)
    # The result must stay sparse (no dense blow-up) ...
    assert sp.issparse(ded.A)
    # ... and the redundant rows are gone: 5 rows -> 3 kept (dup + rank drop).
    assert ded.A.shape == (3, 4), ded.A.shape

    # Dense input keeps the dense contract (unchanged behaviour).
    lp_dense = _mk_lp("ASSIGN_DENSE", A, b, c, ("E",) * 5,
                      np.zeros(4), np.ones(4))
    ded_dense = _drop_redundant_rows(lp_dense)
    assert not sp.issparse(ded_dense.A)
    assert ded_dense.A.shape == (3, 4)
    assert np.allclose(np.asarray(ded_dense.A), np.asarray(ded.A.toarray()))


def test_sparse_milp_end_to_end_matches_dense():
    A, b, c = _sparse_assignment_with_duplicate_row()
    sparse_lp = _mk_lp("ASSIGN_SPARSE", sp.csr_matrix(A), b, c,
                       ("E",) * 5, np.zeros(4), np.ones(4))
    dense_lp = _mk_lp("ASSIGN_DENSE", A, b, c, ("E",) * 5,
                      np.zeros(4), np.ones(4))
    mask = np.ones(4, dtype=bool)

    res_sparse = solve_milp(sparse_lp, integer_mask=mask, gap_tol=1e-6)
    res_dense = solve_milp(dense_lp, integer_mask=mask, gap_tol=1e-6)

    assert res_sparse.status == "optimal", (res_sparse.status, res_sparse.message)
    assert res_sparse.nodes_failed == 0
    assert abs(res_sparse.objective - 4.0) <= 1e-6, res_sparse.objective
    assert abs(res_sparse.objective - res_dense.objective) <= 1e-6
    assert res_sparse.x is not None and res_sparse.x.shape == (4,)


# ---------------------------------------------------------------------------
# 5. Regression: the demo refinery MILP is still solved to 23320, honestly
# ---------------------------------------------------------------------------
def test_refinery_milp_regression_23320():
    """The 4-crude fixed-charge refinery planning MILP: optimum 23320.

    This instance stalls the IPM at a degenerate tail on the root and on some
    children, so it exercises the bounded fallback (including the near-feasible
    Phase-I rescue) and must still report a genuine ``"optimal"`` with zero
    dropped nodes -- the previous silent-prune path claimed optimality while
    discarding a node whose LP bound this test now proves (by bound).
    """
    from demo.demo_sih import build_production_planning_milp

    lp, mask, _info = build_production_planning_milp()
    res = solve_milp(lp, integer_mask=np.asarray(mask, dtype=bool),
                     gap_tol=1e-4)

    assert res.status == "optimal", (res.status, res.message)
    assert res.nodes_failed == 0, (res.nodes_failed, res.node_failure_statuses)
    assert res.nodes_infeasible == 0, res.nodes_infeasible
    assert res.objective is not None
    assert abs(res.objective - 23320.0) <= 1e-4, res.objective
    assert res.x is not None
    assert res.gap is not None and res.gap <= 1e-4


# ---------------------------------------------------------------------------
# 6. Global deadline propagation: a tiny time_limit terminates honestly
# ---------------------------------------------------------------------------
import time as _time_mod


def test_global_deadline_overrides_local_budgets():
    """A tiny global ``time_limit`` overrides a large local budget.

    With ``node_solve_max_iter=0`` the IPM cannot converge and the bounded
    crossover fallback is used; ``node_solve_time_limit=100.0`` would let it
    run for 100 seconds locally.  The global ``time_limit=1e-6`` must cap the
    entire solve and terminate honestly (never ``"optimal"``, never
    ``"infeasible"``), returning in well under one second of wall-clock time.
    """
    import time as _time_mod
    lp = _knapsack_min()
    mask = np.ones(4, dtype=bool)
    t0 = _time_mod.perf_counter()
    res = solve_milp(lp, integer_mask=mask, gap_tol=1e-6,
                     node_solve_max_iter=0, node_solve_time_limit=100.0,
                     time_limit=1e-6)
    elapsed = _time_mod.perf_counter() - t0
    # Honesty: never claims a solution or infeasibility.
    assert res.status != "optimal", res.message
    assert res.status != "infeasible", res.message
    # Must be time-limit related (root failure via deadline).
    assert "time" in res.status.lower() or "time" in res.message.lower()
    # No solution returned.
    assert res.objective is None and res.x is None
    # Must terminate well within the 100s local budget.
    assert elapsed < 5.0, f"global deadline ignored: {elapsed:.2f}s"
    # Deadline propagated: message contains the deadline text.
    assert "deadline" in res.message.lower(), res.message


def test_global_deadline_mid_search(monkeypatch):
    """A global ``time_limit`` that expires *after* the search has begun.

    The earlier version of this test relied on a real 0.05 s wall-clock race:
    on fast CI machines the tiny knapsack tree finished before the deadline
    and the solver legitimately reported ``"optimal"``.  Here the wall clock
    and every node relaxation are mocked, so the interleaving is fixed:

    1. the root relaxation returns a fixed fractional optimum,
    2. the real branch-and-bound code branches normally on it,
    3. the first child relaxation is allowed to complete (refining the
       incumbent the root rounding heuristic already found),
    4. *only then* the mocked clock jumps past the global deadline,
    5. so the very next loop-top deadline check must break the search.

    Production code is untouched: only module-local references used by
    ``solve_milp`` (``branch_bound.time`` and ``branch_bound._relax``) are
    replaced, and the real branching / heap / termination logic runs.
    """
    lp = _knapsack_min()
    mask = np.ones(4, dtype=bool)
    time_limit = 0.05  # unchanged; with a mocked clock its value is not a race

    class _FakeClock:
        """Stand-in for the ``time`` module: only ``perf_counter`` is used."""

        def __init__(self):
            self.now = 0.0

        def perf_counter(self):
            return self.now

    clock = _FakeClock()
    monkeypatch.setattr(bb, "time", clock)

    node_relaxations = 0
    relax_calls = 0

    def _fake_relax(node, *, deadline=None, **kwargs):
        """Deterministic stand-in for every relaxation in the search.

        The three call sites are told apart by the *bounds* of the LP handed
        in, never by call order, so the fake cannot silently mistake the root
        rounding heuristic's repair LP for a branch-and-bound node:

        * root          -- no variable bound differs from ``[lb0, ub0]``
        * heuristic fix -- every integer variable is pinned (``lb == ub``)
        * B&B node      -- exactly one integer variable has a tightened bound
        """
        nonlocal node_relaxations, relax_calls
        relax_calls += 1
        lb_n = np.asarray(node.lower_bounds, dtype=np.float64)
        ub_n = np.asarray(node.upper_bounds, dtype=np.float64)
        is_root = (np.allclose(lb_n, lp.lower_bounds)
                   and np.allclose(ub_n, lp.upper_bounds))
        is_repair = bool(np.all(np.isclose(lb_n[mask], ub_n[mask])))
        if not (is_root or is_repair):
            node_relaxations += 1

        if is_root:
            # Root: the fixed fractional optimum of the knapsack relaxation,
            # so the real branching code splits it (x1 = 0.75) into two children.
            x = np.array([1.0, 0.75, 1.0, 0.0])  # objective -10.75
        elif is_repair:
            # Heuristic integer-fixing repair: hand back the pinned point so
            # the heuristic's own independent verification decides.
            x = np.clip(np.array([0.0, 0.0, 1.0, 1.0]), lb_n, ub_n)
        else:
            # First branch-and-bound node (x1 <= 0): the relaxation completes
            # normally and yields the integer optimum -10 as a new incumbent.
            x = np.clip(np.array([0.0, 0.0, 1.0, 1.0]), lb_n, ub_n)
            if node_relaxations == 1:
                # The deadline expires *after* this relaxation completes, so
                # the next loop-top check is the one that must terminate.
                assert deadline is not None
                clock.now = deadline + 1.0
        return bb._RelaxResult(
            status="optimal",
            objective=float(np.asarray(node.c, dtype=np.float64) @ x),
            x=x,
            message="deterministic mocked relaxation",
        )

    monkeypatch.setattr(bb, "_relax", _fake_relax)

    res = solve_milp(lp, integer_mask=mask, gap_tol=1e-6, time_limit=time_limit)

    # Root + exactly one completed node relaxation: the deadline stopped the
    # *next* node, before any further work.  ``relax_calls`` also counts the
    # root rounding heuristic's integer-fixing repair LPs, whose number is an
    # implementation detail of the heuristic, hence only a lower bound here.
    assert node_relaxations == 1, node_relaxations
    assert relax_calls >= 2, relax_calls
    # Search really began: at least one node was popped and expanded.
    assert res.nodes_explored >= 1, res.nodes_explored
    assert res.lp_solves >= 2, res.lp_solves
    # Honesty: never claims a proof or infeasibility.
    assert res.status not in ("optimal", "infeasible"), (
        f"status={res.status}; the deadline should have terminated the solve"
    )
    # Termination is time-limit related (status is "time_limit" here).
    assert "time" in res.status.lower() or "time" in res.message.lower(), (
        f"status={res.status}; message={res.message}"
    )
    # An incumbent may exist; when it does it is consistent and feasible.
    assert (res.x is None) == (res.objective is None), (res.x, res.objective)
    if res.x is not None:
        assert res.objective is not None
        assert res.x.shape == (4,), res.x.shape
        assert np.all(res.x >= -1e-9), res.x
        assert np.all(res.x <= np.asarray(lp.upper_bounds) + 1e-9), res.x
        # The reported incumbent must actually satisfy the model's rows.
        for i, rt in enumerate(lp.row_types):
            lhs = float(lp.A[i] @ res.x)
            if rt == "E":
                assert abs(lhs - lp.b[i]) <= 1e-6, (i, lhs, lp.b[i])
            elif rt == "L":
                assert lhs <= lp.b[i] + 1e-6, (i, lhs, lp.b[i])
            elif rt == "G":
                assert lhs >= lp.b[i] - 1e-6, (i, lhs, lp.b[i])
        # ...and every integer variable must really be integral.
        assert np.allclose(res.x[mask], np.round(res.x[mask]), atol=1e-6), res.x
        # The objective must match the solution vector.
        assert abs(float(lp.c @ res.x) - res.objective) <= 1e-6, (
            float(lp.c @ res.x),
            res.objective,
        )


def test_expired_deadline_unit_checks():
    """Every helper bails immediately when the deadline is already expired.

    Exercises ``_relax``, ``_rounding_heuristic``,
    ``_repair_via_continuous_lp``, ``_strong_branch_select``, and
    ``_root_infeasible`` with ``deadline=time.perf_counter()`` (already
    passed), verifying each returns its honest "nothing done" response
    instantly.
    """
    import time as _time_mod
    from opticore.lp.branch_bound import (_relax, _rounding_heuristic,
                                          _repair_via_continuous_lp,
                                          _strong_branch_select, _root_infeasible)

    lp = _knapsack_min()
    mask = np.ones(4, dtype=bool)
    lb0 = np.zeros(4)
    ub0 = np.ones(4)
    b_dense = np.asarray(lp.b, dtype=np.float64)
    c_dense = np.asarray(lp.c, dtype=np.float64)
    x = np.array([0.2, 0.8, 0.1, 0.9])

    expired = _time_mod.perf_counter()

    # 1. _relax: must not start IPM or crossover.
    rel = _relax(lp, deadline=expired)
    assert rel.status == "time_limit", (rel.status, rel.message)
    assert "deadline" in rel.message.lower()

    # 2. _rounding_heuristic: must return (False, None, inf) immediately.
    stats = {"attempts": 0, "successes": 0, "lp_solves": 0, "best_obj": None}
    h_feas, h_x, h_obj = _rounding_heuristic(
        x, mask, lb0, ub0, lp.A, b_dense, lp.row_types, c_dense,
        lp=lp, stats=stats, deadline=expired,
    )
    assert h_feas is False

    # 3. _repair_via_continuous_lp: must return (False, ...).
    ok = _repair_via_continuous_lp(lp, x, mask, deadline=expired)
    assert ok[0] is False

    # 4. _strong_branch_select: must return None.
    j = _strong_branch_select(lp, x, mask, lb0, ub0, deadline=expired)
    assert j is None

    # 5. _root_infeasible: must return False (no Phase I started).
    assert _root_infeasible(lp, deadline=expired) is False

