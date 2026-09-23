"""Convex QP skeleton tests.

M1 gate for the QP vertical slice: an indigenous primal-dual IPM on a
standard-form convex QP with independent KKT verification.  No external solver
is used; every optimum below is hand-verifiable.
"""

from __future__ import annotations

import os

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

# The installed ``opticore.lp.qp`` module is the legacy dense standard-form QP
# path; the canonical sparse convex-QP package is ``opticore.qp``.
from opticore.lp.qp import NumericalQP, solve_qp, verify_qp_kkt, QpError


def _mk_qp(Q, c, A, b, rt, lb=None, ub=None, name="QP"):
    n = len(c)
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.full(n, np.inf)
    return NumericalQP(
        name=name,
        Q=np.asarray(Q, dtype=np.float64),
        c=np.asarray(c, dtype=np.float64),
        A=np.asarray(A, dtype=np.float64),
        b=np.asarray(b, dtype=np.float64),
        row_types=tuple(rt),
        lower_bounds=np.asarray(lb, dtype=np.float64),
        upper_bounds=np.asarray(ub, dtype=np.float64),
    )


def test_tiny_qp_kkt_verified():
    """min 1/2(x^2+y^2) + x - 2y s.t. x+y>=1, x-y<=0.5, x,y>=0."""
    qp = _mk_qp(
        [[1.0, 0.0], [0.0, 1.0]], [1.0, -2.0],
        [[1.0, 1.0], [1.0, -1.0]], [1.0, 0.5], ("G", "L"),
    )
    res = solve_qp(qp)
    assert res.status == "optimal", (res.status, res.message)
    assert np.all(np.isfinite(res.x))
    # Feasibility: x+y>=1 and x-y<=0.5.
    assert res.x[0] + res.x[1] >= 1.0 - 1e-7
    assert res.x[0] - res.x[1] <= 0.5 + 1e-7
    check = verify_qp_kkt(qp, res)
    assert check["PASS"], check
    assert check["rel_primal"] < 1e-6 and check["rel_dual"] < 1e-6


def test_equality_qp_closed_form():
    """min 1/2 x^2 + x s.t. x = 2.  KKT: x + 1 - y = 0, x = 2 => y = 3.

    Objective = 2 + 2 = 4.  (1/2*4 + 2 = 4.)
    """
    qp = _mk_qp([[1.0]], [1.0], [[1.0]], [2.0], ("E",))
    res = solve_qp(qp)
    assert res.status == "optimal", (res.status, res.message)
    assert abs(res.x[0] - 2.0) <= 1e-6, res.x
    assert abs(res.objective - 4.0) <= 1e-6, res.objective
    assert abs(res.y[0] - 3.0) <= 1e-4, res.y   # dual value
    check = verify_qp_kkt(qp, res)
    assert check["PASS"], check


def test_upper_bounded_variable():
    """min 1/2 x^2 - 3x s.t. 0 <= x <= 1.  Optimum at x = 1: obj = 0.5 - 3 = -2.5."""
    qp = _mk_qp(
        [[1.0]], [-3.0], np.zeros((0, 1)), np.zeros(0), (),
        lb=[0.0], ub=[1.0],
    )
    res = solve_qp(qp)
    assert res.status == "optimal", (res.status, res.message)
    assert abs(res.objective - (-2.5)) <= 1e-6, res.objective
    assert abs(res.x[0] - 1.0) <= 1e-6, res.x
    check = verify_qp_kkt(qp, res)
    assert check["PASS"], check


def test_nonconvex_q_rejected():
    qp = _mk_qp([[-1.0]], [0.0], [[1.0]], [0.0], ("E",))
    try:
        solve_qp(qp)
    except QpError:
        pass
    else:
        raise AssertionError("expected QpError for indefinite Q")


def test_free_variable_rejected():
    qp = _mk_qp([[1.0]], [0.0], [[1.0]], [0.0], ("E",), lb=[-np.inf], ub=[np.inf])
    try:
        solve_qp(qp)
    except QpError:
        pass
    else:
        raise AssertionError("expected QpError for free variable")