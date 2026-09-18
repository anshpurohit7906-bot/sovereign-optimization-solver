"""Sparse convex QP package tests (canonical path: src/qp).

Adapted from the teammate drop.  src/lp/qp.py remains the legacy
standard-form path and has its own tests in test_qp_skeleton.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_LP = os.path.join(_ROOT, "src", "lp")
_SRC = os.path.join(_ROOT, "src")
# NOTE: order enforcement, not first-insert.  Other test modules
# (e.g. test_qp_skeleton.py) put "src/lp" BEFORE "src" on sys.path, which
# makes bare "import qp" resolve to the legacy src/lp/qp.py module.
# Rebuild the path so "src" precedes "src/lp" and "qp" resolves to the
# canonical src/qp package.  "lp.qp" imports remain unaffected.
for _p in (_ROOT, _LP, _SRC):
    while _p in sys.path:
        sys.path.remove(_p)
for _p in (_LP, _SRC, _ROOT):
    sys.path.insert(0, _p)

from qp import QPProblem, QPValidationError, QPSolverError, solve_qp, certificate  # noqa: E402
from qp import solver as solver_module  # noqa: E402
from qp.linear_system import solve_kkt  # noqa: E402


def test_unconstrained():
    p = QPProblem(2*np.eye(2), [-4, -6])
    r = solve_qp(p, tol=1e-8)
    assert r.status == "optimal"
    assert np.allclose(r.x, [2, 3], atol=1e-5)


def test_inequality():
    p = QPProblem(2*np.eye(2), [-4, -6], G=[[1, 1]], h=[4])
    r = solve_qp(p, tol=1e-7)
    assert r.status == "optimal"
    assert np.allclose(r.x, [1.5, 2.5], atol=1e-4)


def test_equality():
    p = QPProblem(2*np.eye(2), [-4, -6], A=[[1, 1]], b=[4])
    r = solve_qp(p, tol=1e-7)
    assert r.status == "optimal"
    assert np.allclose(r.x, [1.5, 2.5], atol=1e-4)


def test_bounds():
    p = QPProblem(2*np.eye(2), [-4, -6], lb=[0, 0], ub=[1, 3])
    r = solve_qp(p, tol=1e-7)
    assert r.status == "optimal"
    assert np.allclose(r.x, [1, 3], atol=5e-4)


def test_certificate():
    p = QPProblem(2*np.eye(2), [-4, -6], G=[[1, 1]], h=[4])
    r = solve_qp(p, tol=1e-7)
    c = certificate(p, r.x, r.y, r.z, r.s, tol=1e-4)
    assert c["ok"]
    # scaled_stationarity must be present (benchmark harness consumes it).
    assert "scaled_stationarity" in c


def test_sparse_inequality_only_qp():
    P = sp.diags([2.0, 3.0, 4.0], format="csr")
    G = sp.csr_matrix([[1.0, 1.0, 0.0]])
    p = QPProblem(P, [-2, -3, -1], G=G, h=[2.0])
    r = solve_qp(p, tol=1e-7, sparse=True)
    assert r.status == "optimal"
    assert r.primal_residual < 1e-6


def test_sparse_equality_constrained_qp():
    P = sp.eye(4, format="csr")
    A = sp.csr_matrix([[1., 1., 1., 1.]])
    p = QPProblem(P, [-1., -2., -3., -4.], A=A, b=[2.])
    r = solve_qp(p, tol=1e-7, sparse=True)
    assert r.status == "optimal"
    assert np.allclose(r.x, [-1.0, .0, 1.0, 2.0], atol=1e-5)


def test_sparse_kkt_inequality_path_never_calls_dense_solver(monkeypatch):
    import qp.linear_system as ls

    def fail_dense(*args, **kwargs):
        raise AssertionError("dense solver was used")

    monkeypatch.setattr(ls, "dense_solve", fail_dense)
    H = sp.diags([1., 2., 3.], format="csc")
    dx, dy = solve_kkt(H, None, np.array([1., 2., 3.]), sparse=True)
    assert np.allclose(dx, [1., 1., 1.])
    assert dy.size == 0


def test_invalid_dimensions():
    with pytest.raises(QPValidationError):
        QPProblem(np.eye(2), [1, 2, 3])


def test_nonconvex_qp_rejected():
    with pytest.raises(QPValidationError):
        QPProblem(np.diag([1., -1.]), [0., 0.]).check_convexity()


def test_sparse_nonconvex_qp_rejected():
    P = sp.diags([1., -2.], format="csr")
    with pytest.raises(QPValidationError):
        QPProblem(P, [0., 0.]).check_convexity()


def test_negative_kkt_certificate():
    p = QPProblem(np.eye(2), [0., 0.], G=[[1., 0.]], h=[1.])
    bad = certificate(p, np.array([2., 0.]), z=np.array([1.]), s=np.array([1.]),
                      tol=1e-8)
    assert not bad["ok"]
    assert bad["inequality_violation"] > 0


def test_negative_certificate_negative_multiplier():
    p = QPProblem(np.eye(2), [0., 0.], G=[[1., 0.]], h=[1.])
    bad = certificate(p, np.array([0., 0.]), z=np.array([-1.]), s=np.array([1.]),
                      tol=1e-8)
    assert not bad["ok"]
    assert bad["dual_violation"] > 0


def test_ill_conditioned_nontrivial_qp():
    P = np.diag([1e-4, 1., 1e4])
    q = np.array([-1e-4, -1., -1e4])
    G = np.array([[1., 1., 1.], [-1., 0., 0.]])
    h = np.array([2., 0.])
    r = solve_qp(QPProblem(P, q, G=G, h=h), tol=1e-6)
    assert r.status == "optimal"
    assert r.primal_residual < 1e-5
    assert r.dual_residual < 1e-5


def test_best_state_restores_complete_state(monkeypatch):
    p = QPProblem(np.eye(1), [-1.], G=[[1.]], h=[2.])
    initial = solve_qp(p, max_iterations=1, tol=0.1)  # reference state
    initial_s = initial.s.copy()

    def degraded_direction(P, q, G, A, s, z, rd, rg, rp, rc, **kwargs):
        return np.array([1e3]), np.empty(0), np.array([-1e3]), np.array([-1e3])

    monkeypatch.setattr(solver_module, "_solve_direction", degraded_direction)
    r = solve_qp(p, max_iterations=1, tol=0.1)
    assert np.allclose(r.x, 0.0, atol=1e-12)
    assert np.allclose(r.s, initial_s, atol=1e-12)
    assert np.all(r.z > 0)


def test_deterministic_end_to_end_kkt_certificate():
    """Deterministic e2e QP validated with the independent certificate.

    min 1/2 x'Px + q'x  s.t.  x1 + x2 = 2, x >= 0, x1 <= 1.5
    P = diag(2, 2), q = (-2, -4): the linear term prefers a large x2, so
    with x1 + x2 = 2 and x1 <= 1.5 the certified optimum is
    x = (0.5, 1.5) with objective 1/2(0.5 + 4.5) - 1 - 6 = -4.5.
    """
    p = QPProblem(2*np.eye(2), [-2., -4.], lb=[0., 0.], ub=[1.5, np.inf],
                  A=[[1., 1.]], b=[2.], name="DET_E2E")
    r = solve_qp(p, tol=1e-8, max_iterations=200)
    assert r.status == "optimal", (r.status, r.message)
    assert np.allclose(r.x, [0.5, 1.5], atol=1e-5), r.x
    assert abs(r.objective - (-4.5)) <= 1e-6, r.objective
    c = certificate(p, r.x, r.y, r.z, r.s, tol=1e-6)
    assert c["ok"], c
    assert c["stationarity"] < 1e-6
    assert c["complementarity"] < 1e-6
