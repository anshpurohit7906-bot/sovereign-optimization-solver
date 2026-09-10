"""Tests for the sparse crossover engine (src/lp/crossover.py).

Covers Phase I (all-artificial crash), Phase II (Devex revised simplex),
the ``crossover_from_ipm`` orchestrator, and the mehrotra.py fallback gate.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import crossover  # noqa: E402


def _small_feasible_toy():
    """min -x1 - x2  s.t.  x1 + x2 + s = 3, x1 + 2 s = 4."""
    A = sp.csc_matrix(np.array([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 2.0],
    ]))
    b = np.array([3.0, 4.0])
    c = np.array([-1.0, -1.0, 0.0, 0.0])
    return A, b, c


def _infeasible_toy():
    """min -x  s.t.  x + s = -1, x >= 0, s >= 0  (b < 0 => infeasible)."""
    A = sp.csc_matrix(np.array([[1.0, 1.0]]))
    b = np.array([-1.0])
    c = np.array([-1.0, 0.0])
    return A, b, c


class TestPhase1:
    def test_feasible_toy(self):
        A, b, _ = _small_feasible_toy()
        basis, iters, status, info = crossover.sparse_phase1(A, b, max_iter=50000)
        assert status == "feasible", f"Phase I failed: {status} {info}"
        assert iters > 0
        assert all(0 <= j < A.shape[1] for j in basis)

    def test_infeasible_detected(self):
        A, b, _ = _infeasible_toy()
        _, _, status, _ = crossover.sparse_phase1(A, b, max_iter=10000)
        assert status == "infeasible"

    def test_identity_basis_trivial(self):
        m = 5
        A = sp.eye(m, format="csc")
        b = np.ones(m)
        basis, _, status, _ = crossover.sparse_phase1(A, b, max_iter=10000)
        assert status == "feasible"
        xb = sp.linalg.splu(A[:, basis]).solve(b)
        assert np.all(xb >= -1e-7)


class TestPhase2:
    def test_small_toy_optimal(self):
        A, b, c = _small_feasible_toy()
        basis, _, p1_status, _ = crossover.sparse_phase1(A, b, max_iter=50000)
        assert p1_status == "feasible"
        result = crossover.sparse_phase2(
            A, b, c, list(basis), max_iter=5000, pricing="devex")
        assert result["status"] == "optimal", f"Phase II: {result['status']}"
        assert abs(result["objective"] - (-3.0)) < 1e-6
        assert result["rel_primal"] < 1e-6
        assert result["rel_dual"] < 1e-6
        assert result["rel_gap"] < 1e-6

    def test_dantzig_pricing(self):
        A, b, c = _small_feasible_toy()
        basis, _, p1_status, _ = crossover.sparse_phase1(A, b, max_iter=50000)
        assert p1_status == "feasible"
        result = crossover.sparse_phase2(
            A, b, c, list(basis), max_iter=5000, pricing="dantzig")
        assert result["status"] == "optimal"
        assert abs(result["objective"] - (-3.0)) < 1e-6


class TestCrossoverOrchestrator:
    def test_end_to_end(self):
        A, b, c = _small_feasible_toy()
        result = crossover.crossover_from_ipm(A, b, c, max_iter=5000)
        assert result["status"] == "optimal", result.get("message")
        assert abs(result["objective"] - (-3.0)) < 1e-6
        assert "phase1" in result
        assert result["phase1"]["status"] == "feasible"

    def test_infeasible_returns_phase1_fail(self):
        A, b, c = _infeasible_toy()
        result = crossover.crossover_from_ipm(A, b, c)
        assert result["phase1"]["status"] == "infeasible"
        assert result["phase2"] is None


class TestMehrotraFallback:
    def test_pilot4_end_to_end_optimal(self):
        """crossover_fallback=True rescues stalled PILOT4 IPM."""
        from numerical_model import load_numeric_mps
        from mehrotra import solve_lp
        lp = load_numeric_mps(os.path.join(_ROOT, "data", "pilot4_plain.mps"))
        res = solve_lp(lp, tol=1e-7, max_iter=100, crossover_fallback=True)
        assert res.status == "optimal", f"{res.status}: {res.message}"
        assert abs(res.objective - (-2581.139258884)) < 1e-6
        assert res.rel_primal < 1e-6
        assert "crossover" in res.message.lower()

    def test_pilot4_default_no_crossover(self):
        """Default solve_lp (crossover_fallback=False) returns stalled."""
        from numerical_model import load_numeric_mps
        from mehrotra import solve_lp
        lp = load_numeric_mps(os.path.join(_ROOT, "data", "pilot4_plain.mps"))
        res = solve_lp(lp, tol=1e-7, max_iter=100)
        assert res.status == "stalled", f"expected stalled, got {res.status}"
        assert "crossover" not in res.message.lower()

    def test_pilot4_high_merit_skips_crossover(self):
        """When best_merit > CROSSOVER_MERIT_RATIO * tol, crossover is
        never called even with crossover_fallback=True."""
        from unittest.mock import patch
        from numerical_model import load_numeric_mps
        from mehrotra import solve_lp
        lp = load_numeric_mps(os.path.join(_ROOT, "data", "pilot4_plain.mps"))
        # Force early stall by raising mu_floor well above typical mu:
        # IPM stops almost immediately and best_merit stays large.
        with patch("crossover.crossover_from_ipm") as mock_crossover:
            res = solve_lp(lp, tol=1e-7, max_iter=100,
                           crossover_fallback=True, mu_floor=1e6)
            assert res.status in ("stalled", "numerical_tail")
            mock_crossover.assert_not_called()

    def test_small_lp_no_crossover_needed(self):
        from numerical_model import NumericalLP
        from mehrotra import solve_lp
        lp = NumericalLP(
            name="TINY", objective_name="COST",
            A=np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 0.0], [3.0, 1.0, 0.0]]),
            b=np.array([10.0, 8.0, 9.0]),
            c=np.array([-1.0, -1.0, 0.0]),
            lower_bounds=np.zeros(3), upper_bounds=np.full(3, np.inf),
            row_types=("E", "L", "L"),
            var_names=("x1", "x2", "x3"),
            row_names=("E1", "L1", "L2"),
        )
        res = solve_lp(lp, tol=1e-8, max_iter=100)
        assert res.status == "optimal"
        assert abs(res.objective - (-5.0)) < 1e-6
        assert "crossover" not in res.message.lower()
