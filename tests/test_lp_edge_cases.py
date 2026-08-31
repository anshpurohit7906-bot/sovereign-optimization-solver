"""Automated LP correctness and edge-case test suite.

Covers all 18 SIH acceptance criteria using hand-verifiable LPs:
1. Simple equality LP
2. Mixed E/L/G LP
3. Minimization
4. Maximization
5. Lower bounded variable (LO)
6. Upper bounded variable (UP, free lower)
7. Free variable (FR)
8. Fixed variable (FX)
9. Boxed variable (Box)
10. Redundant constraints
11. Degenerate LP
12. Infeasible LP
13. Unbounded LP
14. Badly scaled LP
15. Sparse larger synthetic LP
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

# Ensure src and src/lp are importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import NumericalLP, load_numeric_mps
from lp.mehrotra import solve_lp, MehrotraResult, MehrotraError


@dataclass
class LPTestCase:
    name: str
    description: str
    build_lp: Callable[[], tuple[NumericalLP, bool]]  # returns (lp, maximize)
    expected_status: str  # "optimal" | "infeasible" | "unbounded" | "non_optimal"
    expected_objective: Optional[float] = None
    tol: float = 1e-7


# ---------------------------------------------------------------------------
# Test Case Definitions
# ---------------------------------------------------------------------------

def _case_simple_equality() -> tuple[NumericalLP, bool]:
    """min 2x1 + 3x2 + x3 s.t. x1 + x2 + x3 = 6, 2x1 + x2 = 4, x >= 0.
    Optimum: (2, 0, 4), obj = 8.0
    """
    return NumericalLP(
        name="SIMPLE_EQ",
        objective_name="COST",
        A=np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0]]),
        b=np.array([6.0, 4.0]),
        c=np.array([2.0, 3.0, 1.0]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.full(3, np.inf),
        row_types=("E", "E"),
        var_names=("x1", "x2", "x3"),
        row_names=("E1", "E2"),
    ), False


def _case_mixed_elg() -> tuple[NumericalLP, bool]:
    """min 3x1 + 2x2 + 4x3 s.t. x1+x2+x3=10 (E), x1+2x2<=12 (L), 2x1+x3>=6 (G), x>=0.
    Optimum: (4/3, 16/3, 10/3), obj = 28.0
    """
    return NumericalLP(
        name="MIXED_ELG",
        objective_name="COST",
        A=np.array([
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
        ]),
        b=np.array([10.0, 12.0, 6.0]),
        c=np.array([3.0, 2.0, 4.0]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.full(3, np.inf),
        row_types=("E", "L", "G"),
        var_names=("x1", "x2", "x3"),
        row_names=("E1", "L1", "G1"),
    ), False


def _case_minimization() -> tuple[NumericalLP, bool]:
    """min -2x1 - 5x2 s.t. x1+4x2<=24, 3x1+x2<=21, x1+x2<=9, x>=0.
    Optimum: (4, 5), obj = -33.0
    """
    return NumericalLP(
        name="MINIMIZATION",
        objective_name="COST",
        A=np.array([
            [1.0, 4.0],
            [3.0, 1.0],
            [1.0, 1.0],
        ]),
        b=np.array([24.0, 21.0, 9.0]),
        c=np.array([-2.0, -5.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("L", "L", "L"),
        var_names=("x1", "x2"),
        row_names=("L1", "L2", "L3"),
    ), False


def _case_maximization() -> tuple[NumericalLP, bool]:
    """max 5x1 + 4x2 s.t. 6x1+4x2<=24, x1+2x2<=6, -x1+x2<=1, x2<=2, x>=0.
    Optimum: (3, 1.5), obj = 21.0
    """
    return NumericalLP(
        name="MAXIMIZATION",
        objective_name="PROFIT",
        A=np.array([
            [6.0, 4.0],
            [1.0, 2.0],
            [-1.0, 1.0],
            [0.0, 1.0],
        ]),
        b=np.array([24.0, 6.0, 1.0, 2.0]),
        c=np.array([5.0, 4.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("L", "L", "L", "L"),
        var_names=("x1", "x2"),
        row_names=("L1", "L2", "L3", "L4"),
    ), True


def _case_lower_bound() -> tuple[NumericalLP, bool]:
    """min x1 + 2x2 s.t. x1 + x2 >= 10, x1 >= 3, x2 >= 4.
    Optimum: (6, 4), obj = 14.0
    """
    return NumericalLP(
        name="LOWER_BOUND",
        objective_name="COST",
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        c=np.array([1.0, 2.0]),
        lower_bounds=np.array([3.0, 4.0]),
        upper_bounds=np.full(2, np.inf),
        row_types=("G",),
        var_names=("x1", "x2"),
        row_names=("G1",),
    ), False


def _case_upper_bound() -> tuple[NumericalLP, bool]:
    """min 2x1 - 3x2 s.t. x1 - x2 >= -2, -inf < x1 <= 3, -inf < x2 <= 5.
    Optimum: (3, 5), obj = -9.0
    """
    return NumericalLP(
        name="UPPER_BOUND",
        objective_name="COST",
        A=np.array([[1.0, -1.0]]),
        b=np.array([-2.0]),
        c=np.array([2.0, -3.0]),
        lower_bounds=np.full(2, -np.inf),
        upper_bounds=np.array([3.0, 5.0]),
        row_types=("G",),
        var_names=("x1", "x2"),
        row_names=("G1",),
    ), False


def _case_free_variable() -> tuple[NumericalLP, bool]:
    """min 2x1 + x2 s.t. x1 + 2x2 = 4, x1 >= 0, -inf < x2 < +inf.
    Optimum: (0, 2), obj = 2.0
    """
    return NumericalLP(
        name="FREE_VAR",
        objective_name="COST",
        A=np.array([[1.0, 2.0]]),
        b=np.array([4.0]),
        c=np.array([2.0, 1.0]),
        lower_bounds=np.array([0.0, -np.inf]),
        upper_bounds=np.full(2, np.inf),
        row_types=("E",),
        var_names=("x1", "x2"),
        row_names=("E1",),
    ), False


def _case_fixed_variable() -> tuple[NumericalLP, bool]:
    """min 3x1 + 2x2 + 5x3 s.t. x1 + x2 + x3 >= 10, x1 >= 0, x2 >= 0, x3 == 4.
    Optimum: (0, 6, 4), obj = 32.0
    """
    return NumericalLP(
        name="FIXED_VAR",
        objective_name="COST",
        A=np.array([[1.0, 1.0, 1.0]]),
        b=np.array([10.0]),
        c=np.array([3.0, 2.0, 5.0]),
        lower_bounds=np.array([0.0, 0.0, 4.0]),
        upper_bounds=np.array([np.inf, np.inf, 4.0]),
        row_types=("G",),
        var_names=("x1", "x2", "x3"),
        row_names=("G1",),
    ), False


def _case_boxed_variable() -> tuple[NumericalLP, bool]:
    """min -3x1 - 2x2 s.t. x1 + x2 <= 7, 1 <= x1 <= 4, 2 <= x2 <= 5.
    Optimum: (4, 3), obj = -18.0
    """
    return NumericalLP(
        name="BOXED_VAR",
        objective_name="COST",
        A=np.array([[1.0, 1.0]]),
        b=np.array([7.0]),
        c=np.array([-3.0, -2.0]),
        lower_bounds=np.array([1.0, 2.0]),
        upper_bounds=np.array([4.0, 5.0]),
        row_types=("L",),
        var_names=("x1", "x2"),
        row_names=("L1",),
    ), False


def _case_redundant_constraints() -> tuple[NumericalLP, bool]:
    """min x1 + x2 s.t. x1 + x2 >= 4, 2x1 + 2x2 >= 8 (redundant), x1 + 2x2 >= 6, x >= 0.
    Optimum: (2, 2), obj = 4.0
    """
    return NumericalLP(
        name="REDUNDANT_ROWS",
        objective_name="COST",
        A=np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ]),
        b=np.array([4.0, 8.0, 6.0]),
        c=np.array([1.0, 1.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("G", "G", "G"),
        var_names=("x1", "x2"),
        row_names=("G1", "G2_redundant", "G3"),
    ), False


def _case_degenerate_lp() -> tuple[NumericalLP, bool]:
    """min x1 + x2 s.t. x1+x2>=2, 2x1+x2>=2, x1+2x2>=2, x1+x2<=4, x>=0.
    Multiple optima along x1+x2=2, obj = 2.0
    """
    return NumericalLP(
        name="DEGENERATE",
        objective_name="COST",
        A=np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [1.0, 1.0],
        ]),
        b=np.array([2.0, 2.0, 2.0, 4.0]),
        c=np.array([1.0, 1.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("G", "G", "G", "L"),
        var_names=("x1", "x2"),
        row_names=("G1", "G2", "G3", "L1"),
    ), False


def _case_infeasible_lp() -> tuple[NumericalLP, bool]:
    """min x1 + x2 s.t. x1 + x2 <= 2, x1 + x2 >= 5, x >= 0.
    Infeasible!
    """
    return NumericalLP(
        name="INFEASIBLE",
        objective_name="COST",
        A=np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ]),
        b=np.array([2.0, 5.0]),
        c=np.array([1.0, 1.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("L", "G"),
        var_names=("x1", "x2"),
        row_names=("L1", "G1"),
    ), False


def _case_unbounded_lp() -> tuple[NumericalLP, bool]:
    """min -2x1 + x2 s.t. -x1 + x2 <= 1, x1 - 2x2 <= 2, x >= 0.
    Unbounded (obj -> -inf)!
    """
    return NumericalLP(
        name="UNBOUNDED",
        objective_name="COST",
        A=np.array([
            [-1.0, 1.0],
            [1.0, -2.0],
        ]),
        b=np.array([1.0, 2.0]),
        c=np.array([-2.0, 1.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("L", "L"),
        var_names=("x1", "x2"),
        row_names=("L1", "L2"),
    ), False


def _case_badly_scaled_lp() -> tuple[NumericalLP, bool]:
    """min 10^4 x1 + x2 s.t. 10^4 x1 + x2 >= 10^4, x1 + 10^4 x2 >= 10^4, x >= 0.
    Optimum: x1 = 10000/10001, x2 = 10000/10001, obj = 10000.0
    """
    return NumericalLP(
        name="BADLY_SCALED",
        objective_name="COST",
        A=np.array([
            [1e4, 1.0],
            [1.0, 1e4],
        ]),
        b=np.array([1e4, 1e4]),
        c=np.array([1e4, 1.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("G", "G"),
        var_names=("x1", "x2"),
        row_names=("G1", "G2"),
    ), False


def _case_sparse_synthetic_lp() -> tuple[NumericalLP, bool]:
    """Tridiagonal synthetic LP with N=50 variables:
    min sum(x_i) s.t. x1 = 1, -x_{i-1} + 2x_i - x_{i+1} >= 0, x_N >= N, x >= 0.
    Solution: x_i = i, obj = 50*51/2 = 1275.0
    """
    N = 50
    m = N
    A = np.zeros((m, N))
    b = np.zeros(m)
    row_types = []

    # Row 0: x1 = 1 (E)
    A[0, 0] = 1.0
    b[0] = 1.0
    row_types.append("E")

    # Rows 1 to N-2: -x_{i-1} + 2x_i - x_{i+1} >= 0 (G)
    for i in range(1, N - 1):
        A[i, i - 1] = -1.0
        A[i, i] = 2.0
        A[i, i + 1] = -1.0
        b[i] = 0.0
        row_types.append("G")

    # Row N-1: x_N >= N (G)
    A[N - 1, N - 1] = 1.0
    b[N - 1] = float(N)
    row_types.append("G")

    c = np.ones(N)
    return NumericalLP(
        name=f"SPARSE_TRIDIAG_{N}",
        objective_name="COST",
        A=A,
        b=b,
        c=c,
        lower_bounds=np.zeros(N),
        upper_bounds=np.full(N, np.inf),
        row_types=tuple(row_types),
        var_names=tuple(f"x{i+1}" for i in range(N)),
        row_names=tuple(f"R{i+1}" for i in range(m)),
    ), False


def _case_negative_rhs() -> tuple[NumericalLP, bool]:
    """min 2x1 + x2 s.t. -x1 - 2x2 <= -4 (L, negative RHS), x1 + x2 <= 5 (L), x >= 0.
    -x1 - 2x2 <= -4 is x1 + 2x2 >= 4.
    Optimum: (0, 2), obj = 2.0
    """
    return NumericalLP(
        name="NEGATIVE_RHS",
        objective_name="COST",
        A=np.array([
            [-1.0, -2.0],
            [1.0, 1.0],
        ]),
        b=np.array([-4.0, 5.0]),
        c=np.array([2.0, 1.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("L", "L"),
        var_names=("x1", "x2"),
        row_names=("L_neg", "L_pos"),
    ), False


def _case_negative_box_bounds() -> tuple[NumericalLP, bool]:
    """min x1 + 2x2 s.t. x1 + x2 >= -6 (G), -5 <= x1 <= -2, -4 <= x2 <= -1 (negative box).
    Optimum: x1 = -2, x2 = -4 (since -2 + -4 = -6 >= -6), obj = -2 + 2(-4) = -10.0
    """
    return NumericalLP(
        name="NEGATIVE_BOX",
        objective_name="COST",
        A=np.array([[1.0, 1.0]]),
        b=np.array([-6.0]),
        c=np.array([1.0, 2.0]),
        lower_bounds=np.array([-5.0, -4.0]),
        upper_bounds=np.array([-2.0, -1.0]),
        row_types=("G",),
        var_names=("x1", "x2"),
        row_names=("G1",),
    ), False


def _case_negative_upper_bound() -> tuple[NumericalLP, bool]:
    """min 2x1 + 3x2 s.t. x1 + x2 >= -10 (G), -inf < x1 <= -3, -inf < x2 <= -2.
    To minimize 2x1 + 3x2 with x1 + x2 >= -10:
    Make x2 as negative as possible (x2 = -7), x1 = -3.
    Obj = 2(-3) + 3(-7) = -6 - 21 = -27.0
    """
    return NumericalLP(
        name="NEGATIVE_UP",
        objective_name="COST",
        A=np.array([[1.0, 1.0]]),
        b=np.array([-10.0]),
        c=np.array([2.0, 3.0]),
        lower_bounds=np.full(2, -np.inf),
        upper_bounds=np.array([-3.0, -2.0]),
        row_types=("G",),
        var_names=("x1", "x2"),
        row_names=("G1",),
    ), False


def _case_feasibility_zero_obj() -> tuple[NumericalLP, bool]:
    """min 0x1 + 0x2 s.t. x1 + 2x2 = 6 (E), 2x1 + x2 <= 6 (L), x >= 0.
    Feasible region: segment between (0, 3) and (2, 2).
    Obj = 0.0
    """
    return NumericalLP(
        name="ZERO_OBJ_FEAS",
        objective_name="ZERO",
        A=np.array([
            [1.0, 2.0],
            [2.0, 1.0],
        ]),
        b=np.array([6.0, 6.0]),
        c=np.array([0.0, 0.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("E", "L"),
        var_names=("x1", "x2"),
        row_names=("E1", "L1"),
    ), False


TEST_CASES = [
    LPTestCase("Simple Equality", "Pure equality constraints with nonnegative vars",
               _case_simple_equality, "optimal", 8.0),
    LPTestCase("Mixed E/L/G", "Mixed equality, <=, and >= constraints in one model",
               _case_mixed_elg, "optimal", 28.0),
    LPTestCase("Minimization", "Standard linear minimization with inequality constraints",
               _case_minimization, "optimal", -33.0),
    LPTestCase("Maximization", "Linear maximization (maximize=True)",
               _case_maximization, "optimal", 21.0),
    LPTestCase("Lower Bound", "Variables with non-zero lower bounds (LO)",
               _case_lower_bound, "optimal", 14.0),
    LPTestCase("Upper Bound", "Variables with upper bounds and free lower bounds (UP)",
               _case_upper_bound, "optimal", -9.0),
    LPTestCase("Free Variable", "Variables with (-inf, +inf) bounds (FR)",
               _case_free_variable, "optimal", 2.0),
    LPTestCase("Fixed Variable", "Variables fixed to exact value (FX: lb == ub)",
               _case_fixed_variable, "optimal", 32.0),
    LPTestCase("Boxed Variable", "Variables with finite [lb, ub] box bounds",
               _case_boxed_variable, "optimal", -18.0),
    LPTestCase("Negative RHS", "Constraints with negative right-hand-side values",
               _case_negative_rhs, "optimal", 2.0),
    LPTestCase("Negative Box", "Variables with strictly negative box bounds [L, U] < 0",
               _case_negative_box_bounds, "optimal", -10.0),
    LPTestCase("Negative UP", "Variables with negative upper bounds (-inf, U] with U < 0",
               _case_negative_upper_bound, "optimal", -27.0),
    LPTestCase("Zero Objective", "Pure LP feasibility problem with c = 0",
               _case_feasibility_zero_obj, "optimal", 0.0),
    LPTestCase("Redundant Rows", "Rank-deficient constraint matrix with redundant rows",
               _case_redundant_constraints, "optimal", 4.0),
    LPTestCase("Degenerate LP", "Primal degenerate LP with multiple/flat optima",
               _case_degenerate_lp, "optimal", 2.0),
    LPTestCase("Infeasible LP", "Infeasible constraints (no feasible point exists)",
               _case_infeasible_lp, "non_optimal", None),
    LPTestCase("Unbounded LP", "Unbounded objective along recession cone",
               _case_unbounded_lp, "non_optimal", None),
    LPTestCase("Badly Scaled LP", "Large dynamic range in coefficients (10^4)",
               _case_badly_scaled_lp, "optimal", 10000.0),
    LPTestCase("Sparse Synthetic", "50-variable sparse tridiagonal structured LP",
               _case_sparse_synthetic_lp, "optimal", 1275.0),
]


def run_all_tests(verbose: bool = True) -> tuple[int, int, list[dict]]:
    """Run all test cases and collect detailed results."""
    passed = 0
    failed = 0
    results = []

    for tc in TEST_CASES:
        lp, maximize = tc.build_lp()
        actual_status = "UNKNOWN"
        actual_obj = None
        error_msg = ""
        is_pass = False
        root_cause = ""

        try:
            res = solve_lp(lp, maximize=maximize, tol=tc.tol, max_iter=100)
            actual_status = res.status
            actual_obj = res.objective

            if tc.expected_status == "optimal":
                if res.status == "optimal":
                    if tc.expected_objective is not None:
                        obj_err = abs(actual_obj - tc.expected_objective)
                        rel_err = obj_err / (1.0 + abs(tc.expected_objective))
                        if rel_err <= 1e-4:
                            is_pass = True
                        else:
                            is_pass = False
                            root_cause = f"Objective mismatch: got {actual_obj:.6f}, expected {tc.expected_objective:.6f} (rel_err={rel_err:.2e})"
                    else:
                        is_pass = True
                else:
                    is_pass = False
                    root_cause = f"Solver returned '{res.status}' ({res.message})"
            elif tc.expected_status == "non_optimal":
                # For infeasible or unbounded, solver must NOT claim optimal
                if res.status in ("infeasible", "unbounded", "stalled", "max_iterations", "numerical_tail"):
                    is_pass = True
                elif res.status == "optimal":
                    is_pass = False
                    root_cause = "Solver falsely claimed 'optimal' on an infeasible/unbounded problem"
                else:
                    is_pass = True  # e.g. numerical_failure / stalled
        except MehrotraError as exc:
            actual_status = "MehrotraError"
            error_msg = str(exc)
            if tc.expected_status == "optimal":
                is_pass = False
                root_cause = f"Conversion error: {exc}"
            else:
                is_pass = True
        except Exception as exc:
            actual_status = type(exc).__name__
            error_msg = str(exc)
            is_pass = False
            root_cause = f"Unexpected exception: {exc}"

        if is_pass:
            passed += 1
        else:
            failed += 1

        results.append({
            "test": tc.name,
            "expected_status": tc.expected_status,
            "expected_obj": tc.expected_objective,
            "actual_status": actual_status,
            "actual_obj": actual_obj,
            "pass": is_pass,
            "root_cause": root_cause,
        })

        if verbose:
            tag = "PASS" if is_pass else "FAIL"
            print(f"[{tag:4s}] {tc.name:22s} | Exp: {tc.expected_status:11s} | Act: {actual_status:15s} | Obj: {str(actual_obj):15s} | {root_cause}")

    return passed, failed, results


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING AUTOMATED LP CORRECTNESS TEST SUITE")
    print("=" * 80)
    p, f, res = run_all_tests(verbose=True)
    print("=" * 80)
    print(f"SUMMARY: {p} PASSED, {f} FAILED out of {len(TEST_CASES)} tests")
    print("=" * 80)
