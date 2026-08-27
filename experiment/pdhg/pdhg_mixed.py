"""Mixed equality/inequality PDHG solver for parsed MPS linear programs.

The supported primal form is::

    minimize    c^T x
    subject to  A_E x = b_E
                A_L x <= b_L
                lower_bounds <= x <= upper_bounds

Equality multipliers are free.  Multipliers for ``L`` rows are nonnegative.
This module intentionally uses NumPy only: external LP solvers are not used as
part of the solving path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from numerical_model import NumericalLP, load_numeric_mps, validate_numeric_lp


class MixedPDHGError(ValueError):
    """Raised when a numerical LP is outside this solver's supported form."""


@dataclass(frozen=True)
class MixedPDHGResult:
    """Solution and KKT-style termination diagnostics."""

    x: np.ndarray
    y_eq: np.ndarray
    y_ub: np.ndarray
    iterations: int
    converged: bool
    status: str
    objective: float
    equality_residual: float
    inequality_violation: float
    dual_feasibility: float
    complementarity: float

    @property
    def primal_feasibility(self) -> float:
        """Largest primal feasibility residual across both row classes."""
        return max(self.equality_residual, self.inequality_violation)


def _spectral_norm(A: np.ndarray, *, max_iter: int = 200, tol: float = 1e-12) -> float:
    """Estimate ``||A||_2`` using a deterministic power iteration."""
    if A.size == 0:
        return 0.0

    v = np.ones(A.shape[1], dtype=np.float64)
    v /= np.linalg.norm(v)
    previous = 0.0
    for _ in range(max_iter):
        w = A.T @ (A @ v)
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            return 0.0
        v = w / norm_w
        estimate = float(np.linalg.norm(A @ v))
        if abs(estimate - previous) <= tol * max(1.0, estimate):
            return estimate
        previous = estimate
    return previous


def _project_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Project onto finite or infinite componentwise variable bounds."""
    return np.minimum(np.maximum(x, lower), upper)


def _partition_rows(lp: NumericalLP) -> tuple[np.ndarray, np.ndarray]:
    """Return indices for equality and <= rows, rejecting unsupported row types."""
    row_types = np.asarray(lp.row_types)
    unsupported = set(row_types).difference({"E", "L"})
    if unsupported:
        raise MixedPDHGError(
            "mixed PDHG currently supports only E and L rows; found "
            f"{sorted(unsupported)}"
        )
    return np.flatnonzero(row_types == "E"), np.flatnonzero(row_types == "L")


def _diagnostics(
    c: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    x: np.ndarray,
    y_eq: np.ndarray,
    y_ub: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute primal, stationarity, and complementarity KKT residuals."""
    equality_residual = float(np.linalg.norm(A_eq @ x - b_eq, ord=np.inf)) if A_eq.shape[0] else 0.0
    inequality_slack = A_ub @ x - b_ub
    inequality_violation = float(np.linalg.norm(np.maximum(inequality_slack, 0.0), ord=np.inf)) if A_ub.shape[0] else 0.0

    # This projected-gradient residual is zero iff stationarity holds with
    # the box normal cone, including finite, free, and one-sided bounds.
    lagrangian_gradient = c + A_eq.T @ y_eq + A_ub.T @ y_ub
    dual_feasibility = float(np.linalg.norm(x - _project_box(x - lagrangian_gradient, lower, upper), ord=np.inf))
    complementarity = float(np.linalg.norm(y_ub * inequality_slack, ord=np.inf)) if A_ub.shape[0] else 0.0
    return equality_residual, inequality_violation, dual_feasibility, complementarity


def pdhg_mixed(
    lp: NumericalLP,
    *,
    max_iter: int = 200_000,
    tol: float = 1e-7,
    check_every: int = 250,
    verbose: bool = False,
) -> MixedPDHGResult:
    """Solve a NumericalLP containing E and L rows using vanilla PDHG.

    ``converged`` is true only when equality feasibility, inequality
    feasibility, box-aware stationarity, and complementarity are each below
    ``tol``.  The routine makes no claim of convergence otherwise.
    """
    if max_iter <= 0 or check_every <= 0 or tol <= 0.0:
        raise MixedPDHGError("max_iter, check_every, and tol must be positive")

    validate_numeric_lp(lp)
    eq_rows, ub_rows = _partition_rows(lp)
    A_eq, b_eq = lp.A[eq_rows], lp.b[eq_rows]
    A_ub, b_ub = lp.A[ub_rows], lp.b[ub_rows]
    lower, upper = lp.lower_bounds, lp.upper_bounds

    norm_A = _spectral_norm(lp.A)
    # A zero constraint matrix has no coupling; these finite values avoid a
    # division by zero while preserving valid proximal updates.
    tau = sigma = 1.0 if norm_A == 0.0 else 0.9 / norm_A

    x = _project_box(np.zeros(lp.num_vars, dtype=np.float64), lower, upper)
    x_bar = x.copy()
    y_eq = np.zeros(A_eq.shape[0], dtype=np.float64)
    y_ub = np.zeros(A_ub.shape[0], dtype=np.float64)

    diagnostics = (float("inf"),) * 4
    for iteration in range(1, max_iter + 1):
        # Equality duals are unconstrained; <= duals belong to R_+.  The
        # dual-first Chambolle--Pock ordering is important for equality rows:
        # the primal-first explicit ordering can cycle on a pure equality LP.
        y_eq = y_eq + sigma * (A_eq @ x_bar - b_eq)
        y_ub = np.maximum(y_ub + sigma * (A_ub @ x_bar - b_ub), 0.0)

        previous_x = x
        gradient = lp.c + A_eq.T @ y_eq + A_ub.T @ y_ub
        x = _project_box(x - tau * gradient, lower, upper)
        x_bar = 2.0 * x - previous_x

        if iteration % check_every == 0 or iteration == max_iter:
            diagnostics = _diagnostics(
                lp.c, A_eq, b_eq, A_ub, b_ub, lower, upper, x, y_eq, y_ub
            )
            equality, inequality, dual, complementarity = diagnostics
            if verbose:
                print(
                    f"iter {iteration:6d}  objective={lp.c @ x: .9g}  "
                    f"primal={max(equality, inequality):.2e}  dual={dual:.2e}  "
                    f"complementarity={complementarity:.2e}"
                )
            if max(*diagnostics) <= tol:
                return MixedPDHGResult(
                    x, y_eq, y_ub, iteration, True, "optimal", float(lp.c @ x), *diagnostics
                )

    equality, inequality, dual, complementarity = diagnostics
    return MixedPDHGResult(
        x, y_eq, y_ub, max_iter, False, "iteration_limit", float(lp.c @ x), *diagnostics
    )


def _synthetic_sign_test() -> None:
    """Verify free equality and nonnegative inequality dual signs on a tiny LP.

    minimize x subject to x = 2 and x <= 3.  The equality multiplier must be
    negative at the solution (approximately -1), while the slack inequality
    multiplier must stay zero.
    """
    lp = NumericalLP(
        name="mixed-sign-test",
        objective_name="obj",
        A=np.array([[1.0], [1.0]]),
        b=np.array([2.0, 3.0]),
        c=np.array([1.0]),
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([np.inf]),
        row_types=("E", "L"),
        var_names=("x",),
        row_names=("balance", "capacity"),
    )
    result = pdhg_mixed(lp, max_iter=50_000, tol=1e-8, check_every=25)
    if not result.converged:
        raise AssertionError(f"synthetic mixed PDHG did not converge: {result}")
    if not np.isclose(result.x[0], 2.0, atol=1e-6):
        raise AssertionError(f"wrong primal solution: {result.x}")
    if not result.y_eq[0] < -0.9:
        raise AssertionError(f"equality dual should be free and negative: {result.y_eq}")
    if not np.isclose(result.y_ub[0], 0.0, atol=1e-7):
        raise AssertionError(f"slack inequality dual should be zero: {result.y_ub}")


def main(argv: Optional[list[str]] = None) -> None:
    """Run the synthetic sign test and solve AFIRO (or an explicit MPS path)."""
    parser = argparse.ArgumentParser(description="Solve an E/L-bounded LP using mixed PDHG.")
    default_afiro = Path(__file__).resolve().parents[1] / "data" / "afiro.mps"
    parser.add_argument("mps_file", nargs="?", type=Path, default=default_afiro)
    parser.add_argument("--max-iter", type=int, default=200_000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _synthetic_sign_test()
    print("Synthetic mixed-sign test: PASS")

    lp = load_numeric_mps(args.mps_file)
    result = pdhg_mixed(lp, max_iter=args.max_iter, tol=args.tol, verbose=args.verbose)
    print(f"Problem:              {lp.name}")
    print(f"Objective:            {result.objective:.10g}")
    print(f"Iterations:           {result.iterations}")
    print(f"Primal feasibility:   {result.primal_feasibility:.3e}")
    print(f"  equality residual:  {result.equality_residual:.3e}")
    print(f"  inequality residual:{result.inequality_violation:.3e}")
    print(f"Dual feasibility:     {result.dual_feasibility:.3e}")
    print(f"Complementarity:      {result.complementarity:.3e}")
    print(f"Solver status:        {result.status}")


if __name__ == "__main__":
    main()
