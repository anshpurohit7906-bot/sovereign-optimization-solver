"""Diagonally preconditioned PDHG for mixed equality/inequality LPs.

Replaces scalar step sizes with per-coordinate (diagonal) step sizes:
  τ_i = θ / sum_j |A_ij|   (primal, per-variable)
  σ_j = θ / sum_i |A_ij|   (dual, per-constraint)

Sufficient convergence condition: τ_i σ_j A_ij^2 ≤ θ² < 1 for all i,j
(standard PDHG diagonal preconditioning, e.g., Pock & Chambolle 2011).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from numerical_model import NumericalLP, load_numeric_mps, validate_numeric_lp


class PrecondPDHGError(ValueError):
    """Raised when a numerical LP is outside this solver's supported form."""


@dataclass(frozen=True)
class PrecondPDHGResult:
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
        return max(self.equality_residual, self.inequality_violation)


def _project_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lower), upper)


def _partition_rows(lp: NumericalLP) -> tuple[np.ndarray, np.ndarray]:
    row_types = np.asarray(lp.row_types)
    unsupported = set(row_types).difference({"E", "L"})
    if unsupported:
        raise PrecondPDHGError(
            "preconditioned PDHG supports only E and L rows; found "
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
    equality_residual = float(np.linalg.norm(A_eq @ x - b_eq, ord=np.inf)) if A_eq.shape[0] else 0.0
    inequality_slack = A_ub @ x - b_ub
    inequality_violation = float(np.linalg.norm(np.maximum(inequality_slack, 0.0), ord=np.inf)) if A_ub.shape[0] else 0.0
    lagrangian_gradient = c + A_eq.T @ y_eq + A_ub.T @ y_ub
    dual_feasibility = float(np.linalg.norm(x - _project_box(x - lagrangian_gradient, lower, upper), ord=np.inf))
    complementarity = float(np.linalg.norm(y_ub * inequality_slack, ord=np.inf)) if A_ub.shape[0] else 0.0
    return equality_residual, inequality_violation, dual_feasibility, complementarity


def pdhg_preconditioned(
    lp: NumericalLP,
    *,
    max_iter: int = 200_000,
    tol: float = 1e-7,
    check_every: int = 250,
    theta: float = 0.9,
    verbose: bool = False,
    checkpoints: tuple[int, ...] = (1000, 5000, 10000, 20000, 50000, 100000, 200000),
) -> PrecondPDHGResult:
    """Solve NumericalLP with diagonal preconditioned PDHG.

    Step sizes:
      tau[i] = theta / sum_j |A_ij|   (primal, per variable)
      sigma[j] = theta / sum_i |A_ij|  (dual, per constraint)

    Convergence: for all i,j, tau[i] * sigma[j] * A_ij^2 <= theta^2 < 1.
    This is the standard diagonal PDHG condition (Pock-Chambolle 2011).
    """
    if max_iter <= 0 or check_every <= 0 or tol <= 0.0 or theta <= 0.0 or theta >= 1.0:
        raise PrecondPDHGError("max_iter, check_every, tol must be positive; theta in (0,1)")

    validate_numeric_lp(lp)
    eq_rows, ub_rows = _partition_rows(lp)
    A_eq, b_eq = lp.A[eq_rows], lp.b[eq_rows]
    A_ub, b_ub = lp.A[ub_rows], lp.b[ub_rows]
    A_all = lp.A
    lower, upper = lp.lower_bounds, lp.upper_bounds

    # Diagonal preconditioning: column sums (primal) and row sums (dual)
    col_sums = np.sum(np.abs(A_all), axis=0)  # shape (n,)
    row_sums = np.sum(np.abs(A_all), axis=1)  # shape (m,)

    # Avoid division by zero
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)

    tau = theta / col_sums          # shape (n,) - primal step sizes
    sigma = theta / row_sums        # shape (m,) - dual step sizes

    x = _project_box(np.zeros(lp.num_vars, dtype=np.float64), lower, upper)
    x_bar = x.copy()
    y_eq = np.zeros(A_eq.shape[0], dtype=np.float64)
    y_ub = np.zeros(A_ub.shape[0], dtype=np.float64)

    diagnostics = (float("inf"),) * 4
    checkpoints_set = set(c for c in checkpoints if c <= max_iter)
    checkpoint_data = []

    for iteration in range(1, max_iter + 1):
        # Dual-first updates with per-coordinate step sizes
        sigma_eq = sigma[eq_rows]
        sigma_ub = sigma[ub_rows]

        y_eq = y_eq + sigma_eq * (A_eq @ x_bar - b_eq)
        y_ub = np.maximum(y_ub + sigma_ub * (A_ub @ x_bar - b_ub), 0.0)

        previous_x = x
        gradient = lp.c + A_eq.T @ y_eq + A_ub.T @ y_ub
        x = _project_box(x - tau * gradient, lower, upper)
        x_bar = 2.0 * x - previous_x

        if iteration in checkpoints_set or iteration % check_every == 0 or iteration == max_iter:
            diagnostics = _diagnostics(
                lp.c, A_eq, b_eq, A_ub, b_ub, lower, upper, x, y_eq, y_ub
            )
            equality, inequality, stationarity, complementarity = diagnostics
            objective = float(lp.c @ x)
            if iteration in checkpoints_set:
                checkpoint_data.append((iteration, objective, stationarity))
            if verbose or iteration in checkpoints_set:
                print(
                    f"iter {iteration:6d}  objective={objective: .9g}  "
                    f"primal={max(equality, inequality):.2e}  stationarity={stationarity:.2e}  "
                    f"complementarity={complementarity:.2e}"
                )
            if max(*diagnostics) <= tol:
                if checkpoint_data:
                    _print_checkpoint_table(checkpoint_data)
                return PrecondPDHGResult(
                    x, y_eq, y_ub, iteration, True, "optimal", objective,
                    *diagnostics
                )

    equality, inequality, stationarity, complementarity = diagnostics
    objective = float(lp.c @ x)
    if checkpoint_data:
        _print_checkpoint_table(checkpoint_data)
    return PrecondPDHGResult(
        x, y_eq, y_ub, max_iter, False, "iteration_limit", objective,
        *diagnostics
    )


def _print_checkpoint_table(data: list[tuple[int, float, float]]) -> None:
    """Print checkpoint table with iteration, objective, stationarity."""
    print("\n=== Checkpoint Summary ===")
    print(f"{'Iteration':>10}  {'Objective':>14}  {'Stationarity':>12}")
    print("-" * 42)
    for it, obj, stat in data:
        print(f"{it:>10d}  {obj:>14.6e}  {stat:>12.2e}")
    print("-" * 42)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Solve E/L LP with diagonally preconditioned PDHG.")
    default_sc205 = Path(__file__).resolve().parents[1] / "data" / "sc205.mps"
    parser.add_argument("mps_file", nargs="?", type=Path, default=default_sc205)
    parser.add_argument("--max-iter", type=int, default=200_000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--theta", type=float, default=0.9, help="Safety factor in (0,1), default 0.9")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoints", type=str, default="",
                        help="Comma-separated iteration checkpoints (e.g., 1000,5000,10000)")
    args = parser.parse_args(argv)

    if args.checkpoints:
        checkpoints = tuple(int(c.strip()) for c in args.checkpoints.split(","))
    else:
        checkpoints = (1000, 5000, 10000, 20000, 50000, 100000, 200000)

    lp = load_numeric_mps(args.mps_file)
    start = time.perf_counter()
    result = pdhg_preconditioned(
        lp,
        max_iter=args.max_iter,
        tol=args.tol,
        check_every=args.check_every,
        theta=args.theta,
        verbose=args.verbose,
        checkpoints=checkpoints,
    )
    runtime = time.perf_counter() - start

    print(f"Problem:              {lp.name}")
    print(f"Objective:            {result.objective:.10g}")
    print(f"Iterations:           {result.iterations}")
    print(f"Primal feasibility:   {result.primal_feasibility:.3e}")
    print(f"  equality residual:  {result.equality_residual:.3e}")
    print(f"  inequality residual:{result.inequality_violation:.3e}")
    print(f"Stationarity:         {result.dual_feasibility:.3e}")
    print(f"Complementarity:      {result.complementarity:.3e}")
    print(f"Solver status:        {result.status}")
    print(f"Runtime:              {runtime:.3f} s")


if __name__ == "__main__":
    main()