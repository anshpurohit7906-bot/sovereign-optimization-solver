"""PDHG with fixed-period restart for mixed equality/inequality LPs.

Extends the vanilla mixed PDHG with a conservative restart mechanism:
when iteration % restart_period == 0, the extrapolation state is reset
(x_bar = x). This dampens oscillations without changing step sizes or LP formulation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from numerical_model import NumericalLP, load_numeric_mps, validate_numeric_lp
from experiment.pdhg.pdhg_mixed import _spectral_norm, _project_box, _partition_rows, _diagnostics, MixedPDHGResult


class MixedPDHGRestartError(ValueError):
    """Raised when a numerical LP is outside this solver's supported form."""


@dataclass(frozen=True)
class PDHGRestartResult:
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
    restarts_triggered: int

    @property
    def primal_feasibility(self) -> float:
        return max(self.equality_residual, self.inequality_violation)


def pdhg_mixed_restart(
    lp: NumericalLP,
    *,
    max_iter: int = 200_000,
    tol: float = 1e-7,
    check_every: int = 250,
    restart_period: int = 10_000,
    verbose: bool = False,
) -> PDHGRestartResult:
    """Solve NumericalLP with E/L rows using PDHG with fixed-period restart.

    Restart: every `restart_period` iterations, reset extrapolation (x_bar = x).
    This is conservative - it doesn't reset duals or change step sizes.
    """
    if max_iter <= 0 or check_every <= 0 or tol <= 0.0 or restart_period <= 0:
        raise MixedPDHGRestartError("max_iter, check_every, tol, and restart_period must be positive")

    validate_numeric_lp(lp)
    eq_rows, ub_rows = _partition_rows(lp)
    A_eq, b_eq = lp.A[eq_rows], lp.b[eq_rows]
    A_ub, b_ub = lp.A[ub_rows], lp.b[ub_rows]
    lower, upper = lp.lower_bounds, lp.upper_bounds

    norm_A = _spectral_norm(lp.A)
    tau = sigma = 1.0 if norm_A == 0.0 else 0.9 / norm_A

    x = _project_box(np.zeros(lp.num_vars, dtype=np.float64), lower, upper)
    x_bar = x.copy()
    y_eq = np.zeros(A_eq.shape[0], dtype=np.float64)
    y_ub = np.zeros(A_ub.shape[0], dtype=np.float64)

    diagnostics = (float("inf"),) * 4
    restarts_triggered = 0

    for iteration in range(1, max_iter + 1):
        # FIXED-PERIOD RESTART: reset extrapolation momentum
        if iteration % restart_period == 0:
            x_bar = x.copy()
            restarts_triggered += 1
            if verbose:
                print(f"  [restart] iteration {iteration}: reset x_bar = x")

        # Dual-first updates (equality: free; inequality: nonnegative)
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
                    f"complementarity={complementarity:.2e}  restarts={restarts_triggered}"
                )
            if max(*diagnostics) <= tol:
                return PDHGRestartResult(
                    x, y_eq, y_ub, iteration, True, "optimal", float(lp.c @ x),
                    *diagnostics, restarts_triggered
                )

    equality, inequality, dual, complementarity = diagnostics
    return PDHGRestartResult(
        x, y_eq, y_ub, max_iter, False, "iteration_limit", float(lp.c @ x),
        *diagnostics, restarts_triggered
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Solve E/L LP with PDHG + fixed-period restart.")
    default_sc205 = Path(__file__).resolve().parents[1] / "data" / "sc205.mps"
    parser.add_argument("mps_file", nargs="?", type=Path, default=default_sc205)
    parser.add_argument("--max-iter", type=int, default=200_000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--restart-period", type=int, default=10_000,
                        help="Iterations between extrapolation resets (default: 10000)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    lp = load_numeric_mps(args.mps_file)
    start = time.perf_counter()
    result = pdhg_mixed_restart(
        lp,
        max_iter=args.max_iter,
        tol=args.tol,
        check_every=args.check_every,
        restart_period=args.restart_period,
        verbose=args.verbose,
    )
    runtime = time.perf_counter() - start

    print(f"Problem:              {lp.name}")
    print(f"Objective:            {result.objective:.10g}")
    print(f"Iterations:           {result.iterations}")
    print(f"Restarts triggered:   {result.restarts_triggered}")
    print(f"Primal feasibility:   {result.primal_feasibility:.3e}")
    print(f"  equality residual:  {result.equality_residual:.3e}")
    print(f"  inequality residual:{result.inequality_violation:.3e}")
    print(f"Dual feasibility:     {result.dual_feasibility:.3e}")
    print(f"Complementarity:      {result.complementarity:.3e}")
    print(f"Solver status:        {result.status}")
    print(f"Runtime:              {runtime:.3f} s")


if __name__ == "__main__":
    main()