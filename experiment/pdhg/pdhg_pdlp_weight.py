"""PDHG with PDLP-style primal-weight adaptation.

Exact implementation of OR-Tools PDLP primal weight mechanism:
- w = primal_weight (ratio of primal to dual step size scaling)
- τ_j = s / (w * col_sum_j), σ_i = s * w / row_sum_i
- Update at major iterations: w_new = exp(α log(dual_dist/primal_dist) + (1-α) log w)
- Distance measured from last restart point (x_start, y_start)
- Nonzero tolerance safeguard: keep w if distances too small/large
- Smoothing: α = primal_weight_update_smoothing (default 0.5)

Preserves:
- Mixed E/L formulation (equality duals free, inequality duals ≥ 0)
- Box bounds via projection
- Diagonal preconditioning
- Current extrapolation
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from numerical_model import NumericalLP, load_numeric_mps, validate_numeric_lp


class PDLPWeightError(ValueError):
    """Raised when a numerical LP is outside this solver's supported form."""


@dataclass(frozen=True)
class PDLPWeightResult:
    """Solution and diagnostics."""
    x: np.ndarray
    y_eq: np.ndarray
    y_ub: np.ndarray
    iterations: int
    converged: bool
    status: str
    objective: float
    equality_residual: float
    inequality_violation: float
    dual_cone_violation: float
    stationarity: float
    complementarity: float
    primal_weight: float
    tau_min: float
    tau_max: float
    sigma_min: float
    sigma_max: float
    runtime_seconds: float

    @property
    def primal_feasibility(self) -> float:
        return max(self.equality_residual, self.inequality_violation)


def _project_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lower), upper)


def _partition_rows(lp: NumericalLP) -> tuple[np.ndarray, np.ndarray]:
    row_types = np.asarray(lp.row_types)
    unsupported = set(row_types).difference({"E", "L"})
    if unsupported:
        raise PDLPWeightError(
            "PDLP-weight PDHG supports only E and L rows; found "
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
) -> tuple[float, float, float, float, float, float]:
    """Return (eq_res, ineq_viol, dual_cone, stationarity, complementarity, obj)."""
    equality_residual = float(np.linalg.norm(A_eq @ x - b_eq, ord=np.inf)) if A_eq.shape[0] else 0.0
    inequality_slack = A_ub @ x - b_ub
    inequality_violation = float(np.linalg.norm(np.maximum(inequality_slack, 0.0), ord=np.inf)) if A_ub.shape[0] else 0.0
    dual_cone_violation = float(np.linalg.norm(np.minimum(y_ub, 0.0), ord=np.inf)) if y_ub.size else 0.0
    lagrangian_gradient = c + A_eq.T @ y_eq + A_ub.T @ y_ub
    stationarity = float(np.linalg.norm(x - _project_box(x - lagrangian_gradient, lower, upper), ord=np.inf))
    complementarity = float(np.linalg.norm(y_ub * inequality_slack, ord=np.inf)) if A_ub.shape[0] else 0.0
    objective = float(c @ x)
    return equality_residual, inequality_violation, dual_cone_violation, stationarity, complementarity, objective


def pdhg_pdlp_weight(
    lp: NumericalLP,
    *,
    max_iter: int = 200_000,
    tol: float = 1e-7,
    check_every: int = 250,
    theta: float = 0.9,
    major_iteration_frequency: int = 64,
    primal_weight_update_smoothing: float = 0.5,
    initial_primal_weight: Optional[float] = None,
    verbose: bool = False,
    checkpoints: tuple[int, ...] = (1000, 5000, 10000, 20000, 50000, 100000, 200000),
) -> PDLPWeightResult:
    """Solve with diagonal preconditioning + PDLP primal-weight adaptation.

    Step sizes:
      τ_j = theta / (w * col_sum_j)
      σ_i = theta * w / row_sum_i

    Primal weight updated every major_iteration_frequency iterations:
      w_new = exp(α log(||y - y_start||/||x - x_start||) + (1-α) log w)
      where α = primal_weight_update_smoothing, distances from last restart point.
    """
    if max_iter <= 0 or check_every <= 0 or tol <= 0.0 or theta <= 0.0 or theta >= 1.0:
        raise PDLPWeightError("max_iter, check_every, tol must be positive; theta in (0,1)")
    if major_iteration_frequency <= 0:
        raise PDLPWeightError("major_iteration_frequency must be positive")
    if not (0.0 <= primal_weight_update_smoothing <= 1.0):
        raise PDLPWeightError("primal_weight_update_smoothing must be in [0, 1]")

    validate_numeric_lp(lp)
    eq_rows, ub_rows = _partition_rows(lp)
    A_eq, b_eq = lp.A[eq_rows], lp.b[eq_rows]
    A_ub, b_ub = lp.A[ub_rows], lp.b[ub_rows]
    A_all = lp.A
    lower, upper = lp.lower_bounds, lp.upper_bounds

    # Diagonal preconditioning base
    col_sums = np.sum(np.abs(A_all), axis=0)
    row_sums = np.sum(np.abs(A_all), axis=1)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)

    # Initial primal weight (OR-Tools default: ||c|| / ||b||_2, fallback 1.0)
    if initial_primal_weight is not None and initial_primal_weight > 0:
        w = initial_primal_weight
    else:
        c_norm = np.linalg.norm(lp.c)
        b_norm = np.linalg.norm(lp.b)
        if b_norm > 0 and c_norm > 0:
            w = c_norm / b_norm
        else:
            w = 1.0

    alpha = primal_weight_update_smoothing
    k_nonzero_tol = 1.0e-10

    # Conservative weight bounds to prevent collapse/explosion
    w_min = 1e-4
    w_max = 1e4

    # Primal/dual variables
    x = _project_box(np.zeros(lp.num_vars, dtype=np.float64), lower, upper)
    x_bar = x.copy()
    y_eq = np.zeros(A_eq.shape[0], dtype=np.float64)
    y_ub = np.zeros(A_ub.shape[0], dtype=np.float64)

    # Restart points (for distance computation) - updated at major iterations
    x_start = x.copy()
    y_eq_start = y_eq.copy()
    y_ub_start = y_ub.copy()

    diagnostics = (float("inf"),) * 6
    checkpoints_set = set(c for c in checkpoints if c <= max_iter)
    checkpoint_data = []

    start_time = time.perf_counter()

    for iteration in range(1, max_iter + 1):
        # Current step sizes with primal weight
        tau = theta / (w * col_sums)
        sigma = theta * w / row_sums
        sigma_eq = sigma[eq_rows]
        sigma_ub = sigma[ub_rows]

        # Dual-first updates
        y_eq = y_eq + sigma_eq * (A_eq @ x_bar - b_eq)
        y_ub = np.maximum(y_ub + sigma_ub * (A_ub @ x_bar - b_ub), 0.0)

        previous_x = x
        grad = lp.c + A_eq.T @ y_eq + A_ub.T @ y_ub
        x = _project_box(x - tau * grad, lower, upper)
        x_bar = 2.0 * x - previous_x

        # Major iteration: update primal weight and restart points
        if iteration % major_iteration_frequency == 0:
            # Distance from restart points
            y = np.concatenate([y_eq, y_ub]) if (y_eq.size or y_ub.size) else np.array([])
            y_start = np.concatenate([y_eq_start, y_ub_start]) if (y_eq_start.size or y_ub_start.size) else np.array([])

            primal_dist = float(np.linalg.norm(x - x_start, ord=2))
            dual_dist = float(np.linalg.norm(y - y_start, ord=2))

            # Nonzero tolerance safeguard
            if (primal_dist > k_nonzero_tol and primal_dist < 1.0 / k_nonzero_tol and
                dual_dist > k_nonzero_tol and dual_dist < 1.0 / k_nonzero_tol):
                unsmoothed = dual_dist / primal_dist
                if unsmoothed > 0:
                    # EMA in log space: w_new = exp(α log(unsmoothed) + (1-α) log w)
                    w = float(np.exp(alpha * np.log(unsmoothed) + (1.0 - alpha) * np.log(w)))
                    # Conservative bounds
                    w = float(np.clip(w, w_min, w_max))
                    if verbose:
                        print(f"  [weight] iter {iteration}: w={w:.3e}, primal_dist={primal_dist:.2e}, dual_dist={dual_dist:.2e}")

            # Update restart points
            x_start = x.copy()
            y_eq_start = y_eq.copy()
            y_ub_start = y_ub.copy()

        if iteration in checkpoints_set or iteration % check_every == 0 or iteration == max_iter:
            tau = theta / (w * col_sums)
            sigma = theta * w / row_sums
            diagnostics = _diagnostics(
                lp.c, A_eq, b_eq, A_ub, b_ub, lower, upper, x, y_eq, y_ub
            )
            eq_res, ineq_viol, dual_cone, stationarity, complementarity, objective = diagnostics

            if iteration in checkpoints_set:
                checkpoint_data.append((iteration, objective, stationarity, w))

            if verbose or iteration in checkpoints_set:
                print(
                    f"iter {iteration:6d}  obj={objective: .6e}  "
                    f"primal={max(eq_res, ineq_viol):.2e}  stationarity={stationarity:.2e}  "
                    f"comp={complementarity:.2e}  w={w:.3e}  "
                    f"tau in [{tau.min():.2e},{tau.max():.2e}]  "
                    f"sigma in [{sigma.min():.2e},{sigma.max():.2e}]"
                )

            if max(eq_res, ineq_viol, stationarity, dual_cone, complementarity) <= tol:
                tau = theta / (w * col_sums)
                sigma = theta * w / row_sums
                if checkpoint_data:
                    _print_checkpoint_table(checkpoint_data)
                return PDLPWeightResult(
                    x, y_eq, y_ub, iteration, True, "optimal", objective,
                    eq_res, ineq_viol, dual_cone, stationarity, complementarity,
                    w, float(tau.min()), float(tau.max()), float(sigma.min()), float(sigma.max()),
                    time.perf_counter() - start_time
                )

    tau = theta / (w * col_sums)
    sigma = theta * w / row_sums
    eq_res, ineq_viol, dual_cone, stationarity, complementarity, objective = diagnostics
    if checkpoint_data:
        _print_checkpoint_table(checkpoint_data)
    return PDLPWeightResult(
        x, y_eq, y_ub, max_iter, False, "iteration_limit", objective,
        eq_res, ineq_viol, dual_cone, stationarity, complementarity,
        w, float(tau.min()), float(tau.max()), float(sigma.min()), float(sigma.max()),
        time.perf_counter() - start_time
    )


def _print_checkpoint_table(data: list[tuple[int, float, float, float]]) -> None:
    print("\n=== Checkpoint Summary ===")
    print(f"{'Iter':>6}  {'Objective':>12}  {'Stationarity':>12}  {'Primal Weight':>14}")
    print("-" * 50)
    for it, obj, stat, w in data:
        print(f"{it:>6d}  {obj:>12.6e}  {stat:>12.2e}  {w:>14.6e}")
    print("-" * 50)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="PDHG with PDLP primal-weight adaptation.")
    default_sc205 = Path(__file__).resolve().parents[1] / "data" / "sc205.mps"
    parser.add_argument("mps_file", nargs="?", type=Path, default=default_sc205)
    parser.add_argument("--max-iter", type=int, default=200_000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--theta", type=float, default=0.9)
    parser.add_argument("--major-iter-freq", type=int, default=64,
                        help="Major iteration frequency for weight updates (default 64)")
    parser.add_argument("--smoothing", type=float, default=0.5,
                        help="Primal weight update smoothing α (default 0.5)")
    parser.add_argument("--initial-weight", type=float, default=None,
                        help="Initial primal weight (default: ||c||/||b||)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoints", type=str, default="",
                        help="Comma-separated iteration checkpoints")
    args = parser.parse_args(argv)

    if args.checkpoints:
        checkpoints = tuple(int(c.strip()) for c in args.checkpoints.split(","))
    else:
        checkpoints = (1000, 5000, 10000, 20000, 50000, 100000, 200000)

    lp = load_numeric_mps(args.mps_file)
    result = pdhg_pdlp_weight(
        lp,
        max_iter=args.max_iter,
        tol=args.tol,
        check_every=args.check_every,
        theta=args.theta,
        major_iteration_frequency=args.major_iter_freq,
        primal_weight_update_smoothing=args.smoothing,
        initial_primal_weight=args.initial_weight,
        verbose=args.verbose,
        checkpoints=checkpoints,
    )

    print(f"Problem:              {lp.name}")
    print(f"Iterations:           {result.iterations}")
    print(f"Objective:            {result.objective:.10g}")
    print(f"Primal feasibility:   {result.primal_feasibility:.3e}")
    print(f"  equality residual:  {result.equality_residual:.3e}")
    print(f"  inequality viol:    {result.inequality_violation:.3e}")
    print(f"Dual cone violation:  {result.dual_cone_violation:.3e}")
    print(f"Stationarity:         {result.stationarity:.3e}")
    print(f"Complementarity:      {result.complementarity:.3e}")
    print(f"Primal weight:        {result.primal_weight:.3e}")
    print(f"Primal step-size range:  [{result.tau_min:.2e}, {result.tau_max:.2e}]")
    print(f"Dual step-size range:    [{result.sigma_min:.2e}, {result.sigma_max:.2e}]")
    print(f"Status:               {result.status}")
    print(f"Runtime:              {result.runtime_seconds:.3f} s")


if __name__ == "__main__":
    main()