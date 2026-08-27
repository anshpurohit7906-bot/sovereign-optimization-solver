"""PDHG with Barzilai-Borwein/local-spectral step size adaptation (src/pdhg_bb.py).

Diagonal preconditioning base + BB scalar multiplier on top.
Preserves tau_i * sigma_j * A_ij^2 <= theta^2 invariant exactly.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from numerical_model import NumericalLP, load_numeric_mps, validate_numeric_lp


class BBPDHGError(ValueError):
    """Raised when a numerical LP is outside this solver's supported form."""


@dataclass(frozen=True)
class BBPDHGResult:
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
    stationarity: float
    dual_cone_violation: float
    complementarity: float
    final_tau: np.ndarray
    final_sigma: np.ndarray
    bb_alpha: float

    @property
    def primal_feasibility(self) -> float:
        return max(self.equality_residual, self.inequality_violation)


def _project_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lower), upper)


def _partition_rows(lp: NumericalLP) -> tuple[np.ndarray, np.ndarray]:
    row_types = np.asarray(lp.row_types)
    unsupported = set(row_types).difference({"E", "L"})
    if unsupported:
        raise BBPDHGError(
            "BB PDHG supports only E and L rows; found "
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
) -> tuple[float, float, float, float, float]:
    equality_residual = float(np.linalg.norm(A_eq @ x - b_eq, ord=np.inf)) if A_eq.shape[0] else 0.0
    inequality_slack = A_ub @ x - b_ub
    inequality_violation = float(np.linalg.norm(np.maximum(inequality_slack, 0.0), ord=np.inf)) if A_ub.shape[0] else 0.0
    lagrangian_gradient = c + A_eq.T @ y_eq + A_ub.T @ y_ub
    stationarity = float(np.linalg.norm(x - _project_box(x - lagrangian_gradient, lower, upper), ord=np.inf))
    # Dual cone feasibility: y_L >= 0
    dual_cone_violation = float(np.linalg.norm(np.minimum(y_ub, 0.0), ord=np.inf)) if y_ub.size else 0.0
    complementarity = float(np.linalg.norm(y_ub * inequality_slack, ord=np.inf)) if A_ub.shape[0] else 0.0
    return equality_residual, inequality_violation, stationarity, dual_cone_violation, complementarity


def _compute_bb_alpha(
    x: np.ndarray,
    x_prev: np.ndarray,
    grad: np.ndarray,
    grad_prev: np.ndarray,
    r: np.ndarray,
    r_prev: np.ndarray,
) -> float:
    """Compute BB scalar step size multiplier.

    Tries multiple BB estimates:
    1. Primal BB1: <s, s> / <s, y> where s = x - x_prev, y = grad - grad_prev
    2. Dual BB1: <r, r> / <r, A s> where r = primal residual
    Returns median of valid estimates, or 1.0 if none valid.
    """
    s = x - x_prev
    y = grad - grad_prev

    estimates = []

    # Primal BB1
    s_dot_s = float(np.dot(s, s))
    s_dot_y = float(np.dot(s, y))
    if s_dot_s > 1e-16 and s_dot_y > 1e-16:
        alpha1 = s_dot_s / s_dot_y
        if 0.1 <= alpha1 <= 10.0:
            estimates.append(alpha1)

    # Dual BB1 (if residual available)
    if r_prev is not None and r is not None:
        r_change = r - r_prev
        # A s ≈ r_change, so dual BB: <r, r> / <r, r_change>
        r_dot_r = float(np.dot(r, r))
        r_dot_rc = float(np.dot(r, r_change))
        if r_dot_r > 1e-16 and r_dot_rc > 1e-16:
            alpha2 = r_dot_r / r_dot_rc
            if 0.1 <= alpha2 <= 10.0:
                estimates.append(alpha2)

    if not estimates:
        return 1.0

    # Use median for robustness
    return float(np.median(estimates))


def pdhg_bb(
    lp: NumericalLP,
    *,
    max_iter: int = 200_000,
    tol: float = 1e-7,
    check_every: int = 250,
    theta: float = 0.9,
    bb_every: int = 50,
    ema_decay: float = 0.9,
    verbose: bool = False,
    checkpoints: tuple[int, ...] = (1000, 5000, 10000, 20000, 50000, 100000, 200000),
) -> BBPDHGResult:
    """Solve NumericalLP with diagonal preconditioning + BB scalar adaptation.

    Base step sizes (fixed ratios from diagonal preconditioning):
      tau_base[i] = theta / sum_j |A_ij|
      sigma_base[j] = theta / sum_i |A_ij|

    BB scalar multiplier alpha (reciprocal for dual):
      tau = tau_base * alpha
      sigma = sigma_base / alpha

    Every bb_every iterations:
      alpha_BB1 = <s, s> / <s, y>  where s = x - x_prev, y = grad - grad_prev
      alpha = ema_decay * alpha + (1 - ema_decay) * alpha_BB1 (if valid)
      alpha clipped to [alpha_min, alpha_max]

    This preserves tau_i * sigma_j * A_ij^2 = tau_base_i * sigma_base_j * A_ij^2 <= theta^2 exactly.
    """
    if max_iter <= 0 or check_every <= 0 or tol <= 0.0 or theta <= 0.0 or theta >= 1.0 or bb_every <= 0:
        raise BBPDHGError("max_iter, check_every, tol, theta, bb_every must be positive; theta in (0,1)")

    validate_numeric_lp(lp)
    eq_rows, ub_rows = _partition_rows(lp)
    A_eq, b_eq = lp.A[eq_rows], lp.b[eq_rows]
    A_ub, b_ub = lp.A[ub_rows], lp.b[ub_rows]
    A_all = lp.A
    lower, upper = lp.lower_bounds, lp.upper_bounds

    # Diagonal base step sizes (fixed ratios)
    col_sums = np.sum(np.abs(A_all), axis=0)
    row_sums = np.sum(np.abs(A_all), axis=1)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)

    tau_base = theta / col_sums          # shape (n,)
    sigma_base = theta / row_sums        # shape (m,)

    # BB scalar multiplier
    alpha = 1.0
    alpha_min = 0.3
    alpha_max = 3.0

    x = _project_box(np.zeros(lp.num_vars, dtype=np.float64), lower, upper)
    x_bar = x.copy()
    x_prev = x.copy()
    y_eq = np.zeros(A_eq.shape[0], dtype=np.float64)
    y_ub = np.zeros(A_ub.shape[0], dtype=np.float64)
    grad_prev = None

    diagnostics = (float("inf"),) * 5
    checkpoints_set = set(c for c in checkpoints if c <= max_iter)
    checkpoint_data = []

    for iteration in range(1, max_iter + 1):
        tau = tau_base * alpha
        sigma = sigma_base / alpha

        # Dual-first updates
        sigma_eq = sigma[eq_rows]
        sigma_ub = sigma[ub_rows]

        y_eq = y_eq + sigma_eq * (A_eq @ x_bar - b_eq)
        y_ub = np.maximum(y_ub + sigma_ub * (A_ub @ x_bar - b_ub), 0.0)

        previous_x = x
        grad = lp.c + A_eq.T @ y_eq + A_ub.T @ y_ub
        x = _project_box(x - tau * grad, lower, upper)
        x_bar = 2.0 * x - previous_x

        # BB adaptation
        if iteration % bb_every == 0 and grad_prev is not None:
            # Compute current primal residual
            r_eq = A_eq @ x - b_eq if A_eq.shape[0] else np.array([])
            slack = A_ub @ x - b_ub
            r_ub = np.maximum(slack, 0.0) if A_ub.shape[0] else np.array([])
            r = np.concatenate([r_eq, r_ub]) if (r_eq.size or r_ub.size) else None
            
            r_eq_prev = A_eq @ x_prev - b_eq if A_eq.shape[0] else np.array([])
            slack_prev = A_ub @ x_prev - b_ub
            r_ub_prev = np.maximum(slack_prev, 0.0) if A_ub.shape[0] else np.array([])
            r_prev_vec = np.concatenate([r_eq_prev, r_ub_prev]) if (r_eq_prev.size or r_ub_prev.size) else None

            alpha_bb = _compute_bb_alpha(x, x_prev, grad, grad_prev, r, r_prev_vec)
            if alpha_bb != 1.0:
                alpha = float(np.clip(ema_decay * alpha + (1.0 - ema_decay) * alpha_bb, alpha_min, alpha_max))
                if verbose:
                    print(f"  [BB] iter {iteration}: alpha_bb={alpha_bb:.3f}, alpha={alpha:.3f}")

        if iteration in checkpoints_set or iteration % check_every == 0 or iteration == max_iter:
            diagnostics = _diagnostics(
                lp.c, A_eq, b_eq, A_ub, b_ub, lower, upper, x, y_eq, y_ub
            )
            equality, inequality, stationarity, dual_cone, complementarity = diagnostics
            objective = float(lp.c @ x)
            if iteration in checkpoints_set:
                checkpoint_data.append((iteration, objective, stationarity))
            if verbose or iteration in checkpoints_set:
                print(
                    f"iter {iteration:6d}  objective={objective: .9g}  "
                    f"primal={max(equality, inequality):.2e}  stationarity={stationarity:.2e}  "
                    f"dual_cone={dual_cone:.2e}  complementarity={complementarity:.2e}  alpha={alpha:.3f}"
                )
            if max(*diagnostics) <= tol:
                if checkpoint_data:
                    _print_checkpoint_table(checkpoint_data)
                tau = tau_base * alpha
                sigma = sigma_base / alpha
                return BBPDHGResult(
                    x, y_eq, y_ub, iteration, True, "optimal", objective,
                    equality, inequality, stationarity, dual_cone, complementarity,
                    tau, sigma, alpha
                )

        x_prev = x.copy()
        grad_prev = grad.copy()

    equality, inequality, stationarity, dual_cone, complementarity = diagnostics
    objective = float(lp.c @ x)
    tau = tau_base * alpha
    sigma = sigma_base / alpha
    if checkpoint_data:
        _print_checkpoint_table(checkpoint_data)
    return BBPDHGResult(
        x, y_eq, y_ub, max_iter, False, "iteration_limit", objective,
        equality, inequality, stationarity, dual_cone, complementarity,
        tau, sigma, alpha
    )


def _print_checkpoint_table(data: list[tuple[int, float, float]]) -> None:
    print("\n=== Checkpoint Summary ===")
    print(f"{'Iteration':>10}  {'Objective':>14}  {'Stationarity':>12}")
    print("-" * 42)
    for it, obj, stat in data:
        print(f"{it:>10d}  {obj:>14.6e}  {stat:>12.2e}")
    print("-" * 42)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Solve E/L LP with BB-adaptive PDHG.")
    default_sc205 = Path(__file__).resolve().parents[1] / "data" / "sc205.mps"
    parser.add_argument("mps_file", nargs="?", type=Path, default=default_sc205)
    parser.add_argument("--max-iter", type=int, default=200_000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--theta", type=float, default=0.9, help="Safety factor in (0,1), default 0.9")
    parser.add_argument("--bb-every", type=int, default=50, help="Iterations between BB adaptations")
    parser.add_argument("--ema-decay", type=float, default=0.9, help="EMA decay for alpha (0.9 default)")
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
    result = pdhg_bb(
        lp,
        max_iter=args.max_iter,
        tol=args.tol,
        check_every=args.check_every,
        theta=args.theta,
        bb_every=args.bb_every,
        ema_decay=args.ema_decay,
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
    print(f"Stationarity:         {result.stationarity:.3e}")
    print(f"Dual cone violation:  {result.dual_cone_violation:.3e}")
    print(f"Complementarity:      {result.complementarity:.3e}")
    print(f"Final tau range:      [{result.final_tau.min():.2e}, {result.final_tau.max():.2e}]")
    print(f"Final sigma range:    [{result.final_sigma.min():.2e}, {result.final_sigma.max():.2e}]")
    print(f"BB alpha:             {result.bb_alpha:.3f}")
    print(f"Solver status:        {result.status}")
    print(f"Runtime:              {runtime:.3f} s")


if __name__ == "__main__":
    main()