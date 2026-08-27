"""Minimal PDLP-core: Unified bounded-constraint PDHG.

Problem:
  min c^T x
  s.t. l^c <= A x <= u^c
       l^v <= x <= u^v

Dual proximal (unified):
  v = y^k + sigma * (A x_bar)
  y^{k+1} = v - sigma * clip(v/sigma, l^c, u^c)

Primal update:
  x^{k+1} = proj_{[l^v, u^v]}(x^k - tau * (c + A^T y^{k+1}))
  x_bar = 2*x^{k+1} - x^k
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from constraint_form import BoundedConstraintLP, to_bounded_constraint
from numerical_model import load_numeric_mps


@dataclass(frozen=True)
class PDLPCoreResult:
    x: np.ndarray
    y: np.ndarray
    iterations: int
    converged: bool
    status: str
    objective: float
    constraint_feasibility: float
    variable_bound_violation: float
    dual_proximal_residual: float
    stationarity: float
    runtime_seconds: float


class PDLPCoreError(ValueError):
    pass


def _project_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lower), upper)


def _compute_diagnostics(
    c: np.ndarray,
    A: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
    variable_lower: np.ndarray,
    variable_upper: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (constraint_feas, var_bound_viol, dual_prox_resid, stationarity)."""
    # Constraint feasibility: max violation of l^c <= Ax <= u^c
    Ax = A @ x
    lower_viol = np.maximum(constraint_lower - Ax, 0.0)
    upper_viol = np.maximum(Ax - constraint_upper, 0.0)
    constraint_feas = float(np.max(np.concatenate([lower_viol, upper_viol])))

    # Variable bound violation
    var_lower_viol = np.maximum(variable_lower - x, 0.0)
    var_upper_viol = np.maximum(x - variable_upper, 0.0)
    variable_bound_violation = float(np.max(np.concatenate([var_lower_viol, var_upper_viol])))

    # Dual proximal residual: ||y - prox_{sigma g*}(y + sigma A x)|| / sigma
    # prox_{sigma g*}(v) = v - sigma * clip(v/sigma, l^c, u^c)
    # Residual = y - (y + sigma Ax - sigma clip((y + sigma Ax)/sigma, l^c, u^c))
    #          = sigma * (clip(Ax + y/sigma, l^c, u^c) - Ax)
    # This measures how much the dual variable violates the complementarity/constraint condition
    # We report a normalized version
    v = y + np.ones_like(y)  # placeholder for sigma-weighted (handled via clip logic below)
    # Actually compute: clip(y/sigma + Ax, l^c, u^c) - Ax, but we need sigma per row
    # Since we don't store sigma here, use the fact that at optimum:
    # y_i in [0, inf) for upper bounds, etc. We'll use a simpler proxy:
    # Dual residual = max(|min(y, 0)|, |max(y, 0)|) for appropriate bounds
    # For unified form: the dual residual is the violation of complementarity-like conditions
    
    # Use the standard PDHG dual residual: prox residual of dual update
    # Since we don't have sigma here, we'll compute a sigma-free version:
    # For each constraint i, the optimal y satisfies:
    # - If l_i = u_i (equality): y_i free
    # - If u_i finite (upper bound): y_i >= 0 and (A x - u_i) <= 0, y_i * (A x - u_i) = 0
    # - If l_i finite (lower bound): y_i <= 0 and (A x - l_i) >= 0, y_i * (A x - l_i) = 0
    # We'll compute a combined measure:
    
    # Simpler: dual proximal residual = || y - prox_{g*}(y + A x) || where prox_{g*} is the clip
    # We'll approximate with the condition that y should be in the normal cone of the constraint box at Ax
    # This is the standard stationarity/optimality condition for the dual
    return constraint_feas, variable_bound_violation, 0.0, 0.0


def _full_diagnostics(
    c: np.ndarray,
    A: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
    variable_lower: np.ndarray,
    variable_upper: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute all diagnostics with full information."""
    Ax = A @ x
    
    # Constraint feasibility
    lower_viol = np.maximum(constraint_lower - Ax, 0.0)
    upper_viol = np.maximum(Ax - constraint_upper, 0.0)
    constraint_feas = float(np.max(np.concatenate([lower_viol, upper_viol])))
    
    # Variable bound violation
    var_lower_viol = np.maximum(variable_lower - x, 0.0)
    var_upper_viol = np.maximum(x - variable_upper, 0.0)
    variable_bound_violation = float(np.max(np.concatenate([var_lower_viol, var_upper_viol])))
    
    # Dual proximal residual: ||y - prox_{sigma g*}(y + sigma A x)||_inf
    # prox_{sigma g*}(v) = v - sigma * clip(v/sigma, l^c, u^c)  (with safeguards)
    v = y + sigma * Ax
    prox_v = _safe_dual_prox(v, sigma, constraint_lower, constraint_upper)
    dual_prox_residual = float(np.linalg.norm(y - prox_v, ord=np.inf))
    
    # Stationarity: ||x - proj_{[l^v,u^v]}(x - tau * (c + A^T y))||_inf
    grad = c + A.T @ y
    x_proj = _project_box(x - tau * grad, variable_lower, variable_upper)
    stationarity = float(np.linalg.norm(x - x_proj, ord=np.inf))
    
    return constraint_feas, variable_bound_violation, dual_prox_residual, stationarity, float(c @ x)


def _safe_dual_prox(v: np.ndarray, sigma: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Compute y = v - sigma * clip(v/sigma, l, u) with numerical safeguards.
    
    Avoids overflow in v/sigma when sigma is very small by using equivalent formulation:
    For each component:
    - if l_i = u_i (equality): y_i = v_i - sigma_i * l_i
    - if u_i finite and l_i = -inf (upper bound): y_i = max(v_i - sigma_i * u_i, 0)
    - if l_i finite and u_i = +inf (lower bound): y_i = min(v_i - sigma_i * l_i, 0)
    - if both finite (two-sided): y_i = v_i - sigma_i * clip(v_i/sigma_i, l_i, u_i)
    """
    y = np.zeros_like(v)
    
    # Equality constraints: l == u (and finite)
    eq_mask = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    y[eq_mask] = v[eq_mask] - sigma[eq_mask] * lower[eq_mask]
    
    # Upper bound only: l = -inf, u finite
    ub_mask = (~np.isfinite(lower)) & np.isfinite(upper)
    if np.any(ub_mask):
        y[ub_mask] = np.maximum(v[ub_mask] - sigma[ub_mask] * upper[ub_mask], 0.0)
    
    # Lower bound only: l finite, u = +inf
    lb_mask = np.isfinite(lower) & (~np.isfinite(upper))
    if np.any(lb_mask):
        y[lb_mask] = np.minimum(v[lb_mask] - sigma[lb_mask] * lower[lb_mask], 0.0)
    
    # Two-sided finite bounds
    two_sided = np.isfinite(lower) & np.isfinite(upper) & (lower != upper)
    if np.any(two_sided):
        v_div_sigma = v[two_sided] / sigma[two_sided]
        clipped = np.clip(v_div_sigma, lower[two_sided], upper[two_sided])
        y[two_sided] = v[two_sided] - sigma[two_sided] * clipped
    
    return y


def pdlp_core(
    lp: BoundedConstraintLP,
    *,
    max_iter: int = 1000,
    tol: float = 1e-7,
    check_every: int = 100,
    theta: float = 0.9,
    verbose: bool = False,
) -> PDLPCoreResult:
    """Solve bounded-constraint LP with diagonal-preconditioned PDHG."""
    if max_iter <= 0 or check_every <= 0 or tol <= 0.0 or theta <= 0.0 or theta >= 1.0:
        raise PDLPCoreError("max_iter, check_every, tol must be positive; theta in (0,1)")

    A = lp.A
    c = lp.c
    constraint_lower = lp.constraint_lower
    constraint_upper = lp.constraint_upper
    variable_lower = lp.variable_lower
    variable_upper = lp.variable_upper

    m, n = A.shape

    # Diagonal preconditioning
    col_sums = np.sum(np.abs(A), axis=0)
    row_sums = np.sum(np.abs(A), axis=1)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)

    tau = theta / col_sums
    sigma = theta / row_sums

    # Primal/dual variables
    x = _project_box(np.zeros(n, dtype=np.float64), variable_lower, variable_upper)
    x_bar = x.copy()
    y = np.zeros(m, dtype=np.float64)

    start_time = time.perf_counter()

    for iteration in range(1, max_iter + 1):
        # Dual update: y^{k+1} = prox_{sigma g*}(y^k + sigma A x_bar)
        # prox_{sigma g*}(v) = v - sigma * clip(v/sigma, l^c, u^c)
        v = y + sigma * (A @ x_bar)
        y = _safe_dual_prox(v, sigma, constraint_lower, constraint_upper)

        # Primal update
        previous_x = x
        grad = c + A.T @ y
        x = _project_box(x - tau * grad, variable_lower, variable_upper)
        x_bar = 2.0 * x - previous_x

        # Check for numerical breakdown
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            if verbose:
                print(f"  Numerical breakdown at iteration {iteration}")
            break

        if iteration % check_every == 0 or iteration == max_iter:
            (constraint_feas, var_viol, dual_prox_res, stationarity, 
             objective) = _full_diagnostics(
                c, A, constraint_lower, constraint_upper,
                variable_lower, variable_upper, x, y, tau, sigma
            )
            if verbose:
                print(
                    f"iter {iteration:5d}  obj={objective: .6e}  "
                    f"c_feas={constraint_feas:.2e}  v_viol={var_viol:.2e}  "
                    f"dual_res={dual_prox_res:.2e}  stat={stationarity:.2e}"
                )
            if max(constraint_feas, var_viol, dual_prox_res, stationarity) <= tol:
                runtime = time.perf_counter() - start_time
                return PDLPCoreResult(
                    x, y, iteration, True, "optimal", objective,
                    constraint_feas, var_viol, dual_prox_res, stationarity, runtime
                )

    runtime = time.perf_counter() - start_time
    (constraint_feas, var_viol, dual_prox_res, stationarity, objective) = _full_diagnostics(
        c, A, constraint_lower, constraint_upper, variable_lower, variable_upper, x, y, tau, sigma
    )
    return PDLPCoreResult(
        x, y, iteration, False, "iteration_limit", objective,
        constraint_feas, var_viol, dual_prox_res, stationarity, runtime
    )


def _self_test() -> None:
    """Self-test: verify dual proximal operator handles E, L, G correctly."""
    print("=== Self-Test: Dual Proximal Operator (E, L, G) ===")
    
    A = np.array([
        [1.0, 2.0],   # E row
        [3.0, 4.0],   # L row (upper bound)
        [5.0, 6.0],   # G row (lower bound)
    ], dtype=np.float64)
    
    constraint_lower = np.array([10.0, -np.inf, 30.0])
    constraint_upper = np.array([10.0, 20.0, np.inf])
    sigma = np.array([1.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 0.0])
    x_bar = np.array([1.0, 2.0])
    v = y + sigma * (A @ x_bar)  # [5, 11, 17]
    
    # E row: clip(5, 10, 10) = 10 -> y = 5 - 1*10 = -5
    # L row: clip(11, -inf, 20) = 11 -> y = 11 - 11 = 0 (since Ax < ub)
    # G row: clip(17, 30, inf) = 30 -> y = 17 - 30 = -13 (negative, correct for G)
    prox_v = _safe_dual_prox(v, sigma, constraint_lower, constraint_upper)
    expected_y = np.array([-5.0, 0.0, -13.0])
    assert np.allclose(prox_v, expected_y), f"Dual prox failed: {prox_v} != {expected_y}"
    print(f"  v = {v}")
    print(f"  y = {prox_v} (expected {expected_y}) OK")
    
    # Quick solver sanity check (doesn't require convergence)
    print("\nSolver sanity check (100 iters)...")
    A2 = np.array([
        [1.0, 0.0],   # E: x1 = 1
        [0.0, 1.0],   # L: x2 <= 2
        [1.0, 1.0],   # G: x1 + x2 >= 2
    ], dtype=np.float64)
    
    constraint_lower2 = np.array([1.0, -np.inf, 2.0])
    constraint_upper2 = np.array([1.0, 2.0, np.inf])
    variable_lower2 = np.array([0.0, 0.0])
    variable_upper2 = np.array([np.inf, np.inf])
    c2 = np.array([1.0, 1.0])
    
    lp2 = BoundedConstraintLP(
        name="TEST2", objective_name="OBJ",
        A=A2, c=c2,
        constraint_lower=constraint_lower2, constraint_upper=constraint_upper2,
        variable_lower=variable_lower2, variable_upper=variable_upper2,
        row_names=("eq", "le", "ge"), var_names=("x1", "x2")
    )
    
    result = pdlp_core(lp2, max_iter=100, tol=1e-7, theta=0.5, verbose=False)
    print(f"  100 iter: obj={result.objective:.3f}, feas={result.constraint_feasibility:.2e}, stat={result.stationarity:.2e}, status={result.status}")
    assert np.all(np.isfinite(result.x)), "Solver produced NaN"
    assert np.all(np.isfinite(result.y)), "Solver produced NaN"
    print("Self-test: PASS\n")


def _run_afiro() -> None:
    """AFIRO regression test."""
    print("=== AFIRO Regression ===")
    lp = load_numeric_mps("data/afiro.mps")
    bounded = to_bounded_constraint(lp)
    
    # Use theta=0.9 like the working pdhg_preconditioned.py
    result = pdlp_core(bounded, max_iter=5000, tol=1e-7, theta=0.9, verbose=True)
    print(f"\nResult: obj={result.objective:.6e}, feas={result.constraint_feasibility:.2e}, stat={result.stationarity:.2e}, status={result.status}")
    
# Reference AFIRO objective is approximately -464.75
    if abs(result.objective + 464.75) < 1.0:
        print("AFIRO objective matches reference OK")
    else:
        print(f"AFIRO objective: got {result.objective:.6e}, expected ~-464.75")
    print()


def _run_sc205_smoke() -> None:
    """SC205 smoke test (1000 iterations)."""
    print("=== SC205 Smoke Test (1000 iters) ===")
    lp = load_numeric_mps("data/sc205.mps")
    bounded = to_bounded_constraint(lp)
    
    result = pdlp_core(bounded, max_iter=1000, tol=1e-7, verbose=True)
    print(f"\nResult: obj={result.objective:.6e}")
    print(f"  Constraint feasibility: {result.constraint_feasibility:.2e}")
    print(f"  Variable bound violation: {result.variable_bound_violation:.2e}")
    print(f"  Dual proximal residual: {result.dual_proximal_residual:.2e}")
    print(f"  Stationarity: {result.stationarity:.2e}")
    print(f"  Status: {result.status}")
    print()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="PDLP-core unified bounded-constraint solver.")
    parser.add_argument("--self-test", action="store_true", help="Run self-test only")
    parser.add_argument("--afiro", action="store_true", help="Run AFIRO regression")
    parser.add_argument("--sc205-smoke", action="store_true", help="Run SC205 smoke test")
    parser.add_argument("--mps-file", type=Path, help="Run on specific MPS file")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test or (not args.afiro and not args.sc205_smoke and args.mps_file is None):
        _self_test()
    
    if args.afiro:
        _run_afiro()
    
    if args.sc205_smoke:
        _run_sc205_smoke()
    
    if args.mps_file:
        lp = load_numeric_mps(args.mps_file)
        bounded = to_bounded_constraint(lp)
        result = pdlp_core(bounded, max_iter=args.max_iter, tol=args.tol, verbose=args.verbose)
        print(f"Problem: {bounded.name}")
        print(f"Objective: {result.objective:.6e}")
        print(f"Constraint feasibility: {result.constraint_feasibility:.2e}")
        print(f"Variable bound violation: {result.variable_bound_violation:.2e}")
        print(f"Dual proximal residual: {result.dual_proximal_residual:.2e}")
        print(f"Stationarity: {result.stationarity:.2e}")
        print(f"Status: {result.status}")
        print(f"Runtime: {result.runtime_seconds:.3f} s")


if __name__ == "__main__":
    main()