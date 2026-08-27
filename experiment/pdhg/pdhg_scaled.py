"""Scaled PDHG solver: load MPS -> scale -> solve -> unscale -> evaluate."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mps_parser import MPSParser
from numerical_model import to_numeric, NumericalLP
from experiment.pdhg.pdhg_mixed import pdhg_mixed, MixedPDHGResult
from scaling import scale_lp, unscale_solution, ScaledLP


@dataclass(frozen=True)
class ScaledPDHGResult:
    """Complete result including scaled and unscaled diagnostics."""
    original_objective: float
    scaled_objective: float
    unscaled_x: np.ndarray
    scaled_iterations: int
    converged: bool
    status: str
    primal_feasibility_unscaled: float
    dual_feasibility_unscaled: float
    equality_residual_unscaled: float
    inequality_violation_unscaled: float
    complementarity_unscaled: float
    runtime_seconds: float


def evaluate_unscaled_solution(
    lp: NumericalLP,
    x: np.ndarray,
    y_eq: np.ndarray,
    y_ub: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Evaluate primal/dual residuals for a candidate solution x on the original LP with given duals."""
    row_types = np.asarray(lp.row_types)
    eq_mask = row_types == "E"
    ub_mask = row_types == "L"

    A_eq, b_eq = lp.A[eq_mask], lp.b[eq_mask]
    A_ub, b_ub = lp.A[ub_mask], lp.b[ub_mask]

    equality_residual = float(np.linalg.norm(A_eq @ x - b_eq, ord=np.inf)) if A_eq.shape[0] else 0.0
    inequality_slack = A_ub @ x - b_ub
    inequality_violation = float(np.linalg.norm(np.maximum(inequality_slack, 0.0), ord=np.inf)) if A_ub.shape[0] else 0.0

    lagrangian_gradient = lp.c + A_eq.T @ y_eq + A_ub.T @ y_ub
    dual_feasibility = float(np.linalg.norm(
        x - np.clip(x - lagrangian_gradient, lp.lower_bounds, lp.upper_bounds), ord=np.inf
    ))
    complementarity = float(np.linalg.norm(y_ub * inequality_slack, ord=np.inf)) if A_ub.shape[0] else 0.0

    return equality_residual, inequality_violation, dual_feasibility, complementarity


def solve_scaled(
    mps_path: str | Path,
    *,
    max_iter: int = 200_000,
    tol: float = 1e-7,
    check_every: int = 250,
    verbose: bool = False,
) -> ScaledPDHGResult:
    """Load MPS, scale, solve with PDHG, unscale, and evaluate on original LP."""
    start_time = time.perf_counter()

    parser = MPSParser()
    model = parser.parse_file(str(mps_path))
    lp = to_numeric(model)

    scaled = scale_lp(lp.A, lp.b, lp.c, lp.lower_bounds, lp.upper_bounds)

    row_types = np.asarray(lp.row_types)
    eq_mask = row_types == "E"
    ub_mask = row_types == "L"

    scaled_lp = NumericalLP(
        name=f"{lp.name}_scaled",
        objective_name=lp.objective_name,
        A=scaled.A,
        b=scaled.b,
        c=scaled.c,
        lower_bounds=scaled.lower,
        upper_bounds=scaled.upper,
        row_types=lp.row_types,
        var_names=lp.var_names,
        row_names=lp.row_names,
    )

    result = pdhg_mixed(
        scaled_lp,
        max_iter=max_iter,
        tol=tol,
        check_every=check_every,
        verbose=verbose,
    )

    x_unscaled = unscale_solution(result.x, scaled.column_scale)
    y_eq_unscaled = scaled.row_scale[eq_mask] * result.y_eq
    y_ub_unscaled = scaled.row_scale[ub_mask] * result.y_ub

    obj_unscaled = float(lp.c @ x_unscaled)
    obj_scaled = result.objective

    eq_res, ineq_viol, dual_feas, comp = evaluate_unscaled_solution(lp, x_unscaled, y_eq_unscaled, y_ub_unscaled)

    runtime = time.perf_counter() - start_time

    return ScaledPDHGResult(
        original_objective=obj_unscaled,
        scaled_objective=obj_scaled,
        unscaled_x=x_unscaled,
        scaled_iterations=result.iterations,
        converged=result.converged,
        status=result.status,
        primal_feasibility_unscaled=max(eq_res, ineq_viol),
        dual_feasibility_unscaled=dual_feas,
        equality_residual_unscaled=eq_res,
        inequality_violation_unscaled=ineq_viol,
        complementarity_unscaled=comp,
        runtime_seconds=runtime,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Solve scaled LP with PDHG and evaluate on original.")
    default_sc205 = Path(__file__).resolve().parents[1] / "data" / "sc205.mps"
    parser.add_argument("mps_file", nargs="?", type=Path, default=default_sc205)
    parser.add_argument("--max-iter", type=int, default=200_000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = solve_scaled(
        args.mps_file,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=args.verbose,
    )

    print(f"Problem:              {args.mps_path.name if hasattr(args, 'mps_path') else args.mps_file.name}")
    print(f"Original objective:   {result.original_objective:.10g}")
    print(f"Scaled objective:     {result.scaled_objective:.10g}")
    print(f"Iterations (scaled):  {result.scaled_iterations}")
    print(f"Primal feasibility:   {result.primal_feasibility_unscaled:.3e}")
    print(f"  equality residual:  {result.equality_residual_unscaled:.3e}")
    print(f"  inequality residual:{result.inequality_violation_unscaled:.3e}")
    print(f"Dual feasibility:     {result.dual_feasibility_unscaled:.3e}")
    print(f"Complementarity:      {result.complementarity_unscaled:.3e}")
    print(f"Solver status:        {result.status}")
    print(f"Runtime:              {result.runtime_seconds:.3f} s")


if __name__ == "__main__":
    main()