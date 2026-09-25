"""External production-path AFIRO diagnostic (untracked).

Runs the production ``opticore.lp.mehrotra.solve_lp`` on ``data/afiro.mps`` and
independently evaluates the returned solution on the original unscaled LP.

This script does NOT invoke the experimental PDHG verifier
(``verify_afiro.py`` -> ``experiment.pdhg.pdhg_mixed``).
No tracked solver source is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opticore.numerical_model import load_numeric_mps
from opticore.lp.mehrotra import solve_lp, to_standard_form

AFIRO = ROOT / "data" / "afiro.mps"
TOL = 1e-7
MAX_ITER = 100
DUAL_FEAS_TOL = 1e-7


def _dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def _inf_norm(vector) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    return float(np.max(np.abs(vector))) if vector.size else 0.0


def _l2_norm(vector) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    return float(np.linalg.norm(vector, ord=2)) if vector.size else 0.0


def original_primal_metrics(lp, x):
    A = _dense(lp.A)
    row_types = np.asarray(lp.row_types)
    x = np.asarray(x, dtype=np.float64)

    eq_mask = row_types == "E"
    l_mask = row_types == "L"
    g_mask = row_types == "G"

    eq = A[eq_mask] @ x - np.asarray(lp.b)[eq_mask] if eq_mask.any() else np.zeros(0)
    violations = []
    if l_mask.any():
        violations.append(np.maximum(A[l_mask] @ x - np.asarray(lp.b)[l_mask], 0.0))
    if g_mask.any():
        violations.append(np.maximum(np.asarray(lp.b)[g_mask] - A[g_mask] @ x, 0.0))
    inequality_violation = np.concatenate(violations) if violations else np.zeros(0)

    lower_violation = np.maximum(lp.lower_bounds - x, 0.0)
    upper_violation = np.maximum(x - lp.upper_bounds, 0.0)
    bound_violation = np.concatenate([lower_violation, upper_violation])

    return {
        "eq_l2": _l2_norm(eq),
        "eq_inf": _inf_norm(eq),
        "ineq_l2": _l2_norm(inequality_violation),
        "ineq_inf": _inf_norm(inequality_violation),
        "bound_max": _inf_norm(bound_violation),
        "objective": float(np.asarray(lp.c) @ x),
    }


def highs_reference(lp):
    A = _dense(lp.A)
    b = np.asarray(lp.b, dtype=np.float64)
    c = np.asarray(lp.c, dtype=np.float64)
    row_types = np.asarray(lp.row_types)

    eq_mask = row_types == "E"
    l_mask = row_types == "L"
    g_mask = row_types == "G"

    A_eq = A[eq_mask] if eq_mask.any() else None
    b_eq = b[eq_mask] if eq_mask.any() else None
    A_ub_parts = []
    b_ub_parts = []
    if l_mask.any():
        A_ub_parts.append(A[l_mask])
        b_ub_parts.append(b[l_mask])
    if g_mask.any():
        A_ub_parts.append(-A[g_mask])
        b_ub_parts.append(-b[g_mask])
    A_ub = np.vstack(A_ub_parts) if A_ub_parts else None
    b_ub = np.concatenate(b_ub_parts) if b_ub_parts else None

    bounds = list(zip(lp.lower_bounds, lp.upper_bounds))
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")


def original_dual_recovery(lp, result):
    """Reconstruct recoverable original-dual quantities from standard-form output.

    Sign convention derived from ``to_standard_form``:
      * original equality multiplier  y_orig = -y_standard
      * original inequality multiplier mu_orig = z_standard_slack (L and G rows)
      * original reduced cost z_orig = sum_k block_sign[k] * z_standard[k]
      * stationarity residual = c + A_E^T y_orig + A_L^T mu_L - A_G^T mu_G - z_orig
    """
    sf = to_standard_form(lp)
    A = _dense(lp.A)
    n_orig = int(lp.num_vars)
    m_orig = int(lp.num_constraints)
    row_types = np.asarray(lp.row_types)

    y_std = np.asarray(result.y, dtype=np.float64)
    z_std = np.asarray(result.z_standard, dtype=np.float64)
    if y_std.shape[0] < m_orig or z_std.shape[0] != sf.n:
        raise ValueError("result standard-form dimensions do not match to_standard_form")

    z_orig = np.zeros(n_orig, dtype=np.float64)
    np.add.at(z_orig, sf.block_to_orig, sf.block_sign * z_std[:sf.n_block])

    slack_rows = np.asarray(sf.slack_row_indices, dtype=np.intp)
    z_slack = z_std[sf.n_block:]
    mu_orig = np.zeros(m_orig, dtype=np.float64)
    for position, row in enumerate(slack_rows):
        if row < m_orig:
            mu_orig[int(row)] = z_slack[position]

    eq_mask = row_types == "E"
    l_mask = row_types == "L"
    g_mask = row_types == "G"
    y_orig_eq = -y_std[:m_orig][eq_mask]

    stationarity = np.asarray(lp.c, dtype=np.float64).copy()
    if eq_mask.any():
        stationarity = stationarity + A[eq_mask].T @ y_orig_eq
    if l_mask.any():
        stationarity = stationarity + A[l_mask].T @ mu_orig[l_mask]
    if g_mask.any():
        stationarity = stationarity - A[g_mask].T @ mu_orig[g_mask]
    dual_residual = stationarity - z_orig

    return {
        "sf": sf,
        "z_orig": z_orig,
        "mu_orig": mu_orig,
        "y_orig_eq": y_orig_eq,
        "dual_residual_inf": _inf_norm(dual_residual),
        "dual_residual_l2": _l2_norm(dual_residual),
        "mu_min": float(np.min(mu_orig[l_mask | g_mask])) if (l_mask | g_mask).any() else 0.0,
        "z_orig_min": float(np.min(z_orig)) if z_orig.size else 0.0,
    }



def main() -> int:
    print("=" * 82)
    print("PRODUCTION PATH: opticore.lp.mehrotra.solve_lp (Mehrotra predictor-corrector IPM)")
    print("EXPERIMENTAL PDHG VERIFIER: verify_afiro.py -> experiment.pdhg.pdhg_mixed (NOT run here)")
    print("=" * 82)

    t_load_start = perf_counter()
    lp = load_numeric_mps(AFIRO)
    load_seconds = perf_counter() - t_load_start

    row_types = np.asarray(lp.row_types)
    print(f"Input file                : {AFIRO}")
    print(f"Problem name              : {lp.name}")
    print(f"Variables / constraints   : {lp.num_vars} / {lp.num_constraints}")
    print(f"Rows E/L/G                : {int((row_types == 'E').sum())} / "
          f"{int((row_types == 'L').sum())} / {int((row_types == 'G').sum())}")
    print(f"Objective sense           : maximize={False} (solve_lp default)")
    print(f"Solver arguments          : tol={TOL!r}, max_iter={MAX_ITER}, backend='cpu' (default)")
    print(f"Model load time           : {load_seconds:.6f} s")

    ref = highs_reference(lp)
    if not ref.success or ref.fun is None:
        print(f"HiGHS reference failed    : {ref.status} / {ref.message}")
        return 2

    solve_start = perf_counter()
    result = solve_lp(lp, tol=TOL, max_iter=MAX_ITER)
    solve_seconds = perf_counter() - solve_start

    primal = original_primal_metrics(lp, result.x)
    dual = original_dual_recovery(lp, result)
    sf = dual["sf"]

    A_sf = _dense(sf.A)
    x_std = np.asarray(result.x_standard, dtype=np.float64)
    y_std = np.asarray(result.y, dtype=np.float64)
    z_std = np.asarray(result.z_standard, dtype=np.float64)
    std_primal_residual = _inf_norm(A_sf @ x_std - sf.b)
    std_dual_residual = _inf_norm(sf.c_min - A_sf.T @ y_std - z_std)

    abs_obj_error = abs(primal["objective"] - float(ref.fun))
    rel_obj_error = abs_obj_error / (1.0 + abs(float(ref.fun)))
    objective_field_delta = float(result.objective) - primal["objective"]

    has_finite_upper = bool(np.isfinite(lp.upper_bounds).any())
    extra_rows = sf.m - lp.num_constraints
    if has_finite_upper:
        dual_feasibility_violation = None
    else:
        dual_feasibility_violation = max(
            0.0, -dual["mu_min"], -dual["z_orig_min"]
        )

    print()
    print("PRODUCTION SOLVER STATUS AND RUNTIME")
    print(f"  status                  : {result.status}")
    print(f"  message                 : {result.message}")
    print(f"  iterations              : {result.iterations}")
    print(f"  solve_lp runtime        : {solve_seconds:.6f} s")
    print(f"  regularized iterations  : {list(result.regularized_iterations)}")

    print()
    print("INDEPENDENT ORIGINAL-UNSCALED-LP EVALUATION")
    print(f"  objective value         : {primal['objective']:.12f}")
    print(f"  result.objective delta  : {objective_field_delta:.3e}")
    print(f"  primal equality residual (L2)  : {primal['eq_l2']:.6e}")
    print(f"  primal equality residual (inf) : {primal['eq_inf']:.6e}")
    print(f"  inequality violation (L2)      : {primal['ineq_l2']:.6e}")
    print(f"  inequality violation (inf)     : {primal['ineq_inf']:.6e}")
    print(f"  bound violation (max)          : {primal['bound_max']:.6e}")

    print()
    print("HIGHS REFERENCE COMPARISON")
    print(f"  HiGHS status            : {ref.status}")
    print(f"  HiGHS objective         : {float(ref.fun):.12f}")
    print(f"  absolute objective error: {abs_obj_error:.6e}")
    print(f"  relative objective error: {rel_obj_error:.6e}  (|diff|/(1+|ref|))")

    print()
    print("SOLVER-INTERNAL RELATIVE CRITERIA")
    print(f"  rel_primal              : {result.rel_primal:.6e}")
    print(f"  rel_dual                : {result.rel_dual:.6e}")
    print(f"  rel_gap                 : {result.rel_gap:.6e}")
    print(f"  primal_residual (abs)   : {result.primal_residual:.6e}")
    print(f"  dual_residual (abs)     : {result.dual_residual:.6e}")
    print(f"  complementarity         : {result.complementarity:.6e}")

    print()
    print("RECONSTRUCTED ORIGINAL DUAL (from to_standard_form mapping)")
    print(f"  standard-form primal residual     : {std_primal_residual:.6e}")
    print(f"  standard-form dual residual       : {std_dual_residual:.6e}")
    print(f"  solver-reported dual residual     : {result.dual_residual:.6e}")
    print(f"  original stationarity residual    : {dual['dual_residual_inf']:.6e}")
    print(f"  min inequality multiplier mu      : {dual['mu_min']:.6e}")
    print(f"  min lower-bound reduced cost      : {dual['z_orig_min']:.6e}")
    if extra_rows:
        print(f"  note: {extra_rows} appended box row(s); box upper-bound duals are not mapped")
    if has_finite_upper:
        print("  dual feasibility        : not reconstructed (finite upper bounds present)")
    else:
        print(f"  dual feasibility violation: {dual_feasibility_violation:.6e}")

    lower_comp = _inf_norm(np.asarray(result.x) * dual["z_orig"])
    print(f"  lower-bound complementarity max   : {lower_comp:.6e}")
    print("=" * 82)
    return 0 if result.status == "optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())

