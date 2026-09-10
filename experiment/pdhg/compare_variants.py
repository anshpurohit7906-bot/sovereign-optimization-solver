"""CPU-only convergence comparison of existing PDHG variants.

Runs all 5 PDHG variants on the same deterministic Case 4 problem (2000x1000)
used in the averaging experiment, with identical:
- initial point (zero projected to bounds)
- tolerance = 1e-7
- max iterations = 50,000
- convergence definitions

Variants tested:
1. pdhg_mixed.py (vanilla)
2. pdhg_restart.py (fixed-period restart)
3. pdhg_preconditioned.py (diagonal preconditioning)
4. pdhg_bb.py (Barzilai-Borwein adaptive step)
5. pdhg_pdlp_weight.py (PDLP primal-weight adaptation)

Does NOT modify any existing file.
Does NOT use GPU.
Stops each variant once it converges.

Reports: converged?, iterations, objective, KKT residuals, runtime.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

# Make src/ importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_EXPERIMENT_PDHG = _REPO_ROOT / "experiment" / "pdhg"

for _p in (_SRC, _REPO_ROOT, _EXPERIMENT_PDHG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from numerical_model import NumericalLP  # noqa: E402

# Import all 5 variants
from pdhg_mixed import pdhg_mixed  # noqa: E402
from pdhg_restart import pdhg_mixed_restart  # noqa: E402
from pdhg_preconditioned import pdhg_preconditioned  # noqa: E402
from pdhg_bb import pdhg_bb  # noqa: E402
from pdhg_pdlp_weight import pdhg_pdlp_weight  # noqa: E402


# ---------------------------------------------------------------------------
# Case 4 LP generation (deterministic, matches previous experiments)
# ---------------------------------------------------------------------------
RNG_SEED = 1234
CASE_IDX = 4
ROWS = 2_000
COLS = 1_000
DENSITY = 0.02

MAX_ITER = 50_000
TOL = 1e-7


def _generate_case4() -> NumericalLP:
    """Build deterministic Case 4: 2000x1000, density 0.02."""
    rng = np.random.default_rng(RNG_SEED + CASE_IDX)
    
    nnz = max(1, int(DENSITY * ROWS * COLS))
    n_draws = int(nnz * 1.5) + 1
    row_idx = rng.integers(0, ROWS, size=n_draws)
    col_idx = rng.integers(0, COLS, size=n_draws)
    data = rng.standard_normal(n_draws).astype(np.float64)
    
    A_csr = sp.coo_matrix((data, (row_idx, col_idx)), shape=(ROWS, COLS)).tocsr()
    A = A_csr.toarray()
    
    n_eq = max(1, ROWS // 3)
    n_ub = ROWS - n_eq
    row_types = ("E",) * n_eq + ("L",) * n_ub
    
    x_feas = rng.uniform(0.5, 1.5, size=COLS).astype(np.float64)
    A_eq = A[:n_eq]
    A_ub = A[n_eq:]
    b_eq = A_eq @ x_feas
    b_ub = A_ub @ x_feas + rng.uniform(0.5, 2.0, size=n_ub).astype(np.float64)
    b = np.concatenate([b_eq, b_ub])
    
    c = rng.standard_normal(COLS).astype(np.float64)
    lower = np.full(COLS, -1.0)
    upper = np.full(COLS, 2.5)
    
    return NumericalLP(
        name="case4",
        objective_name="bench",
        A=A,
        b=b,
        c=c,
        lower_bounds=lower,
        upper_bounds=upper,
        row_types=row_types,
        var_names=tuple(f"x{i}" for i in range(COLS)),
        row_names=tuple(f"r{i}" for i in range(ROWS)),
    )


# ---------------------------------------------------------------------------
# Variant runners with unified output
# ---------------------------------------------------------------------------
def run_vanilla(lp: NumericalLP) -> dict:
    """Run pdhg_mixed (vanilla)."""
    start = time.perf_counter()
    result = pdhg_mixed(lp, max_iter=MAX_ITER, tol=TOL, check_every=250, verbose=False)
    runtime = time.perf_counter() - start
    
    return {
        "name": "vanilla",
        "converged": result.converged,
        "iterations": result.iterations,
        "status": result.status,
        "objective": result.objective,
        "eq_res": result.equality_residual,
        "ineq_viol": result.inequality_violation,
        "dual_feas": result.dual_feasibility,
        "compl": result.complementarity,
        "max_kkt": max(result.equality_residual, result.inequality_violation,
                       result.dual_feasibility, result.complementarity),
        "runtime_sec": runtime,
    }


def run_restart(lp: NumericalLP) -> dict:
    """Run pdhg_restart (fixed-period restart every 10k iterations)."""
    start = time.perf_counter()
    result = pdhg_mixed_restart(
        lp, max_iter=MAX_ITER, tol=TOL, check_every=250,
        restart_period=10_000, verbose=False
    )
    runtime = time.perf_counter() - start
    
    return {
        "name": "restart",
        "converged": result.converged,
        "iterations": result.iterations,
        "status": result.status,
        "objective": result.objective,
        "eq_res": result.equality_residual,
        "ineq_viol": result.inequality_violation,
        "dual_feas": result.dual_feasibility,
        "compl": result.complementarity,
        "max_kkt": max(result.equality_residual, result.inequality_violation,
                       result.dual_feasibility, result.complementarity),
        "runtime_sec": runtime,
        "restarts": result.restarts_triggered,
    }


def run_preconditioned(lp: NumericalLP) -> dict:
    """Run pdhg_preconditioned (diagonal preconditioning)."""
    start = time.perf_counter()
    result = pdhg_preconditioned(
        lp, max_iter=MAX_ITER, tol=TOL, check_every=250,
        theta=0.9, verbose=False
    )
    runtime = time.perf_counter() - start
    
    return {
        "name": "preconditioned",
        "converged": result.converged,
        "iterations": result.iterations,
        "status": result.status,
        "objective": result.objective,
        "eq_res": result.equality_residual,
        "ineq_viol": result.inequality_violation,
        "dual_feas": result.dual_feasibility,
        "compl": result.complementarity,
        "max_kkt": max(result.equality_residual, result.inequality_violation,
                       result.dual_feasibility, result.complementarity),
        "runtime_sec": runtime,
    }


def run_bb(lp: NumericalLP) -> dict:
    """Run pdhg_bb (Barzilai-Borwein adaptive step)."""
    start = time.perf_counter()
    result = pdhg_bb(
        lp, max_iter=MAX_ITER, tol=TOL, check_every=250,
        theta=0.9, verbose=False
    )
    runtime = time.perf_counter() - start
    
    return {
        "name": "barzilai_borwein",
        "converged": result.converged,
        "iterations": result.iterations,
        "status": result.status,
        "objective": result.objective,
        "eq_res": result.equality_residual,
        "ineq_viol": result.inequality_violation,
        "dual_feas": result.stationarity,  # Note: BB uses "stationarity" naming
        "compl": result.complementarity,
        "max_kkt": max(result.equality_residual, result.inequality_violation,
                       result.stationarity, result.complementarity),
        "runtime_sec": runtime,
        "bb_alpha": result.bb_alpha,
    }


def run_pdlp_weight(lp: NumericalLP) -> dict:
    """Run pdhg_pdlp_weight (PDLP primal-weight adaptation)."""
    start = time.perf_counter()
    result = pdhg_pdlp_weight(
        lp, max_iter=MAX_ITER, tol=TOL, check_every=250,
        major_iteration_frequency=64, verbose=False
    )
    runtime = time.perf_counter() - start
    
    return {
        "name": "pdlp_weight",
        "converged": result.converged,
        "iterations": result.iterations,
        "status": result.status,
        "objective": result.objective,
        "eq_res": result.equality_residual,
        "ineq_viol": result.inequality_violation,
        "dual_feas": result.stationarity,  # Note: PDLP uses "stationarity" naming
        "compl": result.complementarity,
        "max_kkt": max(result.equality_residual, result.inequality_violation,
                       result.stationarity, result.complementarity),
        "runtime_sec": runtime,
        "primal_weight": result.primal_weight,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_header():
    print("=" * 120)
    print("PDHG Variant Convergence Comparison - Case 4 (2000x1000, density=0.02)")
    print(f"max_iter={MAX_ITER}  tol={TOL:.0e}")
    print("=" * 120)
    print()


def print_result(res: dict):
    name = res["name"]
    converged = "YES" if res["converged"] else "NO"
    iters = res["iterations"]
    obj = res["objective"]
    eq = res["eq_res"]
    ineq = res["ineq_viol"]
    dual = res["dual_feas"]
    compl = res["compl"]
    max_kkt = res["max_kkt"]
    runtime = res["runtime_sec"]
    
    print(f"Variant: {name}")
    print(f"  Converged (max KKT <= 1e-7): {converged}")
    print(f"  Iterations: {iters:>7}")
    print(f"  Runtime: {runtime:>8.2f} sec")
    print(f"  Objective: {obj:>12.6e}")
    print(f"  Equality residual:     {eq:>12.6e}")
    print(f"  Inequality violation:  {ineq:>12.6e}")
    print(f"  Dual feasibility:      {dual:>12.6e}")
    print(f"  Complementarity:       {compl:>12.6e}")
    print(f"  Max KKT residual:      {max_kkt:>12.6e}")
    
    if "restarts" in res:
        print(f"  Restarts triggered: {res['restarts']}")
    if "bb_alpha" in res:
        print(f"  Final BB alpha: {res['bb_alpha']:.6e}")
    if "primal_weight" in res:
        print(f"  Final primal weight: {res['primal_weight']:.6e}")
    
    print()


def print_ranking(results: list[dict]):
    print("=" * 120)
    print("RANKING")
    print("=" * 120)
    
    # Sort by: 1) converged (True first), 2) iterations, 3) runtime
    sorted_results = sorted(
        results,
        key=lambda r: (not r["converged"], r["iterations"], r["runtime_sec"])
    )
    
    print(f"{'Rank':<6} {'Variant':<20} {'Converged':<12} {'Iterations':<12} {'Runtime (s)':<12} {'Max KKT':<15}")
    print("-" * 120)
    
    for rank, res in enumerate(sorted_results, start=1):
        converged = "YES" if res["converged"] else "NO"
        print(f"{rank:<6} {res['name']:<20} {converged:<12} {res['iterations']:<12} "
              f"{res['runtime_sec']:<12.2f} {res['max_kkt']:<15.6e}")
    
    print()
    
    # Summary
    converged_count = sum(1 for r in results if r["converged"])
    print(f"Converged variants: {converged_count}/{len(results)}")
    
    if converged_count > 0:
        best = sorted_results[0]
        print(f"Best variant: {best['name']} ({best['iterations']} iterations, {best['runtime_sec']:.2f} sec)")
    else:
        print("No variant converged to tolerance 1e-7.")
    
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print_header()
    
    print("Generating Case 4 LP...")
    lp = _generate_case4()
    nnz = int(sp.csr_matrix(lp.A).nnz)
    print(f"  Problem: {lp.num_constraints} constraints x {lp.num_vars} variables")
    print(f"  Nonzeros: {nnz} ({100.0 * nnz / (lp.num_constraints * lp.num_vars):.2f}% dense)")
    print()
    
    results = []
    
    print("Running variant 1/5: vanilla (pdhg_mixed)...")
    try:
        res = run_vanilla(lp)
        results.append(res)
        print_result(res)
    except Exception as e:
        print(f"  ERROR: {e}")
        print()
    
    print("Running variant 2/5: restart (pdhg_restart)...")
    try:
        res = run_restart(lp)
        results.append(res)
        print_result(res)
    except Exception as e:
        print(f"  ERROR: {e}")
        print()
    
    print("Running variant 3/5: preconditioned (pdhg_preconditioned)...")
    try:
        res = run_preconditioned(lp)
        results.append(res)
        print_result(res)
    except Exception as e:
        print(f"  ERROR: {e}")
        print()
    
    print("Running variant 4/5: barzilai_borwein (pdhg_bb)...")
    try:
        res = run_bb(lp)
        results.append(res)
        print_result(res)
    except Exception as e:
        print(f"  ERROR: {e}")
        print()
    
    print("Running variant 5/5: pdlp_weight (pdhg_pdlp_weight)...")
    try:
        res = run_pdlp_weight(lp)
        results.append(res)
        print_result(res)
    except Exception as e:
        print(f"  ERROR: {e}")
        print()
    
    if results:
        print_ranking(results)


if __name__ == "__main__":
    main()
