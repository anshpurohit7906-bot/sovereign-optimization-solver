"""SIH 26119 Demo: indigenous MILP + QP solvers on industrial-style problems.

Runs two segments:
  1. Production-planning MILP (binary + continuous, refinery/blending style)
  2. Convex QP portfolio / resource allocation

Both solved by the from-scratch engine; HiGHS/SciPy used as oracle only.

Usage:
    python demo/demo_sih.py              # run both segments
    python demo/demo_sih.py --milp-only  # MILP segment only
    python demo/demo_sih.py --qp-only    # QP segment only
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import NumericalLP
from lp.branch_bound import solve_milp
from lp.qp import NumericalQP, solve_qp, verify_qp_kkt


# ============================================================
# Segment 1: Production-Planning MILP
# ============================================================

def build_production_planning_milp() -> tuple[NumericalLP, np.ndarray, dict]:
    """Build a 4-crude, 2-product production planning MILP.

    Returns (lp, integer_mask, info) where info contains the known oracle
    objective (scipy.optimize.milp) and the problem description.
    """
    # --- problem data ---
    costs = np.array([30.0, 22.0, 18.0, 35.0])       # $/barrel crude cost
    fixed = np.array([50.0, 40.0, 30.0, 60.0])       # fixed charge $/day if used
    capacity = np.array([500.0, 400.0, 300.0, 250.0]) # max barrels/day
    demand = np.array([600.0, 400.0])                 # product demand
    octane = np.array([88.0, 92.0, 95.0, 90.0])      # crude octane rating
    min_octane = np.array([89.0, 91.5])               # minimum blend octane
    M = np.array([500.0, 400.0, 300.0, 250.0])       # big-M (equals capacity)

    n_crudes = 4
    n_products = 2
    n_x = n_crudes             # x_i : amount of crude i
    n_y = n_crudes             # y_i : binary selection
    n_w = n_crudes * n_products  # w_ij allocation
    n = n_x + n_y + n_w
    n_orig = n  # for bounds

    # Index helpers
    def x_i(i): return i
    def y_i(i): return n_x + i
    def w_ij(i, j): return n_x + n_y + i * n_products + j

    # Bounds: x >= 0, y in {0,1} (bounds 0..1), w >= 0
    lb = np.zeros(n)
    ub = np.zeros(n)
    for i in range(n_crudes):
        ub[x_i(i)] = 1e10          # unbounded continuous
        ub[y_i(i)] = 1.0           # binary
    for i in range(n_crudes):
        for j in range(n_products):
            ub[w_ij(i, j)] = 1e10  # unbounded continuous

    # Row types and constraints
    rows = []
    row_types = []
    b_vals = []
    coeffs = {}  # (var, row) -> value

    def add_row(name: str, rt: str, b_val: float) -> int:
        r = len(rows)
        rows.append(name)
        row_types.append(rt)
        b_vals.append(b_val)
        return r

    # --- Supply: x_i - M_i y_i <= 0   (L-row) ---
    for i in range(n_crudes):
        r = add_row(f"supply_{i}", "L", 0.0)
        coeffs[(x_i(i), r)] = 1.0
        coeffs[(y_i(i), r)] = -M[i]

    # --- Demand: sum_i w_ij >= D_j   (G-row) ---
    for j in range(n_products):
        r = add_row(f"demand_{j}", "G", demand[j])
        for i in range(n_crudes):
            coeffs[(w_ij(i, j), r)] = 1.0

    # --- Balance: x_i - sum_j w_ij = 0  (E-row) ---
    for i in range(n_crudes):
        r = add_row(f"balance_{i}", "E", 0.0)
        coeffs[(x_i(i), r)] = 1.0
        for j in range(n_products):
            coeffs[(w_ij(i, j), r)] = -1.0

    # --- Blend quality: sum_i (oct_i - q_j) w_ij >= 0  (G-row) ---
    for j in range(n_products):
        q_j = min_octane[j]
        r = add_row(f"quality_{j}", "G", 0.0)
        for i in range(n_crudes):
            coeffs[(w_ij(i, j), r)] = octane[i] - q_j

    m = len(rows)
    A = np.zeros((m, n))
    for (var, row), val in coeffs.items():
        A[row, var] = val
    b = np.array(b_vals, dtype=np.float64)

    # Objective: cost * x + fixed * y
    c = np.zeros(n)
    for i in range(n_crudes):
        c[x_i(i)] = costs[i]
        c[y_i(i)] = fixed[i]

    lp = NumericalLP(
        name="PRODUCTION_PLANNING",
        objective_name="COST",
        A=A, b=b, c=c,
        lower_bounds=lb, upper_bounds=ub,
        row_types=tuple(row_types),
        var_names=tuple(f"v{i}" for i in range(n)),
        row_names=tuple(rows),
    )
    integer_mask = np.zeros(n, dtype=bool)
    for i in range(n_crudes):
        integer_mask[y_i(i)] = True

    info = {
        "n_crudes": n_crudes,
        "n_products": n_products,
        "n_vars": n,
        "n_constraints": m,
        "n_integer": int(n_y),
        "description": "4-crude 2-product refinery production planning (binary fixed-charge)",
    }
    return lp, integer_mask, info


# ============================================================
# Segment 2: Convex QP — Portfolio / Resource Allocation
# ============================================================

def build_portfolio_qp() -> tuple[NumericalQP, dict]:
    """Small convex QP: minimize risk (quadratic) subject to return and budget.

    min  0.5 x^T Q x
    s.t. p^T x >= R_min      (minimum return)
         1^T x = B            (budget)
         x >= 0               (no short-selling)

    Q encodes variance/covariance (here diagonal for simplicity).
    """
    n_assets = 5
    np.random.seed(42)
    # Diagonal covariance (risk): each asset has different volatility
    vol = np.array([0.02, 0.04, 0.015, 0.05, 0.03])
    Q = np.diag(vol ** 2)

    # Expected return per asset
    ret = np.array([0.06, 0.12, 0.04, 0.15, 0.08])
    B = 100.0       # total budget
    R_min = 8.0     # minimum expected return

# Equality: 1^T x = B
    A_eq = np.ones((1, n_assets))
    b_eq = np.array([B])

    # Inequality: p^T x >= R_min  (use L-row with negated coeffs: -p^T x <= -R_min)
    A_ub = -ret.reshape(1, -1)
    b_ub = np.array([-R_min])

    A = np.vstack([A_eq, A_ub])
    b = np.concatenate([b_eq, b_ub])
    row_types = ("E", "L")

    qp = NumericalQP(
        name="PORTFOLIO_5ASSET",
        Q=Q,
        c=np.zeros(n_assets),
        A=A, b=b,
        row_types=row_types,
        lower_bounds=np.zeros(n_assets),
        upper_bounds=np.full(n_assets, B),
    )

    info = {
        "n_assets": n_assets,
        "n_vars": n_assets,
        "n_constraints": len(row_types),
        "description": "5-asset portfolio optimization (quadratic risk, linear return constraint)",
    }
    return qp, info


# ============================================================
# Oracle references
# ============================================================

def oracle_milp(lp: NumericalLP, integer_mask: np.ndarray) -> Optional[float]:
    """Solve the MILP relaxation via SciPy/HiGHS (oracle reference)."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        A = np.asarray(lp.A, dtype=np.float64)
        b = np.asarray(lp.b, dtype=np.float64)
        c = np.asarray(lp.c, dtype=np.float64)
        integrality = np.where(integer_mask, 1, 0).astype(int)

        constraints = []
        for i in range(lp.num_constraints):
            row = A[i]
            row_type = lp.row_types[i]
            if row_type == "E":
                constraints.append(LinearConstraint(row, b[i], b[i]))
            elif row_type == "L":
                constraints.append(LinearConstraint(row, -np.inf, b[i]))
            elif row_type == "G":
                constraints.append(LinearConstraint(row, b[i], np.inf))

        lb = np.asarray(lp.lower_bounds, dtype=np.float64)
        ub = np.asarray(lp.upper_bounds, dtype=np.float64)
        bounds = Bounds(lb, ub)
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        return float(res.fun) if res.success else None
    except Exception as e:
        print(f"  [oracle] milp failed: {e}")
        return None


def oracle_qp(qp: NumericalQP) -> Optional[float]:
    """Solve the QP via SciPy (SLSQP, reference)."""
    try:
        from scipy.optimize import minimize, LinearConstraint, Bounds
        Q = np.asarray(qp.Q, dtype=np.float64)
        c = np.asarray(qp.c, dtype=np.float64)
        A = np.asarray(qp.A, dtype=np.float64)
        b = np.asarray(qp.b, dtype=np.float64)

        def objective(x):
            return 0.5 * x @ Q @ x + c @ x

        def grad(x):
            return Q @ x + c

        constraints = []
        for i in range(qp.num_constraints):
            row = A[i]
            row_type = qp.row_types[i]
            bi = b[i]
            if row_type == "E":
                constraints.append(LinearConstraint(row, bi, bi))
            elif row_type == "L":
                constraints.append(LinearConstraint(row, -np.inf, bi))
            elif row_type == "G":
                constraints.append(LinearConstraint(row, bi, np.inf))

        lb = np.asarray(qp.lower_bounds, dtype=np.float64)
        ub = np.asarray(qp.upper_bounds, dtype=np.float64)
        bounds = Bounds(lb, ub)

        x0 = np.ones(qp.num_vars) * (ub / 2.0)
        if np.all(np.isfinite(ub)):
            x0 = np.clip(x0, lb + 0.1, ub - 0.1)
        # Project x0 onto equality constraints if any
        E_idx = [i for i, t in enumerate(qp.row_types) if t == "E"]
        if E_idx:
            A_E = A[E_idx]
            b_E = b[E_idx]
            residual = b_E - A_E @ x0
            try:
                x0 += A_E.T @ np.linalg.solve(A_E @ A_E.T, residual)
            except np.linalg.LinAlgError:
                pass
            x0 = np.maximum(x0, lb + 1e-6)

        res = minimize(objective, x0, jac=grad, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-12})
        return float(res.fun) if res.success else None
    except Exception as e:
        print(f"  [oracle] qp failed: {e}")
        return None


# ============================================================
# Demo runner
# ============================================================

def run_milp_segment() -> bool:
    """Run the production-planning MILP and print results."""
    print("=" * 70)
    print("SEGMENT 1: Production-Planning MILP (indigenous solver vs HiGHS oracle)")
    print("=" * 70)

    lp, mask, info = build_production_planning_milp()
    print(f"Problem: {info['description']}")
    print(f"  Variables: {info['n_vars']} ({info['n_integer']} integer, binary)")
    print(f"  Constraints: {info['n_constraints']}")
    print()

    t0 = time.perf_counter()
    milp_res = solve_milp(lp, integer_mask=mask, gap_tol=1e-4, verbose=False)
    dt = time.perf_counter() - t0
    print(f"  Indigenous MILP solver:")
    print(f"    Status:           {milp_res.status}")
    print(f"    Objective:        {milp_res.objective:.4f}")
    print(f"    Nodes explored:   {milp_res.nodes_explored}")
    print(f"    LP solves:        {milp_res.lp_solves}")
    print(f"    Time:             {dt:.3f}s")
    print(f"    Heur. attempts:   {milp_res.heuristic_attempts}")
    heur_found = (milp_res.heuristic_successes or 0) > 0
    print(f"    Heur. incumbent:  {'yes' if heur_found else 'no'} "
          f"(found={milp_res.heuristic_successes}, "
          f"repair solves={milp_res.heuristic_lp_solves})")
    if milp_res.heuristic_best_objective is not None:
        print(f"    Best heur. obj:   {milp_res.heuristic_best_objective:.4f}")
    if milp_res.x is not None:
        print(f"    Solution x:       {np.round(milp_res.x[:4], 1)} (crudes)")
        print(f"    Selection y:      {np.round(milp_res.x[4:8], 0).astype(int)} (selected crudes)")
    print()

    # Oracle reference
    oracle_obj = oracle_milp(lp, mask)
    if oracle_obj is not None:
        obj_err = abs((milp_res.objective or 0.0) - oracle_obj)
        rel_err = obj_err / (1.0 + abs(oracle_obj))
        match = rel_err < 1e-4
        print(f"  HiGHS oracle:      {oracle_obj:.4f}")
        print(f"  Objective error:   {obj_err:.4e} (rel: {rel_err:.2e})")
        print(f"  Match:             {'PASS' if match else 'FAIL'}")
    else:
        print("  Oracle:            unavailable (install scipy)")
    print()
    return milp_res.status == "optimal"


def run_qp_segment() -> bool:
    """Run the portfolio QP and print results."""
    print("=" * 70)
    print("SEGMENT 2: Convex QP Portfolio (indigenous solver vs SciPy oracle)")
    print("=" * 70)

    qp, info = build_portfolio_qp()
    print(f"Problem: {info['description']}")
    print(f"  Variables: {info['n_vars']}")
    print(f"  Constraints: {info['n_constraints']}")
    print()

    t0 = time.perf_counter()
    qp_res = solve_qp(qp)
    dt = time.perf_counter() - t0
    print(f"  Indigenous QP solver:")
    print(f"    Status:           {qp_res.status}")
    print(f"    Objective:        {qp_res.objective:.6f}")
    print(f"    Iterations:       {qp_res.iterations}")
    print(f"    Time:             {dt:.3f}s")
    print(f"    x:                {np.round(qp_res.x, 4)}")
    print()

    # KKT verification
    kkt = verify_qp_kkt(qp, qp_res)
    print(f"  KKT verification:")
    print(f"    rel_primal:       {kkt['rel_primal']:.2e}")
    print(f"    rel_dual:         {kkt['rel_dual']:.2e}")
    print(f"    rel_gap:          {kkt['rel_gap']:.2e}")
    print(f"    Row violations:   {len(kkt['row_violations'])}")
    print(f"    Bound violations: lb={kkt['lb_violation']:.2e}, ub={kkt['ub_violation']:.2e}")
    print(f"    PASS:             {kkt['PASS']}")
    print()

    # Oracle reference
    oracle_obj = oracle_qp(qp)
    if oracle_obj is not None:
        obj_err = abs(qp_res.objective - oracle_obj)
        rel_err = obj_err / (1.0 + abs(oracle_obj))
        match = rel_err < 1e-4
        print(f"  SciPy oracle:      {oracle_obj:.6f}")
        print(f"  Objective error:   {obj_err:.4e} (rel: {rel_err:.2e})")
        print(f"  Match:             {'PASS' if match else 'FAIL'}")
    else:
        print("  Oracle:            FAIL (oracle solve failed)")
        match = False
    print()

    # QP PASS criteria:
    #   1. solver status == "optimal"
    #   2. independent KKT verification PASS
    #   3. oracle solve succeeded
    #   4. objective matches oracle within 1e-4 relative
    qp_ok = (
        qp_res.status == "optimal" and
        kkt["PASS"] and
        oracle_obj is not None and
        match
    )
    if not qp_ok:
        print(f"  QP segment status: {'PASS' if qp_ok else 'FAIL'}")
        if qp_res.status != "optimal":
            print(f"    -> solver status: {qp_res.status}")
        if not kkt["PASS"]:
            print(f"    -> KKT verification failed")
        if oracle_obj is None:
            print(f"    -> oracle unavailable")
        if oracle_obj is not None and not match:
            obj_err = abs(qp_res.objective - oracle_obj)
            rel_err = obj_err / (1.0 + abs(oracle_obj))
            print(f"    -> objective mismatch (rel_err={rel_err:.2e})")
    return qp_ok


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--milp-only", action="store_true")
    parser.add_argument("--qp-only", action="store_true")
    args = parser.parse_args()

    print("SIH 26119 — Sovereign Optimization Solver Demo")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    try:
        import scipy
        print(f"SciPy: {scipy.__version__}")
    except ImportError:
        print("SciPy: NOT INSTALLED")
    print()

    results = {}
    if not args.qp_only:
        results["MILP"] = run_milp_segment()
    if not args.milp_only:
        results["QP"] = run_qp_segment()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name:8s}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())