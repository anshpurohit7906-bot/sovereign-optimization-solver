"""Direct MILP validation driver: solve_milp only (no HiGHS oracle).

Usage: python -u tools/validate_milp_incumbent.py NAME NODE_LIMIT TIME_LIMIT
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

from opticore.numerical_model import load_numeric_mps
from opticore.lp.branch_bound import solve_milp


def main() -> None:
    name = sys.argv[1]
    node_limit = int(sys.argv[2])
    time_limit = float(sys.argv[3])
    mps = os.path.join(_ROOT, "data", f"{name}.mps")
    lp = load_numeric_mps(mps, sparse=True)
    mask = np.asarray(lp.is_integer, dtype=bool)
    print(f"[{name}] n={lp.num_vars} m={lp.num_constraints} "
          f"int={int(mask.sum())}", flush=True)
    t0 = time.perf_counter()
    res = solve_milp(lp, integer_mask=mask, maximize=False, gap_tol=1e-4,
                     crossover_fallback=True, node_crossover_fallback=False,
                     node_limit=node_limit, time_limit=time_limit,
                     verbose=False)
    dt = time.perf_counter() - t0
    print(f"[{name}] RESULT", flush=True)
    print(f"  status          : {res.status}", flush=True)
    print(f"  message         : {res.message}", flush=True)
    print(f"  objective       : {res.objective}", flush=True)
    print(f"  best_bound      : {res.best_bound}", flush=True)
    print(f"  gap             : {res.gap}", flush=True)
    print(f"  nodes_explored  : {res.nodes_explored} / limit {res.node_limit}", flush=True)
    print(f"  lp_solves       : {res.lp_solves}", flush=True)
    print(f"  time_sec        : {dt:.2f}", flush=True)
    print(f"  heuristic       : attempts={res.heuristic_attempts} "
          f"successes={res.heuristic_successes} "
          f"lp_solves={res.heuristic_lp_solves} "
          f"best={res.heuristic_best_objective}", flush=True)
    print(f"  nodes_infeasible: {res.nodes_infeasible}", flush=True)
    print(f"  nodes_failed    : {res.nodes_failed} "
          f"{res.node_failure_statuses[:6]}", flush=True)

    # ---- Independent post-solve verification of the returned incumbent ----
    if res.x is not None:
        x = np.asarray(res.x, dtype=np.float64)
        A = lp.A
        b = np.asarray(lp.b, dtype=np.float64)
        resid = np.asarray(A @ x) - b
        row_bad = 0
        worst_row = 0.0
        for i, rt in enumerate(lp.row_types):
            r = resid[i]
            if rt == "E":
                v = abs(r)
            elif rt == "L":
                v = max(r, 0.0)
            else:
                v = max(-r, 0.0)
            worst_row = max(worst_row, v)
            if v > 1e-6:
                row_bad += 1
        lb = np.asarray(lp.lower_bounds, dtype=np.float64)
        ub = np.asarray(lp.upper_bounds, dtype=np.float64)
        bnd_bad = int(np.count_nonzero((x < lb - 1e-6) | (x > ub + 1e-6)))
        int_bad = int(np.count_nonzero(
            mask & (np.abs(x - np.round(x)) > 1e-6)))
        obj_recomp = float(np.asarray(lp.c, dtype=np.float64) @ x)
        print(f"  VERIFY          : rows_violated={row_bad} "
              f"bounds_violated={bnd_bad} integrality_violated={int_bad} "
              f"worst_row={worst_row:.3e}", flush=True)
        print(f"  VERIFY obj      : recomputed={obj_recomp:.9f} "
              f"reported={res.objective:.9f} "
              f"delta={abs(obj_recomp - res.objective):.3e}", flush=True)
        print(f"  VERIFY PASS     : {row_bad == 0 and bnd_bad == 0 and int_bad == 0}",
              flush=True)


if __name__ == "__main__":
    main()
