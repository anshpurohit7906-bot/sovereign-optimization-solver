"""MIPLIB benchmark harness: indigenous B&B vs SciPy/HiGHS oracle.

Solves real MIPLIB2017 instances through the production B&B engine and
cross-checks each objective against ``scipy.optimize.milp`` (HiGHS backend).

Supported instances (subset available in data/):
  - pk1.mps      (86 vars, 45 constraints, 55 binary)
  - mas74.mps    (151 vars, 13 constraints, 150 binary)
  - 50v-10.mps   (2013 vars, 233 constraints, 1464 binary + 183 general int)

Usage::
    python tools/benchmark_miplib.py                    # all instances
    python tools/benchmark_miplib.py --only pk1 mas74   # subset

Exit code 0 iff every instance PASSes (objective within 1e-4 rel vs HiGHS).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

from opticore.numerical_model import load_numeric_mps
from opticore.lp.branch_bound import solve_milp

# Real MIPLIB2017 instances available in data/
# All have binary variables; 50v-10 also has general integers (LI/UI bounds)
INSTANCE_SPECS = {
    "pk1":       {"sense": "min"},
    "mas74":     {"sense": "min"},
    "50v-10":    {"sense": "min"},
}

PASS_REL_TOL = 1e-4


def _highs_oracle(lp, maximize: bool) -> float | None:
    """Solve via SciPy HiGHS as independent oracle."""
    try:
        A = np.asarray(lp.A.toarray())  # ensure dense for scipy
        constraints = []
        for i in range(lp.num_constraints):
            row = A[i]
            row_type = lp.row_types[i]
            bi = lp.b[i]
            if row_type == "E":
                constraints.append(LinearConstraint(row, bi, bi))
            elif row_type == "L":
                constraints.append(LinearConstraint(row, -np.inf, bi))
            elif row_type == "G":
                constraints.append(LinearConstraint(row, bi, np.inf))
        lb = np.asarray(lp.lower_bounds, dtype=np.float64)
        ub = np.asarray(lp.upper_bounds, dtype=np.float64)
        bounds = Bounds(lb, ub)
        integrality = np.asarray(lp.is_integer, dtype=int)
        c = np.asarray(lp.c)
        if maximize:
            c = -c
        res = milp(c, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"disp": False})
        return float(res.fun) if res.success else None
    except Exception as e:
        print(f"    [oracle] failed: {e}")
        return None


def _solve_one(name: str, mps_path: Path, maximize: bool,
                *, node_limit: int = 5000, time_limit: float = 60.0,
                verbose: bool = False) -> dict:
    lp = load_numeric_mps(str(mps_path), sparse=True)

    # Extract problem statistics
    n_vars = lp.num_vars
    n_constraints = lp.num_constraints
    n_binaries = int(sum(1 for i in range(n_vars) if lp.is_integer[i]
                         and lp.lower_bounds[i] == 0 and lp.upper_bounds[i] == 1))
    n_general_int = int(sum(1 for i in range(n_vars) if lp.is_integer[i]
                            and not (lp.lower_bounds[i] == 0 and lp.upper_bounds[i] == 1)))
    n_continuous = int(sum(1 for i in range(n_vars) if not lp.is_integer[i]))
    nnz = int(lp.nnz)

    # Use the integer mask from the parser
    integer_mask = np.asarray(lp.is_integer, dtype=bool)

    t0 = time.perf_counter()
    res = solve_milp(lp, integer_mask=integer_mask, maximize=maximize, gap_tol=1e-4,
                     crossover_fallback=True, node_crossover_fallback=False,
                     node_limit=node_limit, time_limit=time_limit,
                     verbose=verbose)
    dt = time.perf_counter() - t0

    oracle_obj = _highs_oracle(lp, maximize)
    if oracle_obj is None:
        rel_err = float("nan")
        match = False
    else:
        obj_err = abs((res.objective or 0.0) - oracle_obj)
        rel_err = obj_err / (1.0 + abs(oracle_obj))
        match = rel_err <= PASS_REL_TOL

    return {
        "instance": name,
        "variables": n_vars,
        "constraints": n_constraints,
        "binaries": n_binaries,
        "general_integers": n_general_int,
        "continuous": n_continuous,
        "nnz": nnz,
        "status": res.status,
        "objective": res.objective,
        "oracle_obj": oracle_obj,
        "rel_err": rel_err,
        "match": match,
        "nodes": res.nodes_explored,
        "lp_solves": res.lp_solves,
        "time_sec": dt,
    }


def run_benchmark(
    data_dir: str,
    out_md: str,
    out_csv: str,
    only: tuple[str, ...] | None = None,
    node_limit: int = 5000,
    time_limit: float = 60.0,
    verbose: bool = False,
) -> list[dict]:
    all_paths = sorted(Path(data_dir).glob("*.mps"))
    if only:
        wanted = set(o.lower() for o in only)
        paths = [p for p in all_paths if p.stem.lower() in wanted]
        missing = wanted - {p.stem.lower() for p in paths}
        if missing:
            raise ValueError(f"requested instance(s) not found: {sorted(missing)}")
    else:
        paths = all_paths

    rows = []
    for p in paths:
        name = p.stem.lower()
        if name not in INSTANCE_SPECS:
            print(f"  Skipping {name}: not in benchmark subset")
            continue
        maximize = INSTANCE_SPECS[name]["sense"] == "max"
        row = _solve_one(name, p, maximize,
                         node_limit=node_limit, time_limit=time_limit,
                         verbose=verbose)
        rows.append(row)
        print(
            f"[{len(rows)}/{len(paths)}] {row['instance']:8s} "
            f"vars={row['variables']:4d} cons={row['constraints']:3d} "
            f"bin={row['binaries']:4d} genint={row['general_integers']:3d} "
            f"cont={row['continuous']:4d} nnz={row['nnz']:6d} "
            f"status={row['status']:12s} "
            f"obj={str(row['objective']):>12s} "
            f"oracle={str(row['oracle_obj']):>12s} "
            f"rel_err={row['rel_err']:.2e} "
            f"nodes={row['nodes']} lp_solves={row['lp_solves']} "
            f"time={row['time_sec']:.2f}s {'PASS' if row['match'] else 'FAIL'}"
        )

    # Write markdown
    lines = [
        "# MIPLIB Benchmark: indigenous B&B vs HiGHS",
        "",
        f"Acceptance: relative objective error <= {PASS_REL_TOL}.",
        "",
        "| instance | variables | constraints | binaries | gen. int | continuous | nnz | status | our_obj | oracle_obj | rel_err | nodes | lp_solves | time (s) | PASS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        obj_str = f"{r['objective']:.6f}" if r['objective'] is not None else "None"
        oracle_str = f"{r['oracle_obj']:.6f}" if r['oracle_obj'] is not None else "None"
        lines.append(
            f"| {r['instance']} | {r['variables']} | {r['constraints']} | {r['binaries']} | "
            f"{r['general_integers']} | {r['continuous']} | {r['nnz']} | {r['status']} | "
            f"{obj_str} | {oracle_str} | {r['rel_err']:.3e} | {r['nodes']} | {r['lp_solves']} | "
            f"{r['time_sec']:.2f} | {'PASS' if r['match'] else 'FAIL'} |"
        )
    passed = sum(1 for r in rows if r["match"])
    lines += ["", f"**Summary:** {passed}/{len(rows)} instances matched HiGHS.", ""]
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")

    # Write CSV
    cols = ["instance", "variables", "constraints", "binaries", "general_integers",
            "continuous", "nnz", "status", "objective", "oracle_obj", "rel_err",
            "match", "nodes", "lp_solves", "time_sec"]
    def _cell(v):
        if v is None:
            return ""
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.12g}"
        return str(v)
    csv_lines = [",".join(cols)]
    for r in rows:
        csv_lines.append(",".join(_cell(r[c]) for c in cols))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.path.join(_ROOT, "data"))
    parser.add_argument("--out-md", default=os.path.join(_ROOT, "results", "benchmark_miplib.md"))
    parser.add_argument("--out-csv", default=os.path.join(_ROOT, "results", "benchmark_miplib.csv"))
    parser.add_argument("--only", nargs="+", default=None,
                        help="restrict to these MPS stems")
    parser.add_argument("--node-limit", type=int, default=5000,
                        help="B&B node limit per instance (default: 5000)")
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="wall-clock time limit per instance in seconds (default: 60)")
    parser.add_argument("--verbose", action="store_true",
                        help="stream per-node B&B iteration lines")
    args = parser.parse_args(argv)

    rows = run_benchmark(args.data_dir, args.out_md, args.out_csv,
                         only=tuple(args.only) if args.only else None,
                         node_limit=args.node_limit, time_limit=args.time_limit,
                         verbose=args.verbose)
    n_pass = sum(1 for r in rows if r["match"])
    print(f"\nPASS {n_pass}/{len(rows)}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())