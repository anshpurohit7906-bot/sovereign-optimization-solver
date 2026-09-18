"""Live demonstration of OPTICORE's canonical sparse convex QP solver
on original Maros-Mezzaros benchmark instances.

Uses the reader, solver, and independent certificate from the src/qp package
without any external optimizers (SciPy/HiG/CVXOPT/Gurobi).
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

# Bootstrap src so `qp` resolves to the canonical package (not legacy src/lp/qp)
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_LP = str(_ROOT / "src" / "lp")
_SRC = str(_ROOT / "src")
# Remove any shadowing paths first
for _p in (_ROOT, _LP, _SRC):
    while _p in sys.path:
        sys.path.remove(_p)
# Insert in reverse priority order so `src` (qp package) takes precedence over `src/lp` (qp module)
for _p in (_LP, _SRC, _ROOT):
    sys.path.insert(0, _p)

from qp.qps import read_qps
from qp.solver import solve_qp
from qp.verify import certificate


BENCHMARKS = [
    "data/qp/real_benchmarks/maros_meszaros/QGROW15.SIF",
    "data/qp/real_benchmarks/maros_meszaros/QSC205.SIF",
]


def run_instance(path: str):
    name = Path(path).name
    print("=" * 60)
    print(f"Benchmark instance: {name}")
    print("Source: Maros-Mezzaros")
    print("Original QPS/SIF input")

    # Parse the original QP
    problem = read_qps(path)
    var_count = problem.n
    eq_count = problem.m_eq
    ineq_count = problem.m_ineq
    print(f"Variable count:   {var_count}")
    print(f"Equality count:   {eq_count}")
    print(f"Inequality count: {ineq_count}")

    # Solve live with verbose iteration reporting
    t0 = time.perf_counter()
    result = solve_qp(
        problem,
        max_iterations=200,
        tol=1e-7,
        verbose=True,
        fraction=0.995,
        regularization=1e-10,
    )
    runtime = time.perf_counter() - t0

    # Print actual solver results
    print(f"Final objective:  {result.objective:.6e}")
    print(f"Convergence status: {result.status}")
    print(f"Iterations:       {result.iterations}")
    print(f"Primal residual:  {result.primal_residual:.2e}")
    print(f"Dual residual:    {result.dual_residual:.2e}")
    print(f"Complementarity:  {result.complementarity:.2e}")
    print(f"Runtime:          {runtime:.3f}s")
    print(f"Message:          {result.message}")

    # Independent KKT certificate
    cert = certificate(
        problem,
        result.x,
        y=result.y if result.y.size > 0 else None,
        z=result.z if result.z.size > 0 else None,
        s=result.s if result.s.size > 0 else None,
        tol=1e-6,
    )

    print()
    kkt_pass = cert["ok"]
    print(f"KKT CERTIFICATE: {'PASS' if kkt_pass else 'FAIL'}")
    print(f"  stationarity:     {cert['stationarity']:.2e}")
    print(f"  scaled_stationarity: {cert['scaled_stationarity']:.2e}")
    print(f"  equality_residual: {cert['equality_residual']:.2e}")
    print(f"  inequality_violation: {cert['inequality_violation']:.2e}")
    print(f"  dual_violation:   {cert['dual_violation']:.2e}")
    print(f"  complementarity:  {cert['complementarity']:.2e}")
    print(f"  objective:        {cert['objective']:.6e}")

    return {
        "name": name,
        "n": var_count,
        "m_eq": eq_count,
        "m_ineq": ineq_count,
        "iterations": result.iterations,
        "objective": result.objective,
        "runtime": runtime,
        "solved": result.status == "optimal",
        "kkt_pass": kkt_pass,
    }


def main():
    results = []
    for path in BENCHMARKS:
        try:
            res = run_instance(path)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] {Path(path).name} failed with exception: {e}")
            results.append({
                "name": Path(path).name,
                "n": None,
                "m_eq": None,
                "m_ineq": None,
                "iterations": None,
                "objective": None,
                "runtime": 0.0,
                "solved": False,
                "kkt_pass": False,
            })

    # Summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"{'Instance':<15} | {'n':>5} | {'constraints':>12} | {'iterations':>10} | "
        f"{'objective':>12} | {'runtime':>8} | {'KKT':>5}"
    )
    print("-" * 60)
    for r in results:
        kkt_str = "PASS" if r["kkt_pass"] else "FAIL"
        n = r["n"] if r["n"] is not None else "?"
        constraints = f"{r['m_eq']}+{r['m_ineq']}" if r["m_eq"] is not None and r["m_ineq"] is not None else "?"
        iters = r["iterations"] if r["iterations"] is not None else "?"
        obj = f"{r['objective']:.6e}" if r["objective"] is not None else "?"
        rt = f"{r['runtime']:.3f}"
        print(
            f"{r['name']:<15} | {n:>5} | {constraints:>12} | {str(iters):>10} | "
            f"{obj:>12} | {rt:>8} | {kkt_str:>5}"
        )

    # Exit nonzero if any benchmark failed to solve or pass certification
    all_ok = all(r["solved"] and r["kkt_pass"] for r in results)
    print()
    if all_ok:
        print("All benchmarks solved and certified PASS.")
        return 0
    else:
        print("One or more benchmarks FAILED (solve or certification).")
        return 1


if __name__ == "__main__":
    sys.exit(main())