"""Benchmark selected Maros-Mészáros QP instances and generated sparse cases."""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qp import read_qps, solve_qp, certificate, generate_sparse_qp


REAL = ["QAFIRO", "QADLITTLE", "QSC205", "QGROW15", "QPCBOEI1", "QSHIP12S"]


def run_one(problem):
    t = time.perf_counter()
    r = solve_qp(problem, max_iterations=200, tol=1e-6, sparse=True)
    elapsed = time.perf_counter() - t
    c = certificate(problem, r.x, r.y, r.z, r.s, tol=1e-5)
    return {
        "name": problem.name,
        "n": problem.n,
        "m_ineq": problem.m_ineq,
        "m_eq": problem.m_eq,
        "status": r.status,
        "iterations": r.iterations,
        "runtime_seconds": elapsed,
        "objective": r.objective,
        "primal_residual": r.primal_residual,
        "dual_residual": r.dual_residual,
        "complementarity": r.complementarity,
        "kkt_certificate": c["ok"],
        "certificate_stationarity": c["stationarity"],
        "certificate_equality_residual": c["equality_residual"],
        "certificate_inequality_violation": c["inequality_violation"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/qp/real_benchmarks")
    ap.add_argument("--out", default="reports/maros_qp_benchmark.csv")
    ap.add_argument("--include-generated", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    rows = []
    for name in REAL:
        candidates = list(data.rglob(name + ".SIF")) + list(data.rglob(name + ".sif"))
        if not candidates:
            print(f"SKIP {name}: benchmark file not found")
            continue
        try:
            rows.append(run_one(read_qps(candidates[0])))
        except Exception as e:
            rows.append({"name": name, "status": "error", "message": str(e)})

    if args.include_generated:
        for n in (500, 1000, 5000):
            try:
                rows.append(run_one(generate_sparse_qp(n=n, seed=26119+n)))
            except Exception as e:
                rows.append({"name": f"sparse_generated_{n}", "status": "error",
                             "message": str(e)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({k for row in rows for k in row})
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
