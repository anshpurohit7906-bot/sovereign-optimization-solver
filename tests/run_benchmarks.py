"""Run Netlib benchmark suite against known reference values."""

from __future__ import annotations

import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import load_numeric_mps
from lp.mehrotra import solve_lp

BENCHMARKS = [
    {"file": "afiro.mps", "expected_obj": -464.7531428571, "tol": 1e-7, "max_iter": 100},
    {"file": "sc205.mps", "expected_obj": -52.2020612117, "tol": 1e-7, "max_iter": 100},
    {"file": "adlittle.mps", "expected_obj": 225494.9631623802, "tol": 1e-7, "max_iter": 100},
    {"file": "share2b.mps", "expected_obj": -415.7322407414, "tol": 1e-7, "max_iter": 100},
    {"file": "blend.mps", "expected_obj": -30.8121498458, "tol": 1e-7, "max_iter": 100},
]


def run_benchmarks(data_dir: str = os.path.join(_ROOT, "data")) -> list[dict]:
    results = []
    print("=" * 110)
    print(f"{'BENCHMARK':12s} | {'STATUS':10s} | {'ITERS':5s} | {'REL PRIMAL':10s} | {'REL DUAL':10s} | {'REL GAP':10s} | {'OBJECTIVE':15s} | {'EXPECTED':15s} | {'REL OBJ ERR':11s}")
    print("=" * 110)

    for bench in BENCHMARKS:
        path = os.path.join(data_dir, bench["file"])
        if not os.path.exists(path):
            print(f"{bench['file']:12s} | NOT FOUND")
            continue

        lp = load_numeric_mps(path)
        t0 = time.time()
        res = solve_lp(lp, tol=bench["tol"], max_iter=bench["max_iter"])
        dt = time.time() - t0

        obj_err = abs(res.objective - bench["expected_obj"])
        rel_obj_err = obj_err / (1.0 + abs(bench["expected_obj"]))
        is_ok = (res.status == "optimal" and res.rel_primal <= bench["tol"]
                 and res.rel_dual <= bench["tol"] and res.rel_gap <= bench["tol"]
                 and rel_obj_err <= 1e-4)

        tag = "PASS" if is_ok else "FAIL"
        print(f"{bench['file']:12s} | {res.status:10s} | {res.iterations:5d} | {res.rel_primal:10.2e} | {res.rel_dual:10.2e} | {res.rel_gap:10.2e} | {res.objective:15.6f} | {bench['expected_obj']:15.6f} | {rel_obj_err:11.2e} [{tag}]")

        results.append({
            "benchmark": bench["file"],
            "status": res.status,
            "iterations": res.iterations,
            "rel_primal": res.rel_primal,
            "rel_dual": res.rel_dual,
            "rel_gap": res.rel_gap,
            "objective": res.objective,
            "expected_obj": bench["expected_obj"],
            "rel_obj_err": rel_obj_err,
            "pass": is_ok,
            "time_sec": dt,
        })

    print("=" * 110)
    passed = sum(1 for r in results if r["pass"])
    print(f"SUMMARY: {passed}/{len(results)} Netlib benchmarks solved to optimality within tolerance.")
    print("=" * 110)
    return results


if __name__ == "__main__":
    run_benchmarks()
