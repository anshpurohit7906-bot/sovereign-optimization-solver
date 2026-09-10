#!/usr/bin/env python
"""Clean production benchmark: MMD_AT_PLUS_A vs COLAMD baseline.

EXPERIMENT ONLY: does not modify src/, tests/, or requirements/.
No monkeypatching. No instrumentation. Pure wall-clock solve_lp() calls.

Runs the real production solve on 5 models and compares against
the COLAMD baseline from profile_ordering_full_solve (same production
solve path, same settings).

Settings: tol=1e-7, max_iter=100 (matching tests/run_benchmarks.py).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _SRC / "lp", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from numerical_model import load_numeric_mps  # noqa: E402
from lp.mehrotra import solve_lp  # noqa: E402

RESULTS_DIR = _ROOT / "results"

MODELS = [
    {"file": "afiro.mps",      "expected_obj": -464.7531428571},
    {"file": "blend.mps",      "expected_obj": -30.8121498458},
    {"file": "sc205.mps",      "expected_obj": -52.2020612117},
    {"file": "pilot4_plain.mps", "expected_obj": -2581.106635},
    {"file": "pilot87.mps",    "expected_obj": 301.744869},
]
TOL = 1e-7
MAX_ITER = 100

# Previous COLAMD baseline timings from profile_ordering_full_solve
# (same production solve_lp path with thin timing instrumentation,
# overhead negligible at <1ms total for small models, <10ms for pilot87).
COLAMD_BASELINE = {
    "afiro.mps":      None,  # no previous COLAMD timing
    "blend.mps":      None,
    "sc205.mps":      None,
    "pilot4_plain.mps": {"time": 4.54, "iters": 72, "status": "stalled",
                         "obj": -2581.11, "rel_p": 4.14e-06, "rel_d": 3.05e-06,
                         "rel_gap": 1.08e-05},
    "pilot87.mps":    {"time": 134.54, "iters": 63, "status": "stalled",
                        "obj": 301.744869, "rel_p": 3.47e-06, "rel_d": 3.11e-06,
                        "rel_gap": 3.53e-06},
}


def run_benchmark() -> None:
    print("=" * 110)
    print("  Production Benchmark: MMD_AT_PLUS_A (current) vs COLAMD baseline")
    print("  Settings: tol=1e-7, max_iter=100")
    print("=" * 110)
    print()

    # Header
    hdr = (f"{'Model':16s} | {'Status':10s} | {'Iters':5s} | "
           f"{'Obj':>15s} | {'RelPrimal':>10s} | {'RelDual':>10s} | "
           f"{'RelGap':>10s} | {'Time(s)':>8s}")
    print(hdr)
    print("-" * len(hdr))

    results = []
    for bench in MODELS:
        model_file = bench["file"]
        expected_obj = bench["expected_obj"]
        data_path = str(_ROOT / "data" / model_file)

        if not os.path.exists(data_path):
            print(f"{model_file:16s} | NOT FOUND")
            continue

        lp = load_numeric_mps(data_path)
        t0 = time.perf_counter()
        res = solve_lp(lp, tol=TOL, max_iter=MAX_ITER)
        dt = time.perf_counter() - t0

        obj_err = abs(res.objective - expected_obj)
        rel_obj_err = obj_err / (1.0 + abs(expected_obj))

        print(f"{model_file:16s} | {res.status:10s} | {res.iterations:5d} | "
              f"{res.objective:15.6f} | {res.rel_primal:10.2e} | "
              f"{res.rel_dual:10.2e} | {res.rel_gap:10.2e} | {dt:8.3f}")

        results.append({
            "model": model_file,
            "status": res.status,
            "iterations": res.iterations,
            "objective": res.objective,
            "expected_obj": expected_obj,
            "rel_obj_err": rel_obj_err,
            "rel_primal": res.rel_primal,
            "rel_dual": res.rel_dual,
            "rel_gap": res.rel_gap,
            "time_sec": dt,
        })

    print("=" * 110)
    print()

    # Comparison table
    print("=" * 110)
    print("  COLAMD vs MMD_AT_PLUS_A Comparison")
    print("=" * 110)
    print()
    print(f"{'Model':16s} | {'COLAMD(s)':>9s} | {'MMD(s)':>9s} | "
          f"{'Speedup':>8s} | {'COL iters':>9s} | {'MMD iters':>9s} | "
          f"{'Iter diff':>9s} | {'Obj diff':>10s}")
    print("-" * 110)

    for r in results:
        model = r["model"]
        baseline = COLAMD_BASELINE.get(model)
        if baseline is None:
            print(f"{model:16s} | {'--':>9s} | {r['time_sec']:9.3f} | "
                  f"{'--':>8s} | {'--':>9s} | {r['iterations']:9d} | "
                  f"{'--':>9s} | {'--':>10s}")
        else:
            speedup = baseline["time"] / r["time_sec"] if r["time_sec"] > 0 else float("inf")
            iter_diff = r["iterations"] - baseline["iters"]
            obj_diff = r["objective"] - baseline["obj"]
            print(f"{model:16s} | {baseline['time']:9.3f} | {r['time_sec']:9.3f} | "
                  f"{speedup:7.2f}x | {baseline['iters']:9d} | {r['iterations']:9d} | "
                  f"{iter_diff:+9d} | {obj_diff:+10.4e}")

    print("=" * 110)
    print()

    # Correctness verification
    print("=" * 110)
    print("  Correctness Verification")
    print("=" * 110)
    for r in results:
        ok = (r["rel_obj_err"] <= 1e-4)
        tag = "OK" if ok else "REGRESSION"
        print(f"  {r['model']:16s}: obj_err={r['rel_obj_err']:.2e}  [{tag}]")
    print("=" * 110)
    print()

    # Save CSV
    csv_path = RESULTS_DIR / "benchmark_production_ordering.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ordering", "status", "iterations", "objective",
                     "rel_primal", "rel_dual", "rel_gap", "time_sec",
                     "expected_obj", "rel_obj_err"])
        for r in results:
            w.writerow([r["model"], "MMD_AT_PLUS_A", r["status"],
                        r["iterations"], r["objective"],
                        r["rel_primal"], r["rel_dual"], r["rel_gap"],
                        r["time_sec"], r["expected_obj"], r["rel_obj_err"]])
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    run_benchmark()
