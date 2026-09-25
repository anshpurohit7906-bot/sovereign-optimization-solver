"""Read-only AFIRO CPU timing diagnostic for OPTICORE.

All wrappers forward original positional/keyword arguments and return values.
No numerical solver source file is modified by this diagnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path
import inspect

from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opticore.lp import linear_system as ls
from opticore.lp import mehrotra as meh
from opticore.lp.mehrotra import solve_lp
from opticore.numerical_model import load_numeric_mps

ORIGINAL_SPLU = ls.splu
ORIGINAL_FACTOR = meh.factor_reduced_system
FACTOR_LINES, FACTOR_FIRST_LINE = inspect.getsourcelines(ORIGINAL_FACTOR)
W_LINE = next(i for i, line in enumerate(FACTOR_LINES) if "W_sp =" in line)
S_LINE = next(i for i, line in enumerate(FACTOR_LINES) if "S_sp = A_sp @ W_sp" in line)
ASSEMBLY_START_LINE = FACTOR_FIRST_LINE + W_LINE
ASSEMBLY_END_LINE = FACTOR_FIRST_LINE + S_LINE + 1
METRICS = {}


def reset_metrics():
    METRICS.clear()
    METRICS.update(
        factor_time=0.0,
        assembly_time=0.0,
        splu_time=0.0,
        solve_time=0.0,
        factor_calls=0,
        splu_calls=0,
        solve_calls=0,
    )


class TimedSuperLU:
    """Delegate to an unmodified SuperLU object; time only solve()."""

    def __init__(self, delegate):
        self._delegate = delegate

    def solve(self, *args, **kwargs):
        METRICS["solve_calls"] += 1
        start = perf_counter()
        try:
            return self._delegate.solve(*args, **kwargs)
        finally:
            METRICS["solve_time"] += perf_counter() - start

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def timed_splu(matrix, **kwargs):
    """Time the original SuperLU factorization and return a timed delegate."""
    METRICS["splu_calls"] += 1
    start = perf_counter()
    try:
        return TimedSuperLU(ORIGINAL_SPLU(matrix, **kwargs))
    finally:
        METRICS["splu_time"] += perf_counter() - start


class AssemblyTracer:
    """Time W_sp/S_sp only within the unmodified factor_reduced_system frame."""

    def __init__(self):
        self.start = None
        self.total = 0.0

    def trace(self, frame, event, arg):
        if frame.f_code is ORIGINAL_FACTOR.__code__:
            return self._trace_factor
        return None

    def _trace_factor(self, frame, event, arg):
        if event == "line":
            if frame.f_lineno == ASSEMBLY_START_LINE:
                self.start = perf_counter()
            elif self.start is not None and frame.f_lineno == ASSEMBLY_END_LINE:
                self.total += perf_counter() - self.start
                self.start = None
        elif event == "return" and self.start is not None:
            self.total += perf_counter() - self.start
            self.start = None
        return self._trace_factor


ASSEMBLY_TRACER = AssemblyTracer()


def timed_factor(*args, **kwargs):
    """Time original factorization, exact W/S assembly, and its splu call."""
    METRICS["factor_calls"] += 1
    splu_before = METRICS["splu_time"]
    ASSEMBLY_TRACER.start = None
    ASSEMBLY_TRACER.total = 0.0
    start = perf_counter()
    sys.settrace(ASSEMBLY_TRACER.trace)
    try:
        return ORIGINAL_FACTOR(*args, **kwargs)
    finally:
        sys.settrace(None)
        METRICS["factor_time"] += perf_counter() - start
        METRICS["assembly_time"] += ASSEMBLY_TRACER.total
        # Record the complete non-splu factor time separately for comparison.
        METRICS["nonsplu_factor_time"] = (
            METRICS.get("nonsplu_factor_time", 0.0)
            + (perf_counter() - start)
            - (METRICS["splu_time"] - splu_before)
        )


def results_match(reference, observed):
    fields = (
        ("status", lambda r: r.status),
        ("iterations", lambda r: r.iterations),
        ("objective", lambda r: r.objective),
        ("x_standard", lambda r: np.array_equal(r.x_standard, reference.x_standard)),
        ("y", lambda r: np.array_equal(r.y, reference.y)),
        ("z_standard", lambda r: np.array_equal(r.z_standard, reference.z_standard)),
    )
    return {name: func(observed) == func(reference) for name, func in fields}


def run_timed(lp):
    reset_metrics()
    start = perf_counter()
    result = solve_lp(lp, tol=1e-7, max_iter=100, backend="cpu")
    total = perf_counter() - start
    remaining = total - (METRICS["assembly_time"] + METRICS["splu_time"] + METRICS["solve_time"])
    return result, total, remaining, dict(METRICS)


def main():
    lp = load_numeric_mps(ROOT / "data" / "afiro.mps", sparse=True)

    # Baseline is run without any instrumentation and is excluded from timings.
    reference = solve_lp(lp, tol=1e-7, max_iter=100, backend="cpu")

    ls.splu = timed_splu
    meh.factor_reduced_system = timed_factor
    runs = []
    try:
        for _ in range(5):
            runs.append(run_timed(lp))
    finally:
        ls.splu = ORIGINAL_SPLU
        meh.factor_reduced_system = ORIGINAL_FACTOR

    print("AFIRO CPU timing diagnostic (5 consecutive instrumented solves)")
    print("solve arguments: tol=1e-7 max_iter=100 backend=cpu")
    print("scope: solve_lp only; model loading and baseline are excluded")
    print("assembly definition: W_sp construction plus S_sp = A_sp @ W_sp only")
    print("excluded from assembly: CSR conversion, H regularization, sym(S), diagonal regularization, factor-object construction")
    print()
    print("run,total_s,assembly_s,splu_s,lu_solve_s,remaining_s,factor_calls,splu_calls,lu_solve_calls")
    for index, (_, total, remaining, metrics) in enumerate(runs, 1):

        print(
            f"{index},{total:.9f},{metrics['assembly_time']:.9f},"
            f"{metrics['splu_time']:.9f},{metrics['solve_time']:.9f},"
            f"{remaining:.9f},{metrics['factor_calls']},{metrics['splu_calls']},{metrics['solve_calls']}"
        )

    labels = (
        ("total", "total_s"),
        ("assembly", "assembly_s"),
        ("splu", "splu_s"),
        ("lu_solve", "lu_solve_s"),
        ("remaining", "remaining_s"),
    )
    median_total = float(np.median([total for _, total, _, _ in runs]))
    print()
    print("median total solve time_s:", f"{median_total:.9f}")
    for key, label in labels[1:]:
        values = np.array([
            total - remaining if key == "assembly" else
            remaining if key == "remaining" else
            metrics[{"splu": "splu_time", "lu_solve": "solve_time"}[key]]
            for _, total, remaining, metrics in runs
        ], dtype=float)
        median = float(np.median(values))
        print(f"{label}: median={median:.9f} percent_of_median_total={median / median_total * 100:.3f}%")

    total_values = np.array([total for _, total, _, _ in runs], dtype=float)
    component_values = {
        "assembly": np.array([m["assembly_time"] for _, _, _, m in runs], dtype=float),
        "splu": np.array([m["splu_time"] for _, _, _, m in runs], dtype=float),
        "lu_solve": np.array([m["solve_time"] for _, _, _, m in runs], dtype=float),
        "remaining": np.array([remaining for _, _, remaining, _ in runs], dtype=float),
    }
    print()
    print("median total solve time_s:", f"{float(np.median(total_values)):.9f}")
    for key, label in (("assembly", "assembly_s"), ("splu", "splu_s"),
                       ("lu_solve", "lu_solve_s"), ("remaining", "remaining_s")):
        values = component_values[key]
        shares = values / total_values * 100.0
        print(
            f"{label}: median_cumulative={float(np.median(values)):.9f} "
            f"median_percent={float(np.median(shares)):.3f}% "
            f"aggregate_percent={float(values.sum() / total_values.sum() * 100.0):.3f}%"
        )
    print("aggregate_total_s:", f"{float(total_values.sum()):.9f}")
    print("aggregate_splu_calls:", sum(m["splu_calls"] for _, _, _, m in runs))
    print("aggregate_lu_solve_calls:", sum(m["solve_calls"] for _, _, _, m in runs))

    first_result = runs[0][0]
    print()
    print("instrumented-vs-baseline result match:", results_match(reference, first_result))
    print("status:", first_result.status, "iterations:", first_result.iterations,
          "objective:", f"{first_result.objective:.12f}")


if __name__ == "__main__":
    main()
