"""External, read-only AFIRO CPU breakdown diagnostic.

No tracked solver source is edited. Runtime wrappers delegate to the original
functions and objects; the line tracer only observes selected production lines.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opticore.lp import linear_system as ls
from opticore.lp import mehrotra as meh
from opticore.numerical_model import load_numeric_mps

ORIGINAL_SPLU = ls.splu
ORIGINAL_SPLU_REG = ls._splu_regularized
ORIGINAL_RESIDUAL = ls._schur_residual
ORIGINAL_FACTOR_OBJECT = ls.ReducedNewtonFactorization
ORIGINAL_FACTOR = meh.factor_reduced_system
ORIGINAL_TO_STANDARD_FORM = meh.to_standard_form
ORIGINAL_SOLVE_STANDARD_FORM = meh.solve_standard_form

TOL = 1e-7
MAX_ITER = 100


def source_line(function, needle: str, *, after: int = -1) -> int:
    lines, first = inspect.getsourcelines(function)
    for index, line in enumerate(lines):
        number = first + index
        if number > after and needle in line:
            return number
    raise RuntimeError(f"source line not found: {needle!r}")


FACTOR_H = source_line(ls.factor_reduced_system, "scale = max(1.0, float(np.mean(np.abs(diag))))")
FACTOR_CSR = source_line(ls.factor_reduced_system, "A_sp = sp.csr_matrix(A_eq)")
FACTOR_W = source_line(ls.factor_reduced_system, "W_sp = A_sp.T.multiply")
FACTOR_S = source_line(ls.factor_reduced_system, "S_sp = A_sp @ W_sp")
REG_SYM = source_line(ls._splu_regularized, "M = (0.5 * (S + S.T)).tocsc()")
REG_DIAG_CALL_LINE = source_line(ls._splu_regularized, "scale = max(1.0, float(np.mean(np.abs(M.diagonal()))))")
MEH_RHS_CALL_LINES = {
    source_line(meh.solve_standard_form, "rc = x * z", after=900),
    source_line(meh.solve_standard_form, "rc = x * z - sigma * mu", after=900),
}
SOLVE_RHS_CALL_LINE = source_line(ls.solve_reduced_system, "t = rhs_x / fac.h_diag")
REG_DIAG_END = source_line(ls._splu_regularized, "permc_spec=\"MMD_AT_PLUS_A\")")
REG_DIAG_START = REG_DIAG_CALL_LINE
SOLVE_T = SOLVE_RHS_CALL_LINE
SOLVE_SCHUR_RHS = source_line(ls.solve_reduced_system, "b = rhs_eq - fac.A_sp @ t")
MEH_RHS_LINES = {
    source_line(meh.solve_standard_form, "rc = x * z", after=900),
    source_line(meh.solve_standard_form, "rhs_x = r_d - rc / x", after=900),
    source_line(meh.solve_standard_form, "rc = x * z - sigma * mu", after=900),
    source_line(meh.solve_standard_form, "rhs_x = r_d - rc / x", after=930),
}

TARGET_CODES = {
    meh.solve_standard_form.__code__,
    ls.factor_reduced_system.__code__,
    ls._splu_regularized.__code__,
    ls.solve_reduced_system.__code__,
}
METRICS: dict[str, float | int] = {}


def reset_metrics() -> None:
    METRICS.clear()
    for name in (
        "total", "conversion", "core", "h_reg", "csr", "w_sp", "s_sp",
        "sym_s", "diag_reg", "splu", "lu_solve", "factor_object",
        "rhs", "schur_rhs", "residual", "other_core", "wrapper_residual",
        "factor_calls", "splu_calls", "lu_solve_calls", "factor_object_calls",
        "rhs_calls", "schur_rhs_calls", "residual_calls",
    ):
        METRICS[name] = 0


def add(name: str, seconds: float) -> None:
    METRICS[name] = float(METRICS.get(name, 0.0)) + seconds


def count(name: str) -> None:
    METRICS[name] = int(METRICS.get(name, 0)) + 1


def line_category(frame, line: int) -> str | None:
    if frame.f_code is ORIGINAL_SOLVE_STANDARD_FORM.__code__:
        return "rhs" if line in MEH_RHS_LINES else None
    if frame.f_code is ORIGINAL_FACTOR.__code__:
        if FACTOR_H <= line < FACTOR_CSR:
            return "h_reg"
        if line == FACTOR_CSR:
            return "csr"
        if line == FACTOR_W:
            return "w_sp"
        if line == FACTOR_S:
            return "s_sp"
    if frame.f_code is ORIGINAL_SPLU_REG.__code__:
        if line == REG_SYM:
            return "sym_s"
        if REG_DIAG_START <= line <= REG_DIAG_END:
            return "diag_reg"
    if frame.f_code is ls.solve_reduced_system.__code__ and (line == SOLVE_T or line == SOLVE_SCHUR_RHS):
        return "schur_rhs"
    return None


SPLU_INTERVALS: list[tuple[float, float]] = []


class TimedSuperLU:
    def __init__(self, delegate):
        self._delegate = delegate

    def solve(self, *args, **kwargs):
        count("lu_solve_calls")
        start = perf_counter()
        try:
            return self._delegate.solve(*args, **kwargs)
        finally:
            add("lu_solve", perf_counter() - start)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def timed_splu(matrix, **kwargs):
    count("splu_calls")
    start = perf_counter()
    try:
        return TimedSuperLU(ORIGINAL_SPLU(matrix, **kwargs))
    finally:
        end = perf_counter()
        add("splu", end - start)
        SPLU_INTERVALS.append((start, end))


def timed_factor_object(*args, **kwargs):
    count("factor_object_calls")
    start = perf_counter()
    try:
        return ORIGINAL_FACTOR_OBJECT(*args, **kwargs)
    finally:
        add("factor_object", perf_counter() - start)


def timed_residual(*args, **kwargs):
    count("residual_calls")
    start = perf_counter()
    try:
        return ORIGINAL_RESIDUAL(*args, **kwargs)
    finally:
        add("residual", perf_counter() - start)


def timed_factor(*args, **kwargs):
    count("factor_calls")
    start = perf_counter()
    try:
        return ORIGINAL_FACTOR(*args, **kwargs)
    finally:
        add("factor_total", perf_counter() - start)


def timed_conversion(*args, **kwargs):
    start = perf_counter()
    try:
        return ORIGINAL_TO_STANDARD_FORM(*args, **kwargs)
    finally:
        add("conversion", perf_counter() - start)


def timed_core(*args, **kwargs):
    start = perf_counter()
    try:
        return ORIGINAL_SOLVE_STANDARD_FORM(*args, **kwargs)
    finally:
        add("core", perf_counter() - start)


class LineTracer:
    def __init__(self):
        self.last: dict[int, tuple[int, float]] = {}

    def trace(self, frame, event, arg):
        if frame.f_code in TARGET_CODES:
            return self.local
        return None

    def local(self, frame, event, arg):
        now = perf_counter()
        key = id(frame)
        previous = self.last.get(key)
        if event == "line":
            if previous is not None:
                previous_line, previous_time = previous
                category = line_category(frame, previous_line)
                if category is not None:
                    duration = now - previous_time
                    if category == "diag_reg":
                        duration = max(0.0, duration - sum(
                            max(0.0, min(now, end) - max(previous_time, start))
                            for start, end in SPLU_INTERVALS
                        ))
                    add(category, duration)
            current_category = line_category(frame, frame.f_lineno)
            if current_category == "h_reg" and frame.f_lineno == FACTOR_H:
                count("h_reg_calls")
            elif current_category == "csr":
                count("csr_calls")
            elif current_category == "w_sp":
                count("w_sp_calls")
            elif current_category == "s_sp":
                count("s_sp_calls")
            elif current_category == "sym_s":
                count("sym_s_calls")
            elif current_category == "diag_reg" and frame.f_lineno == REG_DIAG_CALL_LINE:
                count("diag_reg_calls")
            elif current_category == "rhs" and frame.f_lineno in MEH_RHS_CALL_LINES:
                count("rhs_calls")
            elif current_category == "schur_rhs" and frame.f_lineno == SOLVE_RHS_CALL_LINE:
                count("schur_rhs_calls")
            self.last[key] = (frame.f_lineno, now)
        elif event == "return":
            if previous is not None:
                previous_line, previous_time = previous
                category = line_category(frame, previous_line)
                if category is not None:
                    add(category, now - previous_time)
            self.last.pop(key, None)
        return self.local




def run_breakdown(mps_path: Path):
    reset_metrics()
    SPLU_INTERVALS.clear()

    lp = load_numeric_mps(str(mps_path))

    # Reference uninstrumented baseline
    ref_res = meh.solve_lp(lp, tol=TOL, max_iter=MAX_ITER)

    # Monkeypatch for measurement
    ls.splu = timed_splu
    ls._schur_residual = timed_residual
    ls.ReducedNewtonFactorization = timed_factor_object
    meh.factor_reduced_system = timed_factor
    meh.to_standard_form = timed_conversion
    meh.solve_standard_form = timed_core

    tracer = LineTracer()
    old_trace = sys.gettrace()
    total_start = perf_counter()
    try:
        sys.settrace(tracer.trace)
        inst_res = meh.solve_lp(lp, tol=TOL, max_iter=MAX_ITER)
    finally:
        sys.settrace(old_trace)
        add("total", perf_counter() - total_start)
        # Restore original functions
        ls.splu = ORIGINAL_SPLU
        ls._schur_residual = ORIGINAL_RESIDUAL
        ls.ReducedNewtonFactorization = ORIGINAL_FACTOR_OBJECT
        meh.factor_reduced_system = ORIGINAL_FACTOR
        meh.to_standard_form = ORIGINAL_TO_STANDARD_FORM
        meh.solve_standard_form = ORIGINAL_SOLVE_STANDARD_FORM

    # Validate numerical identicalness
    assert inst_res.status == ref_res.status
    assert inst_res.iterations == ref_res.iterations
    assert np.isclose(inst_res.objective, ref_res.objective, rtol=1e-10, atol=1e-10)
    assert np.allclose(inst_res.x, ref_res.x, rtol=1e-10, atol=1e-10)

    total = METRICS["total"]
    core = METRICS["core"]
    conversion = METRICS["conversion"]

    named_core = (
        METRICS["h_reg"]
        + METRICS["csr"]
        + METRICS["w_sp"]
        + METRICS["s_sp"]
        + METRICS["sym_s"]
        + METRICS["diag_reg"]
        + METRICS["splu"]
        + METRICS["lu_solve"]
        + METRICS["factor_object"]
        + METRICS["rhs"]
        + METRICS["schur_rhs"]
        + METRICS["residual"]
    )
    other_core = max(0.0, core - named_core)
    METRICS["other_core"] = other_core
    METRICS["wrapper_residual"] = max(0.0, total - (conversion + core))

    print("=" * 80)
    print("AFIRO CPU EXECUTION TIME DECOMPOSITION (1 Solve)")
    print("=" * 80)
    print(f"Status: {inst_res.status} | Iterations: {inst_res.iterations} | Obj: {inst_res.objective:.12e}")
    print(f"Numerical match with uninstrumented reference: EXACT (x close, obj close, iters identical)")
    print("-" * 80)
    print(f"{'Component':<35} {'Time (s)':<12} {'% Total':<10} {'Call Count':<10}")
    print("-" * 80)

    rows = [
        ("Conversion (to_standard_form)", METRICS["conversion"], 1),
        ("Schur H Regularization", METRICS["h_reg"], METRICS.get("h_reg_calls", 0)),
        ("Schur CSR Conversion (A_sp)", METRICS["csr"], METRICS.get("csr_calls", 0)),
        ("Schur W_sp Assembly", METRICS["w_sp"], METRICS.get("w_sp_calls", 0)),
        ("Schur S_sp Assembly (A@W)", METRICS["s_sp"], METRICS.get("s_sp_calls", 0)),
        ("Schur Symmetrization sym(S)", METRICS["sym_s"], METRICS.get("sym_s_calls", 0)),
        ("Schur Diagonal Regularization", METRICS["diag_reg"], METRICS.get("diag_reg_calls", 0)),
        ("SuperLU Factorization (splu)", METRICS["splu"], METRICS["splu_calls"]),
        ("SuperLU Solves (solve + refine)", METRICS["lu_solve"], METRICS["lu_solve_calls"]),
        ("Factor-Object Construction", METRICS["factor_object"], METRICS["factor_object_calls"]),
        ("RHS Construction (Mehrotra rc/rhs_x)", METRICS["rhs"], METRICS.get("rhs_calls", 0)),
        ("RHS Construction (Schur t/b)", METRICS["schur_rhs"], METRICS.get("schur_rhs_calls", 0)),
        ("Iterative Refinement Residual", METRICS["residual"], METRICS["residual_calls"]),
        ("Other Mehrotra Iteration Work", METRICS["other_core"], "-"),
        ("Solve_lp Wrapper / Frame Overhead", METRICS["wrapper_residual"], "-"),
    ]

    for label, sec, cnt in rows:
        pct = (sec / total) * 100.0 if total > 0 else 0.0
        cnt_str = str(cnt) if isinstance(cnt, int) else cnt
        print(f"{label:<35} {sec:10.6f}s  {pct:8.2f}%  {cnt_str:<10}")

    print("-" * 80)
    print(f"{'Total solve_lp() Time':<35} {total:10.6f}s  {100.00:8.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    mps = ROOT / "tests" / "fixtures" / "afiro.mps"
    if not mps.exists():
        mps = ROOT / "data" / "afiro.mps"
    run_breakdown(mps)
