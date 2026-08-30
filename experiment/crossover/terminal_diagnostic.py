"""Read-only capture of production Mehrotra's terminal local iterate.

The public result intentionally may contain a best-gap iterate.  This module
uses Python tracing only around a production call to snapshot local state just
before its `_finish()` return paths; it changes no solver data or behavior.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass

import numpy as np

from src.lp.linear_system import factor_reduced_system, solve_reduced_system
from src.lp.mehrotra import MehrotraResult, StandardFormLP, solve_standard_form
from src.scaling import unscale_solution


@dataclass(frozen=True)
class TerminalCapture:
    result: MehrotraResult
    terminal_x_standard: np.ndarray
    terminal_z_standard: np.ndarray
    terminal_iteration: int
    return_line: int
    _scaled_x: np.ndarray
    _scaled_y: np.ndarray
    _scaled_z: np.ndarray
    _scaled_A: np.ndarray
    _scaled_b: np.ndarray
    _scaled_c: np.ndarray


@dataclass(frozen=True)
class NewtonResidualComparison:
    exact_factored_reduced_relative: float
    regularized_top_relative: float
    ideal_newton_block_relative: float
    h_regularization_shift_max: float
    schur_regularization: float


def run_with_terminal_capture(sf: StandardFormLP, **kwargs) -> TerminalCapture:
    """Run production Mehrotra once and capture its pre-`_finish` local state."""
    target = solve_standard_form.__code__
    source, first_line = inspect.getsourcelines(solve_standard_form)
    return_lines = {
        first_line + offset for offset, line in enumerate(source)
        if "return _finish" in line
    }
    captured: dict[str, object] = {}

    def trace(frame, event, arg):
        if frame.f_code is target and event == "line" and frame.f_lineno in return_lines:
            local = frame.f_locals
            if all(name in local for name in ("x", "y", "z", "A", "b", "c", "col_scale", "k")):
                captured.update({
                    "x": local["x"].copy(), "y": local["y"].copy(), "z": local["z"].copy(),
                    "A": local["A"].copy(), "b": local["b"].copy(), "c": local["c"].copy(),
                    "col_scale": local["col_scale"].copy(), "k": int(local["k"]),
                    "line": frame.f_lineno,
                })
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        result = solve_standard_form(sf, **kwargs)
    finally:
        sys.settrace(previous)
    if not captured:
        raise RuntimeError("terminal Mehrotra state was not captured")
    col_scale = captured["col_scale"]
    return TerminalCapture(
        result=result,
        terminal_x_standard=unscale_solution(captured["x"], col_scale),
        terminal_z_standard=captured["z"] / col_scale,
        terminal_iteration=captured["k"], return_line=captured["line"],
        _scaled_x=captured["x"], _scaled_y=captured["y"], _scaled_z=captured["z"],
        _scaled_A=captured["A"], _scaled_b=captured["b"], _scaled_c=captured["c"],
    )


def compare_newton_residuals(capture: TerminalCapture) -> NewtonResidualComparison:
    """Compare a predictor solve against factored and ideal Newton equations."""
    A, b, c = capture._scaled_A, capture._scaled_b, capture._scaled_c
    x, y, z = capture._scaled_x, capture._scaled_y, capture._scaled_z
    h = z / x
    r_p, r_d, r_c = A @ x - b, A.T @ y + z - c, x * z
    rhs_x, rhs_eq = r_d - r_c / x, -r_p
    fac = factor_reduced_system(h, A)
    dx, dy = solve_reduced_system(fac, rhs_x, rhs_eq)

    # The actual sparse factorization solves sym(S_reg)+schur_reg*I, where
    # S_reg=A diag(1/h_reg) A^T.  Measure that exact reduced equation.
    h_reg = fac.h_diag
    schur = A @ ((A.T / h_reg[:, None]))
    rhs_schur = rhs_eq - A @ (rhs_x / h_reg)
    schur_reg = float(fac.schur_reg or 0.0)
    reduced_residual = rhs_schur - (0.5 * (schur + schur.T) @ dy + schur_reg * dy)
    exact_relative = float(np.linalg.norm(reduced_residual, ord=np.inf)) / max(1.0, float(np.linalg.norm(rhs_schur, ord=np.inf)))

    regularized_top = h_reg * dx - A.T @ dy - rhs_x
    regularized_top_relative = float(np.linalg.norm(regularized_top, ord=np.inf)) / max(1.0, float(np.linalg.norm(rhs_x, ord=np.inf)))
    ideal_top = h * dx - A.T @ dy - rhs_x
    ideal_bottom = A @ dx - rhs_eq
    ideal_relative = max(float(np.linalg.norm(ideal_top, ord=np.inf)), float(np.linalg.norm(ideal_bottom, ord=np.inf))) / max(
        1.0, float(np.linalg.norm(rhs_x, ord=np.inf)), float(np.linalg.norm(rhs_eq, ord=np.inf))
    )
    return NewtonResidualComparison(
        exact_factored_reduced_relative=exact_relative,
        regularized_top_relative=regularized_top_relative,
        ideal_newton_block_relative=ideal_relative,
        h_regularization_shift_max=float(np.max(np.abs(h_reg - h))),
        schur_regularization=schur_reg,
    )
