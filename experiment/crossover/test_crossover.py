"""Runtime harness for the minimal crossover feasibility probe."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _path in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.lp.mehrotra import to_standard_form
from src.numerical_model import load_numeric_mps
from experiment.crossover.solver import evaluate_rank_aware_basis
from experiment.crossover.terminal_diagnostic import compare_newton_residuals, run_with_terminal_capture


def main() -> None:
    path = os.path.join(_ROOT, "data", "pilot4_plain.mps")
    sf = to_standard_form(load_numeric_mps(path))
    captured = run_with_terminal_capture(sf, tol=1e-8, max_iter=100)
    ipm = captured.result
    print("PILOT4 terminal-versus-best-gap rank-aware basis diagnostic")
    print(f"Mehrotra: status={ipm.status}, objective={ipm.objective:.12g}, "
          f"iterations={ipm.iterations}, rel_primal={ipm.rel_primal:.3e}, "
          f"rel_dual={ipm.rel_dual:.3e}, rel_gap={ipm.rel_gap:.3e}")
    print(f"Standard form: rows={sf.m}, columns={sf.n}; terminal iteration={captured.terminal_iteration}")
    for label, x, z in (
        ("best_gap", ipm.x_standard, ipm.z_standard),
        ("terminal", captured.terminal_x_standard, captured.terminal_z_standard),
    ):
        size, rank, condition, selected, rejected, replacements, inspected, valid, message = (
            evaluate_rank_aware_basis(sf, x, z)
        )
        print(f"{label}: size={size}, rank={rank}/{sf.m}, condition={condition:.3e}, "
              f"selected={selected}, rejected={rejected}, replacements={replacements}, "
              f"inspected={inspected}, simplex_valid={valid}")
        print(f"  {message}")
    newton = compare_newton_residuals(captured)
    print("Newton predictor residual comparison at terminal iterate:")
    print(f"  exact factored reduced system: {newton.exact_factored_reduced_relative:.3e}")
    print(f"  regularized top equation:      {newton.regularized_top_relative:.3e}")
    print(f"  ideal Newton block system:     {newton.ideal_newton_block_relative:.3e}")
    print(f"  max H regularization shift:    {newton.h_regularization_shift_max:.3e}")
    print(f"  Schur regularization:          {newton.schur_regularization:.3e}")
    print("No Phase II, feasibility repair, or crossover decision was run.")


if __name__ == "__main__":
    main()
