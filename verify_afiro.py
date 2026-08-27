from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from numerical_model import load_numeric_mps
from experiment.pdhg.pdhg_mixed import pdhg_mixed


def verify() -> int:
    mps = ROOT / 'data' / 'afiro.mps'
    lp = load_numeric_mps(mps)
    result = pdhg_mixed(lp, max_iter=200_000, tol=1e-7, check_every=250)

    Aeq_mask = np.asarray(lp.row_types) == 'E'
    Aub_mask = np.asarray(lp.row_types) == 'L'
    Aeq, beq = lp.A[Aeq_mask], lp.b[Aeq_mask]
    Aub, bub = lp.A[Aub_mask], lp.b[Aub_mask]

    x = result.x
    eq_abs = float(np.linalg.norm(Aeq @ x - beq, ord=2)) if Aeq.shape[0] else 0.0
    ub_violation = np.maximum(Aub @ x - bub, 0.0)
    ub_abs = float(np.linalg.norm(ub_violation, ord=2)) if Aub.shape[0] else 0.0
    bound_violation = np.maximum(lp.lower_bounds - x, 0.0)
    bound_violation = np.maximum(bound_violation, x - lp.upper_bounds)
    bound_abs = float(np.max(np.abs(bound_violation))) if bound_violation.size else 0.0
    obj = float(lp.c @ x)

    ref = linprog(lp.c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                  bounds=list(zip(lp.lower_bounds, lp.upper_bounds)), method='highs')
    if not ref.success:
        print('Reference HiGHS solve failed:', ref.message)
        return 2

    obj_err = abs(obj - float(ref.fun))
    rel_obj_err = obj_err / (1.0 + abs(float(ref.fun)))

    print(f'Solver status        : {result.status}')
    print(f'Solver objective     : {obj:.12f}')
    print(f'Reference objective  : {ref.fun:.12f}')
    print(f'Absolute obj error   : {obj_err:.3e}')
    print(f'Relative obj error   : {rel_obj_err:.3e}')
    print(f'Equality L2 residual : {eq_abs:.3e}')
    print(f'Ineq violation L2    : {ub_abs:.3e}')
    print(f'Bound violation max  : {bound_abs:.3e}')
    print(f'Iterations           : {result.iterations}')

    # These are local verification gates, intentionally separate from the solver's stopping logic.
    pass_checks = (
        result.converged and
        eq_abs <= 1e-6 * (1.0 + float(np.linalg.norm(beq, ord=2))) and
        ub_abs <= 1e-6 * (1.0 + float(np.linalg.norm(bub, ord=2))) and
        bound_abs <= 1e-8 and
        rel_obj_err <= 1e-6
    )
    print('Independent verification:', 'PASS' if pass_checks else 'FAIL')
    return 0 if pass_checks else 1


if __name__ == '__main__':
    raise SystemExit(verify())
