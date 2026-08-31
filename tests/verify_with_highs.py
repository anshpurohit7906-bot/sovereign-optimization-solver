"""Check all test cases against SciPy/HiGHS reference oracle to verify expected values."""

import numpy as np
from scipy.optimize import linprog
from test_lp_edge_cases import TEST_CASES

print(f"{'TEST':22s} | {'EXP STATUS':10s} | {'HIGHS STATUS':12s} | {'EXP OBJ':12s} | {'HIGHS OBJ':12s} | MATCH")
print("-" * 80)

for tc in TEST_CASES:
    lp, maximize = tc.build_lp()

    # Build linprog inputs
    c = lp.c.copy()
    if maximize:
        c_lp = -c
    else:
        c_lp = c

    m, n = lp.A.shape
    A_ub_list, b_ub_list = [], []
    A_eq_list, b_eq_list = [], []

    for i, rtype in enumerate(lp.row_types):
        if rtype == "E":
            A_eq_list.append(lp.A[i])
            b_eq_list.append(lp.b[i])
        elif rtype == "L":
            A_ub_list.append(lp.A[i])
            b_ub_list.append(lp.b[i])
        elif rtype == "G":
            A_ub_list.append(-lp.A[i])
            b_ub_list.append(-lp.b[i])

    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub = np.array(b_ub_list) if b_ub_list else None
    A_eq = np.array(A_eq_list) if A_eq_list else None
    b_eq = np.array(b_eq_list) if b_eq_list else None

    bounds = [(lb if np.isfinite(lb) else None, ub if np.isfinite(ub) else None)
              for lb, ub in zip(lp.lower_bounds, lp.upper_bounds)]

    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    highs_status = "optimal" if res.success else ("infeasible" if res.status == 2 else ("unbounded" if res.status == 3 else "other"))
    highs_obj = float(-res.fun if maximize else res.fun) if res.success else None

    obj_match = False
    if tc.expected_objective is not None and highs_obj is not None:
        obj_match = abs(tc.expected_objective - highs_obj) <= 1e-4 * (1.0 + abs(tc.expected_objective))
    elif tc.expected_status != "optimal" and not res.success:
        obj_match = True

    print(f"{tc.name:22s} | {tc.expected_status:10s} | {highs_status:12s} | {str(tc.expected_objective):12s} | {str(round(highs_obj, 6) if highs_obj is not None else None):12s} | {'OK' if obj_match else 'DIFF'}")
