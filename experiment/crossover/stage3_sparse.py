"""STAGE 3 - sparse composite-Phase-I basis repair (experimental).

Ports the proven Stage 2 algorithm (monotone composite primal Phase I,
Devex pricing, lexicographic ratio test) to a sparse SuperLU backend so
it scales to PILOT87 (m = 3608, 920 infeasible basics).

Production code is imported READ-ONLY:
    - simplex._solve_basis  : production numerical basis gate
    - simplex._simplex_iterations : production Revised-Simplex loop (Phase II)

The standard-form A is < 1% dense, so sparse ops dominate the cost:
    splu(B)  ~ 0.02 s  (PILOT87),  triangular solve ~ 4e-4 s,
    reduced costs d = -A^T y  is a sparse matvec (O(nnz)).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import load_numeric_mps  # noqa: E402
from mehrotra import to_standard_form  # noqa: E402
from simplex import _inf_norm, _solve_basis, _simplex_iterations  # noqa: E402
from stage1_audit_rrqr import rrqr_basis  # noqa: E402

CONDITION_LIMIT = 1e12
SIMPLEX_TOL = 1e-8
PIVOT_REL_TOL = 1e-7
RESID_LIMIT = 1e-7
MAX_COND_EVALS = 100
MAX_DEGENERATE = 50
INFEAS_TOL = 1e-7


def _cond2_estimate_dense(Bd):
    """True 2-norm condition via SVD - used only for small accepted bases."""
    s = np.linalg.svd(Bd, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def _lex_tiebreak(tied, alpha_q, piv_abs, lu, m, t_min):
    """Lexicographic ratio test among tied min-ratio rows."""
    if tied.size == 1:
        return int(tied[0])
    r = -1
    best_w = None
    for rc in tied:
        ac = float(alpha_q[rc])
        if abs(ac) <= piv_abs:
            continue
        er = np.zeros(m); er[rc] = 1.0
        z = lu.solve(er, trans="T") / ac
        w = np.concatenate(([t_min], z))
        if best_w is None:
            r, best_w = int(rc), w
            continue
        diff = w - best_w
        scale = np.maximum(np.maximum(1.0, np.abs(w)), np.abs(best_w))
        nz = np.flatnonzero(np.abs(diff) > 1e-12 * scale)
        if nz.size and diff[nz[0]] < 0.0:
            r, best_w = int(rc), w
    if r < 0:
        r = int(tied[np.argmax(np.abs(alpha_q[tied]))])
    return r
