"""Independent KKT certificate checks for convex QP solutions."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def certificate(problem, x, y=None, z=None, s=None, tol=1e-6):
    x = np.asarray(x, float)
    G, h = problem.with_bounds_as_inequalities()
    A, b = problem.A, problem.b
    y = np.zeros(0) if y is None else np.asarray(y, float)
    z = np.zeros(0) if z is None else np.asarray(z, float)

    rd = np.asarray(problem.P @ x + problem.q, dtype=float).ravel()
    if A is not None:
        rd += np.asarray(A.T @ y, dtype=float).ravel()
    if G is not None:
        rd += np.asarray(G.T @ z, dtype=float).ravel()

    rp = 0.0 if A is None else float(np.linalg.norm(A @ x - b, np.inf))
    gi = np.empty(0) if G is None else np.asarray(G @ x - h, dtype=float).ravel()
    primal_ineq = 0.0 if G is None else max(0.0, float(np.max(gi)))
    dual_ineq = 0.0 if G is None else max(0.0, float(-np.min(z)))

    if G is None:
        slack = np.empty(0)
    elif s is not None:
        slack = np.asarray(s, float)
    else:
        slack = np.asarray(h - G @ x, dtype=float).ravel()

    comp = 0.0 if G is None else float(np.max(np.abs(z * slack)))
    stat = float(np.linalg.norm(rd, np.inf))

    # Scale-relative stationarity: infinity-norm of the dual residual
    # divided by (1 + max|q| + max|P|) -- robust to badly-scaled
    # Maros-Meszaros instances where absolute infinity-norms are huge
    # even at a KKT-consistent optimum.
    pscale = (
        float(np.max(np.abs(problem.P.data))) if sp.issparse(problem.P)
        else float(np.max(np.abs(problem.P)))
    )
    qscale = float(np.max(np.abs(problem.q)))
    scaled_stat = stat / (1.0 + pscale + qscale)

    ok = max(stat, rp, primal_ineq, dual_ineq, comp) <= tol
    return {
        "ok": bool(ok),
        "stationarity": stat,
        "scaled_stationarity": scaled_stat,
        "equality_residual": rp,
        "inequality_violation": primal_ineq,
        "dual_violation": dual_ineq,
        "complementarity": comp,
        "objective": problem.objective(x),
    }
