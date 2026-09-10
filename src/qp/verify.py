"""Independent KKT certificate checks for convex QP solutions."""
from __future__ import annotations
import numpy as np


def certificate(problem, x, y=None, z=None, tol=1e-6):
    x = np.asarray(x, float)
    G, h = problem.with_bounds_as_inequalities()
    A, b = problem.A, problem.b
    if A is not None and hasattr(A, 'toarray'): A = A.toarray()
    if G is not None and hasattr(G, 'toarray'): G = G.toarray()
    if y is None: y = np.zeros(0)
    if z is None: z = np.zeros(0)
    y = np.asarray(y, float); z = np.asarray(z, float)
    rd = problem.P @ x + problem.q
    if A is not None: rd = rd + A.T @ y
    if G is not None: rd = rd + G.T @ z
    rp = 0.0 if A is None else np.linalg.norm(A @ x - b, np.inf)
    gi = np.empty(0) if G is None else G @ x - h
    primal_ineq = 0.0 if G is None else max(0.0, float(np.max(gi)))
    dual_ineq = 0.0 if G is None else max(0.0, float(-np.min(z)))
    slack = np.empty(0) if G is None else h - G @ x
    comp = 0.0 if G is None else float(np.max(np.abs(z * slack)))
    stat = float(np.linalg.norm(rd, np.inf))
    ok = max(stat, rp, primal_ineq, dual_ineq, comp) <= tol
    return {"ok": bool(ok), "stationarity": stat, "equality_residual": rp,
            "inequality_violation": primal_ineq, "dual_violation": dual_ineq,
            "complementarity": comp, "objective": problem.objective(x)}
