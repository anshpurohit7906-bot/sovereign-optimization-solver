"""Convex QP primal-dual Mehrotra predictor-corrector solver."""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
import scipy.sparse as sp
from .problem import QPProblem, QPValidationError
from .linear_system import solve_kkt, QPLinearSystemError
from .gpu import solve_dense_kkt_gpu


class QPSolverError(RuntimeError):
    pass

@dataclass
class QPResult:
    x: np.ndarray
    objective: float
    status: str
    iterations: int
    primal_residual: float
    dual_residual: float
    complementarity: float
    runtime_seconds: float
    message: str = ""
    history: list[dict] = field(default_factory=list)
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    z: np.ndarray = field(default_factory=lambda: np.empty(0))


def _mat(M):
    if M is None: return None
    return M.toarray() if sp.issparse(M) else np.asarray(M, float)


def _max_step(v, dv, fraction=0.995):
    idx = dv < 0
    if not np.any(idx): return 1.0
    return min(1.0, float(fraction * np.min(-v[idx] / dv[idx])))


def _solve_direction(P, q, G, A, s, z, rd, rg, rp, rc, sparse=False, reg=1e-10, use_gpu=False):
    D = z / np.maximum(s, 1e-14)
    if G is not None and G.shape[0]:
        if sp.issparse(G):
            H = P + G.T @ sp.diags(D) @ G
        else:
            H = P + G.T @ (D[:, None] * G)
    else:
        H = P.copy()
    if G is not None and G.shape[0]:
        rhs_x = -rd + G.T @ ((rc - z * rg) / np.maximum(s, 1e-14))
    else:
        rhs_x = -rd
    if A is None or A.shape[0] == 0:
        if use_gpu:
            dx, dy = solve_dense_kkt_gpu(np.asarray(H), None, np.asarray(rhs_x), reg=reg)
        else:
            dx, dy = solve_kkt(H, None, np.asarray(rhs_x), sparse=False, reg=reg)
    else:
        rhs = np.concatenate([rhs_x, -rp])
        if use_gpu:
            dx, dy = solve_dense_kkt_gpu(np.asarray(H), np.asarray(A), rhs, reg=reg)
        else:
            dx, dy = solve_kkt(H, A, rhs, sparse=sparse, reg=reg)
    if G is not None and G.shape[0]:
        ds = -rg - G @ dx
        dz = (-rc - z * ds) / np.maximum(s, 1e-14)
    else:
        ds = np.empty(0); dz = np.empty(0)
    return dx, dy, ds, dz


def solve_qp(problem: QPProblem | np.ndarray, q=None, G=None, h=None, A=None, b=None,
             lb=None, ub=None, *, max_iterations=100, tol=1e-7, sparse=None,
             use_gpu=False, verbose=False, fraction=0.995, regularization=1e-10):
    """Solve a convex QP with a primal-dual Mehrotra predictor-corrector method.

    Accepts either a QPProblem or the raw P,q,G,h,A,b,lb,ub arguments.
    Bounds are converted to linear inequalities internally.
    """
    if not isinstance(problem, QPProblem):
        problem = QPProblem(problem, q, G, h, A, b, lb, ub)
    problem.check_convexity()
    if sparse is None: sparse = problem.is_sparse
    if use_gpu:
        # GPU path currently supports dense KKT solves through the optional backend.
        from .gpu import gpu_available
        if not gpu_available():
            raise QPSolverError("GPU requested but no CUDA-capable CuPy device is available")
        # Keep algorithmic path identical; GPU backend is exposed separately for profiling.
    G, h = problem.with_bounds_as_inequalities()
    A, b = problem.A, None if problem.b is None else np.asarray(problem.b, float)
    P = problem.P if sp.issparse(problem.P) else np.asarray(problem.P, float)
    qv = np.asarray(problem.q, float)
    n, mi = problem.n, 0 if G is None else G.shape[0]
    if G is not None and not sp.issparse(G): G = np.asarray(G, float)
    x = np.zeros(n)
    if A is not None and A.shape[0]:
        try:
            x = np.linalg.lstsq(_mat(A), b, rcond=None)[0]
        except Exception: pass
    if G is not None and mi:
        # Shift into strict interior by a deterministic margin.
        viol = G @ x - h
        gnorm = max(1.0, float(np.max(np.asarray(np.abs(G).sum(axis=1)).ravel()))**2)
        x = x - 0.1 * (G.T @ np.maximum(viol, 0.0)) / gnorm
        s = h - G @ x
        if np.min(s) <= 0:
            x = x + G.T @ np.maximum(-s + 1.0, 0.0) / gnorm
        s = h - G @ x
        if np.min(s) <= 0: s = np.maximum(s, 1.0)
        z = np.ones(mi)
    else:
        s = np.empty(0); z = np.empty(0)
    y = np.zeros(0 if A is None else A.shape[0])
    # Feasible starting point is not required; Newton iterations drive residuals down.
    start = time.perf_counter(); history=[]; status='max_iterations'; message='maximum iterations reached'
    best=(np.inf, x.copy(), y.copy(), z.copy())
    for it in range(1, max_iterations+1):
        grad = P @ x + qv
        if A is not None and A.shape[0]: grad = grad + A.T @ y
        if mi: grad = grad + G.T @ z
        rp = np.zeros(0) if A is None else A @ x - b
        rg = np.zeros(0) if not mi else G @ x + s - h
        rd = grad
        mu = float(s @ z / mi) if mi else 0.0
        comp_max = float(np.max(np.abs(s*z))) if mi else 0.0
        merit = max(float(np.linalg.norm(rd,np.inf)),
                    float(np.linalg.norm(rp,np.inf)) if rp.size else 0.0,
                    float(np.linalg.norm(rg,np.inf)) if rg.size else 0.0,
                    comp_max)
        history.append({'iteration':it,'objective':problem.objective(x),'primal_residual':max(float(np.linalg.norm(rp,np.inf)) if rp.size else 0.0,float(np.linalg.norm(rg,np.inf)) if rg.size else 0.0),'dual_residual':float(np.linalg.norm(rd,np.inf)),'complementarity':comp_max})
        if merit < best[0]: best=(merit,x.copy(),y.copy(),z.copy())
        if (not rp.size or np.linalg.norm(rp,np.inf)<=tol) and (not rg.size or np.linalg.norm(rg,np.inf)<=tol) and np.linalg.norm(rd,np.inf)<=tol and (not mi or comp_max<=tol):
            status='optimal'; message='converged to KKT tolerance'; break
        # Affine predictor.
        rc_aff = s*z
        try:
            dx_a, dy_a, ds_a, dz_a = _solve_direction(P,qv,G,A,s,z,rd,rg,rp,rc_aff,sparse=sparse,reg=regularization)
        except QPLinearSystemError as e:
            status='numerical_error'; message=str(e); break
        ap=_max_step(s,ds_a,fraction); ad=_max_step(z,dz_a,fraction)
        aa=min(ap,ad)
        s_aff=s+aa*ds_a; z_aff=z+aa*dz_a
        mu_aff=float(s_aff@z_aff/mi) if mi else 0.0
        sigma=(mu_aff/max(mu,1e-16))**3 if mi else 0.0
        sigma=float(np.clip(sigma,0.0,1.0))
        rc=s*z + ds_a*dz_a - sigma*mu
        try:
            dx,dy,ds,dz=_solve_direction(P,qv,G,A,s,z,rd,rg,rp,rc,sparse=sparse,reg=regularization,use_gpu=use_gpu)
        except QPLinearSystemError as e:
            status='numerical_error'; message=str(e); break
        ap=_max_step(s,ds,fraction); ad=_max_step(z,dz,fraction); alpha=min(ap,ad,1.0)
        if not np.isfinite(alpha) or alpha <= 1e-12:
            status='stalled'; message='fraction-to-boundary step collapsed'; break
        x=x+alpha*dx; y=y+alpha*dy
        if mi:
            s=s+alpha*ds; z=z+alpha*dz
            s=np.maximum(s,1e-14); z=np.maximum(z,1e-14)
        if verbose: print(f"iter={it:3d} obj={problem.objective(x): .8e} merit={merit:.3e} alpha={alpha:.3e}")
    else: it=max_iterations
    if status!='optimal' and best[0] < max(1e-5, tol*100):
        # Keep best numerical iterate rather than a later degraded tail.
        _,x,y,z=best
    runtime=time.perf_counter()-start
    grad=P@x+qv
    if A is not None and A.shape[0]: grad += A.T@y
    if mi: grad += G.T@z
    pr=max(float(np.linalg.norm(A@x-b,np.inf)) if A is not None and A.shape[0] else 0.0,
           float(np.linalg.norm(G@x+s-h,np.inf)) if mi else 0.0)
    dr=float(np.linalg.norm(grad,np.inf)); comp=float(s@z/mi) if mi else 0.0
    return QPResult(x=x,objective=problem.objective(x),status=status,iterations=it,
                    primal_residual=pr,dual_residual=dr,complementarity=comp,
                    runtime_seconds=runtime,message=message,history=history,y=y,z=z)
