"""Convex quadratic-programming model and validation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


class QPValidationError(ValueError):
    """Invalid or unsupported QP model."""


def _arr(a, *, sparse=False):
    if a is None:
        return None
    if sparse:
        return a.tocsr().astype(float)
    return np.asarray(a, dtype=float)


@dataclass
class QPProblem:
    P: np.ndarray | sp.spmatrix
    q: np.ndarray
    G: Optional[np.ndarray | sp.spmatrix] = None
    h: Optional[np.ndarray] = None
    A: Optional[np.ndarray | sp.spmatrix] = None
    b: Optional[np.ndarray] = None
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None
    name: str = "QP"

    def __post_init__(self):
        self.P = _arr(self.P, sparse=sp.issparse(self.P))
        self.q = _arr(self.q).reshape(-1)
        n = self.q.size
        if self.P.shape != (n, n):
            raise QPValidationError(f"P must have shape {(n,n)}, got {self.P.shape}")

        if sp.issparse(self.P):
            if not np.all(np.isfinite(self.P.data)) or not np.all(np.isfinite(self.q)):
                raise QPValidationError("P and q must be finite")
            delta = (self.P - self.P.T).tocsr()
            if delta.nnz and np.max(np.abs(delta.data)) > 1e-9:
                raise QPValidationError("P must be symmetric")
            self.P = (0.5 * (self.P + self.P.T)).tocsr()
        else:
            if not np.all(np.isfinite(self.P)) or not np.all(np.isfinite(self.q)):
                raise QPValidationError("P and q must be finite")
            if not np.allclose(self.P, self.P.T, rtol=1e-9, atol=1e-12):
                raise QPValidationError("P must be symmetric")
            self.P = 0.5 * (self.P + self.P.T)

        self.G = _arr(self.G, sparse=sp.issparse(self.G))
        self.A = _arr(self.A, sparse=sp.issparse(self.A))
        self.h = None if self.h is None else _arr(self.h).reshape(-1)
        self.b = None if self.b is None else _arr(self.b).reshape(-1)
        self.lb = None if self.lb is None else _arr(self.lb).reshape(-1)
        self.ub = None if self.ub is None else _arr(self.ub).reshape(-1)

        if self.G is not None:
            if self.G.shape[1] != n or self.h is None or self.G.shape[0] != self.h.size:
                raise QPValidationError("G/h dimensions are inconsistent")
        if self.A is not None:
            if self.A.shape[1] != n or self.b is None or self.A.shape[0] != self.b.size:
                raise QPValidationError("A/b dimensions are inconsistent")
        for name, bound in (("lb", self.lb), ("ub", self.ub)):
            if bound is not None and bound.size != n:
                raise QPValidationError(f"{name} must have length {n}")
            if bound is not None and np.any(np.isnan(bound)):
                raise QPValidationError(f"{name} contains NaN")
        if self.lb is not None and self.ub is not None and np.any(self.lb > self.ub):
            raise QPValidationError("lb must not exceed ub")
        if self.G is not None and not np.all(np.isfinite(self.h)):
            raise QPValidationError("h must be finite")
        if self.A is not None and not np.all(np.isfinite(self.b)):
            raise QPValidationError("b must be finite")

    @property
    def n(self): return self.q.size
    @property
    def m_ineq(self): return 0 if self.G is None else self.G.shape[0]
    @property
    def m_eq(self): return 0 if self.A is None else self.A.shape[0]
    @property
    def is_sparse(self):
        return sp.issparse(self.P) or sp.issparse(self.G) or sp.issparse(self.A)

    def check_convexity(self, tol=1e-9):
        if self.n == 0:
            raise QPValidationError("QP must contain at least one variable")
        if sp.issparse(self.P):
            if self.n == 1:
                eig_min = float(self.P[0, 0])
            else:
                try:
                    eig_min = float(eigsh(self.P, k=1, which="SA",
                                          return_eigenvectors=False,
                                          tol=1e-8, maxiter=max(1000, 5*self.n))[0])
                except Exception as e:
                    raise QPValidationError(
                        f"sparse PSD check failed without densifying P: {e}"
                    ) from e
        else:
            eig_min = float(np.linalg.eigvalsh(self.P)[0])
        scale = max(1.0, float(np.max(np.abs(self.P.data)))
                     if sp.issparse(self.P) else float(np.max(np.abs(self.P))))
        if eig_min < -tol * scale:
            raise QPValidationError(
                f"P is not positive semidefinite; min eigenvalue={eig_min:.3e}"
            )
        return eig_min

    def objective(self, x):
        x = np.asarray(x, dtype=float)
        return float(0.5 * x @ (self.P @ x) + self.q @ x)

    def with_bounds_as_inequalities(self):
        """Return dense/sparse G,h with finite bounds folded into inequalities."""
        rows, rhs = [], []
        sparse_mode = sp.issparse(self.G) or self.is_sparse
        if self.G is not None:
            rows.append(self.G.tocsr() if sp.issparse(self.G) else np.asarray(self.G))
            rhs.append(self.h)
        n = self.n
        if self.ub is not None:
            for j, u in enumerate(self.ub):
                if np.isfinite(u):
                    if sparse_mode:
                        rows.append(sp.csr_matrix(([1.0], ([0], [j])), shape=(1,n)))
                    else:
                        r = np.zeros(n); r[j] = 1.0; rows.append(r[None, :])
                    rhs.append(np.array([u]))
        if self.lb is not None:
            for j, l in enumerate(self.lb):
                if np.isfinite(l):
                    if sparse_mode:
                        rows.append(sp.csr_matrix(([-1.0], ([0], [j])), shape=(1,n)))
                    else:
                        r = np.zeros(n); r[j] = -1.0; rows.append(r[None, :])
                    rhs.append(np.array([-l]))
        if not rows:
            return None, None
        if sparse_mode:
            return sp.vstack([
                r if sp.issparse(r) else sp.csr_matrix(r) for r in rows
            ], format="csr"), np.concatenate(rhs)
        return np.vstack(rows), np.concatenate(rhs)
