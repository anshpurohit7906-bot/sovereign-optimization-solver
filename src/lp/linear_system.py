"""Linear-system kernel for the Mehrotra interior-point method.

Solves the reduced Newton system

    [ H    -A_E^T ] [ dx  ]   [ rhs_x  ]
    [ A_E    0    ] [ dyE ] = [ rhs_eq ]

with H symmetric positive definite (the Mehrotra solver always passes
H = diag(z / x)), by block elimination onto the Schur complement
S = A_E H^{-1} A_E^T:

    S dyE = rhs_eq - A_E H^{-1} rhs_x
    dx    = H^{-1} (rhs_x + A_E^T dyE)

When H is diagonal (always the case for the Mehrotra solver) the
factorization runs on a sparse SciPy backend: A_E is stored once as CSR,
H is kept as its regularized diagonal (so H-solves are elementwise
divisions and no dense H/W/S matrices are built), and the symmetrized
Schur complement S = A_E (H + reg I)^{-1} A_E^T is factorized with
SuperLU (``scipy.sparse.linalg.splu``).  One factorization is reused for
the predictor and corrector right-hand sides of an iteration.  A
non-diagonal SPD H falls back to the previous dense Cholesky path.

Both backends apply escalating diagonal regularization if a factorization
breaks down (``reg = base_reg * max(1, mean|diag|)``, multiplied by 10 up
to 8 times), raising ``LinearSystemError`` if that is still not enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import SuperLU, splu


class LinearSystemError(RuntimeError):
    """Raised when the reduced Newton system cannot be factored."""


def _cholesky_regularized(M: np.ndarray, base_reg: float) -> np.ndarray:
    """Lower Cholesky factor of sym(M) + reg*I with escalating reg."""
    M = 0.5 * (M + M.T)
    scale = max(1.0, float(np.mean(np.abs(np.diag(M)))))
    reg = base_reg * scale
    for _ in range(8):
        try:
            return np.linalg.cholesky(M + reg * np.eye(M.shape[0]))
        except np.linalg.LinAlgError:
            reg *= 10.0
    raise LinearSystemError("Cholesky factorization failed even with regularization")


def _chol_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve (L L^T) X = B for X (B may be a vector or a matrix)."""
    Y = np.linalg.solve(L, B)
    return np.linalg.solve(L.T, Y)


def _splu_regularized(S: sp.csr_matrix, base_reg: float) -> SuperLU:
    """SuperLU factorization of sym(S) + reg*I with escalating reg (sparse).

    Sparse mirror of ``_cholesky_regularized``: same start
    ``reg = base_reg * max(1, mean|diag(sym(S))|)`` and the same 8-step x10
    escalation, applied to the symmetrized sparse Schur complement.
    """
    M = (0.5 * (S + S.T)).tocsc()
    scale = max(1.0, float(np.mean(np.abs(M.diagonal()))))
    reg = base_reg * scale
    for _ in range(8):
        try:
            return splu(M + sp.eye(M.shape[0], format="csc") * reg)
        except RuntimeError:
            reg *= 10.0
    raise LinearSystemError("Sparse factorization failed even with regularization")


@dataclass
class ReducedNewtonFactorization:
    """Reusable factorization of one iteration's reduced Newton system.

    Two equivalent backends, selected at factorization time:
    - sparse (H diagonal, the form the Mehrotra solver produces):
      ``h_diag`` is the regularized H diagonal, ``A_sp`` the CSR copy of
      A_eq and ``schur_lu`` the SuperLU factors of the symmetrized Schur
      complement S = A_eq (H + reg I)^{-1} A_eq^T (+ reg I).
    - dense fallback (general SPD H): the previous dense Cholesky factors
      ``H_chol`` / ``schur_chol`` with the dense ``A_eq``.
    """

    h_diag: Optional[np.ndarray] = None
    A_sp: Optional[sp.csr_matrix] = None
    schur_lu: Optional[SuperLU] = None
    H_chol: Optional[np.ndarray] = None
    A_eq: Optional[np.ndarray] = None
    schur_chol: Optional[np.ndarray] = None

    @property
    def num_eq(self) -> int:
        if self.A_sp is not None:
            return self.A_sp.shape[0]
        return self.A_eq.shape[0]


def factor_reduced_system(
    H: np.ndarray, A_eq: np.ndarray, reg: float = 1e-12
) -> ReducedNewtonFactorization:
    """Factor H and, when equality rows exist, the Schur complement.

    H = diag(z / x) as produced by the Mehrotra solver is diagonal and
    takes the sparse backend: the regularized diagonal replaces the dense
    Cholesky of H (identical solves by elementwise division), A_eq is kept
    sparse, and the Schur complement is constructed and factorized
    sparsely.  A non-diagonal SPD H uses the dense fallback.
    """
    diag = np.diag(H)
    is_diagonal = bool(np.array_equal(H, np.diag(diag)))

    if not is_diagonal:
        # Dense fallback: general SPD H (previous behavior, unchanged).
        H_chol = _cholesky_regularized(H, reg)
        if A_eq.shape[0] == 0:
            return ReducedNewtonFactorization(H_chol=H_chol, A_eq=A_eq, schur_chol=None)
        W = _chol_solve(H_chol, A_eq.T)  # W = H^{-1} A_E^T
        S = A_eq @ W
        schur_chol = _cholesky_regularized(S, reg)
        return ReducedNewtonFactorization(H_chol=H_chol, A_eq=A_eq, schur_chol=schur_chol)

    # Sparse path: H-solves are divisions by the regularized diagonal,
    # exactly matching the dense path's chol(H + reg*I) solves.
    scale = max(1.0, float(np.mean(np.abs(diag))))
    h_diag = diag + reg * scale

    A_sp = sp.csr_matrix(A_eq)  # A_eq stays sparse from here on
    if A_sp.shape[0] == 0:
        return ReducedNewtonFactorization(h_diag=h_diag, A_sp=A_sp, schur_lu=None)

    W_sp = A_sp.T.multiply((1.0 / h_diag)[:, None]).tocsr()  # (H+regI)^{-1} A_E^T
    S_sp = A_sp @ W_sp                                       # sparse m x m
    schur_lu = _splu_regularized(S_sp, reg)
    return ReducedNewtonFactorization(h_diag=h_diag, A_sp=A_sp, schur_lu=schur_lu)


def solve_reduced_system(
    fac: ReducedNewtonFactorization, rhs_x: np.ndarray, rhs_eq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the reduced Newton system for (dx, dyE) using a factorization."""
    if fac.h_diag is not None:
        # Sparse backend: diagonal H -> elementwise solves (identical to the
        # dense chol(diag) solves), sparse Schur solve via the SuperLU factors.
        t = rhs_x / fac.h_diag  # t = H^{-1} rhs_x
        if fac.num_eq == 0:
            return t, np.zeros(0)
        dy = fac.schur_lu.solve(rhs_eq - fac.A_sp @ t)
        dx = (rhs_x + fac.A_sp.T @ dy) / fac.h_diag
        return dx, dy
    # Dense fallback: general SPD H (previous behavior, unchanged).
    t = _chol_solve(fac.H_chol, rhs_x)  # t = H^{-1} rhs_x
    if fac.num_eq == 0:
        return t, np.zeros(0)
    dy = _chol_solve(fac.schur_chol, rhs_eq - fac.A_eq @ t)
    dx = _chol_solve(fac.H_chol, rhs_x + fac.A_eq.T @ dy)
    return dx, dy
