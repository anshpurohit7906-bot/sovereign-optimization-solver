"""Dense linear-system kernel for the Mehrotra interior-point method.

Solves the reduced Newton system

    [ H    -A_E^T ] [ dx  ]   [ rhs_x  ]
    [ A_E    0    ] [ dyE ] = [ rhs_eq ]

with H symmetric positive definite (diagonal plus low-rank), by block
elimination onto the Schur complement S = A_E H^{-1} A_E^T:

    S dyE = rhs_eq - A_E H^{-1} rhs_x
    dx    = H^{-1} (rhs_x + A_E^T dyE)

All factorizations are dense Cholesky (NumPy only), with escalating
diagonal regularization if a factorization breaks down. One factorization
is reused for the predictor and corrector right-hand sides of an
iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


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


@dataclass
class ReducedNewtonFactorization:
    """Reusable factorization of one iteration's reduced Newton system."""

    H_chol: np.ndarray
    A_eq: np.ndarray
    schur_chol: Optional[np.ndarray]

    @property
    def num_eq(self) -> int:
        return self.A_eq.shape[0]


def factor_reduced_system(
    H: np.ndarray, A_eq: np.ndarray, reg: float = 1e-12
) -> ReducedNewtonFactorization:
    """Factor H and, when equality rows exist, the Schur complement."""
    H_chol = _cholesky_regularized(H, reg)
    if A_eq.shape[0] == 0:
        return ReducedNewtonFactorization(H_chol, A_eq, None)
    W = _chol_solve(H_chol, A_eq.T)  # W = H^{-1} A_E^T
    S = A_eq @ W
    schur_chol = _cholesky_regularized(S, reg)
    return ReducedNewtonFactorization(H_chol, A_eq, schur_chol)


def solve_reduced_system(
    fac: ReducedNewtonFactorization, rhs_x: np.ndarray, rhs_eq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the reduced Newton system for (dx, dyE) using a factorization."""
    t = _chol_solve(fac.H_chol, rhs_x)  # t = H^{-1} rhs_x
    if fac.num_eq == 0:
        return t, np.zeros(0)
    dy = _chol_solve(fac.schur_chol, rhs_eq - fac.A_eq @ t)
    dx = _chol_solve(fac.H_chol, rhs_x + fac.A_eq.T @ dy)
    return dx, dy
