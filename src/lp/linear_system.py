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
the predictor and corrector right-hand sides of an iteration.  When the
Schur system is very ill-conditioned a single LU solve can be
rounding-limited; the sparse path therefore applies at most two
iterative-refinement corrections to ``dy``, each accepted only if it
strictly reduces the residual of the EXACT regularized Schur system that
was factored (the escalated diagonal regularization is stored on the
factorization and used in the residual).  A non-diagonal SPD H falls back
to the previous dense Cholesky path.

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


# Cap for the H-regularization ρ_p = min(reg * max(1, mean|h|), MAX_RHO_P).
# Mathematical rationale (see ``factor_reduced_system`` docstring):
#   1. ρ_p floors the reciprocal weights at 1/ρ_p <= 1/MAX_RHO_P ~ 2e7, which
#      bounds the Schur-complement entries that would otherwise blow up in the
#      degenerate interior-point tail (h_i -> 0).
#   2. ρ_p induces a dual-residual bias of at most ρ_p * ||dx||_inf per step;
#      capping it at 5e-8 keeps that bias at ~1e-7..1e-5 for realistic tail
#      step sizes 1..100 (with occasional 1e3-scale components), instead of the
#      ~1e-4..1e-3 floor produced by the uncapped formula on PILOT4.
# Values above ~1e-6 leave the dual residual stuck above 1e-4 on PILOT4; values
# below ~1e-9 let the reciprocal weights 1/(h_i + ρ_p) exceed what the Schur
# SuperLU factorization can resolve (observed primal-infeasibility blowup).
MAX_RHO_P = 5e-8


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


def _splu_regularized(S: sp.csr_matrix, base_reg: float
                      ) -> tuple[SuperLU, float]:
    """SuperLU factorization of sym(S) + reg*I with escalating reg (sparse).

    Sparse mirror of ``_cholesky_regularized``: same start
    ``reg = base_reg * max(1, mean|diag(sym(S))|)`` and the same 8-step x10
    escalation, applied to the symmetrized sparse Schur complement.

    Returns ``(lu, reg)`` with the ACTUAL diagonal regularization ``reg`` of
    the returned factorization: the exact matrix factored is
    ``sym(S) + reg * I``, and any downstream residual computation (e.g. the
    iterative refinement in ``_schur_refine``) must use this same ``reg``.
    """
    M = (0.5 * (S + S.T)).tocsc()
    scale = max(1.0, float(np.mean(np.abs(M.diagonal()))))
    reg = base_reg * scale
    for _ in range(8):
        try:
            return splu(M + sp.eye(M.shape[0], format="csc") * reg), reg
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
    schur_reg: Optional[float] = None
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

    H may be either a 1-D vector of diagonal entries (the h = z/x vector
    produced by the Mehrotra solver) or a 2-D SPD matrix.

    * 1-D input: taken as diagonal without any O(n²) construction or
      comparison — enters the sparse backend directly.
    * 2-D input: diagonality is tested as before; diagonal matrices take
      the sparse backend, general SPD matrices take the dense fallback.

    Parameters
    ----------
    reg : base regularization scale.  The H-regularization is
          ``ρ_p = reg * max(1, mean|h|)`` capped at ``MAX_RHO_P``; the same
          ``reg`` seeds the Schur diagonal regularization in ``_splu_regularized``.

    H regularization rationale
    -------------------------
    The reduced Newton system replaces ``H = diag(z/x)`` by ``H + ρ_p I`` and the
    Schur complement ``S = A (H+ρ_p I)^{-1} A^T`` is factorized (plus its own
    diagonal regularization).  Two independent numerical consequences follow:

    1. The least reciprocal weight is floored: ``1/(h_i + ρ_p) <= 1/ρ_p``, which
       bounds the Schur entries for variables whose ``h_i`` collapses to ~0 in
       the degenerate tail (where ``z_i -> 0`` and/or ``x_i`` stays positive).
    2. A dual-side bias ``ρ_p * dx`` enters every Newton step (in exact
       arithmetic ``A^T dy + dz = c - r_d + ρ_p dx`` for the regularized
       system), so ``ρ_p`` must stay small enough that ``ρ_p * ||dx||_inf`` does
       not dominate the achievable dual accuracy.

    The old formula ``ρ_p = reg * max(1, mean|h|)`` grew with ``mean|h|``
    (which itself grows like ``h_max = O(1/μ)`` in the tail) up to 1e-2 on
    PILOT4, producing a dual-bias floor of ~1e-4..1e-3 that stalled the dual
    residual above the requested tolerance.  The cap below bounds the bias
    (``MAX_RHO_P * ||dx||_inf ~ 1e-7..1e-5`` for realistic tail ``||dx||``)
    while still keeping the reciprocal weights bounded by ``1/MAX_RHO_P ~ 2e7``
    so the Schur factorization remains usable (validated on PILOT4/PILOT87:
    dual residual and gap improve by 40-500x).  ``mu`` is deliberately NOT used:
    a μ-proportional scheme performed worse empirically because the scaled μ is
    large early, and a fixed small cap already keeps the bias bounded.
    """
    if H.ndim == 1:
        diag = H
        is_diagonal = True
    else:
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

    # Sparse path: H-regularization with a bias cap.
    scale = max(1.0, float(np.mean(np.abs(diag))))
    reg_h = reg * scale
    if reg_h > MAX_RHO_P:
        reg_h = MAX_RHO_P
    h_diag = diag + reg_h

    A_sp = sp.csr_matrix(A_eq)
    if A_sp.shape[0] == 0:
        return ReducedNewtonFactorization(h_diag=h_diag, A_sp=A_sp, schur_lu=None)

    W_sp = A_sp.T.multiply((1.0 / h_diag)[:, None]).tocsr()
    S_sp = A_sp @ W_sp
    schur_lu, schur_reg = _splu_regularized(S_sp, reg)
    return ReducedNewtonFactorization(h_diag=h_diag, A_sp=A_sp, schur_lu=schur_lu,
                                      schur_reg=schur_reg)


def _schur_residual(fac: ReducedNewtonFactorization, b: np.ndarray,
                    v: np.ndarray) -> tuple[np.ndarray, float]:
    """Residual ``r = b - (sym(S) + schur_reg*I) v`` of the regularized Schur
    system, computed without building a dense S: ``sym(S) v = A (A^T v / h)``.
    The regularization is the EXACT escalated value stored on the
    factorization (``fac.schur_reg``) -- the system actually factored.
    """
    q = (fac.A_sp.T @ v) / fac.h_diag
    r = b - (fac.A_sp @ q) - fac.schur_reg * v
    return r, float(np.linalg.norm(r, ord=np.inf))


def _schur_refine(fac: ReducedNewtonFactorization, b: np.ndarray,
                  dy: np.ndarray) -> np.ndarray:
    """At most two iterative-refinement corrections of the sparse Schur solve.

    Each candidate ``dy + dc`` (``dc`` from the existing SuperLU factors) is
    accepted only if its residual against the EXACT regularized Schur system
    ``sym(S) + schur_reg*I`` is strictly smaller than the current iterate's;
    otherwise refinement is stopped and the previous iterate is kept.  When
    correction does not help (e.g. cond ~ 1/eps), the original ``dy`` is
    returned unchanged.
    """
    best = dy
    r, rnorm = _schur_residual(fac, b, best)
    for _ in range(2):
        if rnorm == 0.0:
            break
        cand = best + fac.schur_lu.solve(r)
        rc, rn = _schur_residual(fac, b, cand)
        if rn < rnorm:
            best, r, rnorm = cand, rc, rn
        else:
            break
    return best


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
        b = rhs_eq - fac.A_sp @ t
        dy = fac.schur_lu.solve(b)
        dy = _schur_refine(fac, b, dy)
        dx = (rhs_x + fac.A_sp.T @ dy) / fac.h_diag
        return dx, dy
    # Dense fallback: general SPD H (previous behavior, unchanged).
    t = _chol_solve(fac.H_chol, rhs_x)  # t = H^{-1} rhs_x
    if fac.num_eq == 0:
        return t, np.zeros(0)
    dy = _chol_solve(fac.schur_chol, rhs_eq - fac.A_eq @ t)
    dx = _chol_solve(fac.H_chol, rhs_x + fac.A_eq.T @ dy)
    return dx, dy
