"""Revised Simplex LP solver.

This module implements a two-phase Revised Simplex method for the project's
standard-form representation:

    minimize c^T x
    subject to A x = b
               x >= 0

The implementation is intentionally isolated from the Mehrotra IPM so the
project can maintain multiple independent LP algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mehrotra import StandardFormLP


class SimplexError(RuntimeError):
    """Raised when the Revised Simplex method cannot continue safely."""


@dataclass
class SimplexResult:
    """Result returned by the Revised Simplex solver."""

    status: str
    message: str
    objective: float
    x: np.ndarray
    x_standard: np.ndarray
    iterations: int
    phase_one_iterations: int
    phase_two_iterations: int
    primal_residual: float
    rel_primal: float
    rel_gap: float
    history: tuple[dict, ...]


def _inf_norm(v: np.ndarray) -> float:
    return float(np.max(np.abs(v))) if v.size else 0.0


def _clean_zero(v: np.ndarray, tol: float) -> np.ndarray:
    """Remove tiny numerical noise without changing meaningful values."""
    out = np.asarray(v, dtype=np.float64).copy()
    out[np.abs(out) <= tol] = 0.0
    return out


def _solve_basis(
    B: np.ndarray,
    rhs: np.ndarray,
    *,
    condition_limit: float,
) -> np.ndarray:
    """Solve Bx=rhs while rejecting numerically useless bases."""
    if B.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    try:
        cond = np.linalg.cond(B)
    except np.linalg.LinAlgError as exc:
        raise SimplexError("Unable to estimate basis conditioning") from exc

    if not np.isfinite(cond) or cond > condition_limit:
        raise SimplexError(
            f"basis is numerically singular/ill-conditioned (cond={cond:.3e})"
        )

    try:
        return np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError as exc:
        raise SimplexError("Basis solve failed") from exc


def _normalize_rhs(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize constraints so b_i >= 0 for all rows.

    If b_i < 0, multiply row i of A and b_i by -1.
    """
    row_sign = np.where(b < 0.0, -1.0, 1.0)
    A_norm = row_sign[:, None] * A
    b_norm = row_sign * b
    return A_norm, b_norm, row_sign


def _phase_one_setup(
    A_norm: np.ndarray,
    b_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Construct deterministic Phase-I problem [A_norm | I] [x; a] = b_norm with B=I, a=b_norm >= 0."""
    m, n = A_norm.shape
    A_phase1 = np.hstack([A_norm, np.eye(m, dtype=np.float64)])
    c_phase1 = np.zeros(n + m, dtype=np.float64)
    c_phase1[n:] = 1.0
    basis = list(range(n, n + m))
    return A_phase1, c_phase1, basis


def _simplex_iterations(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    basis: list[int],
    *,
    tol: float,
    max_iter: int,
    condition_limit: float,
    phase: int,
    artificial_start: Optional[int] = None,
) -> tuple[str, str, np.ndarray, list[int], int, tuple[dict, ...]]:
    """Run the core Revised Simplex iterations."""
    m, n = A.shape
    basis = list(basis)

    if len(basis) != m:
        raise SimplexError("Basis size does not match number of constraints")

    history: list[dict] = []

    for iteration in range(max_iter + 1):
        if len(set(basis)) != len(basis):
            raise SimplexError("Basis contains duplicate columns")

        nonbasic = np.array(
            [j for j in range(n) if j not in set(basis)],
            dtype=np.intp,
        )

        B = A[:, basis]

        x_basic = _solve_basis(
            B,
            b,
            condition_limit=condition_limit,
        )

        if np.any(x_basic < -tol):
            return (
                "numerical_failure",
                "current basis is not primal feasible",
                np.zeros(n),
                basis,
                iteration,
                tuple(history),
            )

        x_basic = _clean_zero(x_basic, tol)

        c_basic = c[basis]

        # B^T y = c_B
        y = _solve_basis(
            B.T,
            c_basic,
            condition_limit=condition_limit,
        )

        reduced = c[nonbasic] - A[:, nonbasic].T @ y

        reduced = _clean_zero(reduced, tol)

        # Bland's rule: first negative reduced cost enters.
        entering_candidates = nonbasic[reduced < -tol]

        objective = float(c_basic @ x_basic)

        history.append(
            {
                "phase": phase,
                "iteration": iteration,
                "objective": objective,
                "min_reduced_cost": (
                    float(np.min(reduced))
                    if reduced.size
                    else 0.0
                ),
                "basis_size": len(basis),
            }
        )

        if entering_candidates.size == 0:
            x = np.zeros(n, dtype=np.float64)
            x[basis] = x_basic

            return (
                "optimal",
                "simplex optimum reached",
                x,
                basis,
                iteration,
                tuple(history),
            )

        entering = int(entering_candidates[0])

        # Direction: B d = a_entering
        d = _solve_basis(
            B,
            A[:, entering],
            condition_limit=condition_limit,
        )

        positive = d > tol

        if not np.any(positive):
            return (
                "unbounded",
                f"objective is unbounded in column {entering}",
                np.zeros(n),
                basis,
                iteration,
                tuple(history),
            )

        ratios = np.full(m, np.inf, dtype=np.float64)
        ratios[positive] = x_basic[positive] / d[positive]

        theta = float(np.min(ratios))

        leaving_candidates = np.flatnonzero(
            np.abs(ratios - theta) <= tol * max(1.0, abs(theta))
        )

        # Bland tie breaking on the actual basic-column index.
        leaving_row = int(
            min(
                leaving_candidates,
                key=lambda row: basis[int(row)],
            )
        )

        leaving = basis[leaving_row]

        basis[leaving_row] = entering

        history[-1]["entering"] = entering
        history[-1]["leaving"] = leaving
        history[-1]["step"] = theta

    x = np.zeros(n, dtype=np.float64)

    return (
        "max_iterations",
        f"maximum simplex iterations ({max_iter}) reached",
        x,
        basis,
        max_iter,
        tuple(history),
    )


def _remove_artificial_variables(
    A_phase1: np.ndarray,
    A_norm: np.ndarray,
    b_norm: np.ndarray,
    basis: list[int],
    n_original: int,
    tol: float,
    condition_limit: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Drive artificial variables out of the basis and eliminate redundant rows."""
    basis = list(basis)
    m = A_norm.shape[0]

    for row in range(m):
        if basis[row] < n_original:
            continue

        B = A_phase1[:, basis]
        e_row = np.zeros(m, dtype=np.float64)
        e_row[row] = 1.0
        try:
            v = _solve_basis(B.T, e_row, condition_limit=condition_limit)
        except SimplexError:
            v = np.linalg.lstsq(B.T, e_row, rcond=None)[0]

        row_coeffs = v @ A_norm  # length n_original

        candidates = [
            j
            for j in range(n_original)
            if j not in set(basis) and abs(row_coeffs[j]) > tol
        ]
        candidates.sort(key=lambda j: abs(row_coeffs[j]), reverse=True)

        replacement = None
        for cand in candidates:
            trial = list(basis)
            trial[row] = cand
            try:
                B_trial = A_phase1[:, trial]
                _solve_basis(
                    B_trial,
                    b_norm,
                    condition_limit=condition_limit,
                )
                replacement = cand
                break
            except SimplexError:
                continue

        if replacement is not None:
            basis[row] = replacement

    # Identify valid (original) vs redundant rows
    valid_rows = [i for i in range(m) if basis[i] < n_original]
    redundant_rows = [i for i in range(m) if basis[i] >= n_original]

    if redundant_rows:
        A_phase2 = A_norm[valid_rows, :]
        b_phase2 = b_norm[valid_rows]
        basis_phase2 = [basis[i] for i in valid_rows]
    else:
        A_phase2 = A_norm
        b_phase2 = b_norm
        basis_phase2 = list(basis)

    return A_phase2, b_phase2, basis_phase2


def solve_simplex(
    sf: StandardFormLP,
    *,
    tol: float = 1e-8,
    max_iter: int = 1000,
    condition_limit: float = 1e12,
) -> SimplexResult:
    """Solve a StandardFormLP using two-phase Revised Simplex."""

    A = np.asarray(sf.A, dtype=np.float64)
    b = np.asarray(sf.b, dtype=np.float64)
    c = np.asarray(sf.c_min, dtype=np.float64)

    m, n = A.shape

    if b.shape != (m,):
        raise SimplexError("A/b shape mismatch")

    if c.shape != (n,):
        raise SimplexError("A/c shape mismatch")

    if not np.all(np.isfinite(A)):
        raise SimplexError("A contains non-finite values")

    if not np.all(np.isfinite(b)):
        raise SimplexError("b contains non-finite values")

    if not np.all(np.isfinite(c)):
        raise SimplexError("c contains non-finite values")

    if m == 0:
        x_std = np.zeros(n)
        positive_cost = c < -tol

        if np.any(positive_cost):
            return SimplexResult(
                status="unbounded",
                message="unconstrained negative objective direction",
                objective=-np.inf,
                x=sf.recover_original(x_std),
                x_standard=x_std,
                iterations=0,
                phase_one_iterations=0,
                phase_two_iterations=0,
                primal_residual=0.0,
                rel_primal=0.0,
                rel_gap=np.inf,
                history=(),
            )

        recovered = sf.recover_original(x_std)
        # Report the objective in ORIGINAL coordinates (see the note at the
        # main return site below): c_min @ x_standard omits the constant
        # c_orig @ orig_offset introduced by bound shifts.
        objective_orig = float(sf.c_orig @ recovered)

        return SimplexResult(
            status="optimal",
            message="empty constraint system solved",
            objective=objective_orig,
            x=recovered,
            x_standard=x_std,
            iterations=0,
            phase_one_iterations=0,
            phase_two_iterations=0,
            primal_residual=0.0,
            rel_primal=0.0,
            rel_gap=0.0,
            history=(),
        )

    A_norm, b_norm, row_sign = _normalize_rhs(A, b)
    A_phase1, c_phase1, basis = _phase_one_setup(A_norm, b_norm)

    phase1_status, phase1_message, x1, basis, phase1_iters, h1 = (
        _simplex_iterations(
            A_phase1,
            b_norm,
            c_phase1,
            basis,
            tol=tol,
            max_iter=max_iter,
            condition_limit=condition_limit,
            phase=1,
        )
    )

    if phase1_status != "optimal":
        return SimplexResult(
            status=phase1_status,
            message=f"Phase I failed: {phase1_message}",
            objective=np.nan,
            x=np.full(sf.n_orig, np.nan),
            x_standard=np.full(n, np.nan),
            iterations=phase1_iters,
            phase_one_iterations=phase1_iters,
            phase_two_iterations=0,
            primal_residual=np.inf,
            rel_primal=np.inf,
            rel_gap=np.inf,
            history=h1,
        )

    phase1_objective = float(c_phase1 @ x1)

    if phase1_objective > tol:
        return SimplexResult(
            status="infeasible",
            message=(
                f"Phase I minimum {phase1_objective:.6e} "
                "is positive"
            ),
            objective=np.nan,
            x=np.full(sf.n_orig, np.nan),
            x_standard=np.full(n, np.nan),
            iterations=phase1_iters,
            phase_one_iterations=phase1_iters,
            phase_two_iterations=0,
            primal_residual=np.inf,
            rel_primal=np.inf,
            rel_gap=np.inf,
            history=h1,
        )

    A_phase2, b_phase2, basis = _remove_artificial_variables(
        A_phase1,
        A_norm,
        b_norm,
        basis,
        n,
        tol,
        condition_limit,
    )

    phase2_status, phase2_message, x2, basis, phase2_iters, h2 = (
        _simplex_iterations(
            A_phase2,
            b_phase2,
            c,
            basis,
            tol=tol,
            max_iter=max_iter,
            condition_limit=condition_limit,
            phase=2,
        )
    )

    history = h1 + h2

    if phase2_status != "optimal":
        return SimplexResult(
            status=phase2_status,
            message=phase2_message,
            objective=np.nan,
            x=np.full(sf.n_orig, np.nan),
            x_standard=x2,
            iterations=phase1_iters + phase2_iters,
            phase_one_iterations=phase1_iters,
            phase_two_iterations=phase2_iters,
            primal_residual=np.inf,
            rel_primal=np.inf,
            rel_gap=np.inf,
            history=history,
        )

    x2 = _clean_zero(x2, tol)

    primal_residual = _inf_norm(A @ x2 - b)
    rel_primal = primal_residual / max(1.0, _inf_norm(b))

    recovered = sf.recover_original(x2)

    # Report the objective in ORIGINAL coordinates.  The standard-form
    # objective c_min @ x2 omits the constant c_orig @ orig_offset that
    # bound conversions introduce (x = offset +/- x' for LO/UP/box bounds;
    # FX variables carry only an offset and no column at all), so e.g.
    # "min 2x s.t. x + s = 5, x >= 3" would report 0 instead of 6.
    # sf.c_orig is already in the model's original sense (Mehrotra reports
    # sf.c_orig @ x_orig without a sign flip); no maximize handling is
    # needed here.
    objective = float(sf.c_orig @ recovered)

    # A Simplex solution is vertex-based, so complementarity is not the
    # primary quality metric. We report a primal-based gap proxy here.
    rel_gap = primal_residual / max(
        1.0,
        abs(objective),
        _inf_norm(b),
    )

    return SimplexResult(
        status="optimal",
        message=(
            f"converged in {phase1_iters + phase2_iters} "
            "Revised Simplex iterations"
        ),
        objective=objective,
        x=recovered,
        x_standard=x2,
        iterations=phase1_iters + phase2_iters,
        phase_one_iterations=phase1_iters,
        phase_two_iterations=phase2_iters,
        primal_residual=primal_residual,
        rel_primal=rel_primal,
        rel_gap=rel_gap,
        history=history,
    )


__all__ = [
    "SimplexError",
    "SimplexResult",
    "solve_simplex",
]