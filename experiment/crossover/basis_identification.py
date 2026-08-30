"""Candidate basis rankings from a Mehrotra standard-form iterate.

This module deliberately identifies candidates only.  It contains no pivot
repair, Phase-I logic, or replacement factorization method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasisCandidate:
    name: str
    columns: tuple[int, ...]
    scores: np.ndarray


@dataclass(frozen=True)
class RankAwareBasis:
    """A rank-incremental selection result, before any feasibility attempt."""

    columns: tuple[int, ...]
    numerical_rank: int
    rejected_columns: int
    replacement_columns: int
    inspected_columns: int
    relative_rank_tolerance: float


def candidate_rankings(x: np.ndarray, z: np.ndarray, basis_size: int) -> tuple[BasisCandidate, ...]:
    """Return deterministic size-m rankings based on complementary IPM data.

    At a nondegenerate optimum, basic variables tend to have comparatively
    large primal values and small dual slacks.  We therefore test three simple
    orderings: primal descending, dual slack ascending, and a guarded x/z
    complementarity score descending.  This is a feasibility probe, not a
    claim that any ordering identifies the true optimal basis.
    """
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    if x.ndim != 1 or z.shape != x.shape:
        raise ValueError("x and z must be same-length vectors")
    if not (0 <= basis_size <= x.size):
        raise ValueError("basis_size must lie between zero and len(x)")
    guarded_z = np.maximum(np.abs(z), np.finfo(np.float64).tiny)
    score_x_over_z = np.abs(x) / guarded_z
    rankings = (
        ("primal_desc", -np.abs(x)),
        ("dual_asc", np.abs(z)),
        ("x_over_z_desc", -score_x_over_z),
    )
    output: list[BasisCandidate] = []
    for name, ordering_score in rankings:
        # mergesort makes ties deterministic by original column index.
        columns = tuple(int(j) for j in np.argsort(ordering_score, kind="mergesort")[:basis_size])
        output.append(BasisCandidate(name=name, columns=columns, scores=ordering_score.copy()))
    return tuple(output)


def rank_aware_x_over_z_basis(
    A: np.ndarray, x: np.ndarray, z: np.ndarray, basis_size: int, *,
    relative_rank_tolerance: float = 1e-10,
) -> RankAwareBasis:
    """Greedily construct an independent basis in descending ``|x|/|z|`` order.

    A column is accepted only when its twice-reorthogonalized residual against
    the already selected orthonormal directions has norm greater than
    ``relative_rank_tolerance * ||a_j||_2``.  Rejected preferred columns do
    not end the experiment: subsequent, lower-scored columns are considered.
    This is rank-aware selection only, not pivot repair or a factorization
    replacement for Simplex.
    """
    A = np.asarray(A, dtype=np.float64)
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    m, n = A.shape
    if x.shape != (n,) or z.shape != (n,) or basis_size != m:
        raise ValueError("A/x/z shapes or basis_size do not match standard form")
    if relative_rank_tolerance <= 0.0:
        raise ValueError("relative_rank_tolerance must be positive")

    score = np.abs(x) / np.maximum(np.abs(z), np.finfo(np.float64).tiny)
    order = np.argsort(-score, kind="mergesort")
    Q = np.empty((m, 0), dtype=np.float64)
    selected: list[int] = []
    rejected = 0
    inspected = 0

    for column in order:
        inspected += 1
        a = A[:, column]
        a_norm = float(np.linalg.norm(a))
        if a_norm == 0.0:
            rejected += 1
            continue
        residual = a.copy()
        # Reorthogonalization keeps this test materially more reliable than a
        # single projection in the very ill-scaled PILOT4 standard form.
        if Q.shape[1]:
            residual -= Q @ (Q.T @ residual)
            residual -= Q @ (Q.T @ residual)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= relative_rank_tolerance * a_norm:
            rejected += 1
            continue
        Q = np.column_stack((Q, residual / residual_norm))
        selected.append(int(column))
        if len(selected) == basis_size:
            break

    # Columns after index m-1 in the preference order are replacements for
    # rank-deficient members of the naive first-m selection.
    replacement = sum(int(np.flatnonzero(order == col)[0]) >= basis_size for col in selected)
    return RankAwareBasis(
        columns=tuple(selected), numerical_rank=len(selected), rejected_columns=rejected,
        replacement_columns=replacement, inspected_columns=inspected,
        relative_rank_tolerance=relative_rank_tolerance,
    )
