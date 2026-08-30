"""Minimal crossover feasibility probe using production Mehrotra and Simplex."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _path in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.lp.mehrotra import MehrotraResult, StandardFormLP, solve_standard_form, to_standard_form
from src.lp.simplex import SimplexError, _inf_norm, _simplex_iterations, _solve_basis
from src.numerical_model import NumericalLP
from experiment.crossover.basis_identification import (
    BasisCandidate,
    RankAwareBasis,
    candidate_rankings,
    rank_aware_x_over_z_basis,
)


@dataclass(frozen=True)
class CandidateReport:
    name: str
    basis: tuple[int, ...]
    basis_size: int
    rank: int
    condition_number: float
    basis_valid: bool
    minimum_basic_value: Optional[float]
    maximum_basic_value: Optional[float]
    negative_basic_variables: Optional[int]
    primal_feasible: bool
    phase_two_launched: bool
    phase_two_status: Optional[str]
    phase_two_iterations: int
    objective: Optional[float]
    primal_residual: Optional[float]
    minimum_reduced_cost: Optional[float]
    message: str


@dataclass(frozen=True)
class CrossoverResult:
    mehrotra: MehrotraResult
    rows: int
    columns: int
    candidate_basis_size: int
    reports: tuple[CandidateReport, ...]
    runtime: float


@dataclass(frozen=True)
class RankAwareBasisResult:
    """Outcome of the rank-aware basis-construction experiment only."""

    mehrotra: MehrotraResult
    rows: int
    columns: int
    final_basis_size: int
    numerical_rank: int
    condition_number: float
    selected_columns: int
    rejected_columns: int
    replacement_columns: int
    inspected_columns: int
    simplex_basis_valid: bool
    validation_message: str
    runtime: float


def _evaluate_candidate(
    sf: StandardFormLP, candidate: BasisCandidate, *, tol: float,
    simplex_max_iter: int, condition_limit: float,
) -> CandidateReport:
    A, b, c = sf.A, sf.b, sf.c_min
    basis = list(candidate.columns)
    B = A[:, basis]
    rank = int(np.linalg.matrix_rank(B))
    try:
        condition = float(np.linalg.cond(B))
    except np.linalg.LinAlgError:
        condition = float("inf")
    try:
        x_basic = _solve_basis(B, b, condition_limit=condition_limit)
    except SimplexError as exc:
        return CandidateReport(
            candidate.name, tuple(basis), len(basis), rank, condition, False,
            None, None, None, False, False, None, 0, None, None, None,
            f"basis rejected by production Simplex validation: {exc}",
        )

    min_basic, max_basic = float(np.min(x_basic)), float(np.max(x_basic))
    negative = int(np.count_nonzero(x_basic < -tol))
    feasible = negative == 0
    if not feasible:
        return CandidateReport(
            candidate.name, tuple(basis), len(basis), rank, condition, True,
            min_basic, max_basic, negative, False, False, None, 0, None, None, None,
            "candidate basis is not primal feasible; no repair or Phase I is attempted",
        )

    try:
        status, message, x_phase2, _, iterations, history = _simplex_iterations(
            A, b, c, basis, tol=tol, max_iter=simplex_max_iter,
            condition_limit=condition_limit, phase=2,
        )
    except SimplexError as exc:
        return CandidateReport(
            candidate.name, tuple(basis), len(basis), rank, condition, True,
            min_basic, max_basic, negative, True, True, "numerical_failure", 0,
            None, None, None, f"Phase II raised SimplexError: {exc}",
        )

    if status != "optimal":
        min_reduced = history[-1]["min_reduced_cost"] if history else None
        return CandidateReport(
            candidate.name, tuple(basis), len(basis), rank, condition, True,
            min_basic, max_basic, negative, True, True, status, iterations,
            None, None, min_reduced, f"Phase II did not complete: {message}",
        )

    primal_residual = _inf_norm(A @ x_phase2 - b)
    # Match production's original-sense objective convention rather than
    # exposing the internal minimization objective for maximization models.
    objective = float(sf.c_orig @ sf.recover_original(x_phase2))
    min_reduced = history[-1]["min_reduced_cost"] if history else None
    return CandidateReport(
        candidate.name, tuple(basis), len(basis), rank, condition, True,
        min_basic, max_basic, negative, True, True, status, iterations,
        objective, primal_residual, min_reduced, message,
    )


def run_crossover_probe(
    lp: NumericalLP, *, tol: float = 1e-8, mehrotra_max_iter: int = 100,
    simplex_max_iter: int = 1_000, condition_limit: float = 1e12,
) -> CrossoverResult:
    """Run a one-way, non-production IPM-to-basis feasibility experiment.

    The production Mehrotra result is never replaced.  For a non-optimal IPM
    termination, its public result fields already contain production's
    best-trusted iterate (per Mehrotra's `_finish` behavior), so no solver
    mathematics is duplicated here.
    """
    started = time.perf_counter()
    sf = to_standard_form(lp)
    mehrotra = solve_standard_form(sf, tol=tol, max_iter=mehrotra_max_iter)
    A = sf.A
    m, n = A.shape
    candidates = candidate_rankings(mehrotra.x_standard, mehrotra.z_standard, m)
    reports = tuple(
        _evaluate_candidate(sf, candidate, tol=tol, simplex_max_iter=simplex_max_iter,
                            condition_limit=condition_limit)
        for candidate in candidates
    )
    return CrossoverResult(mehrotra, m, n, m, reports, time.perf_counter() - started)


def run_rank_aware_basis_probe(
    lp: NumericalLP, *, tol: float = 1e-8, mehrotra_max_iter: int = 100,
    condition_limit: float = 1e12, relative_rank_tolerance: float = 1e-10,
) -> RankAwareBasisResult:
    """Test whether a usable basis is cheaply selectable; never run Phase II.

    This function deliberately stops after production Simplex basis validation.
    It does not calculate ``B^-1 b`` feasibility, repair a basis, or invoke
    `_simplex_iterations`; those are outside this rank-construction question.
    """
    started = time.perf_counter()
    sf = to_standard_form(lp)
    mehrotra = solve_standard_form(sf, tol=tol, max_iter=mehrotra_max_iter)
    report = evaluate_rank_aware_basis(
        sf, mehrotra.x_standard, mehrotra.z_standard,
        condition_limit=condition_limit, relative_rank_tolerance=relative_rank_tolerance,
    )
    return RankAwareBasisResult(
        mehrotra, sf.m, sf.n, *report, time.perf_counter() - started,
    )


def evaluate_rank_aware_basis(
    sf: StandardFormLP, x_standard: np.ndarray, z_standard: np.ndarray, *,
    condition_limit: float = 1e12, relative_rank_tolerance: float = 1e-10,
) -> tuple[int, int, float, int, int, int, int, bool, str]:
    """Evaluate rank-aware selection for supplied standard-form IPM vectors.

    Returns only construction/validation metrics; it never runs Phase II or a
    feasibility-repair procedure.
    """
    m, n = sf.A.shape
    selected: RankAwareBasis = rank_aware_x_over_z_basis(
        sf.A, x_standard, z_standard, m,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    if selected.numerical_rank != m:
        return (
            len(selected.columns), selected.numerical_rank, float("inf"), len(selected.columns),
            selected.rejected_columns, selected.replacement_columns, selected.inspected_columns, False,
            "rank-aware scan could not construct a full m-column numerical basis",
        )

    B = sf.A[:, list(selected.columns)]
    rank = int(np.linalg.matrix_rank(B))
    try:
        condition = float(np.linalg.cond(B))
    except np.linalg.LinAlgError:
        condition = float("inf")
    try:
        # Invoke the existing Simplex condition-limit validation, but discard
        # the solve result.  No feasibility or Phase-II behavior occurs here.
        _solve_basis(B, sf.b, condition_limit=condition_limit)
        valid, message = True, "basis passes production Simplex condition validation"
    except SimplexError as exc:
        valid, message = False, f"basis rejected by production Simplex validation: {exc}"
    return (
        len(selected.columns), rank, condition, len(selected.columns), selected.rejected_columns,
        selected.replacement_columns, selected.inspected_columns, valid, message,
    )
