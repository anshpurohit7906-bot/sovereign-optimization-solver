"""Minimal, isolated IPM-to-Simplex crossover feasibility experiment."""

from .solver import (
    CrossoverResult,
    RankAwareBasisResult,
    run_crossover_probe,
    run_rank_aware_basis_probe,
)

__all__ = ["CrossoverResult", "RankAwareBasisResult", "run_crossover_probe", "run_rank_aware_basis_probe"]
