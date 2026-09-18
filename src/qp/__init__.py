"""Canonical sparse convex QP package (adapted from teammate drop).

Layout mirrors src/lp/: problem -> solver -> linear_system -> verify ->
benchmark.  src/lp/qp.py is retained as the legacy standard-form path.
"""
from .problem import QPProblem, QPValidationError
from .solver import QPResult, QPSolverError, solve_qp
from .verify import certificate
from .benchmark import generate_sparse_qp, make_suite, run_benchmark
from .qps import read_qps

__all__ = [
    'QPProblem', 'QPValidationError', 'QPResult', 'QPSolverError',
    'solve_qp', 'certificate', 'generate_sparse_qp', 'make_suite',
    'run_benchmark', 'read_qps',
]
