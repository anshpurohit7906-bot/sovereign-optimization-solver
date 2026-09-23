"""OptiCore -- indigenous LP / MILP / QP optimization stack (SIH 26119).

Public solver API
-----------------
``LP(lp, ...)``       linear program      -> ``MehrotraResult``
``MILP(lp, ...)``     mixed-integer LP    -> ``MilpResult``
``QP(problem, ...)``  convex quadratic    -> ``QPResult``

``LP`` / ``MILP`` / ``QP`` are thin, behaviour-preserving entry points over the
production implementations -- they add no numerical logic of their own:

* ``LP``   -> ``opticore.lp.mehrotra.solve_lp`` (Mehrotra predictor-corrector IPM)
* ``MILP`` -> ``opticore.lp.branch_bound.solve_milp`` (branch-and-bound)
* ``QP``   -> ``opticore.qp.solver.solve_qp`` (canonical sparse convex QP)

The legacy dense standard-form QP solver stays available as
``opticore.lp.qp.solve_qp`` (and is deliberately not aliased here, so that the
bare name ``solve_qp`` is unambiguous).

Package layout
--------------
* ``opticore.mps_parser``      -- MPS -> ``LPModel``
* ``opticore.numerical_model`` -- ``LPModel`` -> ``NumericalLP`` arrays
* ``opticore.constraint_form`` -- bounded-constraint form
* ``opticore.scaling``         -- row/column equilibration
* ``opticore.lp``              -- LP/MILP engines (Mehrotra, simplex, crossover, B&B)
* ``opticore.qp``              -- canonical convex QP package

All internal imports are package-relative; importing ``opticore`` never mutates
``sys.path``.

NOTE (pending, separate step): the console entry point in ``pyproject.toml``
still targets the repository-root module (``opticore = "opticore:main"``); the
CLI relocation is intentionally out of scope for this package/import migration.
"""

from . import constraint_form, lp, mps_parser, numerical_model, qp, scaling
from .constraint_form import (
    BoundedConstraintLP,
    to_bounded_constraint,
    validate_bounded_constraint,
)
from .lp import (
    CROSSOVER_MERIT_RATIO,
    MAX_RHO_P,
    LinearSystemError,
    MehrotraError,
    MehrotraResult,
    MilpError,
    MilpResult,
    SimplexError,
    SimplexResult,
    StandardFormLP,
    crossover_from_ipm,
    factor_reduced_system,
    solve_lp,
    solve_milp,
    solve_reduced_system,
    solve_simplex,
    solve_standard_form,
    sparse_phase1,
    sparse_phase2,
    to_standard_form,
)
from .mps_parser import LPModel, MPSParseError, MPSParser
from .numerical_model import (
    NumericalLP,
    NumericalModelError,
    load_numeric_mps,
    to_numeric,
    to_sparse_numeric,
    validate_numeric_lp,
)
from .qp import (
    QPProblem,
    QPResult,
    QPSolverError,
    QPValidationError,
    solve_qp,
)
from .scaling import ScaledLP, scale_lp, unscale_solution


def LP(lp, *args, **kwargs):
    """Solve a linear program.

    Thin alias of :func:`opticore.lp.solve_lp` (Mehrotra IPM).  ``lp`` is a
    ``NumericalLP``; all keyword arguments are forwarded unchanged, e.g.
    ``maximize=True``, ``tol=``, ``max_iter=``, ``backend="cpu"/"gpu"/"auto"``.
    """
    return solve_lp(lp, *args, **kwargs)


def MILP(lp, *args, **kwargs):
    """Solve a mixed-integer linear program by branch-and-bound.

    Thin alias of :func:`opticore.lp.solve_milp`; see that function for the
    ``integer_mask``/``gap_tol``/``node_limit``/... keyword arguments.
    """
    return solve_milp(lp, *args, **kwargs)


def QP(problem, *args, **kwargs):
    """Solve a convex quadratic program.

    Thin alias of :func:`opticore.qp.solve_qp` (canonical sparse convex-QP
    package).  Accepts a ``QPProblem`` (or the raw ``P, q, G, h, A, b, lb, ub``
    arguments) exactly as the wrapped solver does.
    """
    return solve_qp(problem, *args, **kwargs)


__all__ = [
    # Intended public solver API
    "LP",
    "MILP",
    "QP",
    # Subpackages
    "constraint_form",
    "lp",
    "mps_parser",
    "numerical_model",
    "qp",
    "scaling",
    # MPS parsing
    "LPModel",
    "MPSParseError",
    "MPSParser",
    # Numerical model
    "NumericalLP",
    "NumericalModelError",
    "load_numeric_mps",
    "to_numeric",
    "to_sparse_numeric",
    "validate_numeric_lp",
    # Bounded-constraint form
    "BoundedConstraintLP",
    "to_bounded_constraint",
    "validate_bounded_constraint",
    # Scaling
    "ScaledLP",
    "scale_lp",
    "unscale_solution",
    # LP core
    "CROSSOVER_MERIT_RATIO",
    "MAX_RHO_P",
    "LinearSystemError",
    "MehrotraError",
    "MehrotraResult",
    "SimplexError",
    "SimplexResult",
    "StandardFormLP",
    "crossover_from_ipm",
    "factor_reduced_system",
    "solve_lp",
    "solve_reduced_system",
    "solve_simplex",
    "solve_standard_form",
    "sparse_phase1",
    "sparse_phase2",
    "to_standard_form",
    # MILP
    "MilpError",
    "MilpResult",
    "solve_milp",
    # QP (canonical)
    "QPProblem",
    "QPResult",
    "QPSolverError",
    "QPValidationError",
    "solve_qp",
]
