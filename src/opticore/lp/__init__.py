"""LP subpackage: production LP/MILP engines (plus the legacy standard-form QP).

Module layout (moved here from the old flat ``src/lp`` directory; every module
below now uses package-relative imports and no longer manipulates ``sys.path``):

* ``mehrotra``          -- production CPU Mehrotra predictor-corrector IPM.
* ``linear_system``     -- reduced Newton (Schur) factorization/solve backends.
* ``gpu_linear_system`` -- optional CUDA/CuPy backend (kept lazy on purpose).
* ``simplex``           -- two-phase revised Simplex LP solver.
* ``crossover``         -- sparse Phase I/II crossover engine.
* ``branch_bound``      -- indigenous branch-and-bound MILP on top of the LP core.
* ``qp``                -- LEGACY dense standard-form QP; ``opticore.qp`` is the
                          canonical (sparse) convex-QP package.

Every name re-exported here is defined in one of those modules -- no new names
are invented.  ``gpu_linear_system`` is deliberately NOT imported eagerly so
that CuPy/CUDA stays optional and import-time cost is unchanged; reach it via
``opticore.lp.gpu_linear_system`` (e.g. ``gpu_available``/``gpu_info``).
"""

from .branch_bound import MilpError, MilpResult, solve_milp
from .crossover import (
    CROSSOVER_MERIT_RATIO,
    crossover_from_ipm,
    sparse_phase1,
    sparse_phase2,
)
from .linear_system import (
    MAX_RHO_P,
    LinearSystemError,
    ReducedNewtonFactorization,
    factor_reduced_system,
    solve_reduced_system,
)
from .mehrotra import (
    MehrotraError,
    MehrotraResult,
    StandardFormLP,
    solve_lp,
    solve_standard_form,
    to_standard_form,
)
from .simplex import SimplexError, SimplexResult, solve_simplex

# Legacy path: keep ``opticore.lp.qp`` (dense standard-form QP) importable as a
# submodule.  Its symbols are intentionally NOT flattened into this namespace
# because ``opticore.solve_qp`` is the canonical ``opticore.qp`` solver.
from . import qp  # noqa: F401  (legacy dense standard-form QP module)

__all__ = [
    # Mehrotra interior-point LP core
    "MehrotraError",
    "MehrotraResult",
    "StandardFormLP",
    "solve_lp",
    "solve_standard_form",
    "to_standard_form",
    # Reduced Newton linear-system backend
    "MAX_RHO_P",
    "LinearSystemError",
    "ReducedNewtonFactorization",
    "factor_reduced_system",
    "solve_reduced_system",
    # Revised Simplex
    "SimplexError",
    "SimplexResult",
    "solve_simplex",
    # Sparse crossover
    "CROSSOVER_MERIT_RATIO",
    "crossover_from_ipm",
    "sparse_phase1",
    "sparse_phase2",
    # Branch-and-bound MILP
    "MilpError",
    "MilpResult",
    "solve_milp",
]
