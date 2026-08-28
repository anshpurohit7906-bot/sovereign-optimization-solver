"""Dense Mehrotra predictor-corrector interior-point solver for standard-form LPs.

This module converts the project's E/L row formulation (``NumericalLP``) into
the standard-form LP

    minimize    c^T x
    subject to  A x = b
                x >= 0

by transforming every original variable into one or two nonnegative columns
(identity for ``[0, +inf)``; shifted ``x = L + x'`` for LO bounds; reflected
``x = U - x'`` for UP bounds with a free lower bound; split ``x = x+ - x-``
for free variables; shifted plus one appended upper-bound row ``x' <= U - L``
for box bounds ``[L, U]``; substituted out at its fixed value for FX bounds),
appending one nonnegative slack to every inequality row (+1 column for 'L'
rows: ``a_i^T x + s_i = b_i``; -1 column for 'G' rows: ``a_i^T x - s_i =
b_i``), and keeping every 'E' row as an equality.
The solver MINIMIZES by default (``maximize=False``);
when ``maximize=True`` is passed explicitly the objective is negated internally
and ``MehrotraResult.objective`` is reported in the ORIGINAL sense.  Objective
sense is never inferred from the objective row name.

Each iteration assembles the reduced Newton system

    [ H    -A^T ] [ dx ]   [ rhs_x  ]
    [ A      0  ] [ dy ] = [ rhs_eq ],       H = diag(z / x) > 0

and hands it to the regularized dense Cholesky backend in ``linear_system``:
one ``factor_reduced_system`` call per iteration, reused for the affine
predictor and the Mehrotra corrector right-hand sides.  For the perturbed KKT
residuals

    r_p = A x - b
    r_d = A^T y + z - c
    r_c = x*z - sigma*mu + dx_aff*dz_aff          (all elementwise)

the right-hand sides and the dual-slack direction are

    rhs_x  = r_d - r_c / x
    rhs_eq = -r_p
    dz     = -(r_c + z * dx) / x

which reproduces all three Newton equations
``A dx = -r_p``, ``A^T dy + dz = -r_d``, ``Z dx + X dz = -r_c`` exactly.

Because the fraction-to-boundary rule (tau = 0.995) keeps x > 0 and z > 0
invariantly, H is positive definite by construction, and the Schur complement
S = A H^{-1} A^T is positive definite whenever A has full row rank (verified
at conversion time) -- exactly the system ``linear_system`` solves.

Scaling: the standard-form problem is equilibrated with the existing
``scaling.scale_lp`` (row/column max scaling) and every Newton iteration --
directions, steps, termination tests -- runs in those scaled coordinates;
``MehrotraResult`` fields are mapped back to original units before reporting.

Numerical safeguards: strictly positive Mehrotra initialization, guarded
divisions, finiteness checks on every direction and update, explicit
reporting of iterations that required Cholesky regularization, and no silent
recovery from NaN/Inf directions or collapsed step lengths.

Degenerate-tail safeguard: mu = x^T z / n carries the units of the objective,
and the products x_i*z_i carry absolute rounding noise proportional to the
data scale, so once mu reaches the scale-aware floor
``mu_floor * max(1, ||b||_inf, ||c||_inf)`` (default ``mu_floor = 1e-12``,
roughly 4500 double-precision epsilons at unit scale) the barrier trajectory
can no longer be resolved: H = diag(z/x) spans extreme dynamic range and the
computed Newton directions become rounding-error dominated (the dual residual
then *grows* while mu collapses).  The solver stops at the floor before taking
another step and reports "optimal" only if the existing residual criteria are
met, "numerical_tail" when primal feasibility and the best-seen relative gap
are within tolerance but the dual residual is noise-degraded, and "stalled"
when even that practical accuracy was not attained.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Imports.  The project uses a flat layout: ``linear_system`` lives beside
# this file in ``src/lp`` while ``numerical_model`` lives in ``src``.  Make
# both importable whether this file is run as a script or imported.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.dirname(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from linear_system import (  # noqa: E402
    LinearSystemError,
    factor_reduced_system,
    solve_reduced_system,
)
from numerical_model import NumericalLP, load_numeric_mps  # noqa: E402
from scaling import scale_lp, unscale_solution  # noqa: E402

__all__ = [
    "MehrotraError",
    "StandardFormLP",
    "MehrotraResult",
    "to_standard_form",
    "solve_standard_form",
    "solve_lp",
]


class MehrotraError(ValueError):
    """Raised when the LP cannot be converted to the supported standard form."""


# ---------------------------------------------------------------------------
# Standard-form conversion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StandardFormLP:
    """Standard-form LP: minimize ``c_min @ x`` s.t. ``A @ x = b``, ``x >= 0``.

    Attributes
    ----------
    A, b, c_min : standard-form data (``c_min`` is the MINIMIZATION objective).
    c_orig : original objective coefficients over the original variables only.
    maximize : True when the original model maximizes ``c_orig @ x``.
    n_orig : number of ORIGINAL variables (``len(var_names)``).
    n_block : number of standard-form columns before the slacks (one for each
        non-fixed original variable, two for each free variable).
    block_to_orig, block_sign : column mapping; standard-form column ``k``
        contributes ``block_sign[k] * x_standard[k]`` to original variable
        ``block_to_orig[k]``.
    orig_offset : per-original-variable constant of ``x_j = offset_j +
        sum_k block_sign[k] * x'_k`` (lower bound for LO/box, upper bound for
        UP, fixed value for FX -- which have no standard-form column -- else 0).
    num_slacks : number of inequality-row slacks (original 'L'/'G' rows plus
        one appended upper-bound row per box variable).
    slack_row_indices : row indices (into the extended row list) that received
        a slack (+1 column for 'L', -1 column for 'G').
    var_names, slack_names, row_names : name metadata; ``row_names`` covers
        the extended rows (original rows, then box upper-bound rows).
    """

    A: np.ndarray
    b: np.ndarray
    c_min: np.ndarray
    c_orig: np.ndarray
    maximize: bool
    n_orig: int
    n_block: int
    block_to_orig: np.ndarray
    block_sign: np.ndarray
    orig_offset: np.ndarray
    num_slacks: int
    slack_row_indices: tuple[int, ...]
    var_names: tuple[str, ...]
    slack_names: tuple[str, ...]
    row_names: tuple[str, ...]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

    def recover_original(self, x_std: np.ndarray) -> np.ndarray:
        """Map a standard-form solution back to the original variables.

        Inverts ``x_j = offset_j + sum_k block_sign[k] * x'_k``; fixed (FX)
        variables have no standard-form column and recover their fixed value
        from the offset alone.
        """
        x = self.orig_offset.copy()
        block = np.asarray(x_std, dtype=np.float64)[: self.n_block]
        np.add.at(x, self.block_to_orig, self.block_sign * block)
        return x


def to_standard_form(lp: NumericalLP, *, maximize: bool = False) -> StandardFormLP:
    """Convert an E/L/G ``NumericalLP`` to standard form.

    Every original variable is transformed into nonnegative columns via
    ``x_j = offset_j + sum_k block_sign[k] * x'_k``:

    - ``[0, +inf)`` -> kept as-is (one column, sign +1, offset 0);
    - LO ``x >= L`` -> shifted, ``x = L + x'`` (the RHS absorbs ``A @ L``);
    - UP ``x <= U`` with a free lower bound -> reflected, ``x = U - x'``
      (column negated, the RHS absorbs ``A @ U``);
    - FR (free) -> split, ``x = x+ - x-`` (two columns with signs +1/-1);
    - box ``[L, U]`` -> shifted ``x = L + x'`` plus one appended inequality
      row ``x' + s = U - L`` enforcing the upper bound;
    - FX ``x == v`` -> substituted out (no column; rows and objective absorb
      the constant and recovery restores ``x_j = v``).

    One nonnegative slack is appended for each inequality row of the extended
    row set (+1 for 'L': ``a^T x + s = b``; -1 for 'G': ``a^T x - s = b``),
    and 'E' rows are kept as equalities.  Bound types outside this list are
    rejected with a clear error.

    The objective sense defaults to MINIMIZATION (``maximize=False``) and is
    never inferred from the objective row name; pass ``maximize=True``
    explicitly to maximize ``c_orig @ x``.
    """
    m, n = lp.A.shape
    if len(lp.row_types) != m:
        raise MehrotraError("row_types length does not match A")
    if not np.all(np.isfinite(lp.A)):
        raise MehrotraError("A contains non-finite entries")
    if not np.all(np.isfinite(lp.b)):
        raise MehrotraError("b contains non-finite entries")
    if not np.all(np.isfinite(lp.c)):
        raise MehrotraError("c contains non-finite entries")

    bad_types = sorted({t for t in lp.row_types if t not in ("E", "L", "G")})
    if bad_types:
        raise MehrotraError(
            f"row type(s) {bad_types} unsupported; this converter handles E/L/G models only"
        )

    # ----- bound classification: x_j = offset_j + sum_k sign_k * x'_k -------
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    if lb.shape != (n,) or ub.shape != (n,):
        raise MehrotraError("bound arrays do not match A")

    is_free = np.isneginf(lb) & np.isposinf(ub)                # FR: split
    is_reflected = np.isneginf(lb) & np.isfinite(ub)           # UP: reflect
    is_shifted = np.isfinite(lb) & np.isposinf(ub)             # [0,inf)/LO: shift
    is_fixed = np.isfinite(lb) & np.isfinite(ub) & (lb == ub)  # FX: substitute
    is_boxed = np.isfinite(lb) & np.isfinite(ub) & (lb != ub)  # box: shift + row
    supported = is_free | is_reflected | is_shifted | is_fixed | is_boxed
    if not supported.all():
        j = int(np.flatnonzero(~supported)[0])
        raise MehrotraError(
            f"variable {lp.var_names[j]!r} has unsupported bounds "
            f"[{lb[j]!r}, {ub[j]!r}]; supported bound types: LO (x >= L), "
            "UP with a free lower bound (x <= U), FR (free), box [L, U], "
            "and FX (fixed)"
        )

    # Per-original-variable constant: the lower bound for LO/box columns, the
    # fixed value for FX (recovered directly), the upper bound for reflected
    # UP columns, and 0 for identity and free splits.
    orig_offset = np.where(np.isfinite(lb), lb, np.where(np.isfinite(ub), ub, 0.0))

    single_idx = np.flatnonzero(~is_free & ~is_fixed)   # one column each
    free_idx = np.flatnonzero(is_free)                  # two columns each
    box_idx = np.flatnonzero(is_boxed)                  # appended rows below
    n_extra = int(box_idx.size)                         # appended 'L' rows

    single_sign = np.where(is_reflected[single_idx], -1.0, 1.0)
    block_to_orig = np.concatenate([single_idx, free_idx, free_idx]).astype(np.intp)
    block_sign = np.concatenate(
        [single_sign, np.ones(free_idx.size), -np.ones(free_idx.size)]
    )
    n_block = int(block_to_orig.size)

    # Transformed constraint columns: FR splits duplicate their original
    # column with +/- signs; reflected UP columns negate it.
    A_block = lp.A[:, block_to_orig] * block_sign[None, :]
    # The RHS absorbs every constant term of the transformation (LO shifts,
    # UP reflections, and the fixed values of substituted FX columns).
    b_std = lp.b.astype(np.float64) - lp.A @ orig_offset

    # Extended row set: the original rows plus one 'L' row per box variable.
    std_row_types = tuple(lp.row_types) + ("L",) * n_extra
    m_std = m + n_extra
    std_row_names = tuple(lp.row_names) + tuple(
        f"{lp.var_names[j]}_upper" for j in box_idx
    )

    # One nonnegative slack per inequality row of the extended row set:
    # +1 column for 'L' rows (a^T x + s = b), -1 column for 'G' rows
    # (a^T x - s = b).  Without box variables this is identical to the
    # previous slack construction.
    slack_rows = tuple(i for i, t in enumerate(std_row_types) if t in ("L", "G"))
    num_slacks = len(slack_rows)

    if num_slacks:
        slack_cols = np.eye(m_std)[:, list(slack_rows)]
        signs = np.array([1.0 if std_row_types[i] == "L" else -1.0
                          for i in slack_rows])
        slack_cols = slack_cols * signs
        A_std = np.hstack([A_block, slack_cols[:m]])
    else:
        A_std = A_block

    if n_extra:
        # Box upper-bound rows: shifted column + slack = U - L.  The box rows
        # sit at the end of the extended row list, so their slack columns are
        # the last n_extra slack columns, in the same order.
        col_of = np.full(n, -1, dtype=np.intp)
        col_of[single_idx] = np.arange(single_idx.size)
        A_bottom = np.zeros((n_extra, n_block + num_slacks))
        A_bottom[np.arange(n_extra), col_of[box_idx]] = 1.0
        A_bottom[np.arange(n_extra),
                 n_block + num_slacks - n_extra + np.arange(n_extra)] = 1.0
        A_std = np.vstack([A_std, A_bottom])
        b_std = np.concatenate(
            [b_std, (ub[box_idx] - lb[box_idx]).astype(np.float64)]
        )

    zero_rows = np.flatnonzero(np.abs(A_std).sum(axis=1) == 0.0)
    if zero_rows.size:
        name = std_row_names[int(zero_rows[0])]
        raise MehrotraError(
            f"row {name!r} is all zero in standard form; the Newton system would be singular"
        )
    if np.linalg.matrix_rank(A_std) < m_std:
        raise MehrotraError(
            "standard-form constraint matrix is rank deficient; "
            "the Schur complement A H^-1 A^T would be singular"
        )

    c_orig = lp.c.astype(np.float64).copy()
    base = -c_orig if maximize else c_orig
    c_block = block_sign * base[block_to_orig]
    c_min = np.concatenate([c_block, np.zeros(num_slacks)])
    slack_names = tuple(f"{std_row_names[i]}_slack" for i in slack_rows)

    return StandardFormLP(
        A=A_std,
        b=b_std,
        c_min=c_min,
        c_orig=c_orig,
        maximize=bool(maximize),
        n_orig=n,
        n_block=n_block,
        block_to_orig=block_to_orig,
        block_sign=block_sign,
        orig_offset=orig_offset,
        num_slacks=num_slacks,
        slack_row_indices=slack_rows,
        var_names=tuple(lp.var_names),
        slack_names=slack_names,
        row_names=std_row_names,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class MehrotraResult:
    """Solution and diagnostics of one Mehrotra solve.

    Conventions: ``x_standard``, ``y`` and ``z_standard`` refer to the solved
    MINIMIZATION standard form (``A^T y + z = c_min``); ``objective`` is the
    ORIGINAL-sense objective ``c_orig @ x`` (maximized value when the model
    maximizes).  ``x`` holds the original variables only (slacks dropped).
    The iterations run in row/column-equilibrated coordinates
    (``scaling.scale_lp``); every field is mapped back to ORIGINAL problem
    units before being reported.  Residual fields are absolute infinity-norms
    of the original standard form and ``rel_*`` their relative forms.
    """

    status: str                 # "optimal" | "numerical_tail" | "max_iterations" | "stalled" | "numerical_failure"
    message: str
    objective: float
    x: np.ndarray               # original variables (n_orig,)
    x_standard: np.ndarray      # full standard-form primal (n_block + num_slacks,)
    y: np.ndarray               # equality duals (m,) for the min standard form
    z_standard: np.ndarray      # dual slacks (n_orig + num_slacks,)
    primal_residual: float      # ||A x - b||_inf (absolute)
    dual_residual: float        # ||A^T y + z - c||_inf (absolute)
    rel_primal: float           # primal_residual / (1 + ||b||_inf)
    rel_dual: float             # dual_residual / (1 + ||c||_inf)
    complementarity: float      # mu = x @ z / n
    rel_gap: float              # |c x - b y| / (1 + |c x| + |b y|)
    iterations: int
    regularized_iterations: tuple[int, ...]
    history: tuple[dict, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _inf_norm(v: np.ndarray) -> float:
    return float(np.max(np.abs(v))) if v.size else 0.0


def _shift_positive(v: np.ndarray) -> np.ndarray:
    """Mehrotra's positivity shift: v -> v + d1 + d2 with d1, d2 >= 0 chosen so
    the result is strictly positive and scale-consistent with v."""
    d1 = max(0.0, -1.5 * float(np.min(v)))
    v2 = v + d1
    s = float(np.sum(v2))
    if np.isfinite(s) and s > 1e-300:
        d2 = 0.5 * float(v2 @ v2) / s
    else:
        d2 = 1.0  # v2 is (numerically) the zero vector: fall back to a unit start
    return v2 + d2


def _mehrotra_initial_point(A: np.ndarray, b: np.ndarray, c: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mehrotra's starting heuristic, strictly positive in x and z.

    x0: minimum-norm solution of A x = b (least squares), shifted positive.
    y0: least-squares solution of A^T y ~= c; z0 = c - A^T y, shifted positive.
    """
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    y = np.linalg.lstsq(A.T, c, rcond=None)[0]
    z = c - A.T @ y
    return _shift_positive(x), y, _shift_positive(z)


def _max_step(v: np.ndarray, dv: np.ndarray) -> float:
    """Largest alpha >= 0 with v + alpha * dv >= 0, given v > 0 (may be inf)."""
    neg = dv < 0.0
    if not np.any(neg):
        return np.inf
    # dv[neg] < 0 strictly and v[neg] > 0, so the ratios are positive and finite.
    return float(np.min(-v[neg] / dv[neg]))


def _schur_needs_regularization(A: np.ndarray, H: np.ndarray) -> bool:
    """True when S = A H^-1 A^T is not Cholesky-factorizable without regularization.

    ``linear_system`` always applies (and escalates, if needed) its own
    regularization but cannot report it, and it must not be modified; H is
    diagonal positive definite by construction, so this cheap pre-check on the
    exact Schur complement is the honest way to detect when the backend's
    escalating regularization will actually engage beyond its default jitter.
    """
    W = A.T / H[:, None]          # H^{-1} A^T (H is diagonal and positive)
    S = A @ W
    try:
        np.linalg.cholesky(0.5 * (S + S.T))
        return False
    except np.linalg.LinAlgError:
        return True


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve_standard_form(sf: StandardFormLP, *, tol: float = 1e-8, max_iter: int = 100,
                        tau: float = 0.995, mu_floor: float = 1e-12,
                        verbose: bool = False) -> MehrotraResult:
    """Solve a ``StandardFormLP`` with Mehrotra's predictor-corrector method.

    The iterations run in row/column-equilibrated coordinates produced by
    ``scaling.scale_lp`` (``A_s = R A S``, ``b_s = R b``, ``c_s = S c``):
    directions, steps, the convergence test, and the ``mu_floor`` scale all
    refer to that space, and every ``MehrotraResult`` field is mapped back
    to original units.

    Parameters
    ----------
    tol : relative termination tolerance on primal/dual residuals and gap.
    max_iter : maximum number of Newton iterations.
    tau : fraction-to-boundary parameter (step = tau * max feasible step).
    mu_floor : scale-aware complementarity floor (relative multiplier).  The
        effective floor is ``mu_floor * max(1, ||b||_inf, ||c||_inf)``; once
        mu = x^T z / n reaches it, the trajectory is stopped instead of being
        driven deeper into the rounding-noise-dominated degenerate tail.
    verbose : print one progress line per iteration.
    """
    A, b, c = sf.A, sf.b, sf.c_min
    m, n = A.shape
    if mu_floor <= 0.0:
        raise ValueError("mu_floor must be positive")

    # Row/column equilibration (existing ``scaling.scale_lp``): the Newton
    # iterations run entirely in the scaled coordinates
    #     A_s = R A S,  b_s = R b,  c_s = S c,  x = S x_hat,
    # which shrinks the dynamic range of H = diag(z/x) and of the Schur
    # complement on badly scaled models.  ``_finish`` maps the terminal
    # iterate back to original units (x = S x_hat, y = R y_hat, z = z_hat / S).
    scaled = scale_lp(A, b, c, np.zeros(n), np.full(n, np.inf))
    row_scale, col_scale = scaled.row_scale, scaled.column_scale
    A, b, c = scaled.A, scaled.b, scaled.c
    norm_b = _inf_norm(b)
    norm_c = _inf_norm(c)
    mu_floor_eff = mu_floor * max(1.0, norm_b, norm_c)

    x, y, z = _mehrotra_initial_point(A, b, c)

    history: list[dict] = []
    regularized_iters: list[int] = []
    best_rel_gap = float("inf")
    best_x: Optional[np.ndarray] = None
    best_y: Optional[np.ndarray] = None
    best_z: Optional[np.ndarray] = None
    best_k: Optional[int] = None
    k = 0
    mu = float("nan")
    rel_p = rel_d = rel_gap = float("nan")
    primal_abs = dual_abs = float("nan")
    status = "max_iterations"
    message = f"iteration limit ({max_iter}) reached without convergence"

    def _finish(st: str, msg: str) -> MehrotraResult:
        # For non-optimal exits, report the best trusted iterate observed before
        # the numerical tail rather than the final noise-dominated iterate.
        nonlocal x, y, z
        if st != "optimal" and best_x is not None:
            x = best_x.copy()
            y = best_y.copy()
            z = best_z.copy()
            if best_k is not None:
                msg = f"{msg}; returning best trusted iterate from k={best_k}"

        # Map the equilibrated terminal iterate back to original units:
        # x = S x_hat, y = R y_hat, z = z_hat / S (from A_s = R A S,
        # b_s = R b, c_s = S c), then recompute every reported metric on the
        # original standard form so result fields carry original units.
        x_std = unscale_solution(x, col_scale)
        y_orig = row_scale * y
        z_std = z / col_scale
        primal_abs_o = _inf_norm(sf.A @ x_std - sf.b)
        dual_abs_o = _inf_norm(sf.A.T @ y_orig + z_std - sf.c_min)
        mu_o = float(x_std @ z_std) / n
        cx = float(sf.c_min @ x_std)
        by = float(sf.b @ y_orig)
        rel_gap_o = abs(cx - by) / (1.0 + abs(cx) + abs(by))
        x_orig = sf.recover_original(x_std)
        return MehrotraResult(
            status=st,
            message=msg,
            objective=float(sf.c_orig @ x_orig),
            x=x_orig.copy(),
            x_standard=x_std.copy(),
            y=y_orig.copy(),
            z_standard=z_std.copy(),
            primal_residual=primal_abs_o,
            dual_residual=dual_abs_o,
            rel_primal=primal_abs_o / (1.0 + _inf_norm(sf.b)),
            rel_dual=dual_abs_o / (1.0 + _inf_norm(sf.c_min)),
            complementarity=mu_o,
            rel_gap=rel_gap_o,
            iterations=k,
            regularized_iterations=tuple(regularized_iters),
            history=tuple(history),
        )

    for k in range(max_iter + 1):
        # ----- residuals and convergence test -----------------------------
        r_p = A @ x - b
        r_d = A.T @ y + z - c
        mu = float(x @ z) / n
        primal_abs = _inf_norm(r_p)
        dual_abs = _inf_norm(r_d)
        rel_p = primal_abs / (1.0 + norm_b)
        rel_d = dual_abs / (1.0 + norm_c)
        cx = float(c @ x)
        by = float(b @ y)
        rel_gap = abs(cx - by) / (1.0 + abs(cx) + abs(by))

        history.append({
            "iter": k, "mu": mu, "primal": rel_p, "dual": rel_d, "rel_gap": rel_gap,
            "alpha_p": None, "alpha_d": None, "sigma": None, "regularized": False,
        })
        # Trust the gap measure only while complementarity is above the floor;
        # inside the tail it is corrupted by dual-side rounding noise.
        if mu > mu_floor_eff and rel_gap < best_rel_gap:
            best_rel_gap = rel_gap
            best_x = x.copy()
            best_y = y.copy()
            best_z = z.copy()
            best_k = k

        if not (np.isfinite(mu) and mu > 0.0):
            return _finish("numerical_failure",
                           f"complementarity mu={mu!r} collapsed at iteration {k}")
        if rel_p <= tol and rel_d <= tol and rel_gap <= tol:
            status = "optimal"
            message = f"converged in {k} iterations (rel_p={rel_p:.3e}, rel_d={rel_d:.3e}, rel_gap={rel_gap:.3e})"
            break
        if k == max_iter:
            status = "max_iterations"
            break

        # ----- degenerate-tail safeguard -----------------------------------
        # mu carries the units of the objective and the products x_i*z_i carry
        # absolute rounding noise proportional to the data scale, so once mu
        # is at/below the floor the barrier trajectory cannot be resolved in
        # double precision: H = diag(z/x) spans extreme dynamic range and
        # Newton directions become rounding-error dominated (observed as a
        # *growing* dual residual while mu collapses).  Stop before taking
        # another step instead of driving the trajectory indefinitely.
        if mu <= mu_floor_eff:
            primal_ok = rel_p <= tol
            gap_ok = best_rel_gap <= max(tol, 1e3 * np.finfo(float).eps)
            if primal_ok and gap_ok:
                status = "numerical_tail"
                message = (
                    f"complementarity mu={mu:.3e} reached the scale-aware floor "
                    f"{mu_floor_eff:.3e} at iteration {k}: primal feasibility is "
                    f"within tolerance (rel_p={rel_p:.3e}) and the best relative "
                    f"gap seen was {best_rel_gap:.3e}, but the dual residual "
                    f"(rel_d={rel_d:.3e}) is degraded by rounding noise in the "
                    f"degenerate tail; stopped instead of taking further "
                    f"noise-dominated Newton steps"
                )
            else:
                status = "stalled"
                message = (
                    f"complementarity mu={mu:.3e} reached the scale-aware floor "
                    f"{mu_floor_eff:.3e} at iteration {k} without practical "
                    f"accuracy (rel_p={rel_p:.3e}, best_rel_gap={best_rel_gap:.3e})"
                )
            break

        # ----- invariant sanity: x, z strictly positive and finite --------
        if (not np.all(np.isfinite(x))) or (not np.all(np.isfinite(y))) \
                or (not np.all(np.isfinite(z))) or float(np.min(x)) <= 0.0 \
                or float(np.min(z)) <= 0.0:
            return _finish("numerical_failure",
                           f"iterate lost positivity or finiteness at iteration {k}")

        h = z / x
        if (not np.all(np.isfinite(h))) or np.any(h <= 0.0):
            return _finish("numerical_failure",
                           f"barrier h = z/x is not positive finite at iteration {k}")

        regularized = _schur_needs_regularization(A, h)
        history[-1]["regularized"] = regularized
        if regularized:
            regularized_iters.append(k)

        H_matrix = np.diag(h)
        try:
            fac = factor_reduced_system(H_matrix, A)
        except LinearSystemError as exc:
            return _finish("numerical_failure",
                           f"Newton factorization failed at iteration {k}: {exc}")

        # ----- affine predictor -------------------------------------------
        rc = x * z
        rhs_x = r_d - rc / x
        try:
            dx_a, dy_a = solve_reduced_system(fac, rhs_x, -r_p)
        except LinearSystemError as exc:
            return _finish("numerical_failure",
                           f"predictor solve failed at iteration {k}: {exc}")
        dz_a = -(rc + z * dx_a) / x
        if not (np.all(np.isfinite(dx_a)) and np.all(np.isfinite(dy_a))
                and np.all(np.isfinite(dz_a))):
            return _finish("numerical_failure",
                           f"non-finite affine direction at iteration {k}")

        a_p_aff = min(1.0, _max_step(x, dx_a))
        a_d_aff = min(1.0, _max_step(z, dz_a))
        a_aff = min(a_p_aff, a_d_aff)
        mu_aff = float((x + a_aff * dx_a) @ (z + a_aff * dz_a)) / n
        sigma = min(1.0, max(0.0, (mu_aff / mu) ** 3))
        history[-1]["sigma"] = sigma

        # ----- Mehrotra corrector ------------------------------------------
        rc = x * z - sigma * mu + dx_a * dz_a
        rhs_x = r_d - rc / x
        try:
            dx, dy = solve_reduced_system(fac, rhs_x, -r_p)
        except LinearSystemError as exc:
            return _finish("numerical_failure",
                           f"corrector solve failed at iteration {k}: {exc}")
        dz = -(rc + z * dx) / x
        if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(dy))
                and np.all(np.isfinite(dz))):
            return _finish("numerical_failure",
                           f"non-finite corrector direction at iteration {k}")

        # ----- fraction-to-boundary step -----------------------------------
        a_p = min(1.0, tau * _max_step(x, dx))
        a_d = min(1.0, tau * _max_step(z, dz))
        history[-1]["alpha_p"] = a_p
        history[-1]["alpha_d"] = a_d

        if max(a_p, a_d) < 1e-11:
            return _finish("stalled",
                           f"step lengths collapsed at iteration {k} "
                           f"(alpha_p={a_p:.3e}, alpha_d={a_d:.3e})")

        x = x + a_p * dx
        y = y + a_d * dy
        z = z + a_d * dz

        if verbose:
            print(
                f"  it {k:3d}  mu={mu:9.3e}  rel_p={rel_p:9.3e}  rel_d={rel_d:9.3e}  "
                f"rel_gap={rel_gap:9.3e}  ap={a_p:6.4f}  ad={a_d:6.4f}  "
                f"sigma={sigma:5.3f}  reg={'Y' if regularized else 'n'}"
            )

    if verbose:
        print(f"  -> {status}: {message}")
    return _finish(status, message)


def solve_lp(lp: NumericalLP, *, maximize: bool = False, **kwargs) -> MehrotraResult:
    """Convert ``lp`` to standard form and solve it in one call."""
    return solve_standard_form(to_standard_form(lp, maximize=maximize), **kwargs)


# ---------------------------------------------------------------------------
# Test harness (NOT executed as part of this task; run `python src/lp/mehrotra.py`)
# ---------------------------------------------------------------------------
def _tiny_min_lp() -> NumericalLP:
    """min -x1 - x2 s.t. x1+x2+x3 = 10 (E), x1+2x2 <= 8 (L), 3x1+x2 <= 9 (L), x >= 0.

    Optimum: (x1, x2, x3) = (2, 3, 5), objective -5.
    """
    return NumericalLP(
        name="TINY_MIN",
        objective_name="COST",
        A=np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 0.0], [3.0, 1.0, 0.0]]),
        b=np.array([10.0, 8.0, 9.0]),
        c=np.array([-1.0, -1.0, 0.0]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.full(3, np.inf),
        row_types=("E", "L", "L"),
        var_names=("x1", "x2", "x3"),
        row_names=("E1", "L1", "L2"),
    )


def _tiny_max_lp() -> NumericalLP:
    """maximize x1 + 2 x2 s.t. x1 + x2 <= 4 (L), x2 = 1 (E), x >= 0.

    Optimum: (x1, x2) = (3, 1), objective +5 (exercises explicit maximize=True).
    """
    return NumericalLP(
        name="TINY_MAX",
        objective_name="MAXIM",
        A=np.array([[1.0, 1.0], [0.0, 1.0]]),
        b=np.array([4.0, 1.0]),
        c=np.array([1.0, 2.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        row_types=("L", "E"),
        var_names=("x1", "x2"),
        row_names=("L1", "E1"),
    )


def _run_case(title: str, lp: NumericalLP, expected_objective: Optional[float], tol: float,
              maximize: bool = False) -> bool:
    print(f"\n=== {title} ===")
    result = solve_lp(lp, tol=tol, verbose=True, maximize=maximize)
    print(f"  status            : {result.status} ({result.message})")
    print(f"  objective         : {result.objective:.10f}")
    if expected_objective is not None:
        print(f"  expected objective: {expected_objective:.10f}")
    print(f"  primal residual   : {result.primal_residual:.3e} (rel {result.rel_primal:.3e})")
    print(f"  dual residual     : {result.dual_residual:.3e} (rel {result.rel_dual:.3e})")
    print(f"  complementarity   : {result.complementarity:.3e}")
    print(f"  rel gap           : {result.rel_gap:.3e}")
    print(f"  iterations        : {result.iterations}")
    reg = list(result.regularized_iterations)
    print(f"  regularized iters : {reg if reg else 'none'}")

    ok = (
        result.status == "optimal"
        and result.rel_primal <= tol
        and result.rel_dual <= tol
        and result.rel_gap <= tol
        and bool(np.all(np.isfinite(result.x)))
    )
    if expected_objective is not None:
        obj_ok = abs(result.objective - expected_objective) <= 1e-6 * max(1.0, abs(expected_objective))
        ok = ok and obj_ok
        print(f"  objective check   : {'OK' if obj_ok else 'MISMATCH'}")
    print(f"  residual checks   : {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    tol = 1e-8
    ok = True
    ok &= _run_case("Tiny LP (minimization, E+L rows, known optimum -5)", _tiny_min_lp(), -5.0, tol)
    ok &= _run_case("Tiny LP (explicit maximize=True, known optimum +5)", _tiny_max_lp(), 5.0, tol,
                    maximize=True)

    afiro_path = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "afiro.mps"))
    afiro = load_numeric_mps(afiro_path)
    ok &= _run_case("AFIRO (minimization, known optimum -464.7531428571)", afiro,
                    -464.7531428571, tol)

    print("\nOverall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())