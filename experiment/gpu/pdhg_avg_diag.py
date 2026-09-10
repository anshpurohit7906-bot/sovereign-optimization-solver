"""Ergodic-averaging experiment for the mixed PDHG solver.

Tests the *hypothesis* that reporting the running arithmetic average of the
PDHG iterates improves the practical convergence metrics used by
``pdhg_mixed``.  The original PDHG update is reproduced unchanged
(identical tau/sigma, dual-first ordering, x-bar extrapolation,
initialization, and stopping definitions); the ONLY addition is running
averages of the primal and dual iterates::

    x_avg   = (1/k) sum_{i=1..k} x_i
    y_eq_avg= (1/k) sum_{i=1..k} y_eq_i
    y_ub_avg= (1/k) sum_{i=1..k} y_ub_i

``x_bar`` is NOT averaged.  This does not assume averaging fixes anything;
it measures last-iterate vs averaged-iterate diagnostics side by side.

Runs CPU only, on the same deterministic case-4 / case-5 synthetic problems
used by the GPU PDHG benchmark.

By design nothing here imports or modifies the production/reference files:
``pdhg_mixed.py``, ``pdhg_gpu.py``, ``benchmark_pdhg.py``, ``src/``,
``tests/`` and ``requirements.txt`` are left untouched.  The solver loop and
the LP generator are reproduced locally so this experiment is hermetically
self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

# Make ``numerical_model`` (in ``src/``) importable, mirroring the reference
# solver's ``from numerical_model import ...`` without touching ``src/``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"

# ---------------------------------------------------------------------------
# Sparse spectral norm estimate (copied verbatim from pdhg_mixed)
# ---------------------------------------------------------------------------
def _spectral_norm(
    A: np.ndarray, *, max_iter: int = 200, tol: float = 1e-12
) -> float:
    """Estimate ``||A||_2`` using a deterministic power iteration."""
    if A.size == 0:
        return 0.0

    v = np.ones(A.shape[1], dtype=np.float64)
    v /= np.linalg.norm(v)
    previous = 0.0
    for _ in range(max_iter):
        w = A.T @ (A @ v)
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            return 0.0
        v = w / norm_w
        estimate = float(np.linalg.norm(A @ v))
        if abs(estimate - previous) <= tol * max(1.0, estimate):
            return estimate
        previous = estimate
    return previous


def _project_box(
    x: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Project onto finite or infinite componentwise variable bounds."""
    return np.minimum(np.maximum(x, lower), upper)


def _partition_rows(lp: NumericalLP) -> tuple[np.ndarray, np.ndarray]:
    """Return indices for equality and <= rows (E and L only)."""
    row_types = np.asarray(lp.row_types)
    unsupported = set(row_types).difference({"E", "L"})
    if unsupported:
        raise ValueError(
            "mixed PDHG supports only E and L rows; found "
            f"{sorted(unsupported)}"
        )
    return np.flatnonzero(row_types == "E"), np.flatnonzero(row_types == "L")

for _p in (_SRC, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from numerical_model import NumericalLP, validate_numeric_lp  # noqa: E402

# ---------------------------------------------------------------------------
# Diagnostics (copied verbatim from pdhg_mixed)
# ---------------------------------------------------------------------------
def _diagnostics(
    c: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    x: np.ndarray,
    y_eq: np.ndarray,
    y_ub: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute primal, stationarity, and complementarity KKT residuals."""
    equality_residual = (
        float(np.linalg.norm(A_eq @ x - b_eq, ord=np.inf)) if A_eq.shape[0] else 0.0
    )
    inequality_slack = A_ub @ x - b_ub
    inequality_violation = (
        float(np.linalg.norm(np.maximum(inequality_slack, 0.0), ord=np.inf))
        if A_ub.shape[0]
        else 0.0
    )

    # This projected-gradient residual is zero iff stationarity holds with
    # the box normal cone, including finite, free, and one-sided bounds.
    lagrangian_gradient = c + A_eq.T @ y_eq + A_ub.T @ y_ub
    dual_feasibility = float(
        np.linalg.norm(x - _project_box(x - lagrangian_gradient, lower, upper), ord=np.inf)
    )
    complementarity = (
        float(np.linalg.norm(y_ub * inequality_slack, ord=np.inf))
        if A_ub.shape[0]
        else 0.0
    )
    return equality_residual, inequality_violation, dual_feasibility, complementarity


def _objective(c: np.ndarray, x: np.ndarray) -> float:
    return float(c @ x)


# ---------------------------------------------------------------------------
# Synthetic problem generation (reproduces the benchmark's case-4 / case-5
# problems deterministically: same RNG seed + case index, same dimensions,
# same density, same construction).
# ---------------------------------------------------------------------------
RNG_SEED = 1234

# (rows, cols, density) - must match the benchmark CASES table for 4 & 5.
CASES: list[tuple[int, int, float]] = [
    (2_000, 1_000, 0.02),   # case 4
    (4_000, 2_000, 0.015),  # case 5
]

# Checkpoints at which last-iterate and averaged-iterate diagnostics are
# reported.
CHECKPOINTS = [1_000, 2_500, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000]

MAX_ITER = 50_000
TOL = 1e-7
CHECK_EVERY = 250


def _generate_lp(
    rows: int, cols: int, density: float, rng: np.random.Generator, name: str
) -> NumericalLP:
    """Build the deterministic feasible sparse E/L LP (mirrors benchmark)."""
    nnz = max(1, int(density * rows * cols))
    n_draws = int(nnz * 1.5) + 1
    row_idx = rng.integers(0, rows, size=n_draws)
    col_idx = rng.integers(0, cols, size=n_draws)
    data = rng.standard_normal(n_draws).astype(np.float64)

    A_csr = sp.coo_matrix((data, (row_idx, col_idx)), shape=(rows, cols)).tocsr()
    A = A_csr.toarray()

    n_eq = max(1, rows // 3)
    n_ub = rows - n_eq
    row_types = ("E",) * n_eq + ("L",) * n_ub

    x_feas = rng.uniform(0.5, 1.5, size=cols).astype(np.float64)
    A_eq = A[:n_eq]
    A_ub = A[n_eq:]
    b_eq = A_eq @ x_feas
    b_ub = A_ub @ x_feas + rng.uniform(0.5, 2.0, size=n_ub).astype(np.float64)
    b = np.concatenate([b_eq, b_ub])

    c = rng.standard_normal(cols).astype(np.float64)
    lower = np.full(cols, -1.0)
    upper = np.full(cols, 2.5)

    return NumericalLP(
        name=name,
        objective_name="bench",
        A=A,
        b=b,
        c=c,
        lower_bounds=lower,
        upper_bounds=upper,
        row_types=row_types,
        var_names=tuple(f"x{i}" for i in range(cols)),
        row_names=tuple(f"r{i}" for i in range(rows)),
    )


# ---------------------------------------------------------------------------
# Averages of iterates
# ---------------------------------------------------------------------------
class _RunningAverages:
    """Track running sums of primal/dual iterates (NOT x_bar)."""

    __slots__ = ("n", "x_sum", "y_eq_sum", "y_ub_sum", "count")

    def __init__(self, n: int, n_eq: int, n_ub: int) -> None:
        self.n = n
        self.x_sum = np.zeros(n, dtype=np.float64)
        self.y_eq_sum = np.zeros(n_eq, dtype=np.float64)
        self.y_ub_sum = np.zeros(n_ub, dtype=np.float64)
        self.count = 0

    def update(self, x, y_eq, y_ub) -> None:
        self.x_sum += x
        self.y_eq_sum += y_eq
        self.y_ub_sum += y_ub
        self.count += 1

    def averages(self):
        k = self.count
        return (
            self.x_sum / k,
            self.y_eq_sum / k,
            self.y_ub_sum / k,
        )


# ---------------------------------------------------------------------------
# The experiment: identical PDHG loop + running averages
# ---------------------------------------------------------------------------
def run_with_averaging(
    lp: NumericalLP,
    *,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
    check_every: int = CHECK_EVERY,
) -> dict:
    """Run vanilla PDHG (unchanged from pdhg_mixed) plus ergodic averages.

    Returns, per checkpoint, both last-iterate and averaged-iterate
    diagnostics plus the averaged objective.
    """
    validate_numeric_lp(lp)
    eq_rows, ub_rows = _partition_rows(lp)
    A_eq, b_eq = lp.A[eq_rows], lp.b[eq_rows]
    A_ub, b_ub = lp.A[ub_rows], lp.b[ub_rows]
    lower, upper = lp.lower_bounds, lp.upper_bounds

    norm_A = _spectral_norm(lp.A)
    tau = sigma = 1.0 if norm_A == 0.0 else 0.9 / norm_A

    x = _project_box(np.zeros(lp.num_vars, dtype=np.float64), lower, upper)
    x_bar = x.copy()
    y_eq = np.zeros(A_eq.shape[0], dtype=np.float64)
    y_ub = np.zeros(A_ub.shape[0], dtype=np.float64)

    avg = _RunningAverages(lp.num_vars, A_eq.shape[0], A_ub.shape[0])

    checkpoints = set(CHECKPOINTS)
    checkpoint_rows: dict[int, dict] = {
        cp: {"last": None, "avg": None}
        for cp in CHECKPOINTS
    }
    converged_at = None

    diagnostics = (float("inf"),) * 4
    for iteration in range(1, max_iter + 1):
        # --- UNCHANGED updates from pdhg_mixed ---
        y_eq = y_eq + sigma * (A_eq @ x_bar - b_eq)
        y_ub = np.maximum(y_ub + sigma * (A_ub @ x_bar - b_ub), 0.0)

        previous_x = x
        gradient = lp.c + A_eq.T @ y_eq + A_ub.T @ y_ub
        x = _project_box(x - tau * gradient, lower, upper)
        x_bar = 2.0 * x - previous_x

        # --- ONLY addition: running average of the *fresh* iterates ---
        avg.update(x, y_eq, y_ub)

        if iteration % check_every == 0 or iteration == max_iter:
            diagnostics = _diagnostics(
                lp.c, A_eq, b_eq, A_ub, b_ub, lower, upper, x, y_eq, y_ub
            )
            equality, inequality, dual, complementarity = diagnostics
            if iteration in checkpoints:
                x_avg, y_eq_avg, y_ub_avg = avg.averages()
                checkpoint_rows[iteration]["last"] = diagnostics
                checkpoint_rows[iteration]["avg"] = _diagnostics(
                    lp.c, A_eq, b_eq, A_ub, b_ub,
                    lower, upper, x_avg, y_eq_avg, y_ub_avg,
                )
                checkpoint_rows[iteration]["avg_objective"] = _objective(lp.c, x_avg)
                checkpoint_rows[iteration]["last_objective"] = _objective(lp.c, x)
            if max(*diagnostics) <= tol and converged_at is None:
                converged_at = iteration

    final_last_diag = diagnostics

    # Averaged diagnostics at the final iteration.
    x_avg_final, y_eq_avg_final, y_ub_avg_final = avg.averages()
    final_avg_diag = _diagnostics(
        lp.c, A_eq, b_eq, A_ub, b_ub,
        lower, upper, x_avg_final, y_eq_avg_final, y_ub_avg_final,
    )
    final_avg_objective = _objective(lp.c, x_avg_final)
    final_last_objective = _objective(lp.c, x)

    return {
        "nnz": int(sp.csr_matrix(lp.A).nnz),
        "tau": tau,
        "sigma": sigma,
        "converged_last_at": converged_at,
        "checkpoints": checkpoint_rows,
        "final_last_diag": final_last_diag,
        "final_avg_diag": final_avg_diag,
        "final_avg_objective": final_avg_objective,
        "final_last_objective": final_last_objective,
    }

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_DIAG_NAMES = ("eq_res", "ineq_viol", "dual_feas", "compl")


def _fmt_diag(diag) -> str:
    if diag is None:
        return " " * 52
    parts = [f"{n}={v:.3e}" for n, v in zip(_DIAG_NAMES, diag)]
    return "  ".join(parts)[:52]


def _fmt_obj(v: Optional[float]) -> str:
    return "  --" if v is None else f"{v: .6e}"


def _print_case(idx: int, rows: int, cols: int, res: dict) -> None:
    print("=" * 112)
    print(f"  CASE {idx}: {rows} x {cols}  nnz={res['nnz']}  "
          f"tau=sigma={res['tau']:.4e}")
    print("=" * 112)
    header_last = "last-iterate                          "
    header_avg = "averaged-iterate                      "
    print(f"  {'iter':>7} | {header_last}| {header_avg}")
    print("  " + "-" * 104)

    for cp in CHECKPOINTS:
        row = res["checkpoints"][cp]
        last = row["last"]
        avg = row["avg"]
        last_obj = row.get("last_objective")
        avg_obj = row.get("avg_objective")
        print(f"  {cp:>7} | {_fmt_diag(last):48s} obj={_fmt_obj(last_obj):>12} "
              f"| {_fmt_diag(avg):48s} obj={_fmt_obj(avg_obj):>12}")
    print("  " + "-" * 104)
    print(f"  FINAL  | last : {_fmt_diag(res['final_last_diag'])}  "
          f"obj={_fmt_obj(res['final_last_objective'])}")
    print(f"  (50000) | avg  : {_fmt_diag(res['final_avg_diag'])}  "
          f"obj={_fmt_obj(res['final_avg_objective'])}")
    last_max = max(res["final_last_diag"])
    avg_max = max(res["final_avg_diag"])
    print(f"  max last KKT = {last_max:.3e}   max avg KKT = {avg_max:.3e}")
    print(f"  last-iterate converged (max<=1e-7)? {last_max <= TOL}")
    print(f"  averaged-iterate converged (max<=1e-7)? {avg_max <= TOL}")
    print()
def main(argv: Optional[list[str]] = None) -> None:
    _ = argv
    print("Ergodic-averaging experiment (CPU only) - pdhg_avg_diag.py")
    print(f"max_iter={MAX_ITER}  tol={TOL:.0e}  checkpoints={CHECKPOINTS}")
    print()

    for idx, (rows, cols, density) in enumerate(CASES, start=4):
        rng = np.random.default_rng(RNG_SEED + idx)
        lp = _generate_lp(rows, cols, density, rng, f"case{idx}")
        res = run_with_averaging(
            lp, max_iter=MAX_ITER, tol=TOL, check_every=CHECK_EVERY
        )
        _print_case(idx, rows, cols, res)


if __name__ == "__main__":
    main()
