"""Indigenous convex QP skeleton -- Mehrotra-style primal-dual interior point.

This is the **v1 vertical slice** for quadratic programming, deliberately
minimal but complete: standard-form conversion + a primal-dual predictor-
corrector IPM that mirrors the repo's Mehrotra LP culture (numerical honesty,
statuses, residuals recomputed before the claim).

Scope (v1, documented, not hidden):

* Convex PSD ``Q`` (checked); dense arrays only.
* Original model: ``min 1/2 x'Qx + c'x`` with E/L/G rows and finite-or-0 lower
  bounds.  Finite upper bounds are converted to internal ``L`` rows, so
  bounded variables work.  Free variables (``lb = -inf``) are rejected loudly.
* ``objective`` is recomputed from ``x`` (never a solver-reported value).

Independent KKT verification (``verify_qp_kkt``) recomputes primal/dual
residuals and complementarity from the returned solution without trusting any
solver-reported value -- the same honesty contract as the LP certificates.

Extension points (owned by the QP engineer, in priority order):

1. ``tools/benchmark_qp.py`` -- harness vs an external QP oracle (e.g.
   ``highspy``) on small convex Maros-Meszaros (QPS) instances.
2. Sparse ``Q``/``A`` and reuse of ``lp/linear_system.py`` backends for the
   KKT solve instead of the dense block assembly used here.
3. MIQP / indefinite-QP handling as later phases.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class QpError(ValueError):
    pass


@dataclass
class NumericalQP:
    """Dense convex QP in natural coordinates.

    ``row_types[i]`` is ``'E'``, ``'L'`` or ``'G'`` for row ``i`` of ``A``.
    ``lower_bounds`` / ``upper_bounds`` are per-variable bounds; a finite
    upper bound is converted to an internal ``L`` row (v1).
    """

    name: str
    Q: np.ndarray               # (n, n) symmetric positive semidefinite
    c: np.ndarray               # (n,)
    A: np.ndarray               # (m, n)
    b: np.ndarray               # (m,)
    row_types: tuple[str, ...]  # 'E'/'L'/'G'
    lower_bounds: np.ndarray    # finite (or default 0)
    upper_bounds: np.ndarray    # +inf or finite

    @property
    def num_vars(self) -> int:
        return self.c.shape[0]

    @property
    def num_constraints(self) -> int:
        return self.b.shape[0]


@dataclass
class StandardQP:
    """min 1/2 x'Qx + c'x s.t. A x = b, x >= 0 (all rows equality)."""

    Q: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    slack_cols: np.ndarray      # (>0) for each std column that is a slack
    n_orig: int
    lb_orig: np.ndarray         # maps the first n_orig standard vars back
    obj_const: float            # constant contributed by the x = x' + lb shift


def to_standard_qp(model: NumericalQP) -> StandardQP:
    """Convert an E/L/G + bounds QP to equality/nonnegativity standard form.

    Column layout of the standard form:

        [0, n)                      shifted original variables x' = x - lb
        [n, n + n_row_slacks)       one slack per non-E row (L: +1, G: -1)
        [n + n_row_slacks, n_std)   one slack per finite-UB variable
    """
    Q = np.asarray(model.Q, dtype=np.float64)
    c = np.asarray(model.c, dtype=np.float64)
    A = np.asarray(model.A, dtype=np.float64)
    b = np.asarray(model.b, dtype=np.float64)
    lb = np.asarray(model.lower_bounds, dtype=np.float64)
    ub = np.asarray(model.upper_bounds, dtype=np.float64)
    rt = tuple(model.row_types)

    n = model.num_vars
    m = model.num_constraints
    if Q.shape != (n, n):
        raise QpError(f"Q shape {Q.shape} != (n, n) = {(n, n)}")
    if c.shape != (n,):
        raise QpError(f"c shape {c.shape} != (n,) = {(n,)}")
    if A.shape != (m, n) or b.shape != (m,):
        raise QpError(f"A/b shape mismatch: A={A.shape}, b={b.shape}")
    if len(rt) != m:
        raise QpError("row_types length mismatch")
    if any(t not in ("E", "L", "G") for t in rt):
        raise QpError(f"unsupported row type(s): {rt}")
    if lb.shape != (n,) or ub.shape != (n,):
        raise QpError("bound shape mismatch")
    if np.any(lb == -np.inf):
        raise QpError("v1 QP requires finite lower bounds (free variables unsupported)")
    if np.any(lb > ub):
        raise QpError("at least one lower bound exceeds its upper bound")

    n_row_slacks = sum(1 for t in rt if t != "E")
    n_ub_slacks = int(np.count_nonzero(np.isfinite(ub)))
    n_std = n + n_row_slacks + n_ub_slacks

    A_std = np.zeros((m + n_ub_slacks, n_std))
    b_std = np.zeros(m + n_ub_slacks)
    slack_cols = np.zeros(n_std, dtype=bool)
    row_slack_col = np.zeros(m, dtype=np.int64)

    row_slacks_used = 0
    for i in range(m):
        t = rt[i]
        a = np.zeros(n)
        a[:] = A[i]
        A_std[i, :n] = a
        if t == "E":
            b_std[i] = b[i] - float(a @ lb)
        elif t == "L":
            col = n + row_slacks_used
            A_std[i, col] = 1.0
            b_std[i] = b[i] - float(a @ lb)
            slack_cols[col] = True
            row_slack_col[i] = col
            row_slacks_used += 1
        else:  # G : a x >= b  =>  -a x' - slack = a lb - b
            A_std[i, :n] = -a
            col = n + row_slacks_used
            A_std[i, col] = -1.0
            b_std[i] = float(a @ lb) - b[i]
            slack_cols[col] = True
            row_slack_col[i] = col
            row_slacks_used += 1

    # Upper-bound rows: x_j <= ub_j  =>  x'_j + slack = ub_j - lb_j.
    ub_slacks_used = 0
    for j in range(n):
        if np.isfinite(ub[j]):
            r = m + ub_slacks_used
            A_std[r, j] = 1.0
            col = n + n_row_slacks + ub_slacks_used
            A_std[r, col] = 1.0
            b_std[r] = ub[j] - lb[j]
            slack_cols[col] = True
            ub_slacks_used += 1

    # Objective in shifted coordinates: c' = c + Q lb ; const = lb'Qlb/2 + c lb.
    c_std = np.zeros(n_std)
    c_std[:n] = c + Q @ lb
    obj_const = 0.5 * float(lb @ Q @ lb) + float(c @ lb)
    Q_std = np.zeros((n_std, n_std))
    Q_std[:n, :n] = Q

    return StandardQP(
        Q=Q_std, c=c_std, A=A_std, b=b_std, slack_cols=slack_cols,
        n_orig=n, lb_orig=lb.copy(), obj_const=obj_const,
    )


@dataclass
class QpResult:
    """Solution and diagnostics of one convex-QP solve.

    ``x`` is the ORIGINAL variables; ``objective`` is the ORIGINAL objective
    ``1/2 x'Qx + c'x`` (recomputed, never a solver-reported value).
    ``x_standard`` is the full standard-form primal (shifted variables +
    slacks) used to reconstruct the solve.  Residual fields are absolute
    infinity-norms of the standard form and ``rel_*`` their relative forms,
    matching the LP core's conventions.
    """

    status: str                 # "optimal" | "max_iterations" | "numerical_failure" | "stalled"
    message: str
    objective: float
    x: np.ndarray               # original variables (n,)
    y: np.ndarray               # standard-form equality duals
    s: np.ndarray               # standard-form dual slacks (>= 0)
    x_standard: np.ndarray      # full standard-form primal
    primal_residual: float
    dual_residual: float
    complementarity: float
    rel_primal: float
    rel_dual: float
    rel_gap: float
    iterations: int
    obj_const: float            # constant from the lower-bound shift


def _max_step(d: np.ndarray, dd: np.ndarray) -> float:
    """Fraction-to-boundary step for a nonneg variable with direction dd."""
    neg = dd < 0.0
    if not np.any(neg):
        return float("inf")
    vals = -d[neg] / dd[neg]
    return float(np.min(vals))


def _active_set_polish(sq: StandardQP, x: np.ndarray,
                        x_tol: float = 1e-8, s_tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Exact active-set polish: solve the KKT system for the detected active set.

    Given a primal `x` near the solution, partition variables into:
      F = {i : x_i > x_tol}  (free/basic, should have s_i = 0)
      Z = {i : x_i <= x_tol} (active/nonbasic, should have s_i >= 0)

    The exact KKT system is:
      Qx + c - A'y - s = 0
      Ax = b
      x_i = 0  for i in Z
      s_i = 0  for i in F
      s_i >= 0 for i in Z

    Eliminate s_F = 0: Q_F x + c_F - A_F'y = 0 => A_F'y = (Qx + c)_F
    Solve for y in least-squares, then set s_F = 0 and s_Z = (Qx + c - A'y)_Z.
    This satisfies stationarity exactly, complementarity by construction, and
    primal feasibility by the input x. The only remaining check is dual feasibility
    (s_Z >= -s_tol).
    """
    Q, c, A = sq.Q, sq.c, sq.A
    m = A.shape[0]
    F = np.flatnonzero(x > x_tol)
    if F.size == 0:
        return np.zeros(m), np.maximum(Q @ x + c - A.T @ np.zeros(m), 0.0)
    AF = A[:, F]                        # (m, |F|)
    rhs = (Q @ x + c)[F]                # (|F|,)
    y, *_ = np.linalg.lstsq(AF.T, rhs, rcond=None)  # (m,)
    s = Q @ x + c - A.T @ y
    s[F] = 0.0
    return y, s


def solve_standard_qp(
    sq: StandardQP,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
    tau: float = 0.995,
    reg0: float = 1e-10,
    dual_recover: bool = True,
) -> QpResult:
    """Primal-dual predictor-corrector IPM for a convex equality QP.

    Reduced Newton system (mirrors ``lp/linear_system.py`` when D = s/x is
    diagonal, which it always is here):

        [ Q + D    -A^T ] [ dx ]   [ -r_x - s + (sigma*mu - dx*ds)/x ]
        [ A         0   ] [ dy ] = [ -r_b                          ]

    with ``D = diag(s/x)`` and a tiny diagonal regularization in the (1,1)
    block to stabilise degenerate / dependent-row systems.

    When ``dual_recover`` is true (default), the final iterate is finished by
    ``_recover_dual`` so degenerate exact-complementarity problems can be
    certified even though the raw barrier lets ``s -> 0``.
    """
    Q, c, A, b = sq.Q, sq.c, sq.A, sq.b
    m, n = A.shape
    QD = Q + reg0 * np.eye(n)
    bnorm = float(np.max(np.abs(b))) if b.size else 0.0
    cnorm = float(np.max(np.abs(c))) if c.size else 0.0

    x = np.ones(n)
    s = np.ones(n)
    y = np.zeros(m)

    def _fres(xv, yv, sv) -> tuple[float, float, float]:
        rc = Q @ xv + c - A.T @ yv - sv
        rb = A @ xv - b
        obj = 0.5 * float(xv @ Q @ xv) + float(c @ xv) + sq.obj_const
        rel_p = float(np.max(np.abs(rb))) / (1.0 + bnorm) if rb.size else 0.0
        rel_d = float(np.max(np.abs(rc))) / (1.0 + cnorm) if rc.size else 0.0
        rel_g = float(np.abs(sv @ xv)) / (1.0 + abs(obj) + abs(float(b @ yv)))
        return rel_p, rel_d, rel_g

    def _result(status, msg, it, xv, yv, sv):
        rel_p, rel_d, rel_g = _fres(xv, yv, sv)
        return QpResult(
            status=status, message=msg,
            objective=0.5 * float(xv @ Q @ xv) + float(c @ xv) + sq.obj_const,
            x=xv[: sq.n_orig] + sq.lb_orig, y=yv.copy(), s=sv.copy(),
            x_standard=xv.copy(),
            primal_residual=float(np.max(np.abs(A @ xv - b))) if A.shape[0] else 0.0,
            dual_residual=float(np.max(np.abs(Q @ xv + c - A.T @ yv - sv)))
            if Q.shape[0] else 0.0,
            complementarity=float(sv @ xv),
            rel_primal=rel_p, rel_dual=rel_d, rel_gap=rel_g,
            iterations=it + 1, obj_const=sq.obj_const,
        )

    stalled_cnt = 0
    for it in range(max_iter):
        rc = Q @ x + c - A.T @ y - s
        rb = A @ x - b
        rel_p, rel_d, rel_g = _fres(x, y, s)
        if rel_p <= tol and rel_d <= tol and rel_g <= tol:
            return _result("optimal", "converged to tolerance", it, x, y, s)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(s)) and np.all(np.isfinite(y))):
            return _result("numerical_failure", "non-finite iterate", it, x, y, s)

        mu = float(s @ x) / max(n, 1)
        D = s / x
        M = np.block([[QD + np.diag(D), -A.T], [A, np.zeros((m, m))]])

        # ---- predictor (sigma = 0, cross = 0) ----
        try:
            sol = np.linalg.solve(M, np.concatenate([-rc - s, -rb]))
        except np.linalg.LinAlgError:
            return _result("numerical_failure", "singular predictor KKT", it, x, y, s)
        dxp = sol[:n]
        dyp = sol[n:]
        # Predictor ds: S·dx + X·ds = -xs (sigma=0, cross=0)
        #   ds = -s - (s/x)·dx = -s - Σ·dx
        dsp = -s - D * dxp

        axp = min(1.0, tau * _max_step(x, dxp)) if np.any(dxp < 0) else 1.0
        asp = min(1.0, tau * _max_step(s, dsp)) if np.any(dsp < 0) else 1.0
        mu_aff = float((x + axp * dxp) @ (s + asp * dsp)) / max(n, 1)
        sig = (mu_aff / mu) ** 3 if (mu > 0 < mu_aff) else 0.25

        # ---- corrector ----
        cross = dxp * dsp
        try:
            sol = np.linalg.solve(
                M,
                np.concatenate([-rc - s + (sig * mu - cross) / x, -rb]),
            )
        except np.linalg.LinAlgError:
            return _result("numerical_failure", "singular corrector KKT", it, x, y, s)
        dx = sol[:n]
        dy = sol[n:]
        # Corrector ds formula: from S·dx + X·ds = -xs + σμ - cross
        #   ds = -s + (σμ - cross)/x - (s/x)·dx
        ds = -s + (sig * mu - cross) / x - D * dx

        ax = min(1.0, tau * _max_step(x, dx)) if np.any(dx < 0) else 1.0
        as_ = min(1.0, tau * _max_step(s, ds)) if np.any(ds < 0) else 1.0
        if not (np.isfinite(ax) and np.isfinite(as_)):
            return _result("stalled", "non-finite step", it, x, y, s)
        if ax <= 1e-16 or as_ <= 1e-16:
            stalled_cnt += 1
            if stalled_cnt >= 5:
                break
        else:
            stalled_cnt = 0

        x = x + ax * dx
        y = y + ax * dy
        s = s + as_ * ds

    # ---- polish: active-set exact KKT recovery, then recompute everything ----
    if dual_recover and np.all(np.isfinite(x)) and np.all(x >= 0.0):
        y_r, s_r = _active_set_polish(sq, x)
        rel_p, rel_d, rel_g = _fres(x, y_r, s_r)
        # s >= -tol_s on the active set and complementarity by construction.
        s_min = float(np.min(s_r)) if s_r.size else 0.0
        s_tol = tol * (1.0 + cnorm)
        if s_min >= -s_tol and rel_p <= tol and rel_d <= tol and rel_g <= tol:
            r = _result("optimal", "converged to tolerance (active-set polished)", it, x, y_r, s_r)
            return r
        return _result("max_iterations",
                       f"primal converged but dual recovery failed "
                       f"(s_min={s_min:.2e}, rel_d={rel_d:.2e})",
                       it, x, y, s)
    return _result("max_iterations",
                   f"did not converge within {max_iter} iterations", max_iter - 1,
                   x, y, s)


def solve_qp(
    model: NumericalQP,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
    verbose: bool = False,
) -> QpResult:
    """Solve a convex QP end-to-end (standard-form conversion + IPM)."""
    t0 = time.perf_counter()
    Q = np.asarray(model.Q, dtype=np.float64)
    if not np.allclose(Q, Q.T, atol=1e-12):
        raise QpError("Q must be symmetric")
    w = np.linalg.eigvalsh(Q)
    if np.min(w) < -1e-8:
        raise QpError("Q is not positive semidefinite (indefinite QP not in v1 scope)")
    sq = to_standard_qp(model)
    res = solve_standard_qp(sq, tol=tol, max_iter=max_iter)
    if verbose:
        print(f"[qp] status={res.status} obj={res.objective:.9g} "
              f"rel_p/d/g={res.rel_primal:.2e}/{res.rel_dual:.2e}/{res.rel_gap:.2e} "
              f"iter={res.iterations} ({time.perf_counter() - t0:.3f}s)")
    return res


def verify_qp_kkt(model: NumericalQP, result: QpResult) -> dict:
    """Independently recompute standard-form KKT residuals from the result.

    Returns a dict with ``rel_primal``, ``rel_dual``, ``rel_gap`` recomputed
    from the returned ``x_standard``/``y``/``s`` plus original-coordinate
    checks (constraint satisfaction and bounds).  ``PASS`` is True only when
    every check is below ``1e-6`` -- the QP analogue of the LP certificates.
    """
    Q = np.asarray(model.Q, dtype=np.float64)
    c = np.asarray(model.c, dtype=np.float64)
    A = np.asarray(model.A, dtype=np.float64)
    b = np.asarray(model.b, dtype=np.float64)
    rt = tuple(model.row_types)
    lb = np.asarray(model.lower_bounds, dtype=np.float64)
    ub = np.asarray(model.upper_bounds, dtype=np.float64)
    x = result.x

    sq = to_standard_qp(model)
    xs = result.x_standard

    rc = sq.Q @ xs + sq.c - sq.A.T @ result.y - result.s
    rb = sq.A @ xs - sq.b
    obj = 0.5 * float(x @ Q @ x) + float(c @ x)
    bnorm = float(np.max(np.abs(sq.b))) if sq.b.size else 0.0
    cnorm = float(np.max(np.abs(sq.c))) if sq.c.size else 0.0
    rel_p = float(np.max(np.abs(rb))) / (1.0 + bnorm) if rb.size else 0.0
    rel_d = float(np.max(np.abs(rc))) / (1.0 + cnorm) if rc.size else 0.0
    rel_g = float(np.abs(result.s @ xs)) / (1.0 + abs(obj) + abs(float(sq.b @ result.y)))

    viol_rows = []
    for i in range(model.num_constraints):
        val = float(A[i] @ x)
        if rt[i] == "E" and abs(val - b[i]) > 1e-6:
            viol_rows.append((i, abs(val - b[i])))
        elif rt[i] == "L" and val - b[i] > 1e-6:
            viol_rows.append((i, val - b[i]))
        elif rt[i] == "G" and b[i] - val > 1e-6:
            viol_rows.append((i, b[i] - val))
    lb_viol = float(np.max(np.maximum(0.0, lb - x))) if x.size else 0.0
    ub_viol = float(np.max(np.maximum(0.0, x - ub))) if x.size else 0.0

    passes = (
        rel_p <= 1e-6 and rel_d <= 1e-6 and rel_g <= 1e-6
        and not viol_rows and lb_viol <= 1e-9 and ub_viol <= 1e-9
    )
    return {
        "rel_primal": rel_p,
        "rel_dual": rel_d,
        "rel_gap": rel_g,
        "row_violations": viol_rows,
        "lb_violation": lb_viol,
        "ub_violation": ub_viol,
        "objective": obj,
        "PASS": bool(passes),
    }


def _tiny_qp() -> NumericalQP:
    """min 1/2(x^2+y^2) + x - 2y s.t. x + y >= 1, x - y <= 0.5, x,y >= 0.

    KKT-verifiable; used by the test suite and the demo as a known check.
    """
    return NumericalQP(
        name="TINY_QP",
        Q=np.array([[1.0, 0.0], [0.0, 1.0]]),
        c=np.array([1.0, -2.0]),
        A=np.array([[1.0, 1.0], [1.0, -1.0]]),
        b=np.array([1.0, 0.5]),
        row_types=("G", "L"),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
    )


if __name__ == "__main__":
    _m = _tiny_qp()
    _r = solve_qp(_m, verbose=True)
    print(_r.status, _r.objective, _r.x)
    print(verify_qp_kkt(_m, _r))