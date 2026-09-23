"""Indigenous branch-and-bound MILP solver on top of the production LP core.

This is a **v2** implementation that adds key algorithmic improvements over the
v1 skeleton to handle real MIPLIB2017 instances:

* LP relaxation   : existing ``solve_lp`` (Mehrotra IPM) plus a *bounded*
                    sparse-crossover fallback.  Honesty contract: a node whose
                    relaxation does not report ``status == "optimal"`` is
                    *dropped*, never trusted; only a Phase-I infeasibility
                    certificate maps a failed relaxation to ``"infeasible"``.
                    Every per-node fallback is bounded (pivot caps + an
                    optional shared time budget) so no node can run an
                    effectively unbounded simplex loop.
* Node selection  : best-bound first (heap on relaxation objective).
* Branching       : most-fractional (ties broken by |c|); optional strong
                    branching is available via ``strong_branch_depth`` but is
                    DISABLED by default for this experiment.
* Pruning         : bound >= incumbent (within tolerance) and infeasible nodes.
* Incumbent       : any node whose relaxation is integer-feasible; plus a
                    rounding heuristic applied at the root and periodically
                    during search to find feasible integer solutions early.
* Termination     : proven optimal when the node heap empties, or the relative
                    MIP gap <= ``gap_tol`` (best-bound vs incumbent).

Algorithm upgrades over v1:

* **Rounding + integer-fixing repair heuristic**: after solving each LP
  relaxation, round candidate integer assignments and then REPAIR them by
  fixing every integer variable and re-solving the continuous LP with the
  production core.  Independently verified before being accepted.  Applied at
  the root and every ``heuristic_freq`` nodes during the search.
* **Strong branching** (optional, disabled by default): at nodes below
  ``strong_branch_depth``, evaluate the top ``strong_branch_k`` most-fractional
  candidates by solving both child relaxations and pick the variable that
  maximises bound degradation.
* **Improved node selection**: same best-bound-first heap; children are pushed
  with a proxy bound from the parent until re-solved.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ..numerical_model import NumericalLP
from .mehrotra import solve_lp, to_standard_form, MehrotraError, StandardFormLP
from .crossover import sparse_phase1, sparse_phase2, crossover_from_ipm


# ---------------------------------------------------------------------------
# Robustness bounds for per-node relaxations.  A single B&B node must never
# execute an effectively unbounded IPM/crossover loop, and a relaxation that
# fails is never conflated with infeasibility (only a Phase-I certificate may
# map to "infeasible").
# ---------------------------------------------------------------------------
_NODE_RELAX_TIME_LIMIT = 30.0             # wall-clock budget (s) per relaxation fallback
_NODE_RELAX_MAX_ITER = 100                # IPM Newton iterations per relaxation default
_NODE_FALLBACK_PHASE1_MAX_ITER = 200_000  # Phase-I pivot cap for a relaxation fallback
_NODE_FALLBACK_PHASE2_MAX_ITER = 50_000   # Phase-II pivot cap for a relaxation fallback
_REPAIR_TIME_LIMIT = 10.0                 # wall-clock budget (s) for one repair re-solve


class MilpError(ValueError):
    pass


def _drop_redundant_rows(lp: NumericalLP) -> NumericalLP:
    """Remove provably redundant rows (mini-presolve, safe subset).

    Two safe cases only:

    * **Exact duplicates** of any row type: identical coefficients, type and
      RHS -- dropping the later copy preserves the feasible set exactly.
    * **Equality rows** that are exact linear combinations of previously kept
      equality rows WITH consistent RHS -- likewise preserves feasibility.

    Mixed inequality combinations are intentionally NOT reduced here (a
    nonnegativity constraint is required to dominate, which is presolve's job);
    avoiding false removals is more important than removing all redundancy in
    v1.  The assignment test in the skeleton needs the E-row reduction because
    the production dense Schur backend requires a full-rank standard form.
    Sparse-safe: sparse inputs are inspected one row at a time (per-row
    extraction) and returned in CSR form; the full m×n matrix is never
    densified.
    """
    import scipy.sparse as sp
    A = lp.A
    if sp.issparse(A):
        A_csr = A.tocsr()
    else:
        A_csr = np.asarray(A, dtype=np.float64)
    b = np.asarray(lp.b, dtype=np.float64)
    rt = tuple(lp.row_types)
    m = lp.num_constraints

    def _row(i: int) -> np.ndarray:
        if sp.issparse(A_csr):
            return np.asarray(A_csr.getrow(i).toarray()).ravel()
        return np.asarray(A_csr[i], dtype=np.float64)

    drop: set[int] = set()
    kept_e: list[tuple[np.ndarray, float]] = []      # equality rows
    for i in range(m):
        a = _row(i)
        rhs = float(b[i])
        # Exact duplicate against any earlier row of the same type.
        dup = False
        for k in range(i):
            if k in drop or rt[k] != rt[i]:
                continue
            if (np.allclose(a, _row(k), rtol=1e-8, atol=1e-10)
                    and abs(rhs - b[k]) <= 1e-8):
                dup = True
                break
        if dup:
            drop.add(i)
            continue
        if rt[i] == "E":
            span = False
            for (ak, bk) in kept_e:
                if np.allclose(a, ak, rtol=1e-8, atol=1e-10):
                    span = abs(rhs - bk) <= 1e-8
                    break
            if not span and len(kept_e) >= 2:
                S = np.vstack([ak for ak, _ in kept_e]).T   # (cols, k)
                try:
                    lam, *_ = np.linalg.lstsq(S, a, rcond=None)
                    resid = float(np.linalg.norm(S @ lam - a))
                    if (resid <= 1e-8 * (1.0 + float(np.linalg.norm(a))) and
                            abs(float(np.dot(lam, [bk for _, bk in kept_e])) - rhs) <= 1e-8):
                        span = True
                except np.linalg.LinAlgError:
                    span = False
            if span:
                drop.add(i)
            else:
                kept_e.append((a, rhs))
    if not drop:
        return lp
    keep = sorted(set(range(m)) - drop)
    return replace(
        lp,
        A=A_csr[np.asarray(keep, dtype=np.intp)],
        b=b[keep],
        row_types=tuple(rt[k] for k in keep),
        row_names=tuple(lp.row_names[k] for k in keep),
    )


def _root_infeasible(lp: NumericalLP, max_iter: int = 20_000,
                     deadline: Optional[float] = None) -> bool:
    """True only when the LP relaxation is *proven* infeasible via Phase I.

    Uses the production sparse Phase I (all-artificial crash) on the standard
    form as an independent feasibility oracle; never used as the solve engine.

    ``deadline`` (optional absolute ``time.perf_counter()`` timestamp) caps
    the Phase I wall-clock budget so this verification pass cannot run past
    the global deadline (when it does, it reports ``False`` rather than a
    possibly-wrong verdict).
    """
    if deadline is not None and time.perf_counter() >= deadline:
        return False
    try:
        sf = to_standard_form(lp)
    except MehrotraError:
        return False
    try:
        tl_budget = None
        if deadline is not None:
            tl_budget = max(0.0, deadline - time.perf_counter())
            if tl_budget <= 0.0:
                return False
        _, _, status, _ = sparse_phase1(sf.A, sf.b, max_iter=max_iter,
                                        time_limit=tl_budget)
        return status == "infeasible"
    except Exception:
        return False


@dataclass
class MilpResult:
    """Solution and diagnostics of one branch-and-bound solve.

    ``objective`` is in the ORIGINAL sense requested by ``maximize`` in the
    original model coordinates; ``x`` are the original variables only.
    ``best_bound`` is the best relaxation bound in that same original sense
    (min-objective for minimization, max-bound for maximization), so the
    reported ``gap`` is always nonnegative.
    """

    status: str                 # "optimal" | "infeasible" | "unbounded" | "node_limit" |
                                # "time_limit" | "lp_status:<solver status>"
    message: str
    objective: Optional[float]  # original-sense optimum (None if infeasible or none found)
    x: Optional[np.ndarray]     # original variables (None if none found)
    best_bound: Optional[float] # original-sense relaxation bound
    gap: Optional[float]        # relative MIP gap (incumbent vs best bound)
    nodes_explored: int
    node_limit: int
    lp_solves: int
    time_sec: float
    heuristic_attempts: int = 0
    heuristic_successes: int = 0
    heuristic_lp_solves: int = 0
    heuristic_best_objective: Optional[float] = None
    nodes_infeasible: int = 0                # nodes pruned by a Phase-I infeasibility certificate
    nodes_failed: int = 0                    # nodes dropped because their relaxation failed
    node_failure_statuses: tuple[str, ...] = ()  # e.g. ("time_limit", "max_iterations")


def _relative_gap(incumbent: float, bound: float) -> float:
    return abs(incumbent - bound) / (1.0 + abs(incumbent))


class _RelaxResult:
    """Lightweight adapter returned by the node LP solver.

    Carries only the attributes ``status``, ``objective``, ``x`` and
    ``message`` that the branch-and-bound loop needs.
    """
    __slots__ = ("status", "objective", "x", "message")
    def __init__(self, status, objective, x, message=""):
        self.status = status
        self.objective = objective
        self.x = x
        self.message = message


def _classify_relax_failure(
    ipm_status: str,
    fb_status: str,
    co: "object | None" = None,
) -> tuple[str, str]:
    """Choose the most specific honest failure status for a relaxation.

    Contract: a numerical failure is NEVER conflated with infeasibility —
    only a Phase-I infeasibility certificate (handled by the caller) may map
    a failed relaxation to ``"infeasible"``.  Of the remaining causes, the
    most specific one reported by the bounded fallback wins over the generic
    IPM status, so callers can distinguish ``max_iterations`` from
    ``time_limit`` from ``stalled``.
    """
    fb = fb_status or ""
    note = ""
    if isinstance(co, dict):
        note = str(co.get("message", ""))
    if "time_limit" in fb:
        return "time_limit", f"relaxation time budget exhausted; fallback: {note}"
    if "max_iterations" in fb:
        return "max_iterations", f"relaxation hit its iteration budget; fallback: {note}"
    if "numerical_failure" in fb:
        return "numerical_failure", f"relaxation failed numerically; fallback: {note}"
    if "stalled" in fb:
        return "stalled", f"relaxation stalled; fallback: {note}"
    if ipm_status in ("stalled", "numerical_tail", "numerical_failure",
                      "max_iterations"):
        return ipm_status, f"IPM {ipm_status}; fallback incomplete: {note}"
    return ("numerical_failure",
            f"relaxation failed (IPM {ipm_status}; fallback {fb or 'unknown'}): {note}")


def _relax(
    lp: NumericalLP,
    *,
    crossover_fallback: bool = True,
    verbose: bool = False,
    max_iter: Optional[int] = None,
    time_limit: Optional[float] = None,
    phase1_max_iter: int = _NODE_FALLBACK_PHASE1_MAX_ITER,
    phase2_max_iter: int = _NODE_FALLBACK_PHASE2_MAX_ITER,
    deadline: Optional[float] = None,
) -> _RelaxResult:
    """Solve the LP relaxation of a NumericalLP.

    Primary path: the production Mehrotra IPM with ``crossover_fallback=False``
    — within B&B the LP-core's internal (merit-gated but iteration-unbounded)
    crossover is never used; every fallback here is bounded by construction, so
    no node can run an effectively unbounded simplex loop.

    Secondary path: a bounded from-scratch sparse simplex on the standard form
    via ``crossover_from_ipm`` (Phase I all-artificial crash + Devex Phase II),
    budgeted by ``phase1_max_iter``/``phase2_max_iter`` pivots and an optional
    shared ``time_limit`` (wall-clock seconds) that is propagated into both
    phases.

    ``deadline`` is an optional absolute wall-clock timestamp (as returned by
    ``time.perf_counter()``).  When provided, the local ``time_limit`` passed
    to the crossover fallback is capped by the remaining budget
    ``max(0, deadline - now)`` so the global deadline always takes precedence
    over any local iteration/repair limit.

    Honesty contract:

    * ``"optimal"`` only when the IPM converges, or the fallback passes the
      same independent KKT acceptance gate it did before.
    * ``"infeasible"`` ONLY when the fallback's Phase I produces an
      infeasibility certificate.  A numerical failure is never silently
      reported as infeasible.
    * Any other return is an explicit failure status (``stalled``,
      ``max_iterations``, ``time_limit``, ``numerical_failure``, or the IPM's
      own status), which the branch-and-bound loop counts as a dropped node.

    ``crossover_fallback`` is accepted for call-site compatibility only: the LP
    core's internal crossover is never enabled from B&B regardless of its value
    (that path is merit-gated but iteration-unbounded), so *all* fallbacks here
    are bounded and pass through the same independent KKT gate.
    """
    # ---- Early global-deadline exit, before even the IPM solve. ----
    # The primary Mehrotra solve is bounded by ``max_iter`` Newton steps but
    # has no wall-clock budget of its own, so when the global deadline has
    # already passed we must not start it at all.
    if deadline is not None and time.perf_counter() >= deadline:
        return _RelaxResult("time_limit", float("nan"), None,
                            "global deadline exceeded before relaxation")

    # Primary: production Mehrotra IPM (fast for most LPs).
    ipm_kw = {} if max_iter is None else {"max_iter": max_iter}
    ipm = solve_lp(lp, crossover_fallback=False, **ipm_kw)
    ipm_obj = float(ipm.objective) if np.isfinite(ipm.objective) else float("nan")
    ipm_x = np.asarray(ipm.x) if getattr(ipm, "x", None) is not None else None
    if ipm.status == "optimal":
        return _RelaxResult("optimal", ipm_obj, np.asarray(ipm.x), ipm.message)

    # ---- Cap local time budget by remaining global deadline. ----
    # The global deadline always wins: when it has expired the fallback
    # returns ``"time_limit"`` immediately rather than spinning for the
    # full local budget.
    remaining = (max(0.0, deadline - time.perf_counter())
                 if deadline is not None else None)
    if remaining is not None and remaining <= 0.0:
        return _RelaxResult("time_limit", ipm_obj, ipm_x,
                            "global deadline exceeded before fallback")
    effective_tl = time_limit
    if remaining is not None:
        effective_tl = (remaining if effective_tl is None
                        else min(effective_tl, remaining))

    # Secondary: bounded from-scratch sparse simplex on the standard form.
    # Handles rank-deficient / degenerate relaxations that the dense IPM
    # sometimes fails at iteration 0 via μ-floor termination.
    try:
        sf = to_standard_form(lp)
    except MehrotraError:
        if verbose:
            print(f"    [fallback] to_standard_form failed: {ipm.status} "
                  f"{ipm.message[:80]}")
        return _RelaxResult(ipm.status, ipm_obj, ipm_x, ipm.message)
    try:
        co = crossover_from_ipm(sf.A, sf.b, sf.c_min,
                                max_iter=phase2_max_iter,
                                phase1_max_iter=phase1_max_iter,
                                pricing="devex", verbose=0,
                                time_limit=effective_tl)
    except Exception as exc:
        if verbose:
            print(f"    [fallback] crossover failed: {exc}")
        return _RelaxResult(ipm.status, ipm_obj, ipm_x, ipm.message)

    # 1. A Phase-I infeasibility certificate is authoritative.
    if co.get("status") == "phase1_infeasible":
        return _RelaxResult("infeasible", float("nan"), None,
                            "LP relaxation proven infeasible (Phase I certificate)")

    # 2. Near-feasible rescue: Phase I returned an ALL-ORIGINAL basis whose
    #    certification solve was numerically borderline (small negatives in
    #    xB, not a proof of infeasibility).  Phase II re-solves that basis from
    #    scratch with its soft-clamp safety net; this stays within the same
    #    bounded pivot/time budget and reports through the same KKT gate.
    #    (Phases that return artificials -- stalled/max_iterations -- cannot be
    #    jumped; only the all-original clean-out output is acceptable here.)
    if (co.get("status") == "phase1_near_feasible"
            and co.get("phase1") is not None):
        basis_nf = (co["phase1"] or {}).get("basis")
        if basis_nf is not None and len(basis_nf) > 0:
            try:
                co = sparse_phase2(sf.A, sf.b, sf.c_min, list(basis_nf),
                                   max_iter=phase2_max_iter, pricing="devex",
                                   verbose=0, time_limit=effective_tl)
            except Exception:
                co = {}  # fall through to the honest failure classification

    # 3. Optimality gate (independent KKT acceptance, unchanged).
    if (co.get("status") == "optimal" and co.get("rel_primal", float("inf")) <= 1e-7
            and co.get("rel_dual", float("inf")) <= 1e-7
            and co.get("rel_gap", float("inf")) <= 1e-7):
        x_std = co["x"]
        x_orig = sf.recover_original(x_std)
        obj = float(sf.c_orig @ x_orig)
        return _RelaxResult("optimal", obj, x_orig,
                            f"simplex fallback ({co.get('phase2_iterations', 0)} Phase-II pivots)")

    # 3. Nothing converged: report the most specific honest cause.
    if verbose:
        print(f"    [fallback] simplex failed: {co.get('status')} "
              f"rel_p={co.get('rel_primal', float('nan')):.1e} "
              f"rel_d={co.get('rel_dual', float('nan')):.1e} "
              f"rel_g={co.get('rel_gap', float('nan')):.1e}")
    status, msg = _classify_relax_failure(ipm.status, co.get("status"), co)
    return _RelaxResult(status, ipm_obj, ipm_x, msg)


def _check_feasibility(
    A: np.ndarray,
    b: np.ndarray,
    row_types: tuple[str, ...],
    x: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """Check whether x satisfies Ax compat b given row_types."""
    residual = A @ x - b
    for i, rt in enumerate(row_types):
        r = residual[i]
        if rt == "E" and abs(r) > tol:
            return False
        elif rt == "L" and r > tol:
            return False
        elif rt == "G" and r < -tol:
            return False
    return True


def _build_integer_candidates(
    x_lp: np.ndarray,
    int_idx: np.ndarray,
    frac_vars: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    int_tol: float = 1e-6,
) -> list:
    """Generate a small, bounded set of candidate integer assignments.

    Returns a list of candidate vectors (each a copy) with integer variables
    fixed to specific values.  Candidates, in order:

    1. nearest-round of every integer variable (always first);
    2. round-DOWN-all / round-UP-all of the fractional variables while the
       non-fractional ones keep their nearest rounding (two structurally
       different global candidates -- nearest-only rounding fails on
       equality-structured instances where every variable must move the
       same way);
    3. floor/ceil flips of the most-fractional variables (top-6), all other
       variables nearest-rounded.

    At most 15 candidates total.  Every candidate is subsequently validated
    by the integer-fixing repair LP, so a bad candidate costs only its
    bounded re-solve, never correctness.
    """
    x_base = x_lp.copy()
    for i in int_idx:
        x_base[i] = np.clip(np.round(x_lp[i]), lb[i], ub[i])
    candidates = [x_base]

    if frac_vars.size == 0:
        return candidates

    # Global directional candidates: everything down / everything up.
    for target_fn in (np.floor, np.ceil):
        cand = x_base.copy()
        for j in frac_vars:
            cand[j] = np.clip(target_fn(x_lp[j]), lb[j], ub[j])
        candidates.append(cand)

    frac_vals = np.abs(x_lp[frac_vars] - np.round(x_lp[frac_vars]))
    top_k = min(6, frac_vars.size)
    top_idx = frac_vars[np.argsort(-frac_vals)[:top_k]]

    for j in top_idx:
        for target_fn in (np.floor, np.ceil):
            cand = x_base.copy()
            cand[j] = np.clip(target_fn(x_lp[j]), lb[j], ub[j])
            candidates.append(cand)

    return candidates



def _repair_via_continuous_lp(
    lp: NumericalLP,
    x_candidate: np.ndarray,
    integer_mask: np.ndarray,
    int_tol: float = 1e-6,
    stats: "dict | None" = None,
    deadline: "Optional[float]" = None,
) -> tuple:
    """Fix integer variables and re-solve the resulting continuous LP.

    1. Set lb = ub = rounded value for every integer variable.
    2. Solve the resulting continuous LP via the production solver.
    3. Independently verify: constraints, bounds, integrality, objective.
    4. Return ``(feasible, x_orig, obj)`` or ``(False, candidate, nan)``.

    Never raises on solver failure -- simply returns infeasible.

    ``deadline`` (optional absolute ``time.perf_counter()`` timestamp) caps
    the per-repair ``_REPAIR_TIME_LIMIT`` so the global deadline is never
    overshot by the continuous repair path.
    """
    int_idx = np.flatnonzero(integer_mask)
    n = lp.num_vars

    # --- 1. Fix integer variables ---
    fixed_lb = np.asarray(lp.lower_bounds, dtype=np.float64).copy()
    fixed_ub = np.asarray(lp.upper_bounds, dtype=np.float64).copy()
    for i in int_idx:
        val = np.clip(np.round(x_candidate[i]), fixed_lb[i], fixed_ub[i])
        fixed_lb[i] = val
        fixed_ub[i] = val

    if np.any(fixed_lb > fixed_ub + 1e-12):
        return False, x_candidate, float("nan")

    fixed_lp = replace(lp, lower_bounds=fixed_lb, upper_bounds=fixed_ub)

    # --- 2. Solve the continuous LP (bounded relaxation; never the
    # iteration-unbounded internal crossover) ---
    if stats is not None:
        stats["lp_solves"] = stats.get("lp_solves", 0) + 1
    retry_tl = _REPAIR_TIME_LIMIT
    if deadline is not None:
        remaining = max(0.0, deadline - time.perf_counter())
        if remaining <= 0.0:
            return False, x_candidate, float("nan")
        retry_tl = min(retry_tl, remaining)
    rel = _relax(fixed_lp, crossover_fallback=False,
                 max_iter=_NODE_RELAX_MAX_ITER, time_limit=retry_tl,
                 deadline=deadline)
    if rel.status != "optimal" or rel.x is None:
        return False, x_candidate, float("nan")

    x_sol = np.asarray(rel.x, dtype=np.float64)
    obj = float(rel.objective)

    # --- 3. Independent verification ---
    # 3a. Dimensions.
    if x_sol.shape != (n,):
        return False, x_candidate, float("nan")

    # 3b. Bounds (including fixed integers).
    for i in range(n):
        if x_sol[i] < fixed_lb[i] - 1e-6 or x_sol[i] > fixed_ub[i] + 1e-6:
            return False, x_candidate, float("nan")

    # 3c. Integrality of integer variables.
    for i in int_idx:
        if abs(x_sol[i] - np.round(x_sol[i])) > int_tol:
            return False, x_candidate, float("nan")

    # 3d. Row constraints (original LP constraints; sparse-safe matvec).
    b_vec = np.asarray(lp.b, dtype=np.float64)
    residual = np.asarray(lp.A @ x_sol) - b_vec
    for i, rt in enumerate(lp.row_types):
        r = residual[i]
        if rt == "E" and abs(r) > 1e-6:
            return False, x_candidate, float("nan")
        elif rt == "L" and r > 1e-6:
            return False, x_candidate, float("nan")
        elif rt == "G" and r < -1e-6:
            return False, x_candidate, float("nan")

    # 3e. Objective consistency.
    c_vec = np.asarray(lp.c, dtype=np.float64)
    computed_obj = float(c_vec @ x_sol)
    if abs(computed_obj - obj) > 1e-4 * (1.0 + abs(obj)):
        return False, x_candidate, float("nan")

    return True, x_sol, obj


def _rounding_heuristic(
    x_lp: np.ndarray,
    integer_mask: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    row_types: tuple[str, ...],
    c_min: np.ndarray,
    *,
    lp: "NumericalLP | None" = None,
    max_rounds: int = 50,
    int_tol: float = 1e-6,
    stats: "dict | None" = None,
    deadline: "Optional[float]" = None,
    incumbent: float = float("inf"),
) -> tuple:
    """Try to produce an integer-feasible solution from the LP relaxation.

    Strategy (in order):
    1. Deterministic rounding -- nearest integer, clamped to bounds.
    2. Integer-fixing repair (requires *lp*): for a bounded number of
       candidate integer assignments, fix every integer variable's bounds
       to that value and re-solve the resulting continuous LP using the
       production solver.  Independently verified before return.
    3. Greedy one-flip rounding.
    4. Randomised feasibility-pump rounding for ``max_rounds`` iterations.

    Improvement gate: ``incumbent`` (min-form) is the current best objective.
    A candidate is accepted EARLY only when it strictly improves it
    (``obj < incumbent - 1e-12``); otherwise the best verified-feasible
    candidate is returned and the caller applies its own improvement gate.
    This lets periodic heuristic calls keep trying floor/ceil variants even
    after the nearest rounding is already feasible-but-not-improving, while
    never spending more than the bounded candidate list allows.

    The heuristic never affects correctness; failure means no incumbent found.

    ``deadline`` (optional absolute ``time.perf_counter()`` timestamp)
    bounds the wall-clock work spent across the integer-fixing repairs so the
    global deadline is honoured even when many candidates would each take up
    to ``_REPAIR_TIME_LIMIT`` seconds.

    Returns ``(feasible, x_feas, obj)``.
    """
    int_idx = np.flatnonzero(integer_mask)
    if int_idx.size == 0:
        obj = float(c_min @ x_lp)
        return True, x_lp.copy(), obj

    if deadline is not None and time.perf_counter() >= deadline:
        return False, x_lp.copy(), float(c_min @ x_lp)

    def _viol(xv: np.ndarray) -> float:
        """Infeasibility measure."""
        res = A @ xv - b
        v = 0.0
        for i, rt in enumerate(row_types):
            if rt == "E":
                v += abs(res[i])
            elif rt == "L":
                v += max(res[i], 0.0)
            elif rt == "G":
                v += max(-res[i], 0.0)
        return v

    def _record(obj: float) -> None:
        """Record a successful heuristic round for diagnostics."""
        if stats is not None:
            stats["successes"] = stats.get("successes", 0) + 1
            prev = stats.get("best_obj")
            stats["best_obj"] = obj if prev is None else min(prev, obj)

    # --- 1. Deterministic rounding ---
    x_rd = x_lp.copy()
    for i in int_idx:
        x_rd[i] = np.clip(np.round(x_lp[i]), lb[i], ub[i])
    rounding_ok = _check_feasibility(A, b, row_types, x_rd, tol=int_tol * 10)
    if rounding_ok:
        obj = float(c_min @ x_rd)
        if obj < incumbent - 1e-12:
            # Strict improvement over the current incumbent: accept now.
            _record(obj)
            return True, x_rd, obj
        # Feasible but not improving: remember as a fallback and still try
        # the repair candidates for something better.
        best_alt = (x_rd, obj)
    else:
        best_alt = None

    # --- 2. Integer-fixing repair via continuous LP re-solve ---
    frac_vars = int_idx[
        np.abs(x_lp[int_idx] - np.round(x_lp[int_idx])) > int_tol
    ]
    if lp is not None:
        if stats is not None:
            stats["attempts"] = stats.get("attempts", 0) + 1
        candidates = _build_integer_candidates(x_lp, int_idx, frac_vars,
                                               lb, ub, int_tol)
        for cand in candidates:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            ok, x_repaired, obj_repaired = _repair_via_continuous_lp(
                lp, cand, integer_mask, int_tol=int_tol, stats=stats,
                deadline=deadline,
            )
            if not ok:
                continue
            if obj_repaired < incumbent - 1e-12:
                # Strict improvement over the current incumbent: accept.
                _record(obj_repaired)
                return True, x_repaired, obj_repaired
            # Verified-feasible but not improving: keep the best and let the
            # caller decide (it applies the same improvement gate).
            if best_alt is None or obj_repaired < best_alt[1]:
                best_alt = (x_repaired, obj_repaired)

    if best_alt is not None:
        _record(best_alt[1])
        return True, best_alt[0], best_alt[1]

    if frac_vars.size == 0:
        return False, x_rd, float(c_min @ x_rd)

    # --- 3. Greedy one-flip rounding ---
    best_x = x_rd.copy()
    best_v = _viol(best_x)
    for j in frac_vars:
        for target in (np.floor(x_lp[j]), np.ceil(x_lp[j])):
            target = np.clip(target, lb[j], ub[j])
            cand = best_x.copy()
            cand[j] = target
            v = _viol(cand)
            if v < best_v:
                best_v = v
                best_x = cand
        if best_v <= 1e-6:
            obj = float(c_min @ best_x)
            _record(obj)
            return True, best_x, obj

    if _check_feasibility(A, b, row_types, best_x, tol=int_tol * 10):
        obj = float(c_min @ best_x)
        _record(obj)
        return True, best_x, obj

    # --- 4. Randomised feasibility-pump rounding ---
    rng = np.random.RandomState(42)
    for _ in range(max_rounds):
        xr = x_lp.copy()
        for j in frac_vars:
            fval = x_lp[j]
            if rng.random() < 0.5:
                xr[j] = np.clip(np.floor(fval), lb[j], ub[j])
            else:
                xr[j] = np.clip(np.ceil(fval), lb[j], ub[j])
        for i in int_idx:
            if i not in frac_vars:
                xr[i] = np.clip(np.round(x_lp[i]), lb[i], ub[i])
        if _check_feasibility(A, b, row_types, xr, tol=int_tol * 10):
            obj = float(c_min @ xr)
            _record(obj)
            return True, xr, obj
        v = _viol(xr)
        if v < best_v:
            best_v = v
            best_x = xr.copy()

    return False, best_x, float(c_min @ best_x)


def _strong_branch_select(
    lp: NumericalLP,
    x: np.ndarray,
    integer_mask: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    k: int = 6,
    int_tol: float = 1e-6,
    crossover_fallback: bool = False,
    verbose: bool = False,
    max_iter: Optional[int] = None,
    time_limit: Optional[float] = None,
    deadline: Optional[float] = None,
) -> Optional[int]:
    """Evaluate up to *k* most-fractional variables by solving both children.

    Returns the index of the variable whose split gives the largest bound
    degradation.  Returns ``None`` when no candidates were solved successfully.

    ``deadline`` (optional absolute ``time.perf_counter()`` timestamp) caps the
    per-child ``time_limit`` by the remaining global budget so strong branching
    can never spend more than the global deadline across its 2*k child solves.
    """
    int_idx = np.flatnonzero(integer_mask)
    frac_mask = np.zeros_like(x, dtype=bool)
    frac_mask[int_idx] = np.abs(x[int_idx] - np.round(x[int_idx])) > int_tol
    frac_vars = np.flatnonzero(frac_mask)
    if frac_vars.size == 0:
        return None

    frac_vals = np.abs(x[frac_vars] - np.round(x[frac_vars]))
    order = np.argsort(-frac_vals)
    candidates = frac_vars[order[:k]]

    best_j = None
    best_obj = -np.inf
    for j in candidates:
        fj = x[j]
        lo, hi = int(np.floor(fj)), int(np.ceil(fj))
        for branch_val in (lo, hi):
            if deadline is not None and time.perf_counter() >= deadline:
                return best_j
            clb, cub = lb.copy(), ub.copy()
            if branch_val == lo:
                cub[j] = min(ub[j], float(lo))
            else:
                clb[j] = max(lb[j], float(hi))
            if not np.all(clb <= cub + 1e-12):
                continue
            child_lp = replace(lp, lower_bounds=clb, upper_bounds=cub)
            child_tl = time_limit
            if deadline is not None:
                remaining = max(0.0, deadline - time.perf_counter())
                if remaining <= 0.0:
                    return best_j
                child_tl = (remaining if child_tl is None
                            else min(child_tl, remaining))
            child_rel = _relax(child_lp, crossover_fallback=crossover_fallback,
                               verbose=verbose, max_iter=max_iter,
                               time_limit=child_tl, deadline=deadline)
            if child_rel.status == "optimal":
                if float(child_rel.objective) > best_obj:
                    best_obj = float(child_rel.objective)
                    best_j = j
    return best_j


def solve_milp(
    lp: NumericalLP,
    integer_mask: Optional[np.ndarray] = None,
    *,
    maximize: bool = False,
    int_tol: float = 1e-6,
    gap_tol: float = 1e-4,
    node_limit: int = 50_000,
    time_limit: Optional[float] = None,
    verbose: bool = False,
    crossover_fallback: bool = True,
    node_crossover_fallback: bool = False,
    heuristic_freq: int = 25,
    strong_branch_depth: int = 0,
    strong_branch_k: int = 4,
    node_solve_max_iter: Optional[int] = None,
    node_solve_time_limit: Optional[float] = None,
) -> MilpResult:
    """Solve a mixed-integer linear program by branch-and-bound.

    ``integer_mask`` is a bool array over the ORIGINAL variables; if omitted,
    ``lp.is_integer`` is used when present (the parser extension), otherwise a
    ``MilpError`` is raised.  Only finitely-bounded integer variables are
    supported in v1.

    ``node_solve_max_iter`` bounds the IPM Newton iterations per node
    relaxation; ``node_solve_time_limit`` bounds the wall-clock budget (mainly
    spent in the node's sparse-simplex fallback).  Both default to production
    values (100 IPM iterations; ``_NODE_RELAX_TIME_LIMIT`` seconds of fallback)
    so no single node can run an effectively unbounded loop.

    Returns a ``MilpResult`` whose honesty contract mirrors the LP core: an
    ``"optimal"`` status means proven optimal (heap exhausted or gap closed);
    anything less is reported explicitly.  A node whose relaxation fails
    numerically is counted in ``nodes_failed`` (and never claimed infeasible);
    a node pruned by a Phase-I infeasibility certificate is counted in
    ``nodes_infeasible``.
    """
    if integer_mask is None:
        integer_mask = getattr(lp, "is_integer", None)
    if integer_mask is None:
        raise MilpError(
            "solve_milp requires an explicit integer_mask or lp.is_integer "
            "(MPS integer markers are parser M2 work)"
        )
    mask = np.asarray(integer_mask, dtype=bool)
    if mask.shape != (lp.num_vars,):
        raise MilpError(f"integer_mask shape {mask.shape} != num_vars {lp.num_vars}")

    n0 = lp.num_vars
    lb0 = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub0 = np.asarray(lp.upper_bounds, dtype=np.float64)
    for i in np.flatnonzero(mask):
        if not np.isfinite(ub0[i]):
            # Both children of a free/one-sided integer variable are infinitely
            # branching: reject loudly rather than branch forever.
            raise MilpError(
                f"v1 requires finite upper bounds on integer variables; "
                f"var {lp.var_names[i]} has upper bound {ub0[i]}"
            )
    if not np.all(np.isfinite(lb0[mask])) or np.any(lb0[mask] < 0.0):
        # 0/1 and bounded-single-side-from-zero are the common cases; other
        # lower bounds work but are left to the engineer's M2 tests.
        pass  # finite lower bounds are fine; infinite lower bounds are not
    for i in np.flatnonzero(mask):
        if not np.isfinite(lb0[i]):
            raise MilpError(
                f"v1 requires finite lower bounds on integer variables; "
                f"var {lp.var_names[i]} has lower bound {lb0[i]}"
            )

    # Mini-presolve: drop provably redundant rows so the dense Schur backend
    # gets a full-rank standard form (assignment-type models have dependent
    # equality rows by construction).
    lp_base = _drop_redundant_rows(lp)

    # Solve the minimization form internally: negate c for maximization.
    sign = 1.0
    lp_rel = lp_base
    if maximize:
        sign = -1.0
        lp_rel = replace(lp_base, c=-np.asarray(lp.c, dtype=np.float64).copy())

    t0 = time.perf_counter()
    # Absolute global deadline: every expensive sub-solve (root, node
    # relaxation, heuristic repairs, strong branching, crossover fallback)
    # computes ``max(0, deadline - now)`` and caps its local budget with it,
    # so the global deadline always wins over local iteration/repair limits.
    deadline = (t0 + time_limit) if time_limit is not None else None
    # A may be dense or sparse; every consumer below uses matvec / row
    # extraction that handles both (no full densification).
    A_heuristic = lp_base.A
    b_dense = np.asarray(lp_base.b, dtype=np.float64)
    c_dense = np.asarray(lp_rel.c, dtype=np.float64)

    # Per-relaxation robustness budget: bounded IPM iterations and a wall-clock
    # budget for the sparse-simplex fallback (a node must never run an
    # effectively unbounded loop).  The same budget applies to the root.
    relax_time_limit = (node_solve_time_limit
                        if node_solve_time_limit is not None
                        else _NODE_RELAX_TIME_LIMIT)

    root = _relax(lp_rel, crossover_fallback=crossover_fallback, verbose=verbose,
                  max_iter=node_solve_max_iter, time_limit=relax_time_limit,
                  deadline=deadline)
    if root.status != "optimal":
        if root.status == "infeasible" or _root_infeasible(lp_rel,
                                                           deadline=deadline):
            return MilpResult(
                status="infeasible",
                message="LP relaxation proven infeasible (Phase I certificate)",
                objective=None, x=None, best_bound=None, gap=None,
                nodes_explored=0, node_limit=node_limit, lp_solves=1,
                time_sec=time.perf_counter() - t0,
            )
        return MilpResult(
            status=f"lp_status:{root.status}",
            message=f"root LP relaxation failed: {root.message}",
            objective=None,
            x=None,
            best_bound=None,
            gap=None,
            nodes_explored=0,
            node_limit=node_limit,
            lp_solves=1,
            time_sec=time.perf_counter() - t0,
        )

    incumbent = float("inf")   # min-form incumbent
    best_x: Optional[np.ndarray] = None
    dropped = 0                # nodes whose relaxation failed numerically
    nodes_infeasible = 0       # nodes pruned by a Phase-I infeasibility certificate
    failed_statuses: list[str] = []   # honest per-node failure statuses
    heap: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    seq = 1
    explored = 0
    solves = 1
    best_bound = float(root.objective)
    heuristic_stats = {"attempts": 0, "successes": 0, "lp_solves": 0,
                       "best_obj": None}

    # Root is already solved; branch from it directly.
    x_root = np.asarray(root.x, dtype=np.float64)
    frac = x_root[mask]
    viols = np.abs(frac - np.round(frac))
    max_viol = float(np.max(viols)) if viols.size else 0.0

    if max_viol <= int_tol:
        incumbent = float(root.objective)
        best_x = x_root.copy()
        if verbose:
            print(f"  ROOT INCUMBENT {sign * incumbent:.9g}")
    else:
        # ---- Rounding heuristic at root ----
        h_feas, h_x, h_obj = _rounding_heuristic(
            x_root, mask, lb0, ub0, A_heuristic, b_dense,
            lp_rel.row_types, c_dense, lp=lp_rel, int_tol=int_tol,
            stats=heuristic_stats, deadline=deadline,
            incumbent=incumbent,
        )
        if h_feas and h_obj < incumbent - 1e-12:
            incumbent = h_obj
            best_x = h_x.copy()
            if verbose:
                print(f"  ROOT HEURISTIC INCUMBENT {sign * incumbent:.9g}")

        # ---- Branch variable selection ----
        j = None
        if explored < strong_branch_depth:
            j = _strong_branch_select(
                lp_rel, x_root, mask, lb0, ub0,
                k=strong_branch_k, int_tol=int_tol,
                crossover_fallback=node_crossover_fallback,
                verbose=verbose,
                max_iter=node_solve_max_iter,
                time_limit=relax_time_limit,
                deadline=deadline,
            )
        # Fallback: most-fractional.
        if j is None:
            candidates = np.flatnonzero(mask)
            frac_pos = np.argmax(viols)
            j = candidates[frac_pos]
            cjabs = np.abs(np.asarray(lp.c, dtype=np.float64))
            tie = viols >= max_viol - 1e-12
            if np.count_nonzero(tie) > 1:
                best_t = np.argmax(np.where(tie, cjabs[candidates], -1.0))
                j = candidates[int(best_t)]

        fj = x_root[j]
        lo = int(np.floor(fj))
        hi = int(np.ceil(fj))

        for cub_j, clb_j in ((lo, -float("inf")), (float("inf"), hi)):
            clb = lb0.copy()
            cub = ub0.copy()
            if np.isfinite(cub_j):
                cub[j] = min(ub0[j], float(cub_j))
            if np.isfinite(clb_j):
                clb[j] = max(lb0[j], float(clb_j))
            if np.all(clb <= cub + 1e-12):
                heapq.heappush(heap, (float(root.objective), seq, clb, cub))
                seq += 1

    def _gap() -> Optional[float]:
        if best_x is None:
            return None
        return _relative_gap(incumbent, best_bound)

    while heap:
        if explored >= node_limit:
            break
        # Global deadline: absolute check so a node's expensive work that ran
        # past the limit is never followed by another full solve cycle.
        if deadline is not None and time.perf_counter() >= deadline:
            break

        parent_bound, _, lb, ub = heapq.heappop(heap)
        gap_abs = gap_tol * (1.0 + abs(incumbent))
        explored += 1
        if best_x is not None and parent_bound >= incumbent - gap_abs:
            if verbose:
                print(f"  node {explored - 1}: bound={parent_bound:.9g} pruned")
            continue  # cannot beat the incumbent

        node = replace(lp_rel, lower_bounds=lb, upper_bounds=ub)
        rel = _relax(node, crossover_fallback=node_crossover_fallback, verbose=verbose,
                     max_iter=node_solve_max_iter, time_limit=relax_time_limit,
                     deadline=deadline)
        solves += 1
        if verbose:
            print(f"  node {explored - 1}: parent_bound={parent_bound:.9g} status={rel.status}")
        # Honesty: a Phase-I infeasibility certificate prunes the node; ANY
        # other non-optimal outcome is a failure and is never trusted, never
        # conflated with infeasibility.
        if rel.status == "infeasible":
            nodes_infeasible += 1
            if verbose:
                print(f"  node {explored - 1}: INFEASIBLE (pruned by Phase-I certificate)")
            continue
        if rel.status != "optimal":
            dropped += 1
            failed_statuses.append(rel.status)
            if verbose:
                print(f"  node {explored - 1}: relaxation failed ({rel.status}): "
                      f"{rel.message[:100]}")
            continue

        x = np.asarray(rel.x, dtype=np.float64)
        child_bound = float(rel.objective)
        frac = x[mask]
        viols = np.abs(frac - np.round(frac))
        max_viol = float(np.max(viols)) if viols.size else 0.0

        if max_viol <= int_tol:
            val = float(rel.objective)  # min-form
            if val < incumbent - 1e-12:
                incumbent = val
                best_x = x.copy()
                if verbose:
                    print(f"  NEW INCUMBENT {sign * val:.9g}")
            # Node fully solved: no children.
        else:
            # ---- Periodic rounding heuristic ----
            if heuristic_freq > 0 and explored % heuristic_freq == 0:
                h_feas, h_x, h_obj = _rounding_heuristic(
                    x, mask, lb, ub, A_heuristic, b_dense,
                    lp_rel.row_types, c_dense, lp=lp_rel, int_tol=int_tol,
                    stats=heuristic_stats, deadline=deadline,
                    incumbent=incumbent,
                )
                if h_feas and h_obj < incumbent - 1e-12:
                    incumbent = h_obj
                    best_x = h_x.copy()
                    if verbose:
                        print(f"  HEURISTIC INCUMBENT {sign * incumbent:.9g} "
                              f"at node {explored - 1}")

            # ---- Branch variable selection ----
            j = None
            if explored < strong_branch_depth:
                j = _strong_branch_select(
                    node, x, mask, lb, ub,
                    k=strong_branch_k, int_tol=int_tol,
                    crossover_fallback=node_crossover_fallback,
                    verbose=verbose,
                    max_iter=node_solve_max_iter,
                    time_limit=relax_time_limit,
                    deadline=deadline,
                )
            # Fallback: most-fractional (ties broken by |c| then index).
            if j is None:
                candidates = np.flatnonzero(mask)
                frac_pos = np.argmax(viols)
                j = candidates[frac_pos]
                cjabs = np.abs(np.asarray(lp.c, dtype=np.float64))
                tie = viols >= max_viol - 1e-12
                if np.count_nonzero(tie) > 1:
                    best_t = np.argmax(
                        np.where(tie, cjabs[candidates], -1.0)
                    )
                    j = candidates[int(best_t)]

            fj = x[j]
            lo = int(np.floor(fj))
            hi = int(np.ceil(fj))

            # child A: x_j <= lo ; child B: x_j >= hi
            for cub_j, clb_j in ((lo, -float("inf")), (float("inf"), hi)):
                clb = lb.copy()
                cub = ub.copy()
                if np.isfinite(cub_j):
                    cub[j] = min(ub[j], float(cub_j))
                if np.isfinite(clb_j):
                    clb[j] = max(lb[j], float(clb_j))
                if np.all(clb <= cub + 1e-12):
                    heapq.heappush(heap, (child_bound, seq, clb, cub))
                    seq += 1

        best_bound = heap[0][0] if heap else (incumbent if dropped == 0 else best_bound)
        g = _gap()
        # Gap-close early exit is only trustworthy when no node was dropped:
        # a failed relaxation may conceal a better (or the only) solution, so
        # with dropped > 0 we keep exploring until the heap is honestly empty.
        if dropped == 0 and (g is not None and g <= gap_tol
                             and best_bound >= incumbent - gap_abs):
            break

    elapsed = time.perf_counter() - t0
    _h_best = heuristic_stats.get("best_obj")
    heuristic_best_objective = (
        None if _h_best is None else sign * float(_h_best)
    )
    g = _gap()
    exhausted = not heap
    if exhausted and best_x is not None and dropped == 0:
        # The tree region is empty and no node was dropped, so the relaxation
        # bound equals the incumbent: the gap is provably zero.
        best_bound = incumbent
        g = 0.0

    if best_x is None:
        if exhausted and dropped == 0:
            return MilpResult(
                status="infeasible",
                message="branch-and-bound exhausted the tree without an integer-feasible "
                        "incumbent (the LP relaxation was feasible)",
                objective=None, x=None, best_bound=sign * best_bound, gap=None,
                nodes_explored=explored, node_limit=node_limit, lp_solves=solves,
                time_sec=elapsed,
                heuristic_attempts=heuristic_stats["attempts"],
                heuristic_successes=heuristic_stats["successes"],
                heuristic_lp_solves=heuristic_stats["lp_solves"],
                heuristic_best_objective=heuristic_best_objective,
                nodes_infeasible=nodes_infeasible, nodes_failed=0,
                node_failure_statuses=tuple(failed_statuses),
            )
        if exhausted:
            return MilpResult(
                status="stalled",
                message=(
                    f"tree exhausted without an incumbent; {dropped} node "
                    "relaxation(s) failed"
                ) + (
                    f" ({', '.join(failed_statuses)})" if failed_statuses else ""
                ),
                objective=None, x=None, best_bound=sign * best_bound, gap=None,
                nodes_explored=explored, node_limit=node_limit, lp_solves=solves,
                time_sec=elapsed,
                heuristic_attempts=heuristic_stats["attempts"],
                heuristic_successes=heuristic_stats["successes"],
                heuristic_lp_solves=heuristic_stats["lp_solves"],
                heuristic_best_objective=heuristic_best_objective,
                nodes_infeasible=nodes_infeasible, nodes_failed=dropped,
                node_failure_statuses=tuple(failed_statuses),
            )
        return MilpResult(
            status="node_limit" if explored >= node_limit else "time_limit",
            message=("stopped by limit without an incumbent"
                     + (f"; {dropped} node relaxation(s) failed "
                        f"({', '.join(failed_statuses)})" if dropped else "")),
            objective=None, x=None, best_bound=sign * best_bound, gap=None,
            nodes_explored=explored, node_limit=node_limit, lp_solves=solves,
            time_sec=elapsed,
            heuristic_attempts=heuristic_stats["attempts"],
            heuristic_successes=heuristic_stats["successes"],
            heuristic_lp_solves=heuristic_stats["lp_solves"],
            heuristic_best_objective=heuristic_best_objective,
            nodes_infeasible=nodes_infeasible, nodes_failed=dropped,
            node_failure_statuses=tuple(failed_statuses),
        )

    # With dropped nodes the tree was not fully resolved, so the result can
    # never be claimed optimal even if the (stale) bound suggests gap closure.
    if dropped == 0:
        proven = exhausted or (g is not None and g <= gap_tol
                               and best_bound >= incumbent - gap_abs)
    else:
        proven = False
    if proven:
        status, message = "optimal", "proven optimal by branch-and-bound"
    elif exhausted:
        status = "stalled"
        message = (f"tree exhausted; {dropped} node relaxation(s) failed "
                   f"({', '.join(failed_statuses)}); optimality not claimed")
    else:
        status = "node_limit" if explored >= node_limit else "time_limit"
        message = f"terminated by limit with gap {g:.3g}"
        if dropped:
            message += (f"; {dropped} node relaxation(s) failed "
                        f"({', '.join(failed_statuses)})")
    return MilpResult(
        status=status,
        message=message,
        objective=sign * incumbent,
        x=best_x,
        best_bound=sign * best_bound,
        gap=g if best_x is not None else None,
        nodes_explored=explored,
        node_limit=node_limit,
        lp_solves=solves,
        time_sec=elapsed,
        heuristic_attempts=heuristic_stats["attempts"],
        heuristic_successes=heuristic_stats["successes"],
        heuristic_lp_solves=heuristic_stats["lp_solves"],
        heuristic_best_objective=heuristic_best_objective,
        nodes_infeasible=nodes_infeasible,
        nodes_failed=dropped,
        node_failure_statuses=tuple(failed_statuses),
    )