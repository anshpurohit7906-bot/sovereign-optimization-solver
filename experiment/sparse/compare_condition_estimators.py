"""EXPERIMENT: Compare dense cond2 vs sparse Hager/Higham cond1 estimator.

Goal: Determine whether a sparse 1-norm condition estimator can replace
the dense 2-norm cond2_estimate(B) -> B.toarray() -> np.linalg.cond()
in Phase II without changing the condition-gate decisions.

The estimator uses ONLY sparse LU factorization and triangular solves:
    lu.solve(x)          -- forward solve
    lu.solve(x, trans='T')  -- backward solve (transpose)

The dense reference (cond2_exact) is used ONLY for ground-truth comparison
and is never used by the estimator path.

Outputs:
    results/compare_condition_estimators.csv
    results/compare_condition_estimators.md
"""
from __future__ import annotations

import os
import sys
import time
import csv
import tracemalloc

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"),
           os.path.join(_HERE, "..", "crossover")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import load_numeric_mps   # noqa: E402
from mehrotra import to_standard_form           # noqa: E402
from scaling import scale_lp                    # noqa: E402
from sparse_phase1 import sparse_phase1         # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "compare_condition_estimators.csv")
MD_PATH = os.path.join(RESULTS_DIR, "compare_condition_estimators.md")
CAPTURE_CACHE_DIR = os.path.join(_ROOT, "scratch", "cond_est_captures")

SAFETY_FACTOR = 50.0  # matches p87_phase2_v2.py
EPS = np.finfo(np.float64).eps


# ======================================================================
# Capture caching  (saves/loads Phase II captured bases to avoid re-run)
# ======================================================================

def save_captures_to_cache(captures, model_name):
    """Save captured bases to scratch/ for reuse on next run."""
    model_dir = os.path.join(CAPTURE_CACHE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    for cap in captures:
        it = cap["iteration"]
        sp.save_npz(os.path.join(model_dir, f"cap_{it}_B.npz"), cap["B"])
        np.savez(os.path.join(model_dir, f"cap_{it}_meta.npz"),
                 x_basic=cap["x_basic"],
                 iteration=cap["iteration"],
                 fraction=cap["fraction"],
                 min_xB=cap.get("min_xB", cap["x_basic"].min()))
    # Save b vector (needed for gate simulation)
    np.save(os.path.join(model_dir, "_b.npy"),
            captures[0].get("_b_ref", np.zeros(0)))
    print(f"  [{model_name}] Saved {len(captures)} captures to cache",
          flush=True)


def load_captures_from_cache(model_name):
    """Load cached captured bases; returns list of dicts or None."""
    model_dir = os.path.join(CAPTURE_CACHE_DIR, model_name)
    if not os.path.isdir(model_dir):
        return None
    b_path = os.path.join(model_dir, "_b.npy")
    b_ref = np.load(b_path) if os.path.exists(b_path) else None
    captures = []
    # Find all meta files
    meta_files = sorted(f for f in os.listdir(model_dir)
                        if f.startswith("cap_") and f.endswith("_meta.npz"))
    for mf in meta_files:
        meta = np.load(os.path.join(model_dir, mf))
        B_path = os.path.join(model_dir, mf.replace("_meta.npz", "_B.npz"))
        if not os.path.exists(B_path):
            continue
        B = sp.load_npz(B_path)
        captures.append({
            "B": B,
            "x_basic": np.asarray(meta["x_basic"], float),
            "iteration": int(meta["iteration"]),
            "fraction": float(meta["fraction"]),
            "min_xB": float(meta.get("min_xB", meta["x_basic"].min())),
            "_b_ref": b_ref,
        })
    if captures:
        print(f"  [{model_name}] Loaded {len(captures)} captures from cache",
              flush=True)
    return captures if captures else None


# ======================================================================
# 1. Hager/Higham sparse 1-norm inverse estimator
# ======================================================================

def hager_higham_inv_norm1(lu, n, itmax=5):
    """Estimate ||B^{-1}||_1 using the Hager/Higham power iteration.

    Each iteration requires exactly 2 sparse triangular solves:
        z = lu.solve(x)              # B^{-1} x
        w = lu.solve(e_j, trans='T') # B^{-T} e_j

    The algorithm iteratively searches for the column of B^{-1} with
    the largest 1-norm (L1 operator norm).

    Parameters
    ----------
    lu : scipy.sparse.linalg.SuperLU
        Sparse LU factorization of B.
    n : int
        Matrix dimension.
    itmax : int
        Maximum power iterations (default 5; Hager/Higham prove convergence
        in at most n iterations, but in practice 3-5 suffice).

    Returns
    -------
    est : float
        Estimate of ||B^{-1}||_1 (1-norm condition number component).
    n_solves : int
        Number of triangular solves performed.
    """
    x = np.ones(n, dtype=np.float64) / n
    n_solves = 0

    for _it in range(itmax):
        # z = B^{-1} x
        try:
            z = lu.solve(x)
        except Exception:
            return np.nan, n_solves
        n_solves += 1

        est = np.sum(np.abs(z))
        if est < 1e-300:
            return 0.0, n_solves

        # Find index of largest |z_j|
        j = int(np.argmax(np.abs(z)))

        # e_j: unit vector
        e_j = np.zeros(n, dtype=np.float64)
        e_j[j] = 1.0

        # w = B^{-T} e_j  (backward solve)
        try:
            w = lu.solve(e_j, trans='T')
        except Exception:
            return est, n_solves + 1
        n_solves += 1

        # New iterate: sign(w), with zeros replaced by 1
        x_new = np.sign(w)
        if not np.any(x_new):
            x_new = np.ones(n, dtype=np.float64)

        # Convergence: same sign pattern as previous iterate
        if _it > 0 and np.array_equal(x_new, x):
            break
        x = x_new

    # Final evaluation with converged direction
    try:
        z = lu.solve(x)
        n_solves += 1
        est = np.sum(np.abs(z))
    except Exception:
        pass
    return est, n_solves


def norm1_sparse(B):
    """Compute ||B||_1 (max column 1-norm) directly from sparse B.

    Uses scipy.sparse.linalg.norm which operates on the sparse structure
    without densification.
    """
    return float(sp.linalg.norm(B, ord=1))


def sparse_cond1_estimate(B, lu_override=None):
    """Compute cond1_est = ||B||_1 * est ||B^{-1}||_1, fully sparse.

    Parameters
    ----------
    B : sparse CSC matrix
    lu_override : SuperLU, optional
        Pre-computed LU factorization (avoids re-factoring).

    Returns
    -------
    cond1_est : float
    norm_B_1 : float
    est_inv_norm1 : float
    n_solves : int
    """
    n = B.shape[0]
    norm_B1 = norm1_sparse(B)

    if lu_override is not None:
        lu = lu_override
    else:
        lu = splu(B)

    est_inv, n_solves = hager_higham_inv_norm1(lu, n)
    cond1_est = norm_B1 * est_inv if np.isfinite(est_inv) else np.nan
    return cond1_est, norm_B1, est_inv, n_solves


def dense_cond2_reference(B):
    """Compute exact cond2 via dense SVD -- REFERENCE ONLY.

    This is the expensive operation we aim to replace.  It is allowed
    inside this experiment for ground-truth comparison.
    """
    Bd = B.toarray()
    return float(np.linalg.cond(Bd))


# ======================================================================
# 2. Small-matrix validation (dense -- allowed here)
# ======================================================================

def _validate_on_small_matrices():
    """Validate the Hager/Higham estimator against exact cond1 and cond2
    on small dense test matrices where all norms can be computed exactly.
    """
    results = []
    rng = np.random.RandomState(42)

    # --- Well-conditioned random ---
    A = rng.randn(50, 50)
    cond2 = np.linalg.cond(A, 2)
    cond1_exact = np.linalg.cond(A, 1)
    B = sp.csc_matrix(A)
    lu = splu(B)
    est_inv, ns = hager_higham_inv_norm1(lu, 50)
    norm_B1 = norm1_sparse(B)
    cond1_est = norm_B1 * est_inv
    results.append({
        "name": "random_well_conditioned", "n": 50,
        "cond2_exact": cond2, "cond1_exact": cond1_exact,
        "cond1_est": cond1_est,
        "ratio_est_exact": cond1_est / cond1_exact if cond1_exact > 0 else np.nan,
        "ratio_est_cond2": cond1_est / cond2 if cond2 > 0 else np.nan,
        "n_solves": ns,
    })

    # --- Hilbert matrix (ill-conditioned) ---
    n_h = 20
    A_h = np.array([[1.0 / (i + j + 1) for j in range(n_h)] for i in range(n_h)])
    cond2 = np.linalg.cond(A_h, 2)
    cond1_exact = np.linalg.cond(A_h, 1)
    B = sp.csc_matrix(A_h)
    lu = splu(B)
    est_inv, ns = hager_higham_inv_norm1(lu, n_h)
    norm_B1 = norm1_sparse(B)
    cond1_est = norm_B1 * est_inv
    results.append({
        "name": "hilbert_ill_conditioned", "n": n_h,
        "cond2_exact": cond2, "cond1_exact": cond1_exact,
        "cond1_est": cond1_est,
        "ratio_est_exact": cond1_est / cond1_exact if cond1_exact > 0 else np.nan,
        "ratio_est_cond2": cond1_est / cond2 if cond2 > 0 else np.nan,
        "n_solves": ns,
    })

    # --- Diagonal ---
    d = np.array([1.0, 10.0, 100.0, 1000.0, 0.001])
    A_d = np.diag(d)
    cond2 = np.linalg.cond(A_d, 2)
    cond1_exact = np.linalg.cond(A_d, 1)
    B = sp.csc_matrix(A_d)
    lu = splu(B)
    est_inv, ns = hager_higham_inv_norm1(lu, len(d))
    norm_B1 = norm1_sparse(B)
    cond1_est = norm_B1 * est_inv
    results.append({
        "name": "diagonal", "n": len(d),
        "cond2_exact": cond2, "cond1_exact": cond1_exact,
        "cond1_est": cond1_est,
        "ratio_est_exact": cond1_est / cond1_exact if cond1_exact > 0 else np.nan,
        "ratio_est_cond2": cond1_est / cond2 if cond2 > 0 else np.nan,
        "n_solves": ns,
    })

    # --- Near-singular ---
    A_ns = rng.randn(30, 30)
    U, _, Vt = np.linalg.svd(A_ns)
    sigmas = np.linspace(1.0, 1e-10, 30)
    A_ns = U @ np.diag(sigmas) @ Vt
    cond2 = np.linalg.cond(A_ns, 2)
    cond1_exact = np.linalg.cond(A_ns, 1)
    B = sp.csc_matrix(A_ns)
    lu = splu(B)
    est_inv, ns = hager_higham_inv_norm1(lu, 30)
    norm_B1 = norm1_sparse(B)
    cond1_est = norm_B1 * est_inv
    results.append({
        "name": "near_singular", "n": 30,
        "cond2_exact": cond2, "cond1_exact": cond1_exact,
        "cond1_est": cond1_est,
        "ratio_est_exact": cond1_est / cond1_exact if cond1_exact > 0 else np.nan,
        "ratio_est_cond2": cond1_est / cond2 if cond2 > 0 else np.nan,
        "n_solves": ns,
    })

    # --- Identity (condition = 1) ---
    n_id = 40
    A_id = np.eye(n_id)
    cond2 = np.linalg.cond(A_id, 2)
    cond1_exact = np.linalg.cond(A_id, 1)
    B = sp.csc_matrix(A_id)
    lu = splu(B)
    est_inv, ns = hager_higham_inv_norm1(lu, n_id)
    norm_B1 = norm1_sparse(B)
    cond1_est = norm_B1 * est_inv
    results.append({
        "name": "identity", "n": n_id,
        "cond2_exact": cond2, "cond1_exact": cond1_exact,
        "cond1_est": cond1_est,
        "ratio_est_exact": cond1_est / cond1_exact if cond1_exact > 0 else np.nan,
        "ratio_est_cond2": cond1_est / cond2 if cond2 > 0 else np.nan,
        "n_solves": ns,
    })

    # --- Upper triangular with varying diagonals ---
    n_ut = 40
    A_ut = np.triu(rng.randn(n_ut, n_ut))
    np.fill_diagonal(A_ut, np.abs(np.diag(A_ut)) + 1e-3)
    cond2 = np.linalg.cond(A_ut, 2)
    cond1_exact = np.linalg.cond(A_ut, 1)
    B = sp.csc_matrix(A_ut)
    lu = splu(B)
    est_inv, ns = hager_higham_inv_norm1(lu, n_ut)
    norm_B1 = norm1_sparse(B)
    cond1_est = norm_B1 * est_inv
    results.append({
        "name": "upper_triangular", "n": n_ut,
        "cond2_exact": cond2, "cond1_exact": cond1_exact,
        "cond1_est": cond1_est,
        "ratio_est_exact": cond1_est / cond1_exact if cond1_exact > 0 else np.nan,
        "ratio_est_cond2": cond1_est / cond2 if cond2 > 0 else np.nan,
        "n_solves": ns,
    })

    return results


# ======================================================================
# 3. Basis capture during Phase II
# ======================================================================

def _clean_zero(v, tol=1e-9):
    x = np.asarray(v, dtype=np.float64).copy()
    x[np.abs(x) < tol] = 0.0
    return x


def _refine_solve(B, lu, rhs, n_refine=3, tighten=1e-9):
    """Solve B x = rhs with iterative residual refinement (sparse)."""
    x = lu.solve(rhs)
    resid = B @ x - rhs
    rnorm = float(np.max(np.abs(resid)))
    for _ in range(n_refine):
        if rnorm < tighten:
            break
        try:
            resid_dense = resid.toarray().ravel() if hasattr(resid, "toarray") else np.asarray(resid).ravel()
            dc = lu.solve(resid_dense)
        except Exception:
            break
        cand = x + dc
        rc = B @ cand - rhs
        rn = float(np.max(np.abs(rc)))
        if rn < rnorm:
            x, resid, rnorm = cand, rc, rn
        else:
            break
    return x


def _try_factorize(B):
    """Try sparse LU factorization with fallback."""
    try:
        return splu(B)
    except RuntimeError:
        pass
    try:
        return splu(B, permc_spec='MMD_AT_PLUS_A')
    except RuntimeError:
        pass
    return None


def run_phase2_capture_bases(A, b, c, basis0, *,
                              capture_fracs=(0.0, 0.25, 0.50, 0.75, 1.0),
                              max_iter=50000, tol=1e-7, piv_tol=1e-9,
                              verbose=5000, label=""):
    """Run sparse Phase II and capture basis matrices at specified fractions.

    Returns
    -------
    captures : list of dict
        Each dict has keys: fraction, iteration, B (sparse), basis,
        x_basic, objective
    final_status : str
    final_info : dict
    """
    A = A.tocsc()
    m, n = A.shape
    basis = list(basis0)

    B = A[:, basis].tocsc()
    lu = _try_factorize(B)
    if lu is None:
        return [], "factorization_failure", {"err": "initial factorize failed"}

    x_basic = _refine_solve(B, lu, b)
    basis_set = set(basis)
    nonbasic = np.array([j for j in range(n) if j not in basis_set], dtype=np.intp)
    objective = float(c[basis] @ x_basic)

    total_budget = max_iter
    capture_iters = set()
    for f in capture_fracs:
        capture_iters.add(int(f * total_budget))

    captures = []
    degenerate_run = 0
    MAX_DEGENERATE = 50
    bland_active = False
    iters_since_refactor = 0
    REFACTOR_INTERVAL = 1

    def _capture_if_needed(it):
        if it in capture_iters:
            B_snap = A[:, basis].tocsc()
            captures.append({
                "fraction": it / total_budget if total_budget > 0 else 0,
                "iteration": it,
                "B": B_snap.copy(),
                "basis": list(basis),
                "x_basic": x_basic.copy(),
                "objective": objective,
            })

    _capture_if_needed(0)
    t0 = time.perf_counter()

    for iteration in range(1, total_budget + 1):
        # Check feasibility
        if np.any(x_basic < -tol):
            from sparse_phase1 import sparse_phase1
            if verbose and iteration % verbose == 0:
                print(f"  [{label}] it={iteration}: feasibility lost "
                      f"(neg={(x_basic < -tol).sum()}, min_xB={x_basic.min():.3e})",
                      flush=True)
            basis, r_its, r_status, r_info = sparse_phase1(
                A, b, basis, max_iter=2_000_000, verbose=1 << 30)
            if r_status != "feasible":
                return captures, "repair_failed", {"iter": iteration, "r_status": r_status}
            B = A[:, basis].tocsc()
            lu = _try_factorize(B)
            if lu is None:
                return captures, "factorization_failure", {"iter": iteration}
            x_basic = _refine_solve(B, lu, b)
            basis_set = set(basis)
            nonbasic = np.array([j for j in range(n) if j not in basis_set],
                                dtype=np.intp)
            objective = float(c[basis] @ x_basic)
            degenerate_run = 0
            bland_active = False
            _capture_if_needed(iteration)
            continue

        x_basic = _clean_zero(x_basic, tol)
        c_basic = c[basis]

        # Dual solve
        try:
            y = lu.solve(c_basic, trans='T')
        except Exception:
            y = np.linalg.solve(B.T.toarray(), c_basic)

        A_nb = A[:, nonbasic]
        reduced = c[nonbasic] - A_nb.T @ y
        reduced = _clean_zero(reduced, tol)
        objective = float(c_basic @ x_basic)

        # Optimality check
        neg_mask = reduced < -tol
        if not np.any(neg_mask):
            _capture_if_needed(iteration)
            if verbose:
                print(f"  [{label}] OPTIMAL at it={iteration}: obj={objective:.6f}",
                      flush=True)
            return captures, "optimal", {"iter": iteration, "obj": objective}

        # Entering variable
        if degenerate_run >= MAX_DEGENERATE and not bland_active:
            bland_active = True
            if verbose:
                print(f"  [{label}] it={iteration}: BLAND activated", flush=True)

        if bland_active:
            entering = int(np.min(nonbasic[neg_mask]))
        else:
            neg_indices = np.where(neg_mask)[0]
            best = int(neg_indices[np.argmin(reduced[neg_mask])])
            entering = int(nonbasic[best])

        a_ent = A[:, entering].toarray().ravel()
        d = _clean_zero(_refine_solve(B, lu, a_ent), piv_tol)

        positive = d > piv_tol
        if not np.any(positive):
            return captures, "unbounded", {"iter": iteration, "col": entering}

        ratios = np.full(m, np.inf)
        ratios[positive] = x_basic[positive] / d[positive]
        theta = float(np.min(ratios))
        leaving_cand = np.flatnonzero(
            np.abs(ratios - theta) <= piv_tol * max(1.0, abs(theta)))
        leaving_idx = int(leaving_cand[np.argmin([basis[i] for i in leaving_cand])])

        # Basis update
        basis[leaving_idx] = entering
        basis_set = set(basis)
        nonbasic = np.array([j for j in range(n) if j not in basis_set],
                            dtype=np.intp)

        x_basic = x_basic - theta * d
        x_basic[leaving_idx] = theta

        if theta <= piv_tol:
            degenerate_run += 1
        else:
            degenerate_run = 0
            if bland_active:
                bland_active = False

        # Refactorize every pivot (PILOT87 requires this for stability)
        iters_since_refactor += 1
        if iters_since_refactor >= REFACTOR_INTERVAL:
            B = A[:, basis].tocsc()
            lu = _try_factorize(B)
            if lu is None:
                return captures, "factorization_failure", {"iter": iteration}
            iters_since_refactor = 0
            x_basic = _refine_solve(B, lu, b)

        _capture_if_needed(iteration)

        if verbose and iteration % verbose == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{label}] it={iteration} obj={objective:.6f} "
                  f"min_rc={reduced.min():.3e} "
                  f"neg_rc={int(neg_mask.sum())} "
                  f"elapsed={elapsed:.1f}s", flush=True)

    return captures, "max_iterations", {"iter": total_budget}


# ======================================================================
# 4. Condition-gate decision equivalence simulation
# ======================================================================

def simulate_gate_decision(cond_val, b_norm, min_xB, tol=1e-7):
    """Simulate the Phase II condition-gate logic from p87_phase2_v2.py.

    Returns
    -------
    decision : str  ('no_action', 'soft_clamp', 'repair')
    eff_tol : float
    """
    if min_xB >= -tol:
        return "no_action", tol

    if not np.isfinite(cond_val) or cond_val <= 0:
        eff_tol = tol * 1000.0
    else:
        eff_tol = cond_val * EPS * b_norm * SAFETY_FACTOR
        eff_tol = max(eff_tol, tol)

    if min_xB >= -eff_tol:
        return "soft_clamp", eff_tol
    else:
        return "repair", eff_tol


# ======================================================================
# 5. Main experiment runner
# ======================================================================

def load_model(name):
    """Load, standardize, and scale an MPS model. Returns (A, b, c)."""
    # Check for pre-prepared artifacts (from p87_prepare.py)
    prep_path = os.path.join(_ROOT, "artifacts", "pilot87", "p87_prepared.npz")
    prep_A_path = os.path.join(_ROOT, "artifacts", "pilot87", "p87_prepared_A.npz")
    if name == "pilot87" and os.path.exists(prep_path) and os.path.exists(prep_A_path):
        prep = np.load(prep_path, allow_pickle=True)
        A = sp.load_npz(prep_A_path)
        b = np.asarray(prep["b"], float)
        c = np.asarray(prep["c"], float)
        m, n = A.shape
        return A, b, c, m, n

    # Standard path: load, standardize, scale
    mps_path = os.path.join(_ROOT, "data", f"{name}.mps")
    sf = to_standard_form(load_numeric_mps(mps_path, sparse=True))
    A0 = sf.A
    if not sp.issparse(A0):
        A0 = sp.csc_matrix(A0)
    b0 = np.asarray(sf.b, float)
    c0 = np.asarray(sf.c_min, float)
    m, n = A0.shape

    S = scale_lp(A0, b0, c0, np.zeros(n), np.full(n, np.inf))
    A = sp.csc_matrix(S.A)
    b = np.asarray(S.b, float)
    c = np.asarray(S.c, float)
    return A, b, c, m, n


def run_phase1_and_capture(A, b, c, model_name, capture_fracs,
                            max_phase2_iter):
    """Run Phase I + Phase II and capture bases.

    For PILOT87, uses the pre-prepared Phase I basis from the prepared
    artifacts when available.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # For PILOT87, check for pre-prepared basis from p87_prepare.py
    prep_path = os.path.join(_ROOT, "artifacts", "pilot87", "p87_prepared.npz")
    basis0 = None
    if model_name == "pilot87" and os.path.exists(prep_path):
        prep = np.load(prep_path, allow_pickle=True)
        basis0 = list(np.asarray(prep["basis"], int))
        print(f"  [{model_name}] Using pre-prepared Phase I basis "
              f"(len={len(basis0)})", flush=True)

    if basis0 is None:
        # Run sparse Phase I
        print(f"  [{model_name}] Running sparse Phase I...", flush=True)
        t0 = time.perf_counter()
        basis0, p1_iters, p1_status, p1_info = sparse_phase1(
            A, b, max_iter=2_000_000, verbose=100000)
        t_p1 = time.perf_counter() - t0
        print(f"  [{model_name}] Phase I: iters={p1_iters} status={p1_status} "
              f"t={t_p1:.1f}s", flush=True)

        if p1_status != "feasible":
            print(f"  [{model_name}] Phase I FAILED: {p1_status}", flush=True)
            return []
    else:
        # Verify the prepared basis is feasible
        B = A[:, basis0].tocsc()
        lu = _try_factorize(B)
        if lu is None:
            print(f"  [{model_name}] Prepared basis factorization failed", flush=True)
            return []
        xB = _refine_solve(B, lu, b)
        neg = int((xB < -1e-7).sum())
        minxB = float(xB.min())
        print(f"  [{model_name}] Prepared basis: neg={neg} min_xB={minxB:.3e}",
              flush=True)

    # Phase II with capture
    # Check for cached captures first
    cached = load_captures_from_cache(model_name)
    if cached is not None and len(cached) >= len(capture_fracs):
        print(f"  [{model_name}] Using cached captures (skipping Phase II)",
              flush=True)
        return cached

    print(f"  [{model_name}] Running Phase II with basis capture...", flush=True)
    t0 = time.perf_counter()
    captures, status, info = run_phase2_capture_bases(
        A, b, c, basis0,
        capture_fracs=capture_fracs,
        max_iter=max_phase2_iter,
        verbose=max(5000, max_phase2_iter // 10),
        label=model_name,
    )
    t_p2 = time.perf_counter() - t0
    print(f"  [{model_name}] Phase II: captured {len(captures)} bases "
          f"status={status} t={t_p2:.1f}s", flush=True)

    for cap in captures:
        B = cap["B"]
        xB = cap["x_basic"]
        neg = int((xB < -1e-7).sum())
        minxB = float(xB.min())
        cap["neg_basics"] = neg
        cap["min_xB"] = minxB

    # Save captures to cache for future reuse
    for cap in captures:
        cap["_b_ref"] = b
    save_captures_to_cache(captures, model_name)

    return captures


def compare_estimators_on_bases(captures, model_name, b):
    """For each captured basis, compute cond2_exact vs cond1_est."""
    b_norm = float(np.max(np.abs(b)))
    rows = []

    for cap in captures:
        B = cap["B"]
        m = B.shape[0]
        iteration = cap["iteration"]
        fraction = cap["fraction"]
        min_xB = cap["min_xB"]

        print(f"  [{model_name}] it={iteration} ({fraction:.0%}) "
              f"nnz={B.nnz:,} m={m}...", end="", flush=True)

        # --- Sparse path (no densification) ---
        t0 = time.perf_counter()
        lu = _try_factorize(B)
        t_factor = time.perf_counter() - t0
        if lu is None:
            print(" SKIP (factorize failed)")
            continue

        norm_B1 = norm1_sparse(B)

        t0 = time.perf_counter()
        est_inv, n_solves = hager_higham_inv_norm1(lu, m)
        t_est = time.perf_counter() - t0
        cond1_est = norm_B1 * est_inv if np.isfinite(est_inv) else np.nan

        # --- Dense reference (allowed in experiment) ---
        t0 = time.perf_counter()
        cond2_exact = dense_cond2_reference(B)
        t_dense = time.perf_counter() - t0

        # --- Metrics ---
        ratio = cond1_est / cond2_exact if (np.isfinite(cond1_est) and
                                              cond2_exact > 0) else np.nan
        log_cond2 = np.log10(cond2_exact) if cond2_exact > 0 else np.nan
        log_cond1 = np.log10(cond1_est) if (np.isfinite(cond1_est) and
                                              cond1_est > 0) else np.nan

        # --- Gate simulation ---
        dec_exact, eff_exact = simulate_gate_decision(cond2_exact, b_norm, min_xB)
        dec_est, eff_est = simulate_gate_decision(cond1_est, b_norm, min_xB)
        decisions_match = (dec_exact == dec_est)

        row = {
            "model": model_name,
            "iteration": iteration,
            "fraction": f"{fraction:.0%}",
            "nnz_B": B.nnz,
            "m": m,
            "norm_B_1": norm_B1,
            "cond2_exact": cond2_exact,
            "cond1_est": cond1_est,
            "ratio_cond1_cond2": ratio,
            "log10_cond2": log_cond2,
            "log10_cond1_est": log_cond1,
            "estimator_solves": n_solves,
            "estimator_runtime_s": t_est,
            "factorize_runtime_s": t_factor,
            "dense_cond2_runtime_s": t_dense,
            "min_xB": min_xB,
            "eff_tol_exact": eff_exact,
            "eff_tol_est": eff_est,
            "decision_exact": dec_exact,
            "decision_est": dec_est,
            "decisions_match": decisions_match,
        }
        rows.append(row)

        print(f"  cond2={cond2_exact:.3e}  cond1_est={cond1_est:.3e}  "
              f"ratio={ratio:.2f}  "
              f"gate_exact={dec_exact}  gate_est={dec_est}  "
              f"{'OK' if decisions_match else 'MISMATCH'}")

    return rows


def write_results(all_rows, small_rows):
    """Write CSV and markdown report."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- CSV ---
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nCSV saved: {CSV_PATH}", flush=True)

    # --- Markdown ---
    max_m = max(r["m"] for r in all_rows) if all_rows else 0
    dense_mb = max_m * max_m * 8 / 1e6

    with open(MD_PATH, "w") as f:
        f.write("# Sparse Condition Estimator vs Dense Cond2\n\n")
        f.write("**Date:** " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("## Objective\n\n")
        f.write("Replace the dense `cond2_estimate(B)` -> `B.toarray()` -> "
                "`np.linalg.cond()` in Phase II with a sparse Hager/Higham "
                "1-norm condition estimator that uses only sparse LU solves.\n\n")

        # --- Small matrix validation ---
        f.write("## A. Small-Matrix Validation (dense reference)\n\n")
        f.write("The estimator is validated on small dense matrices where exact "
                "cond1 and cond2 can be computed.\n\n")
        f.write("| Matrix | n | cond2 | cond1_exact | cond1_est | "
                "est/exact | est/cond2 | solves |\n")
        f.write("|--------|---|-------|-------------|-----------|"
                "---------|----------|--------|\n")
        for r in small_rows:
            f.write(f"| {r['name']} | {r['n']} | "
                    f"{r['cond2_exact']:.3e} | {r['cond1_exact']:.3e} | "
                    f"{r['cond1_est']:.3e} | "
                    f"{r['ratio_est_exact']:.3f} | "
                    f"{r['ratio_est_cond2']:.3f} | "
                    f"{r['n_solves']} |\n")
        f.write("\n")

        # --- Real basis comparison ---
        f.write("## B. Real Basis Comparison (PILOT4 & PILOT87)\n\n")
        f.write("Basis matrices captured during Phase II at representative "
                "points (0%, 25%, 50%, 75%, 100%).\n\n")

        for model in ["pilot4_plain", "pilot87"]:
            model_rows = [r for r in all_rows if r["model"] == model]
            if not model_rows:
                continue
            f.write(f"### {model.upper()}\n\n")
            f.write("| it | frac | nnz | cond2_exact | cond1_est | "
                    "ratio | log10(c2) | log10(c1) | "
                    "solves | est_time | dense_time |\n")
            f.write("|----|------|-----|-------------|-----------|"
                    "------|-----------|-----------|"
                    "-------|---------|------------|\n")
            for r in model_rows:
                f.write(f"| {r['iteration']} | {r['fraction']} | "
                        f"{r['nnz_B']:,} | "
                        f"{r['cond2_exact']:.3e} | "
                        f"{r['cond1_est']:.3e} | "
                        f"{r['ratio_cond1_cond2']:.3f} | "
                        f"{r['log10_cond2']:.2f} | "
                        f"{r['log10_cond1_est']:.2f} | "
                        f"{r['estimator_solves']} | "
                        f"{r['estimator_runtime_s']:.6f} | "
                        f"{r['dense_cond2_runtime_s']:.6f} |\n")
            f.write("\n")

        # --- Accuracy summary ---
        f.write("## C. Estimator Accuracy Summary\n\n")
        if all_rows:
            ratios = [r["ratio_cond1_cond2"] for r in all_rows
                      if np.isfinite(r["ratio_cond1_cond2"])]
            if ratios:
                f.write(f"- **ratio = cond1_est / cond2_exact**\n")
                f.write(f"  - min: {min(ratios):.4f}\n")
                f.write(f"  - max: {max(ratios):.4f}\n")
                f.write(f"  - mean: {np.mean(ratios):.4f}\n")
                f.write(f"  - median: {np.median(ratios):.4f}\n")
                f.write(f"  - cond1_est >= cond2_exact in "
                        f"{sum(1 for r in ratios if r >= 1)}/{len(ratios)} "
                        f"cases\n\n")

        # --- Runtime comparison ---
        f.write("## D. Runtime / Cost\n\n")
        if all_rows:
            est_times = [r["estimator_runtime_s"] for r in all_rows]
            dense_times = [r["dense_cond2_runtime_s"] for r in all_rows
                           if r["dense_cond2_runtime_s"] > 0]
            factor_times = [r["factorize_runtime_s"] for r in all_rows]
            if est_times and dense_times:
                speedup = np.mean(dense_times) / np.mean(est_times) \
                    if np.mean(est_times) > 0 else np.inf
                f.write(f"- **Estimator (sparse) mean time:** "
                        f"{np.mean(est_times)*1000:.3f} ms\n")
                f.write(f"- **Dense cond2 mean time:** "
                        f"{np.mean(dense_times)*1000:.3f} ms\n")
                f.write(f"- **LU factorization mean time:** "
                        f"{np.mean(factor_times)*1000:.3f} ms\n")
                f.write(f"- **Speedup (dense / estimator):** "
                        f"{speedup:.1f}x\n")
                f.write(f"- **Memory saved per call:** "
                        f"~{dense_mb:.0f} MB "
                        f"(dense m x m allocation eliminated)\n\n")
            solves = [r["estimator_solves"] for r in all_rows]
            f.write("- **Estimator solves per call:** "
                    f"{np.mean(solves):.1f} "
                    f"(all sparse triangular, no allocation)\n\n")

        # --- Gate decision equivalence ---
        f.write("## E. Condition-Gate Decision Equivalence\n\n")
        f.write("Simulating the Phase II feasibility gate from "
                "`p87_phase2_v2.py`:\n\n")
        f.write("```\n")
        f.write("if min_xB >= -tol:        -> no_action\n")
        f.write("elif min_xB >= -eff_tol:  -> soft_clamp (clamp to 0)\n")
        f.write("else:                     -> repair (Phase I restart)\n")
        f.write("```\n\n")
        f.write(f"where `eff_tol = kappa * eps * ||b|| * {SAFETY_FACTOR}`\n\n")

        if all_rows:
            n_match = sum(1 for r in all_rows if r["decisions_match"])
            n_total = len(all_rows)
            f.write(f"- **Decisions match:** {n_match}/{n_total}\n\n")

            mismatches = [r for r in all_rows if not r["decisions_match"]]
            if mismatches:
                f.write("### Decision Mismatches\n\n")
                f.write("| model | iter | min_xB | cond2_exact | "
                        "eff_tol_exact | dec_exact | "
                        "cond1_est | eff_tol_est | dec_est |\n")
                f.write("|-------|------|--------|-------------|"
                        "---------------|-----------|"
                        "----------|-------------|--------|\n")
                for r in mismatches:
                    f.write(f"| {r['model']} | {r['iteration']} | "
                            f"{r['min_xB']:.3e} | "
                            f"{r['cond2_exact']:.3e} | "
                            f"{r['eff_tol_exact']:.3e} | "
                            f"{r['decision_exact']} | "
                            f"{r['cond1_est']:.3e} | "
                            f"{r['eff_tol_est']:.3e} | "
                            f"{r['decision_est']} |\n")
                f.write("\n")
            else:
                f.write("**No decision mismatches.** "
                        "The sparse estimator would produce identical "
                        "gate decisions in all tested cases.\n\n")

            f.write("### Effective Tolerance Comparison\n\n")
            f.write("| model | iter | eff_tol_exact | eff_tol_est | "
                    "ratio |\n")
            f.write("|-------|------|---------------|-------------|"
                    "------|\n")
            for r in all_rows:
                et_ratio = (r["eff_tol_est"] / r["eff_tol_exact"]
                            if r["eff_tol_exact"] > 0 else np.nan)
                f.write(f"| {r['model']} | {r['iteration']} | "
                        f"{r['eff_tol_exact']:.3e} | "
                        f"{r['eff_tol_est']:.3e} | "
                        f"{et_ratio:.4f} |\n")
            f.write("\n")

        # --- Conclusion ---
        f.write("## F. Conclusion\n\n")
        if all_rows:
            f.write("### Estimator reliability for the safety gate\n\n")
            n_match = sum(1 for r in all_rows if r["decisions_match"])
            n_total = len(all_rows)
            f.write(f"- Gate decisions identical in **{n_match}/{n_total}** "
                    f"cases.\n")

            ratios = [r["ratio_cond1_cond2"] for r in all_rows
                      if np.isfinite(r["ratio_cond1_cond2"])]
            if ratios:
                f.write(f"- cond1_est/cond2 ratio: "
                        f"{min(ratios):.3f} -- {max(ratios):.3f} "
                        f"(mean {np.mean(ratios):.3f})\n")

            illcond_rows = [r for r in all_rows
                            if r["cond2_exact"] > 1e6]
            if illcond_rows:
                ill_ratios = [r["ratio_cond1_cond2"] for r in illcond_rows
                              if np.isfinite(r["ratio_cond1_cond2"])]
                if ill_ratios:
                    f.write(f"- For ill-conditioned bases (cond2 > 1e6): "
                            f"ratio = {min(ill_ratios):.3f} -- "
                            f"{max(ill_ratios):.3f}\n")

            est_times = [r["estimator_runtime_s"] for r in all_rows]
            dense_times = [r["dense_cond2_runtime_s"] for r in all_rows
                           if r["dense_cond2_runtime_s"] > 0]
            if est_times and dense_times:
                speedup = np.mean(dense_times) / np.mean(est_times) \
                    if np.mean(est_times) > 0 else np.inf
                f.write(f"\n### Runtime improvement\n\n")
                f.write(f"- Mean estimator time: "
                        f"{np.mean(est_times)*1000:.3f} ms\n")
                f.write(f"- Mean dense cond2 time: "
                        f"{np.mean(dense_times)*1000:.3f} ms\n")
                f.write(f"- Speedup: **{speedup:.1f}x**\n")
                f.write(f"- Dense 2D allocation per call: eliminated\n")

            f.write(f"\n### Dense 2D allocation elimination\n\n")
            f.write("- The estimator **never calls** `B.toarray()`, "
                    "`np.linalg.cond()`, or `np.linalg.inv()`.\n")
            f.write("- All computation uses sparse SuperLU factorization "
                    "and triangular solves.\n")
            f.write(f"- Memory: 0 MB dense allocation (vs ~{dense_mb:.0f} MB per "
                    f"cond2 call for m={max_m}).\n")

        f.write("\n---\n")
        f.write("*Generated by `experiment/sparse/"
                "compare_condition_estimators.py`*\n")

    print(f"Markdown saved: {MD_PATH}", flush=True)


# ======================================================================
# 6. Entry point
# ======================================================================

def main():
    print("=" * 60, flush=True)
    print("  SPARSE CONDITION ESTIMATOR COMPARISON", flush=True)
    print("=" * 60, flush=True)

    all_rows = []

    # --- Part 1: Small matrix validation ---
    print("\n--- Small-Matrix Validation ---", flush=True)
    small_rows = _validate_on_small_matrices()
    for r in small_rows:
        print(f"  {r['name']:30s}  cond2={r['cond2_exact']:.3e}  "
              f"cond1_est={r['cond1_est']:.3e}  "
              f"ratio={r['ratio_est_exact']:.3f}", flush=True)

    # --- Part 2: Real basis comparison ---
    # Use --models to select which models to process (default: all)
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Model names to process (default: all)")
    ap.add_argument("--no-phase1", action="store_true",
                    help="Skip Phase I for PILOT87 (use existing artifacts)")
    args, _ = ap.parse_known_args()

    models = [
        ("pilot4_plain", 3000),   # PILOT4: ~657x1428, typically <3000 Phase II iters
        ("pilot87", 25000),       # PILOT87: ~3608x8038, up to 22K Phase II iters
    ]
    if args.models is not None:
        models = [m for m in models if m[0] in args.models]

    capture_fracs = (0.0, 0.25, 0.50, 0.75, 1.0)

    for model_name, max_iter in models:
        try:
            A, b, c, m, n = load_model(model_name)
            print(f"  Loaded {model_name}: {m}x{n} nnz={A.nnz:,}", flush=True)

            captures = run_phase1_and_capture(
                A, b, c, model_name, capture_fracs, max_iter)

            if captures:
                rows = compare_estimators_on_bases(captures, model_name, b)
                all_rows.extend(rows)
        except Exception as exc:
            print(f"  ERROR on {model_name}: {exc}", flush=True)
            import traceback
            traceback.print_exc()

    # --- Part 3: Write results ---
    print(f"\n--- Writing results ---", flush=True)
    write_results(all_rows, small_rows)

    # --- Summary ---
    print("\n" + "=" * 60, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 60, flush=True)
    if all_rows:
        n_match = sum(1 for r in all_rows if r["decisions_match"])
        n_total = len(all_rows)
        ratios = [r["ratio_cond1_cond2"] for r in all_rows
                  if np.isfinite(r["ratio_cond1_cond2"])]
        est_times = [r["estimator_runtime_s"] for r in all_rows]
        dense_times = [r["dense_cond2_runtime_s"] for r in all_rows
                       if r["dense_cond2_runtime_s"] > 0]
        print(f"  Real basis tests:  {n_total}")
        print(f"  Gate match:        {n_match}/{n_total}")
        if ratios:
            print(f"  cond1/cond2 ratio: {min(ratios):.3f} -- {max(ratios):.3f} "
                  f"(mean {np.mean(ratios):.3f})")
        if est_times and dense_times:
            speedup = np.mean(dense_times) / np.mean(est_times) \
                if np.mean(est_times) > 0 else float('inf')
            print(f"  Estimator mean:    {np.mean(est_times)*1000:.3f} ms")
            print(f"  Dense cond2 mean:  {np.mean(dense_times)*1000:.3f} ms")
            print(f"  Speedup:           {speedup:.1f}x")
    print(f"  CSV:  {CSV_PATH}")
    print(f"  MD:   {MD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
