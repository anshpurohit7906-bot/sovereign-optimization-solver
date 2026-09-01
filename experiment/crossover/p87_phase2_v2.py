"""Phase II Revised Simplex for PILOT87 with condition-number-aware repair.

Principled improvement over v1 (instrumented):
  - Condition-number-aware feasibility gate: only trigger full Phase-I repair
    when infeasibility exceeds the backward error bound of the basis solve.
  - Soft repair (clamp): when a basic variable is marginally negative but
    within numerical noise (|x_B[i]| < kappa * eps * ||b|| * SAFETY), clamp
    to 0 and continue.  This avoids the catastrophic 100+ unit objective
    loss from unnecessary Phase-I restarts.
  - Diagnostic instrumentation preserved.

The root cause of v1's inefficiency was:
  Mechanism C -> D: A ~1e-7 negative basic (within the cond(B)*eps*||b||~1.4e-6
  error bound) triggered a full Phase-I restart that reset objective progress
  from 326 back to 431 (losing 105 units of objective).

OPENBLAS_NUM_THREADS=1 required.
"""
from __future__ import annotations
import os, sys, time, csv, hashlib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import scipy.linalg as sla

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Tolerances and limits
# ---------------------------------------------------------------------------
PIV_TOL = 1e-9          # minimum pivot ratio
TOL = 1e-7              # feasibility / reduced-cost tolerance
REFINE_TOL = 1e-9       # iterative-refinement tighten threshold
N_REFINE = 5            # more refinement rounds for better accuracy

MAX_ITER = 25000        # generous limit -- aim for optimality
MAX_DEGENERATE = 50     # Bland entering after this many consecutive deg. pivots
LOG_EVERY = 200         # log every N pivots
REPAIR_LIMIT = 30       # cap on full Phase-I repairs
SOFT_CLAMP_LIMIT = 100  # cap on soft clamps per region

# Condition-number-aware repair gate
SAFETY_FACTOR = 50.0    # kappa*eps*||b||*SAFETY is the effective tolerance

_ARTIFACTS = os.path.join(_ROOT, "artifacts", "pilot87")

CSV_PATH = os.path.join(_ARTIFACTS, "p87_phase2_v2_log.csv")
REPAIR_CSV = os.path.join(_ARTIFACTS, "p87_phase2_v2_repair_events.csv")
SOFT_CSV = os.path.join(_ARTIFACTS, "p87_phase2_v2_soft_events.csv")

COLS = ["iter", "obj", "obj_delta", "min_rc", "neg_rc", "min_xB", "neg_basics",
        "theta", "degenerate", "lu_time_s", "nnz_L", "nnz_U", "xB_resid", "bh",
        "repair", "soft_clamp", "eff_tol"]
REPCOLS = ["iter", "obj_before", "obj_before_neg_basics", "obj_before_min_rc",
           "obj_before_cond2", "neg_xB", "min_xB", "repair_iters",
           "repair_status", "obj_after", "neg_xB_after", "min_xB_after", "obj_loss"]
SOFTCOLS = ["iter", "obj", "min_xB", "neg_basics", "eff_tol", "cond2",
            "clamped_count", "max_clamped"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(*a):
    print(*a, flush=True)


def clean_zero(v, tol=1e-9):
    x = np.asarray(v, dtype=np.float64).copy()
    x[np.abs(x) < tol] = 0.0
    return x


def inf_norm(v):
    return float(np.max(np.abs(v))) if v.size else 0.0


def refine_solve(B, lu, rhs, n_refine=N_REFINE, tighten=REFINE_TOL):
    """Solve B x = rhs with iterative residual-refinement."""
    x = lu.solve(rhs)
    resid = B @ x - rhs
    rnorm = float(np.max(np.abs(resid)))
    for _ in range(n_refine):
        if rnorm < tighten:
            break
        try:
            dc = lu.solve(resid.toarray() if hasattr(resid, "toarray") else resid)
            if hasattr(resid, "toarray"):
                dc = dc.ravel()
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


def try_factorize(B):
    """Try multiple factorization strategies."""
    try:
        return splu(B)
    except RuntimeError:
        pass
    try:
        return splu(B, permc_spec='MMD_AT_PLUS_A')
    except RuntimeError:
        pass
    Bd = B.toarray()
    for delta in [0.0, 1e-12, 1e-10, 1e-8]:
        try:
            Bp = Bd + delta * np.eye(Bd.shape[0]) if delta > 0 else Bd
            lu_d = sla.lu_factor(Bp)
            u_diag = np.diag(lu_d[0])
            if np.any(np.abs(u_diag) < 1e-14):
                continue

            class DenseLU:
                def __init__(s, lu, piv):
                    s.lu = lu; s.piv = piv
                def solve(s, b, trans=0):
                    return sla.lu_solve(
                        (s.lu, s.piv), b, trans=2 if trans == 'T' else 0)

            return DenseLU(lu_d[0], lu_d[1])
        except Exception:
            continue
    return None


def cond2_estimate(B):
    try:
        return float(np.linalg.cond(B.toarray()))
    except Exception:
        return np.nan


def basis_hash(basis):
    return int(
        hashlib.md5(np.array(basis, dtype=np.int32)).hexdigest()[:8], 16)


def compute_effective_tol(B, b_norm):
    """Compute condition-number-aware effective feasibility tolerance.

    The backward error bound for Bx = b:
        ||delta x|| <= kappa(B) * eps * ||b||
    Any basic variable within this bound of zero is indistinguishable
    from numerical noise and should NOT trigger a costly Phase-I repair.
    """
    kappa = cond2_estimate(B)
    if not np.isfinite(kappa) or kappa <= 0:
        return TOL * 1000.0, np.nan
    eff = kappa * np.finfo(np.float64).eps * b_norm * SAFETY_FACTOR
    eff = max(eff, TOL)
    return eff, kappa


# ---------------------------------------------------------------------------
# Devex (steepest-edge-like) pricing
# ---------------------------------------------------------------------------
def devex_init_weights(A, nonbasic, lu):
    """Initialize Devex weights for nonbasic columns.

    A Devex weight approximates a normalized direction norm: a larger weight
    penalises columns whose entering direction d = B^{-1} a_j has large norm
    (i.e. shallow effective descent per unit step).  We initialize from the
    current feasible basis by computing ||B^{-1} a_j|| for the first columns
    of the nonbasic set, normalize by the median, and floor at 1.0.

    Weights are returned as a dict {column_index: weight}.  Columns that
    cannot be solved (or are not yet in the dict) default to weight 1.0 at
    selection time, i.e. standard Dantzig behaviour.  This keeps the
    initialization safe when the factorization is unstable.
    """
    n_init = min(150, len(nonbasic))
    weights = {}
    for jj in range(n_init):
        col = int(nonbasic[jj])
        a_col = A[:, col]
        rhs = a_col.toarray().ravel() if hasattr(a_col, "toarray") else a_col
        try:
            dj = lu.solve(rhs)
        except Exception:
            continue
        # floor at 1.0 so no weight is ever invalid/zero
        weights[col] = max(1.0, float(np.linalg.norm(dj)))
    if weights:
        med = float(np.median(list(weights.values())))
        if med > 0:
            for k in weights:
                weights[k] = max(1.0, weights[k] / med)
    return weights


def devex_update_weights(weights, exiting_col, d_norm):
    """Update Devex weights after a pivot.

    The column that left the basis now becomes nonbasic.  Following the Harris
    Devex convention, its new weight is the (floored, normalized) norm of the
    entering column's direction d = B^{-1} a_enter computed for this pivot.
    The entering column, now basic, is simply dropped from the weight dict.
    """
    if not weights:
        return
    w = max(1.0, float(d_norm))
    med = float(np.median(list(weights.values()))) if weights else 1.0
    if med > 0:
        w = max(1.0, w / med)
    weights[int(exiting_col)] = w


def devex_select(reduced, nonbasic, weights):
    """Devex entering-variable selection.

    Minimize reduced_cost / weight over the subset of nonbasic columns with
    negative reduced cost (respecting the existing minimization sign
    convention).  Columns without an initialized weight fall back to
    weight = 1.0 (pure Dantzig), which is always safe.
    Returns the index into `nonbasic`, or the Dantzig argmin if none applies.
    """
    m_ = reduced.size
    neg = reduced < -TOL
    if not np.any(neg):
        return int(np.argmin(reduced))
    scored = np.full(m_, np.inf)
    sub = np.flatnonzero(neg)
    for idx in sub:
        val = weights.get(int(nonbasic[idx]))
        w = val if (val is not None and val > 0) else 1.0
        scored[idx] = reduced[idx] / w
    return int(np.argmin(scored))


def do_repair(iteration, basis, A, b, c, x_basic, objective,
              nonbasic, lu, repw, repf):
    """Execute Phase-I repair; return result tuple or None on failure."""
    from sparse_phase1 import sparse_phase1
    obj_before = objective
    neg_count = int((x_basic < -TOL).sum())
    neg_idx = np.where(x_basic < -TOL)[0]
    min_rc_pre = np.nan
    cond2_before = np.nan
    try:
        A_nb = A[:, nonbasic]
        y_pre = lu.solve(c[basis], trans='T')
        rc_pre = c[nonbasic] - A_nb.T @ y_pre
        min_rc_pre = float(rc_pre.min())
    except Exception:
        pass
    try:
        cond2_before = cond2_estimate(A[:, basis].tocsc())
    except Exception:
        pass

    t_repair = time.perf_counter()
    basis, r_its, r_status, _ = sparse_phase1(
        A, b, basis, max_iter=2_000_000, verbose=1 << 30)
    t_repair = time.perf_counter() - t_repair

    Bpost = A[:, basis].tocsc()
    lu = try_factorize(Bpost)
    if lu is None:
        log(f"it={iteration}: post-repair factorization FAILED")
        return None
    x_basic = refine_solve(Bpost, lu, b)
    basis_set = set(basis)
    nonbasic = np.array([j for j in range(A.shape[1])
                         if j not in basis_set], dtype=np.intp)
    obj_after = float(c[basis] @ x_basic)
    nb_after = int((x_basic < -TOL).sum())
    obj_loss = obj_after - obj_before

    repw.writerow([iteration, f"{obj_before:.9f}", neg_count,
                   f"{min_rc_pre:.3e}" if min_rc_pre == min_rc_pre else "nan",
                   f"{cond2_before:.3e}" if cond2_before == cond2_before else "nan",
                   neg_count,
                   f"{x_basic[neg_idx].min():.3e}" if neg_count else "0",
                   r_its, r_status, f"{obj_after:.9f}", nb_after,
                   f"{x_basic.min():.3e}", f"{obj_loss:.9f}"])
    repf.flush()
    return basis, x_basic, obj_after, nonbasic, lu, obj_loss, r_its, r_status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import argparse as _argparse
    _ap = _argparse.ArgumentParser()
    _ap.add_argument("--pricing", default="devex", choices=["dantzig", "devex"])
    _ap.add_argument("--budget", type=int, default=0,
                     help="optional pivot budget (0 = MAX_ITER)")
    _args, _ = _ap.parse_known_args()
    pricing_mode = _args.pricing
    pivot_budget = _args.budget if _args.budget and _args.budget > 0 else MAX_ITER
    log(f"pricing={pricing_mode}  budget={pivot_budget}")

    d = np.load(os.path.join(_ARTIFACTS, "p87_prepared.npz"),
                allow_pickle=False)
    A = sp.load_npz(os.path.join(_ARTIFACTS, "p87_prepared_A.npz")).tocsc()
    b = d["b"].astype(np.float64)
    c = d["c"].astype(np.float64)
    basis = d["basis"].tolist()
    m, n = A.shape
    b_norm = float(np.max(np.abs(b)))
    log(f"loaded: A={m}x{n} nnz={A.nnz} basis={len(basis)} ||b||_inf={b_norm:.3f}")

    B = A[:, basis].tocsc()
    lu = try_factorize(B)
    if lu is None:
        log("INITIAL FACTORIZATION FAILED"); return 1
    x_basic = refine_solve(B, lu, b)
    objective = float(c[basis] @ x_basic)
    basis_set = set(basis)
    nonbasic = np.array([j for j in range(n) if j not in basis_set], dtype=np.intp)
    log(f"start obj={objective:.9f} min_xB={x_basic.min():.3e}")

    eff_tol, kappa = compute_effective_tol(B, b_norm)
    log(f"initial effective_tol={eff_tol:.3e}  cond2={kappa:.3e}")

    # ---- Devex pricing weight bookkeeping (initialized from current basis) ----
    devex_weights = {}
    if pricing_mode == "devex":
        devex_weights = devex_init_weights(A, nonbasic, lu)
        dvals = list(devex_weights.values())
        log(f"devex weights: initialized n={len(dvals)} "
            f"med={np.median(dvals):.3f} max={max(dvals):.3f}"
            if dvals else "devex weights: none initialized")
    devex_count = 0
    devex_degen = 0

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    csvf = open(CSV_PATH, "w", newline="")
    csvw = csv.writer(csvf)
    csvw.writerow(COLS); csvf.flush()
    repf = open(REPAIR_CSV, "w", newline="")
    repw = csv.writer(repf)
    repw.writerow(REPCOLS); repf.flush()
    softf = open(SOFT_CSV, "w", newline="")
    softw = csv.writer(softf)
    softw.writerow(SOFTCOLS); softf.flush()

    prev_obj = objective
    degenerate_run = 0
    repairs = 0
    soft_clamps = 0
    consecutive_soft = 0
    total_repair_pivots = 0
    t0 = time.perf_counter()
    seen_bases: dict[int, int] = {}
    iters_since_eff_tol = 0
    RECOMPUTE_EFF_TOL = 200

    for iteration in range(pivot_budget):
        iters_since_eff_tol += 1

        # ---- FEASIBILITY CHECK (condition-number-aware) ----
        neg_count = int((x_basic < -TOL).sum())
        is_repair = False
        is_soft = False
        obj_brep = np.nan; nb_before = -1; nb_after = -1
        obj_arep = np.nan; obj_loss = np.nan

        if neg_count > 0:
            if iters_since_eff_tol >= RECOMPUTE_EFF_TOL:
                eff_tol, kappa = compute_effective_tol(A[:, basis].tocsc(), b_norm)
                iters_since_eff_tol = 0

            min_xB = float(x_basic.min())

            if min_xB >= -eff_tol:
                # ---- SOFT REPAIR: infeasibility within numerical noise ----
                n_clamped = int((x_basic < 0).sum())
                max_clamped = float(np.abs(x_basic[x_basic < 0]).max()) if n_clamped > 0 else 0.0
                x_basic[x_basic < 0] = 0.0
                soft_clamps += 1
                consecutive_soft += 1
                is_soft = True

                softw.writerow([iteration, f"{objective:.9f}",
                               f"{min_xB:.6e}", neg_count,
                               f"{eff_tol:.3e}",
                               f"{kappa:.3e}" if kappa == kappa else "nan",
                               n_clamped, f"{max_clamped:.3e}"])
                softf.flush()

                if consecutive_soft >= SOFT_CLAMP_LIMIT:
                    log(f"it={iteration}: WARNING {consecutive_soft} consecutive "
                        f"soft clamps -- basis may be degenerating")
            else:
                # ---- FULL REPAIR: genuine infeasibility ----
                is_repair = True
                consecutive_soft = 0
                obj_brep = objective; nb_before = neg_count
                result = do_repair(iteration, basis, A, b, c, x_basic, objective,
                                   nonbasic, lu, repw, repf)
                if result is None:
                    break
                basis, x_basic, obj_arep, nonbasic, lu, obj_loss, rp, rs = result
                repairs += 1; total_repair_pivots += rp
                objective = obj_arep
                nb_after = int((x_basic < -TOL).sum())
                degenerate_run = 0
                eff_tol, kappa = compute_effective_tol(A[:, basis].tocsc(), b_norm)
                iters_since_eff_tol = 0
                log(f"  it={iteration}: REPAIR #{repairs} obj={obj_brep:.6f} -> "
                    f"{obj_arep:.6f} loss={obj_loss:.6f} (PhaseI={rp} {rs}) "
                    f"eff_tol={eff_tol:.3e}")
                if repairs >= REPAIR_LIMIT:
                    log(f"repair limit at it={iteration}"); break
        else:
            consecutive_soft = 0

        # ---- DUAL SOLVE ----
        x_basic = clean_zero(x_basic, TOL)
        c_basic = c[basis]
        try:
            y = lu.solve(c_basic, trans='T')
        except Exception:
            y = np.linalg.solve(B.T.toarray(), c_basic)

        A_nb = A[:, nonbasic]
        reduced = c[nonbasic] - A_nb.T @ y
        reduced = clean_zero(reduced, TOL)
        objective = float(c_basic @ x_basic)
        obj_delta = objective - prev_obj
        Bcurr = A[:, basis].tocsc()
        xB_resid = float(np.max(np.abs(Bcurr @ x_basic - b)))

        neg_mask = reduced < -TOL
        neg_rc = int(np.sum(neg_mask))
        if neg_rc == 0:
            log(f"\nOPTIMAL at it={iteration}: obj={objective:.9f} "
                f"resid={xB_resid:.3e} repairs={repairs} soft_clamps={soft_clamps}")
            log(f"LIVE OPTIMAL it={iteration} obj={objective:.9f} neg_rc=0")
            # ---- PERSIST FINAL SOLUTION for independent KKT certificate ----
            try:
                final_x = np.zeros(n)
                final_x[basis] = x_basic
                # dual y is already computed (lu.solve(c_basic,'T')); recompute to be safe
                yf = lu.solve(np.asarray(c[basis], float), trans='T')
                xf = final_x.astype(np.float64)
                npsave = os.path.join(_ARTIFACTS, "p87_phase2_v2_final.npz")
                np.savez(
                    npsave,
                    basis=np.asarray(basis, dtype=np.intp),
                    nonbasic=nonbasic,
                    x_basic=x_basic.astype(np.float64),
                    x_full=xf,
                    y=yf.astype(np.float64),
                    reduced=reduced.astype(np.float64),
                    obj_scaled=float(objective),
                    it=int(iteration),
                    neg_rc=int(neg_rc),
                    xB_resid=float(xB_resid),
                )
                log(f"  SAVED final solution -> {npsave}")
            except Exception as e:
                log(f"  WARN failed to save final solution: {e}")
            break

        min_rc = float(reduced.min())
        min_xB = float(x_basic.min())
        neg_basics = int(neg_count)
        bh = basis_hash(basis)

        try:
            t_lu_s = time.perf_counter()
            lu_m = splu(Bcurr.tocsr(), permc_spec="COLAMD")
            nnzL, nnzU = int(lu_m.L.nnz), int(lu_m.U.nnz)
            t_lu_s = time.perf_counter() - t_lu_s
        except Exception:
            nnzL = nnzU = -1; t_lu_s = float('nan')

        # ---- ENTERING VARIABLE (Devex / Dantzig / Bland) ----
        bland = degenerate_run >= MAX_DEGENERATE
        if bland:
            ni_arr = np.flatnonzero(neg_mask)
            e_idx = int(ni_arr[np.argmin(nonbasic[ni_arr])])
        elif pricing_mode == "devex":
            e_idx = devex_select(reduced, nonbasic, devex_weights)
            devex_count += 1
        else:
            e_idx = int(np.argmin(reduced))
        entering = int(nonbasic[e_idx])

        # ---- DIRECTION d = B^{-1} a_enter ----
        a_enter = A[:, entering]
        try:
            d = lu.solve(a_enter.toarray().ravel())
        except Exception:
            d = np.linalg.solve(B.toarray(), a_enter.toarray().ravel())
        d_norm = float(np.linalg.norm(d))

        # ---- RATIO TEST ----
        positive = d > PIV_TOL
        if not np.any(positive):
            log(f"it={iteration}: UNBOUNDED col {entering}"); break
        ratios = np.full(m, np.inf)
        ratios[positive] = x_basic[positive] / d[positive]
        theta = float(np.min(ratios))
        lc = np.flatnonzero(np.abs(ratios - theta) <= PIV_TOL * max(1.0, abs(theta)))
        lidx = int(lc[np.argmin([basis[i] for i in lc])])
        is_degen = 1 if theta <= PIV_TOL else 0

        # ---- LOG ROW ----
        csvw.writerow([
            iteration, f"{objective:.9f}", f"{obj_delta:.9e}",
            f"{min_rc:.6e}", neg_rc, f"{min_xB:.6e}", neg_basics,
            f"{theta:.6e}", is_degen,
            f"{t_lu_s:.6f}" if t_lu_s == t_lu_s else "nan",
            nnzL, nnzU, f"{xB_resid:.6e}", bh,
            int(is_repair), int(is_soft),
            f"{eff_tol:.3e}"])
        if iteration % LOG_EVERY == 0 or is_repair or is_soft:
            csvf.flush()
            log(f"LIVE it={iteration:6d} obj={objective:.6f} "
                f"obj_delta={obj_delta:+.3e} neg_rc={neg_rc} "
                f"min_rc={min_rc:.3e} theta={theta:.3e} "
                f"degen={is_degen} repairs={repairs} soft={soft_clamps}")

        # ---- PIVOT UPDATE ----
        exiting_col = basis[lidx]          # leaves the basis this iteration
        basis[lidx] = entering
        basis_set = set(basis)
        nonbasic = np.array([j for j in range(n) if j not in basis_set], dtype=np.intp)
        x_basic = x_basic - theta * d
        x_basic[lidx] = theta

        # ---- PER-PIVOT REFACTORIZATION ----
        B = A[:, basis].tocsc()
        lu = try_factorize(B)
        if lu is None:
            log(f"it={iteration}: factorization failed after pivot"); break
        x_basic = refine_solve(B, lu, b)
        if np.any(~np.isfinite(x_basic)):
            log(f"it={iteration}: non-finite x_basic after refactor"); break

        # ---- DEVEX WEIGHT UPDATE (exiting col -> nonbasic) ----
        if pricing_mode == "devex" and not bland:
            devex_update_weights(devex_weights, exiting_col, d_norm)
            if is_degen:
                devex_degen += 1

        # ---- PERIODIC EFF_TOL UPDATE ----
        if iters_since_eff_tol >= RECOMPUTE_EFF_TOL:
            eff_tol, kappa = compute_effective_tol(B, b_norm)
            iters_since_eff_tol = 0

        if is_degen:
            degenerate_run += 1
        else:
            degenerate_run = 0
            if iteration % 8 == 0:
                bh_n = basis_hash(basis)
                if bh_n in seen_bases:
                    log(f"  it={iteration}: REPEATED BASIS (prev={seen_bases[bh_n]})")
                seen_bases[bh_n] = iteration
        prev_obj = objective

    # =========================================================================
    # SUMMARY
    # =========================================================================
    csvf.flush(); csvf.close()
    repf.flush(); repf.close()
    softf.flush(); softf.close()
    t_total = time.perf_counter() - t0

    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    ni = len(rows)
    if ni == 0:
        log("no iterations completed"); return 1

    objs = [float(r["obj"]) for r in rows]
    deltas = [float(r["obj_delta"]) for r in rows]
    neg_rcs = [int(r["neg_rc"]) for r in rows]
    degens = [int(r["degenerate"]) for r in rows]
    lu_ts = [float(r["lu_time_s"]) for r in rows
             if r["lu_time_s"] and r["lu_time_s"] != "nan"]
    xB_r = [float(r["xB_resid"]) for r in rows]
    n_pos = sum(1 for dd in deltas if dd > 1e-12)
    n_neg_d = sum(1 for dd in deltas if dd < -1e-12)
    n_zro = sum(1 for dd in deltas if abs(dd) <= 1e-12)

    log("\n========== PHASE II V2 SUMMARY ==========")
    log(f"  iterations:      {ni}")
    log(f"  wall time:       {t_total:.1f}s")
    log(f"  full repairs:    {repairs}  (PhaseI pivots={total_repair_pivots})")
    log(f"  soft clamps:     {soft_clamps}")
    log(f"  obj:             {objs[0]:.6f} -> {objs[-1]:.6f}  delta={objs[-1]-objs[0]:.6f}")
    log(f"  obj_delta:       +{n_pos}  -{n_neg_d}  ~0={n_zro}  "
        f"(+{100*n_pos/ni:.1f}% -{100*n_neg_d/ni:.1f}%)")
    log(f"  degen:           {sum(degens)}/{ni} ({100*sum(degens)/ni:.1f}%)")
    if lu_ts:
        log(f"  LU time:         med={np.median(lu_ts):.4f}s  "
            f"mean={np.mean(lu_ts):.4f}s  sum={np.sum(lu_ts):.1f}s")
    log(f"  xB residual:     mean={np.mean(xB_r):.3e}  max={np.max(xB_r):.3e}")
    e = min(500, len(neg_rcs))
    log(f"  neg_rc:          first_e={np.mean(neg_rcs[:e]):.1f}  "
        f"last_e={np.mean(neg_rcs[-e:]):.1f}  min={min(neg_rcs)}  max={max(neg_rcs)}")

    dr = []; rl = 0
    for dd in degens:
        if dd: rl += 1
        else:
            if rl > 0: dr.append(rl)
            rl = 0
    if rl > 0: dr.append(rl)
    if dr:
        log(f"  degen runs:      count={len(dr)}  mean={np.mean(dr):.1f}  "
            f"max={max(dr)}  sum={sum(dr)}")

    if repairs > 0:
        rr = []
        with open(REPAIR_CSV) as f:
            for row in csv.DictReader(f): rr.append(row)
        losses = [float(r["obj_loss"]) for r in rr if r["obj_loss"]]
        if losses:
            log(f"  repair loss:     mean={np.mean(losses):.6f}  "
                f"max={max(losses):.6f}  total={sum(losses):.6f}")

    if soft_clamps > 0:
        sf = []
        with open(SOFT_CSV) as f:
            for row in csv.DictReader(f): sf.append(row)
        log(f"  soft clamp count: {len(sf)} events")

    # --- Devex pricing note ---
    log(f"  pricing: Devex used for {devex_count}/{ni} iterations "
        f"(degen under Devex={devex_degen})"
        if pricing_mode == "devex" else
        f"  pricing: Dantzig ({ni} iterations)")

    # --- Phase II progress rates ---
    if ni >= 2000:
        log("\n  Progress by 1000-iter blocks:")
        for start in range(0, ni, 1000):
            end = min(start + 1000, ni)
            block_objs = [float(rows[j]["obj"]) for j in range(start, end)]
            log(f"    it {start:5d}-{end-1:5d}: obj {block_objs[0]:.3f} -> "
                f"{block_objs[-1]:.3f} (d={block_objs[-1]-block_objs[0]:+.3f})")

    log(f"\n  CSV:        {CSV_PATH}")
    log(f"  REPAIR CSV: {REPAIR_CSV}")
    log(f"  SOFT CSV:   {SOFT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
