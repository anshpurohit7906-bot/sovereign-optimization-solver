""" here and how to do it to understandDiagnostic only.  Reads production code and data; never modifies it.

Part 1 audits the two previous basis experiments (naive |x|/|z| ranking and
rank-aware |x|/|z| greedy selection) by comparing their ACTUAL selected
column sets and conditioning on the same production standard form.

Part 2 builds a completely independent basis with scipy column-pivoted QR
(no x, z, y, or ratio information) and measures it with the same metrics.

Part 4 checks x_B = B^{-1} b feasibility for the RRQR basis only.
Part 5 runs the same RRQR construction on the small validation set.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _path in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.lp.mehrotra import MehrotraResult, StandardFormLP, solve_standard_form, to_standard_form
from src.lp.simplex import SimplexError, _solve_basis
from src.numerical_model import NumericalLP, load_numeric_mps
from experiment.crossover.basis_identification import (
    candidate_rankings,
    rank_aware_x_over_z_basis,
)

FEAS_TOL = 1e-7          # production solve tolerance used for x_B >= -tol tests
CONDITION_LIMIT = 1e12   # production Simplex condition limit used previously


@dataclass(frozen=True)
class BasisMetrics:
    name: str
    size: int
    rank_default: int          # np.linalg.matrix_rank (default tolerance)
    rank_1e10: int             # rank with relative tolerance 1e-10
    rank_1e7: int              # rank with relative tolerance 1e-7
    sigma_min: float
    sigma_max: float
    cond2: float               # true 2-norm condition number from SVD
    cond1: float               # 1-norm condition estimate (||B||_1 ||B^-1||_1)
    solve_rel_residual: float  # ||B x_B - b||_inf / (1 + ||b||_inf), refined LU solve
    simplex_accepts: bool
    simplex_message: str
    x_min: float
    x_max: float
    negatives: int             # count of x_B < -FEAS_TOL


def measure_basis(A: np.ndarray, b: np.ndarray, columns, name: str) -> BasisMetrics:
    B = A[:, list(columns)]
    s = np.linalg.svd(B, compute_uv=False)
    sigma_max, sigma_min = float(s[0]), float(s[-1])
    rank_default = int(np.linalg.matrix_rank(B))
    rank_1e10 = int(np.sum(s > 1e-10 * sigma_max))
    rank_1e7 = int(np.sum(s > 1e-7 * sigma_max))
    cond2 = float(sigma_max / sigma_min) if sigma_min > 0 else float("inf")
    try:
        cond1 = float(np.linalg.cond(B, 1))
    except np.linalg.LinAlgError:
        cond1 = float("inf")
    # Stable LU solve with one step of iterative refinement (diagnostic only).
    # A basis can be SO ill-conditioned that the LU solve itself overflows to
    # NaN/Inf; that is recorded as a measurement (non-finite solve), not an error.
    try:
        lu, piv = sla.lu_factor(B)
        x_B = sla.lu_solve((lu, piv), b)
        x_B = x_B + sla.lu_solve((lu, piv), b - B @ x_B)
        if not np.all(np.isfinite(x_B)):
            raise ArithmeticError("LU solve produced non-finite x_B")
        rel_res = float(np.linalg.norm(B @ x_B - b, ord=np.inf)) / max(
            1.0, float(np.linalg.norm(b, ord=np.inf)))
        x_min, x_max = float(np.min(x_B)), float(np.max(x_B))
        negatives = int(np.count_nonzero(x_B < -FEAS_TOL))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
        rel_res = float("inf")
        x_min = x_max = float("nan")
        negatives = -1
        solve_note = f" (solve failed: {exc})"
    else:
        solve_note = ""
    try:
        _solve_basis(B, b, condition_limit=CONDITION_LIMIT)
        ok, msg = True, "accepted by production _solve_basis"
    except SimplexError as exc:
        ok, msg = False, str(exc)
    return BasisMetrics(
        name=name + solve_note, size=B.shape[1], rank_default=rank_default,
        rank_1e10=rank_1e10,
        rank_1e7=rank_1e7, sigma_min=sigma_min, sigma_max=sigma_max, cond2=cond2,
        cond1=cond1, solve_rel_residual=rel_res, simplex_accepts=ok,
        simplex_message=msg, x_min=x_min, x_max=x_max,
        negatives=negatives,
    )


def print_metrics(mt: BasisMetrics) -> None:
    print(f"  [{mt.name}] size={mt.size} rank_default={mt.rank_default} "
          f"rank@1e-10={mt.rank_1e10} rank@1e-7={mt.rank_1e7}")
    print(f"    sigma_max={mt.sigma_max:.6e} sigma_min={mt.sigma_min:.6e}")
    print(f"    cond2={mt.cond2:.6e} cond1={mt.cond1:.6e}")
    print(f"    solve rel residual={mt.solve_rel_residual:.3e}  "
          f"simplex={mt.simplex_accepts} ({mt.simplex_message})")
    print(f"    x_B: min={mt.x_min:.6e} max={mt.x_max:.6e} negatives(<-{FEAS_TOL:g})={mt.negatives}")


def rrqr_basis(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure column-pivoted-QR basis: first m pivot columns. No IPM data used."""
    _, _, piv = sla.qr(A, pivoting=True, mode="economic")
    return piv, piv[: A.shape[0]].copy()

def run_instance(path: str, name: str, *, run_ipm_for_audit: bool) -> dict:
    print("=" * 78)
    print(f"INSTANCE: {name}  ({os.path.basename(path)})")
    print("=" * 78)
    started = time.perf_counter()
    sf = to_standard_form(load_numeric_mps(path))
    A, b = sf.A, sf.b
    m, n = A.shape
    print(f"standard form: m={m} n={n} (basis size required = {m})")
    out: dict = {"name": name, "m": m, "n": n}

    # ----- Part 2/3/5: pure RRQR basis (no IPM information) -----------------
    t0 = time.perf_counter()
    _, basis_cols = rrqr_basis(A)
    rrqr = measure_basis(A, b, basis_cols, "RRQR")
    rrqr_time = time.perf_counter() - t0
    print(f"RRQR pivot construction: {rrqr_time:.2f}s; first {m} pivots selected")
    print_metrics(rrqr)
    out["rrqr"] = rrqr
    out["rrqr_time"] = rrqr_time

    # ----- Parts 1/3: prior x/z experiments (PILOT4 audit only) -------------
    if run_ipm_for_audit:
        t0 = time.perf_counter()
        ipm = solve_standard_form(sf, tol=1e-8, max_iter=100)
        ipm_time = time.perf_counter() - t0
        print(f"\nproduction Mehrotra (tol=1e-8, max_iter=100): status={ipm.status} "
              f"obj={ipm.objective:.10g} iters={ipm.iterations} ({ipm_time:.1f}s)")
        x, z = ipm.x_standard, ipm.z_standard

        naive = candidate_rankings(x, z, m)[2]  # 'x_over_z_desc'
        naive_cols = np.array(naive.columns)
        t0 = time.perf_counter()
        ra = rank_aware_x_over_z_basis(A, x, z, m)
        ra_time = time.perf_counter() - t0
        ra_cols = np.array(ra.columns)
        print(f"\nrank-aware greedy scan: {ra_time:.1f}s  selected={len(ra_cols)} "
              f"rejected={ra.rejected_columns} replacements(counter)={ra.replacement_columns} "
              f"inspected={ra.inspected_columns}")

        common = np.intersect1d(naive_cols, ra_cols)
        only_naive = np.setdiff1d(naive_cols, ra_cols)
        only_ra = np.setdiff1d(ra_cols, naive_cols)
        print("\n--- Part 1 audit: actual column-set comparison ---")
        print(f"common columns          : {common.size}")
        print(f"unique to naive x/z     : {only_naive.size}")
        print(f"unique to rank-aware    : {only_ra.size}")
        print(f"bases are identical sets: {common.size == m}")
        print(f"replacement counter={ra.replacement_columns}, actual |RA\\naive|={only_ra.size}, "
              f"|naive\\RA|={only_naive.size}")
        # Replacement-counter semantics: counter counts selected columns whose
        # preference-order position >= m.  Verify it equals |RA \ naive|.
        order = np.argsort(-(np.abs(x) / np.maximum(np.abs(z), np.finfo(float).tiny)),
                           kind="mergesort")
        pos = np.empty(n, dtype=int)
        pos[order] = np.arange(n)
        counter_check = int(np.count_nonzero(pos[ra_cols] >= m))
        print(f"recomputed counter (order position >= m): {counter_check}  "
              f"(matches reported counter: {counter_check == ra.replacement_columns})")
        print(f"counter equals actual net basis change vs naive: "
              f"{counter_check == only_ra.size}")

        print("\n--- conditioning measured from EACH matrix independently ---")
        mt_naive = measure_basis(A, b, naive_cols, "x/z naive")
        mt_ra = measure_basis(A, b, ra_cols, "rank-aware x/z")
        print_metrics(mt_naive)
        print_metrics(mt_ra)
        print(f"cond2 equal for the two DIFFERENT matrices: "
              f"{mt_naive.cond2 == mt_ra.cond2}")
        print(f"rank-aware sigma_min={mt_ra.sigma_min:.3e}: the greedy threshold "
              f"1e-10*||a_j|| accepts columns with relative residual down to 1e-10, "
              f"i.e. it accepts near-dependence at the 1e-10 level, not only "
              f"numerical independence.")
        out.update(naive=mt_naive, rank_aware=mt_ra, ipm=ipm,
                   common=int(common.size), only_naive=int(only_naive.size),
                   only_ra=int(only_ra.size), counter_check=counter_check,
                   ra_counter=ra.replacement_columns)
    out["total_time"] = time.perf_counter() - started
    return out
    out["total_time"] = time.perf_counter() - started
    return out


_DATA = os.path.join(_ROOT, "data")
_SMALL = ["afiro", "sc205", "adlittle", "share2b", "blend"]


def main() -> int:
    # Part 1+2+3+4: PILOT4 with the full prior-experiment audit.
    pilot4 = run_instance(os.path.join(_DATA, "pilot4_plain.mps"), "PILOT4",
                          run_ipm_for_audit=True)
    # Part 5: small validation set, RRQR construction only.
    small = [run_instance(os.path.join(_DATA, f"{n}.mps"), n.upper(),
                          run_ipm_for_audit=False)
             for n in _SMALL]
    print("\n" + "=" * 78)
    print("STAGE-1 SUMMARY (same metrics/tolerances for every basis)")
    print("=" * 78)
    hdr = f"{'instance':10s} {'basis':16s} {'rank@1e-10':>10s} {'sigma_min':>10s} " \
          f"{'sigma_max':>10s} {'cond2':>10s} {'resid':>9s} {'spx':>4s} {'neg':>4s}"
    print(hdr)
    rows = [("PILOT4", pilot4)]
    rows += [(s["name"], s) for s in small]
    for inst_name, d in rows:
        b = d["rrqr"]
        print(f"{inst_name:10s} {'RRQR':16s} {b.rank_1e10:10d} {b.sigma_min:10.3e} "
              f"{b.sigma_max:10.3e} {b.cond2:10.3e} {b.solve_rel_residual:9.1e} "
              f"{'OK' if b.simplex_accepts else 'REJ':>4s} {b.negatives:4d}")
    if "naive" in pilot4:
        for key, label in (("naive", "x/z naive"), ("rank_aware", "rank-aware x/z")):
            b = pilot4[key]
            print(f"{'PILOT4':10s} {label:16s} {b.rank_1e10:10d} {b.sigma_min:10.3e} "
                  f"{b.sigma_max:10.3e} {b.cond2:10.3e} {b.solve_rel_residual:9.1e} "
                  f"{'OK' if b.simplex_accepts else 'REJ':>4s} {b.negatives:4d}")
    print(f"\ntotal wall time: {pilot4['total_time']:.1f}s "
          f"(PILOT4 incl. audit); small set ~{sum(s['rrqr_time'] for s in small):.1f}s RRQR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def run_pilot87_only() -> None:
    """Optional Part-5 extension: PILOT87, RRQR construction only (no IPM)."""
    d = run_instance(os.path.join(_DATA, "pilot87.mps"), "PILOT87",
                     run_ipm_for_audit=False)
    b = d["rrqr"]
    print(f"PILOT87 RRQR summary: rank@1e-10={b.rank_1e10} "
          f"sigma_min={b.sigma_min:.3e} sigma_max={b.sigma_max:.3e} "
          f"cond2={b.cond2:.3e} resid={b.solve_rel_residual:.2e} "
          f"simplex={b.simplex_accepts} x_min={b.x_min:.3e} "
          f"negatives={b.negatives} qr_time={d['rrqr_time']:.1f}s")

