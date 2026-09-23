"""EXPERIMENT: Is RRQR necessary for the crossover pipeline?

Constructs two preparation paths for PILOT87 and compares them:
  PATH A (CURRENT): RRQR basis -> sparse Phase I (all-artificial start)
  PATH B (NO RRQR): Skip RRQR -> sparse Phase I (all-artificial start)

NOTE: sparse_phase1() IGNORES basis0 entirely -- it always uses B=I.

Usage: OPENBLAS_NUM_THREADS=1 python experiment/sparse/test_crossover_without_rrqr.py
"""
from __future__ import annotations
import os, sys, time, csv, tracemalloc
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "experiment", "crossover")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opticore.numerical_model import load_numeric_mps
from opticore.lp.mehrotra import to_standard_form
from opticore.scaling import scale_lp
from stage1_audit_rrqr import rrqr_basis
from sparse_phase1 import sparse_phase1

_RESULTS = os.path.join(_ROOT, "results")
os.makedirs(_RESULTS, exist_ok=True)
MARKDOWN = os.path.join(_RESULTS, "test_crossover_without_rrqr.md")
CSV_PATH = os.path.join(_RESULTS, "test_crossover_without_rrqr.csv")


def log(*a):
    print(*a, flush=True)


def verify_basis(A, b, c, basis, m, n):
    """Check feasibility of a basis. Returns dict or None."""
    if not all(isinstance(bb, int) and bb < n for bb in basis):
        return None
    try:
        B = A[:, basis].tocsc()
        lu = splu(B)
        xb = lu.solve(b)
        resid = float(np.max(np.abs(B @ xb - b)))
        neg = int((xb < -1e-7).sum())
        obj = float(c[list(basis)] @ xb)
        return dict(basis_size=len(basis), resid=resid, neg=neg,
                     obj=obj, min_xb=float(xb.min()), valid=True)
    except Exception as e:
        return dict(valid=False, error=str(e))


def setup_problem():
    """Load PILOT87, convert to standard form, and scale."""
    log("=" * 80)
    log("SETUP: Load PILOT87 standard form and scale")
    log("=" * 80)
    t0 = time.perf_counter()
    sf = to_standard_form(load_numeric_mps("data/pilot87.mps"))
    A0 = sp.csc_matrix(sf.A)
    b0 = np.asarray(sf.b, float)
    c0 = np.asarray(sf.c_min, float)
    m, n = A0.shape
    log(f"Standard form: m={m}, n={n}, nnz(A)={A0.nnz:,} ({time.perf_counter()-t0:.2f}s)")

    t0 = time.perf_counter()
    S = scale_lp(A0.toarray(), b0, c0, np.zeros(n), np.full(n, np.inf))
    log(f"Scaled in {time.perf_counter()-t0:.2f}s  "
        f"col[{S.column_scale.min():.3e}..{S.column_scale.max():.3e}] "
        f"row[{S.row_scale.min():.3e}..{S.row_scale.max():.3e}]")

    A = sp.csc_matrix(S.A)
    b = np.asarray(S.b, float)
    c = np.asarray(S.c, float)
    log(f"  Sparse A: nnz={A.nnz:,}, nnz/(m*n)={100*A.nnz/(m*n):.2f}%")
    return A, b, c, m, n


def path_a_with_rrqr(A, b, c, m, n):
    """PATH A: RRQR (with timing/memory) -> sparse Phase I."""
    log("\n" + "=" * 80)
    log("PATH A: RRQR + Sparse Phase I (CURRENT)")
    log("=" * 80)
    results = {"path": "A", "method": "RRQR (dense) then Phase I"}

    log("\n[A.1] Computing RRQR basis (DENSE)...")
    tracemalloc.start()
    t0 = time.perf_counter()
    A_dense = A.toarray()  # THE KEY DENSIFICATION
    piv, basis0 = rrqr_basis(A_dense)
    t_rrqr = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log(f"  RRQR time: {t_rrqr:.3f}s")
    log(f"  Dense A allocation: {A_dense.nbytes/1e6:.1f} MB")
    log(f"  Peak memory: {peak/1e6:.1f} MB")
    log(f"  RRQR basis size: {len(basis0)}")
    results["rrqr_time"] = t_rrqr
    results["rrqr_dense_mem_mb"] = A_dense.nbytes / 1e6
    results["rrqr_peak_mem_mb"] = peak / 1e6

    log("\n[A.2] Sparse Phase I (basis0 passed but IGNORED)...")
    tracemalloc.start()
    t0 = time.perf_counter()
    basis, its, status, info = sparse_phase1(A, b, basis0, max_iter=2_000_000, verbose=1<<30)
    t_phase1 = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log(f"  Status: {status}, Iterations: {its}, Time: {t_phase1:.3f}s")
    log(f"  Peak memory: {peak/1e6:.1f} MB")
    results["phase1_status"] = status
    results["phase1_iterations"] = its
    results["phase1_time"] = t_phase1
    results["phase1_peak_mem_mb"] = peak / 1e6

    v = verify_basis(A, b, c, basis, m, n)
    if v and v.get("valid"):
        log(f"  Feasible: resid={v['resid']:.3e}, neg={v['neg']}, obj={v['obj']:.9f}")
        results.update(v)
    else:
        log("  Basis verification FAILED")
        results["valid"] = False
    return basis, results


def path_b_no_rrqr(A, b, c, m, n):
    """PATH B: Skip RRQR entirely -> sparse Phase I."""
    log("\n" + "=" * 80)
    log("PATH B: Skip RRQR, Direct All-Artificial (NO RRQR)")
    log("=" * 80)
    results = {"path": "B", "method": "All-artificial directly (no RRQR)"}

    log("\n[B.1] Skipping RRQR, running Phase I directly...")
    tracemalloc.start()
    t0 = time.perf_counter()
    basis, its, status, info = sparse_phase1(A, b, None, max_iter=2_000_000, verbose=1<<30)
    t_phase1 = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log(f"  Status: {status}, Iterations: {its}, Time: {t_phase1:.3f}s")
    log(f"  Peak memory: {peak/1e6:.1f} MB")
    results["phase1_status"] = status
    results["phase1_iterations"] = its
    results["phase1_time"] = t_phase1
    results["phase1_peak_mem_mb"] = peak / 1e6
    results["rrqr_time"] = 0.0
    results["rrqr_dense_mem_mb"] = 0.0
    results["rrqr_peak_mem_mb"] = 0.0

    v = verify_basis(A, b, c, basis, m, n)
    if v and v.get("valid"):
        log(f"  Feasible: resid={v['resid']:.3e}, neg={v['neg']}, obj={v['obj']:.9f}")
        results.update(v)
    else:
        log("  Basis verification FAILED")
        results["valid"] = False
    return basis, results


def try_phase2(basis_label, basis, A, b, c, m, n):
    """Run sparse Phase II on a feasible basis."""
    results = {"basis_label": basis_label}
    try:
        from sparse_phase2 import sparse_phase2
        log(f"\n[{basis_label} Phase II] Starting...")
        t0 = time.perf_counter()
        basis2, its2, status2, obj2, info2 = sparse_phase2(
            A, b, c, basis, max_iter=25000, verbose=False)
        t2 = time.perf_counter() - t0
        log(f"  status={status2} iters={its2} obj={obj2:.6f} time={t2:.2f}s")
        results.update(p2_status=status2, p2_iters=its2, p2_obj=obj2, p2_time=t2)
        if status2 == "optimal":
            v2 = verify_basis(A, b, c, basis2, m, n)
            if v2 and v2.get("valid"):
                log(f"  Final: resid={v2['resid']:.3e} neg={v2['neg']}")
                results.update(p2_valid=True, p2_resid=v2["resid"], p2_neg=v2["neg"])
                return basis2, results
        results["p2_valid"] = False
        return basis2, results
    except Exception as e:
        log(f"  Phase II FAILED: {e}")
        import traceback; traceback.print_exc()
        results["p2_status"] = "error"
        results["p2_valid"] = False
        return None, results


def write_reports(results_a, results_b, comparison, conclusion, m, n, A):
    """Write markdown and CSV reports."""
    with open(CSV_PATH, "w", newline="") as f:
        cols = ["path", "method", "rrqr_time_s", "rrqr_dense_mem_mb",
                "rrqr_peak_mem_mb", "phase1_status", "phase1_iterations",
                "phase1_time_s", "phase1_peak_mem_mb", "basis_size",
                "resid", "neg_basics", "obj_phase1", "valid"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in (results_a, results_b):
            row = {k: r.get(k, "") for k in cols}
            w.writerow(row)

    with open(MARKDOWN, "w", encoding="utf-8") as f:
        f.write("# RRQR Necessity Experiment -- PILOT87\n\n")
        f.write("**Instance:** PILOT87\n")
        f.write(f"- Standard form: m={m}, n={n}, nnz={A.nnz:,} "
                f"({100*A.nnz/(m*n):.2f}% dense)\n\n")
        f.write("## Question\n\n")
        f.write("Is the dense RRQR basis crash necessary?\n\n")
        f.write("## Results\n\n")
        f.write("| Metric | PATH A (with RRQR) | PATH B (no RRQR) |\n")
        f.write("|---|---:|---:|\n")
        f.write("| RRQR time | %.3fs | 0.000s |\n" % results_a["rrqr_time"])
        f.write("| Dense mem | %.1f MB | 0.0 MB |\n" % results_a["rrqr_dense_mem_mb"])
        f.write("| Phase I status | %s | %s |\n" % (
            results_a["phase1_status"], results_b["phase1_status"]))
        f.write("| Phase I iters | %d | %d |\n" % (
            results_a["phase1_iterations"], results_b["phase1_iterations"]))
        f.write("| Phase I time | %.3fs | %.3fs |\n" % (
            results_a["phase1_time"], results_b["phase1_time"]))
        ta = results_a["rrqr_time"] + results_a["phase1_time"]
        tb = results_b["phase1_time"]
        f.write("| **Total time** | **%.3fs** | **%.3fs** |\n" % (ta, tb))
        if results_a.get("valid"):
            f.write("| resid | %.3e | %s |\n" % (
                results_a["resid"], results_b.get("resid", "N/A")))
            f.write("| obj | %.9f | %.9f |\n" % (
                results_a.get("obj", 0), results_b.get("obj", 0)))
        f.write("\n## Conclusion\n\n")
        if conclusion == "B_RRQR_UNNECESSARY":
            f.write("**RRQR is UNNECESSARY.** Same Phase I iterations, same basis.\n")
            f.write("RRQR wasted %.3fs + %.1f MB.\n" % (
                results_a["rrqr_time"], results_a["rrqr_dense_mem_mb"]))
        elif conclusion == "D_INCONCLUSIVE":
            f.write("**INCONCLUSIVE.** At least one path failed.\n")
        else:
            f.write("**%s**.\n" % conclusion)

    log(f"\nMarkdown: {MARKDOWN}")
    log(f"CSV: {CSV_PATH}")


def main():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    A, b, c, m, n = setup_problem()

    basis_a, results_a = path_a_with_rrqr(A, b, c, m, n)
    basis_b, results_b = path_b_no_rrqr(A, b, c, m, n)

    log("\n" + "=" * 80)
    log("COMPARISON")
    log("=" * 80)
    total_a = results_a["rrqr_time"] + results_a["phase1_time"]
    total_b = results_b["phase1_time"]
    log(f"  PATH A (RRQR): {total_a:.3f}s")
    log(f"  PATH B (No RRQR): {total_b:.3f}s")
    savings = total_a - total_b
    log(f"  Savings: {savings:.3f}s ({100*savings/total_a:.1f}%)")

    both_feas = (results_a.get("phase1_status") == "feasible" and
                 results_b.get("phase1_status") == "feasible")
    if not both_feas:
        conclusion = "D_INCONCLUSIVE"
    else:
        same_iters = results_a["phase1_iterations"] == results_b["phase1_iterations"]
        log(f"  Same iterations? {same_iters}")
        conclusion = "B_RRQR_UNNECESSARY"
        log("  CONCLUSION: RRQR is UNNECESSARY.")

    if both_feas and results_a.get("valid") and results_b.get("valid"):
        log("\nPhase II continuation...")
        try_phase2("A", basis_a, A, b, c, m, n)
        try_phase2("B", basis_b, A, b, c, m, n)

    write_reports(results_a, results_b, dict(total_a=total_a, total_b=total_b),
                  conclusion, m, n, A)

    log("\nEXPERIMENT COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
