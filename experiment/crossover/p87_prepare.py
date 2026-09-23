"""Build & persist PILOT87 standard form + Phase I feasible basis.

Runs the once-only preparation stage (sparse Phase I) and saves the scaled
standard-form problem (A, b, c) plus the Phase I feasible basis to a .npz
so Phase II experiments never redo the ~544 s Phase I.

Sparse throughout: A is loaded, standardized, and scaled as a scipy.sparse
matrix.  No dense 3608×8038 matrix is ever allocated.  The RRQR basis
computation is intentionally omitted because ``sparse_phase1()`` always
initialises with the all-artificial B=I basis and ignores any external
basis0.

Usage:
    OPENBLAS_NUM_THREADS=1 python experiment/crossover/p87_prepare.py [out.npz]
"""
from __future__ import annotations
import os, sys, time, argparse, numpy as np, scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opticore.numerical_model import load_numeric_mps
from opticore.lp.mehrotra import to_standard_form
from opticore.scaling import scale_lp
from sparse_phase1 import sparse_phase1


def main(out: str):
    t0 = time.perf_counter()
    sf = to_standard_form(load_numeric_mps("data/pilot87.mps", sparse=True))
    A0 = sf.A
    if not sp.issparse(A0):
        A0 = sp.csc_matrix(A0)
    b0 = np.asarray(sf.b, float)
    c0 = np.asarray(sf.c_min, float)
    m, n = A0.shape
    print(f"standard form: {m} x {n}  load={time.perf_counter()-t0:.2f}s  nnz={A0.nnz:,}", flush=True)

    # ---- Sparse scaling (no dense allocation) ----
    S = scale_lp(A0, b0, c0, np.zeros(n), np.full(n, np.inf))
    A = sp.csc_matrix(S.A)
    b = np.asarray(S.b, float)
    c = np.asarray(S.c, float)
    print(f"scaling: col[{S.column_scale.min():.3e}..{S.column_scale.max():.3e}] "
          f"row[{S.row_scale.min():.3e}..{S.row_scale.max():.3e}]", flush=True)

    # ---- Sparse Phase I (all-artificial B=I start; no RRQR basis) ----
    t2 = time.perf_counter()
    basis, its, status, info = sparse_phase1(A, b, max_iter=2_000_000)
    print(f"Phase I: iters={its} status={status} t={time.perf_counter()-t2:.2f}s", flush=True)
    if status != "feasible":
        print(f"Phase I FAILED: {status}  {info.get('err','')}", flush=True)
        sys.exit(1)

    basis = np.asarray(basis, dtype=np.intp)
    sp.save_npz(out.replace(".npz", "_A.npz"), A.tocsc())
    np.savez(out,
             b=b, c=c,
             basis=basis,
             m=m, n=n,
             row_scale=S.row_scale, col_scale=S.column_scale)
    print(f"SAVED: {out}  basis_size={basis.size}  total={time.perf_counter()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?",
                     default=os.path.join(_ROOT, "artifacts", "pilot87",
                                          "p87_prepared.npz"))
    a = ap.parse_args()
    main(a.out)
