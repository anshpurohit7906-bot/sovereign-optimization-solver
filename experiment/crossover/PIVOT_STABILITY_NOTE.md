# Pivot-path environment sensitivity (PILOT4 RRQR → repair)

**Finding:** the crossover pipeline's step counts are environment-dependent,
not run-to-run nondeterministic. `stage1_audit_rrqr.rrqr_basis` uses
`scipy.linalg.qr(A, pivoting=True)`, whose column tie-breaking depends on
floating-point rounding, which differs with BLAS threading and build.
A different RRQR basis in produces a different (equally valid) composite
Phase-I repair trajectory out.

**Measured on numpy 2.3.4 / scipy 1.18.1 / Python 3.13 (Windows):**

| BLAS threads | RRQR basis hash | repair steps |
|---|---|---|
| default (multithreaded) | `9777a370` | 1161 |
| `OPENBLAS_NUM_THREADS=1` | `6819c2d8` | 1437 |
| `OMP_NUM_THREADS=1` | `6819c2d8` | 1437 |
| `OPENBLAS_NUM_THREADS=2` | `e9347791` | 1437 |

Within a fixed environment the pipeline is fully deterministic (8/8 fresh
processes across two sessions reproduced identical hashes and step counts).

**Invariants that hold across ALL environments tested** (these, not step
counts, are the baseline): `feas=True`, `neg=0`, production `_solve_basis`
gate accepts the basis, cond2 ≈ 1.2e9 << 1e12.

**Implications:**

1. Recorded step counts (e.g. 1028) are only reproducible in the exact
   environment they were recorded in. Do not use them as regression targets.
2. For a pinned, cross-machine-reproducible basis: set
   `OPENBLAS_NUM_THREADS=1` (or equivalent) around the RRQR call, or
   canonicalize pivot tie-breaking explicitly.
3. Reproduce with `python experiment/crossover/thread_probe.py`.
