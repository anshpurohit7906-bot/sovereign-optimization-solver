# Sparse Crossover Without RRQR — Production Path Report

**Date:** 2026-09-04
**Instance:** PILOT87 (m=3608, n=8038, nnz=75,635)

## Goal

Remove RRQR from the validated crossover preparation path for PILOT87,
preserving sparse matrix flow throughout to eliminate ~232 MB dense allocation
and ~21s waste.

## File Changed

`experiment/crossover/p87_prepare.py` — **only file modified**

### Changes

| What | Before | After |
|------|--------|-------|
| MPS load | `load_numeric_mps("data/pilot87.mps")` (dense) | `load_numeric_mps("data/pilot87.mps", sparse=True)` |
| Standard-form A | `sp.csc_matrix(sf.A)` (dense→sparse conversion) | `sf.A` (already sparse from `to_standard_form`) |
| Scaling input | `scale_lp(A0.toarray(), ...)` (sparse→dense→dense scaling) | `scale_lp(A0, ...)` (sparse→sparse scaling) |
| RRQR basis | `rrqr_basis(A.toarray())` (dense 232 MB alloc, ~21s) | **removed** |
| Phase I call | `sparse_phase1(A, b, basis0, ...)` (ignored basis0) | `sparse_phase1(A, b, ...)` (default basis0=None) |
| Import removed | `from stage1_audit_rrqr import rrqr_basis` | — |

## Results Comparison

### Preparation + Phase I

| Metric | Baseline (with RRQR) | New (no RRQR, sparse) | Delta |
|--------|----------------------:|----------------------:|------:|
| Dense memory | 232 MB | **0 MB** | **-232 MB** |
| RRQR time | 21.126 s | **0.000 s** | **-21.126 s** |
| Phase I iterations | 16,346 | 14,628 | -1,718 |
| Phase I time | 538.155 s | **264.74 s** | **-273.42 s** |
| **Total prep time** | **559.281 s** | **264.94 s** | **-294.34 s** |
| Phase I status | feasible | feasible | ✅ identical |

### Phase II + Polish + Certification

| Metric | Baseline | New (no RRQR, sparse) |
|--------|----------|----------------------|
| Phase II iterations | ~18,587 | 22,046 |
| Phase II obj | 301.711 | 301.712 |
| Polish pivots | 48 | 48 |
| Original objective | 301.710347333 | 301.710347333 |

### Certification (STRICT KKT)

| Check | Result |
|-------|--------|
| Primal ‖Ax−b‖∞ | 3.638e-11 ✅ |
| min(z_nonbasic) | −1.070e-18 ✅ |
| Complementarity x'z | 4.716e-12 ✅ |
| Basis residual | 3.638e-11 ✅ |
| Original objective | 301.710347333 ✅ |
| vs HiGHS reference | \|delta\| = 1.034e-10 ✅ |
| Strong duality gap | −2.331e-12 ✅ |
| **VERDICT** | **STRICT VERIFIED OPTIMAL** |

## Summary

- **RRQR is unnecessary.** Same KKT-certified optimal objective (301.710347333) with
  zero dense allocation.
- **Preparation is 2× faster** (265s vs 559s) — the sparse-only path avoids
  all dense matrix materialization.
- **232 MB dense allocation eliminated** — A stays sparse CSC/CSR throughout
  the entire pipeline.
- **Downstream unchanged** — artifact format identical; Phase II, polish, and
  certification tools required zero modifications.

## Artifacts

```
artifacts/pilot87/
├── p87_prepared.npz            # b, c, basis, m, n, row_scale, col_scale
├── p87_prepared_A.npz          # sparse CSC matrix
├── p87_phase2_v2_final.npz     # Phase II terminal basis
├── p87_phase2_v2_log.csv       # 22,046 iterations
├── p87_strict_polished.npz     # polished basis
├── p87_strict_certificate.txt  # STRICT VERIFIED OPTIMAL
└── p87_certificate.txt         # certification report
```

## Test Suite

82/82 pytest tests pass (unchanged).
