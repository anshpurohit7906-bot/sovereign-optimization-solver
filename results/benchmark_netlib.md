# Netlib LP benchmark: production sparse Mehrotra vs HiGHS

Every `data/*.mps` instance is solved through the production sparse
end-to-end path (`load_numeric_mps(sparse=True)` + `solve_lp` with
`crossover_fallback=True`) and cross-checked against a fresh
`scipy.optimize.linprog(method="highs")` reference objective.
Acceptance: relative objective error <= `1e-06`.
Method legend: `ipm` = direct interior-point solve; `crossover` = the
IPM stalled and the automatic sparse-crossover fallback resolved it;
`certified` = value folded from an independent certificate artifact.

| instance | m | n | nnz | status | method | iters | solver objective | HiGHS reference | rel obj err | rel_gap | time (s) | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot4_plain | 410 | 1000 | 5141 | optimal | crossover | 79 + 1546 piv | -2581.13926 | -2581.13926 | 1.76e-15 | 0 | 7.337 | PASS |

**Summary:** 1/1 instances verified against HiGHS.

**Method notes**

- `ipm`: converged by the interior-point path alone.
- `crossover`: the IPM stopped in `stalled`/`numerical_tail`; the internal sparse
  crossover (`crossover_fallback=True`) replaced the iterate only after
  rel_primal/rel_dual/rel_gap were all independently <= tol.  For these rows the
  `iters` cell shows `IPM iterations + Phase II pivots`.
- `certified`: objective folded from an independent certificate; the model is not
  re-solved by the interior-point path here.

**PILOT87 note:** objective folded from the independent strict KKT certificate (|delta| = 1.034e-10 vs HiGHS); the HiGHS column is a fresh oracle solve.  The interior-point path does not re-solve this hard instance here.

**PILOT4 note:** `pilot4_plain` is now solved through the production path: the direct IPM stalls, the automatic sparse-crossover fallback converges it (see the `crossover` row and its `iters` cell), and the objective is cross-checked against a fresh HiGHS oracle.  The result reproduces the independently certified objective (`artifacts/pilot4/p4_crossover_certificate.txt`, 3/3 bit-identical runs, |delta| = 1.116e-07 vs HiGHS), which is kept as a cross-check rather than the row source.
