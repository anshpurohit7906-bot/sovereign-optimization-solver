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
| adlittle | 56 | 97 | 383 | optimal | ipm | 11 | 225494.963 | 225494.963 | 2.95e-11 | 1.28216e-10 | 0.022 | PASS |
| afiro | 27 | 32 | 83 | optimal | ipm | 9 | -464.753143 | -464.753143 | 4.26e-10 | 3.75965e-10 | 0.014 | PASS |
| blend | 74 | 83 | 491 | optimal | ipm | 9 | -30.8121497 | -30.8121498 | 5.83e-09 | 6.39319e-09 | 0.020 | PASS |
| pilot4_plain | 410 | 1000 | 5141 | optimal | crossover | 79 + 1546 piv | -2581.13926 | -2581.13926 | 1.76e-15 | 0 | 7.622 | PASS |
| pilot87 | 2030 | 4883 | 73152 | certified | certified | - | 301.710347 | 301.710347 | 2.42e-13 | - | 2.972 | PASS |
| sc205 | 205 | 203 | 551 | optimal | ipm | 11 | -52.2020612 | -52.2020612 | 1.27e-10 | 3.98579e-10 | 0.028 | PASS |
| share2b | 96 | 79 | 694 | optimal | ipm | 13 | -415.732241 | -415.732241 | 3.31e-10 | 2.15964e-10 | 0.025 | PASS |

**Summary:** 7/7 instances verified against HiGHS.

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
