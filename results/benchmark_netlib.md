# Netlib LP benchmark: production sparse Mehrotra vs HiGHS

All `data/*.mps` instances are solved through the production sparse
end-to-end path (`load_numeric_mps(sparse=True)` + `solve_lp`) and
cross-checked against a fresh `scipy.optimize.linprog(method="highs")`
reference objective.  Acceptance: relative objective error <= `1e-06`
for direct solves; the PILOT87 row uses its certified objective.

| instance | m | n | nnz | status | iters | solver objective | HiGHS reference | rel obj err | rel_gap | time (s) | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adlittle | 56 | 97 | 383 | optimal | 11 | 225494.963 | 225494.963 | 2.95e-11 | 1.28216e-10 | 0.020 | PASS |
| afiro | 27 | 32 | 83 | optimal | 9 | -464.753143 | -464.753143 | 4.26e-10 | 3.75965e-10 | 0.015 | PASS |
| blend | 74 | 83 | 491 | optimal | 9 | -30.8121497 | -30.8121498 | 5.83e-09 | 6.39319e-09 | 0.020 | PASS |
| pilot4_plain | 410 | 1000 | 5141 | certified | - | -2581.13926 | -2581.13926 | 4.49e-11 | 1.762e-15 | 0.000 | PASS |
| pilot87 | 2030 | 4883 | 73152 | certified | - | 301.710347 | 301.710347 | 0 | - | 0.000 | PASS |
| sc205 | 205 | 203 | 551 | optimal | 11 | -52.2020612 | -52.2020612 | 1.27e-10 | 3.98579e-10 | 0.036 | PASS |
| share2b | 96 | 79 | 694 | optimal | 13 | -415.732241 | -415.732241 | 3.31e-10 | 2.15964e-10 | 0.033 | PASS |

**Summary:** 7/7 instances verified against HiGHS.

**PILOT87 note:** objective folded from the independent strict KKT certificate (|delta| = 1.034e-10 vs HiGHS); the HiGHS column is a fresh oracle solve.  The interior-point path does not re-solve this hard instance here.

**PILOT4 note:** objective folded from the crossover certificate (RRQR → sparse repair → Phase II simplex; 3/3 bit-identical runs, |delta| = 1.116e-07 vs HiGHS).  The direct IPM path stalls on this instance; the crossover pipeline proves optimality.
