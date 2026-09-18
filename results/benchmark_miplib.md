# MIPLIB Benchmark: indigenous B&B vs HiGHS

Acceptance: relative objective error <= 0.0001.

| instance | variables | constraints | binaries | gen. int | continuous | nnz | status | our_obj | oracle_obj | rel_err | nodes | lp_solves | time (s) | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pk1 | 86 | 45 | 55 | 0 | 31 | 915 | node_limit | 93.000000 | 11.000000 | 6.833e+00 | 8 | 9 | 0.16 | FAIL |

**Summary:** 0/1 instances matched HiGHS.
