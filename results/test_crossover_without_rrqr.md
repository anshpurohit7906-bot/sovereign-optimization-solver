# RRQR Necessity Experiment -- PILOT87

**Instance:** PILOT87
- Standard form: m=3608, n=8038, nnz=75,635 (0.26% dense)

## Question

Is the dense RRQR basis crash necessary?

## Results

| Metric | PATH A (with RRQR) | PATH B (no RRQR) |
|---|---:|---:|
| RRQR time | 21.126s | 0.000s |
| Dense mem | 232.0 MB | 0.0 MB |
| Phase I status | feasible | feasible |
| Phase I iters | 16346 | 16346 |
| Phase I time | 538.155s | 544.684s |
| **Total time** | **559.281s** | **544.684s** |
| resid | 1.830e-12 | 1.829647544582258e-12 |
| obj | 431.007431804 | 431.007431804 |

## Conclusion

**RRQR is UNNECESSARY.** Same Phase I iterations, same basis.
RRQR wasted 21.126s + 232.0 MB.
