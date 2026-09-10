# SuperLU Ordering Benchmark — Real Models

One representative Newton factorization per model. Same pipeline as the synthetic `benchmark_orderings.py`.

## Per-Model Results

### sc205

- A_eq: `(205, 317)`, nnz 665
- ACTUAL Schur S: shape `(205, 205)`, **nnz(S)=1,107**, sparse=True
- nnz(M)=1,107, reg_M=3.211e-08, build time=0.012s

| Ordering | Time (s) | nnz(L) | nnz(U) | Fill-in | Speedup | Success |
|----------|----------|--------|--------|---------|---------|---------|
| COLAMD | 0.000 | 1430 | 1438 | 2.59× | 1.00× | OK |
| MMD_AT_PLUS_A | 0.001 | 1160 | 1172 | 2.11× | 0.56× | OK |
| MMD_ATA | 0.000 | 1524 | 1638 | 2.86× | 1.24× | OK |
| NATURAL | 0.000 | 1536 | 1651 | 2.88× | 2.05× | OK |

- **Best by time:** NATURAL (0.000s)
- **Best by fill-in:** MMD_AT_PLUS_A (2.11×)

---

### pilot4_plain

- A_eq: `(657, 1428)`, nnz 7,736
- ACTUAL Schur S: shape `(657, 657)`, **nnz(S)=15,289**, sparse=True
- nnz(M)=15,289, reg_M=1.911e-08, build time=0.017s

| Ordering | Time (s) | nnz(L) | nnz(U) | Fill-in | Speedup | Success |
|----------|----------|--------|--------|---------|---------|---------|
| COLAMD | 0.003 | 33025 | 33016 | 4.32× | 1.00× | OK |
| MMD_AT_PLUS_A | 0.002 | 16013 | 16013 | 2.09× | 1.23× | OK |
| MMD_ATA | 0.005 | 26113 | 26113 | 3.42× | 0.55× | OK |
| NATURAL | 0.008 | 101990 | 101218 | 13.29× | 0.36× | OK |

- **Best by time:** MMD_AT_PLUS_A (0.002s)
- **Best by fill-in:** MMD_AT_PLUS_A (2.09×)

---

### pilot87

- A_eq: `(3608, 8038)`, nnz 75,635
- ACTUAL Schur S: shape `(3608, 3608)`, **nnz(S)=251,712**, sparse=True
- nnz(M)=251,712, reg_M=6.811e-09, build time=0.086s

| Ordering | Time (s) | nnz(L) | nnz(U) | Fill-in | Speedup | Success |
|----------|----------|--------|--------|---------|---------|---------|
| COLAMD | 0.349 | 1357359 | 1359179 | 10.79× | 1.00× | OK |
| MMD_AT_PLUS_A | 0.110 | 436513 | 435980 | 3.47× | 3.17× | OK |
| MMD_ATA | 0.525 | 919544 | 924836 | 7.33× | 0.67× | OK |
| NATURAL | 1.439 | 4035308 | 4035724 | 32.06× | 0.24× | OK |

- **Best by time:** MMD_AT_PLUS_A (0.110s)
- **Best by fill-in:** MMD_AT_PLUS_A (3.47×)

---

## Summary Across Models

| Model | Best ordering | Speedup vs COLAMD |
|-------|---------------|-------------------|
| sc205 | NATURAL | 2.05× |
| pilot4_plain | MMD_AT_PLUS_A | 1.23× |
| pilot87 | MMD_AT_PLUS_A | 3.17× |

- **All factorizations numerically successful:** True
- **MMD_AT_PLUS_A beats COLAMD on 2/3 models** (of those evaluated)
- **Average MMD_AT_PLUS_A speedup:** 1.65×
  - sc205: 0.56×
  - pilot4_plain: 1.23×
  - pilot87: 3.17×