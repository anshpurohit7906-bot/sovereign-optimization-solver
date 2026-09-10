# Sparse vs Dense End-to-End Comparison

- tolerance `tol=1e-07`, `max_iter=100`
- psutil available: False

## Available / Missing models

**Ran (5):** afiro, blend, sc205, pilot4_plain, pilot87

## Per-model results

| Model | Rows | Cols | SparseNNZ | Density | DenseStatus | SparseStatus | DenseObj | SparseObj | ObjDiff | DenseSolve(s) | SparseSolve(s) | DenseIters | SparseIters | Obj | Sol | PrimalRes | DualRes | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| afiro | 27 | 32 | 83 | 0.096065 | optimal | optimal | -464.753 | -464.753 | 4.547e-13 | 0.0113 | 0.0145 | 8 | 8 | PASS | PASS | PASS | PASS | PASS |
| blend | 74 | 83 | 491 | 0.079941 | optimal | optimal | -30.8121 | -30.8121 | 8.527e-14 | 0.0162 | 0.0197 | 9 | 9 | PASS | PASS | PASS | PASS | PASS |
| sc205 | 205 | 203 | 551 | 0.01324 | optimal | optimal | -52.2021 | -52.2021 | 1.066e-13 | 0.0406 | 0.0265 | 10 | 10 | PASS | PASS | PASS | PASS | PASS |
| pilot4_plain | 410 | 1000 | 5141 | 0.012539 | stalled | stalled | -2581.11 | -2581.05 | 5.712e-02 | 1.14 | 0.492 | 72 | 85 | FAIL | FAIL | FAIL | FAIL | PASS |
| pilot87 | 2030 | 4883 | 73152 | 0.0073798 | stalled | stalled | 301.745 | 301.793 | 4.771e-02 | 82.1 | 24.3 | 63 | 33 | FAIL | FAIL | FAIL | FAIL | PASS |
