# PDHG CPU vs GPU Benchmark

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB) 
- **Config**: MAX_ITER=50000, TOL=1e-07, CHECK_EVERY=250, solver_reps=3, e2e_reps=3
- **Timestamp**: 2026-09-02 02:18:48

| # | rows | cols | nnz | cpu_iters | gpu_iters | cpu_s | gpu_solver_s | gpu_e2e_s | str_only | str_e2e | conv_match |
|---|-----:|-----:|----:|----------:|----------:|------:|------------:|----------:|--------:|--------:|:----------:|
| 1 | 200 | 100 | 1,434 | 50,000 | 50,000 | 1.26 | 42.16 | 46.57 | 0.03x | 0.03x | True |
| 2 | 500 | 250 | 7,266 | 50,000 | 50,000 | 2.25 | 46.31 | 45.94 | 0.05x | 0.05x | True |
| 3 | 1,000 | 500 | 22,009 | 50,000 | 50,000 | 5.99 | 44.18 | 46.43 | 0.14x | 0.13x | True |
| 4 | 2,000 | 1,000 | 59,103 | 50,000 | 50,000 | 57.53 | 65.71 | 46.80 | 0.88x | 1.23x | True |
| 5 | 4,000 | 2,000 | 178,065 | 50,000 | 50,000 | 286.69 | 47.15 | 46.90 | 6.08x | 6.11x | True |

### KKT residual / objective comparison

| # | metric | CPU | GPU | match |
|---|--------|----:|----:|:-----:|
| 1 | objective |    -22.4637325 |    -22.4637325 | True |
| 1 | eq_res |       0.000483 |       0.000483 | True |
| 1 | ineq_viol |       0.000337 |       0.000337 | True |
| 1 | dual_feas |       9.13e-05 |       9.13e-05 | True |
| 1 | complement |       0.000208 |       0.000208 | True |
| 2 | objective |      -42.46907 |      -42.46907 | True |
| 2 | eq_res |       0.000655 |       0.000655 | True |
| 2 | ineq_viol |       0.000295 |       0.000295 | True |
| 2 | dual_feas |       0.000427 |       0.000427 | True |
| 2 | complement |        0.00108 |        0.00108 | True |
| 3 | objective |    -53.6778508 |    -53.6778508 | True |
| 3 | eq_res |       0.000188 |       0.000188 | True |
| 3 | ineq_viol |        0.00014 |        0.00014 | True |
| 3 | dual_feas |       0.000163 |       0.000163 | True |
| 3 | complement |       0.000118 |       0.000118 | True |
| 4 | objective |    -88.0215376 |    -88.0215376 | True |
| 4 | eq_res |       8.26e-05 |       8.26e-05 | True |
| 4 | ineq_viol |       5.07e-05 |       5.07e-05 | True |
| 4 | dual_feas |       0.000157 |       0.000157 | True |
| 4 | complement |       6.55e-05 |       6.55e-05 | True |
| 5 | objective |    -241.800429 |    -241.800429 | True |
| 5 | eq_res |       0.000114 |       0.000114 | True |
| 5 | ineq_viol |       8.77e-05 |       8.77e-05 | True |
| 5 | dual_feas |       0.000104 |       0.000104 | True |
| 5 | complement |       5.12e-05 |       5.12e-05 | True |
