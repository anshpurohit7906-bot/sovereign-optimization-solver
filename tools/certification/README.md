# Certification tools

Independent verification scripts for the validated LP solvers. These scripts
deliberately **do not trust** any objective reported by a solver — they
reconstruct the solution from persisted artifacts and check each KKT condition
directly.

## `p87_certify.py`

Independent KKT certificate for the PILOT87 sparse-crossover optimum.

- Loads the scaled standard-form data and terminal basis from
  `artifacts/pilot87/`.
- Reconstructs the unscaled standard-form and *original*-LP solution.
- Verifies, in original coordinates: primal feasibility, dual feasibility
  (`A^T y + z = c`, `z >= 0`), complementarity, basis consistency, strong
  duality, and agreement with the HiGHS reference (`301.710347333`) to `< 1e-6`.
- Prints `[PASS]/[FAIL]` per condition and exits nonzero on any failure.

```bash
OPENBLAS_NUM_THREADS=1 python tools/certification/p87_certify.py
```

## `p87_strict_polish.py`

Strict simplex polish applied to the PILOT87 terminal basis before certifying.

- Continues pivoting from the terminal basis with a strict raw
  reduced-cost criterion (no tolerance floor) until all reduced costs are
  `>= 0` at full precision.
- Writes the polished basis/solution to `artifacts/pilot87/p87_strict_polished.npz`.

```bash
OPENBLAS_NUM_THREADS=1 python tools/certification/p87_strict_polish.py
```

Both scripts source their numeric inputs (`p87_prepared*`,
`p87_phase2_v2_final`) and emit results from `artifacts/pilot87/`.
