# SIH26119 architecture

This is a 17-day linear-programming prototype.  The production path is a
CPU, NumPy/SciPy Mehrotra interior-point solver; experiments are isolated
under `experiment/` and are not production dependencies.

## Data flow

```text
data/*.mps
  -> src/mps_parser.py (LPModel)
  -> src/numerical_model.py (NumericalLP: A, b, c, bounds, row types)
  -> src/lp/mehrotra.py (standard-form conversion, scaling, IPM iteration)
  -> src/lp/linear_system.py (reduced Newton factorization/solves)
  -> MehrotraResult (original-variable solution and diagnostics)
```

## Current modules

- `src/mps_parser.py` parses the supported MPS subset (`NAME`, `ROWS`,
  `COLUMNS`, `RHS`, `BOUNDS`) into `LPModel`.  It preserves row types,
  coefficients, objective, and bounds.  RANGES, SOS, quadratic objectives,
  indicators, and integer semantics are unsupported.
- `src/numerical_model.py` converts `LPModel` to dense `NumericalLP` arrays
  and validates shapes and metadata.  It is the boundary between parsing and
  numerical solvers.
- `src/lp/mehrotra.py` is the production LP solver.  It converts E/L/G models
  and supported variable bounds to equality-standard form with nonnegative
  variables/slacks, applies existing equilibration, runs Mehrotra
  predictor-corrector IPM, and maps results back to original units.
- `src/lp/linear_system.py` is the production Newton backend.  It factors the
  reduced Schur system once per IPM iteration and reuses that factorization for
  predictor/corrector right-hand sides.  Its sparse SciPy path is the current
  production backend.  The H-regularization is scale-aware
  (`rho_p = reg * max(1, mean|h|)`, `reg = 1e-12`) and capped at
  `MAX_RHO_P = 5e-8`; the cap bounds the per-step dual-side bias
  `rho_p * ||dx||_inf` that the uncapped formula (which grows like
  `h_max = O(1/mu)`) imposed on degenerate tails.
- `src/scaling.py` provides the existing row/column scaling used by production
  Mehrotra; it is not an experimental tuning surface.
- `src/pdhg_mixed.py` is a separate NumPy mixed E/L PDHG prototype.  It keeps
  equality duals free, L-row duals nonnegative, and projects variable bounds.
  It is not the current production solving path.
- `src/pdhg_lp_solver.py` is the older all-`Ax <= b, x >= 0` toy PDHG reference
  and demonstration.  It is useful for algorithm comparison, not for mixed
  MPS production models.

## Termination and best-iterate policy (Mehrotra)

Iterations run in row/column-equilibrated coordinates, but every reported
metric and the final acceptance test are recomputed in ORIGINAL standard-form
coordinates.  Two consequences are enforced in production:

- An `optimal` status additionally requires the original-coordinate
  `rel_primal`, `rel_dual`, and `rel_gap` to satisfy the tolerance; if only the
  scaled trajectory criterion passed, the solver downgrades the status instead
  of over-reporting.
- On any non-optimal exit (`stalled`, `max_iterations`, `numerical_tail`, ...)
  the returned iterate is the one minimizing the ORIGINAL-coordinate merit
  `max(rel_primal, rel_dual, rel_gap)` — the same scalarization the convergence
  test uses.  Selecting on the scaled gap instead could return a point whose
  original primal residual is materially worse (the original PILOT4 failure
  mode: merit 3.85e-4 selected by scaled gap vs 4.3e-6 after the fix).

## Data and verification

- `data/afiro.mps` and `data/pilot87.mps` are benchmark inputs.
- `tests/test_mps_parser_hardening.py`, `tests/test_lp_edge_cases.py`, and
  `tests/test_mehrotra_reporting.py` provide parser, production LP, and
  termination/best-iterate/H-cap regression coverage.
- `tests/verify_with_highs.py` is a planned/external verifier path: it is for
  comparison and diagnostics, never a production solve engine.
- `tests/run_benchmarks.py` is the planned benchmark harness: it records
  benchmark status, iterations, objective, residuals, and runtime.

## Sparse crossover and PILOT87 certification

Beyond the Mehrotra IPM production path, `experiment/crossover/` holds an
independent sparse crossover path that drives the IPM terminal basis to a
strict simplex-verified optimum:

```text
RRQR basis identification -> sparse Phase I -> sparse Phase II
  -> strict polish (tools/certification/p87_strict_polish.py)
  -> independent KKT certificate (tools/certification/p87_certify.py)
```

This pipeline is validated on `data/pilot87.mps`, reproducing the HiGHS
reference objective `301.710347333` with every residual below `1e-6`. It is
kept isolated under `experiment/crossover/` and `tools/certification/` and is
**not** a production dependency of `src/lp/mehrotra.py`.

### Verification and certification workflow

1. Persist the terminal basis and scaled solution to
   `artifacts/pilot87/` (`.npz` inputs: `p87_prepared*`, `p87_phase2_v2_final`).
2. Run the strict polish on the terminal basis:
   `python tools/certification/p87_strict_polish.py` — pivots any raw
   negative reduced costs to a strict criterion and writes
   `p87_strict_polished.npz`.
3. Run the independent certificate:
   `python tools/certification/p87_certify.py` — reconstructs the unscaled
   standard-form and original-LP solution and checks each KKT condition
   independently (primal, dual, complementarity, basis, original feasibility,
   strong duality, and agreement with the HiGHS reference to `< 1e-6`).
   It **does not trust** the solver-reported objective.

Both scripts emit `[PASS]/[FAIL]` verdicts and exit nonzero on any failure.
Text certificates are archived at `artifacts/pilot87/p87_strict_certificate.txt`
and `artifacts/pilot87/p87_certificate.txt`.

## Experiments

`experiment/` contains deliberately isolated hypotheses.  In particular,
`experiment/regularization/` remains the documented negative HOPDM-style
global-rho experiment and must not be folded into production.  The MCC
experiment in `experiment/mcc/` reuses production conversion, scaling, and
linear-system code, changing only the number of additional centrality
correction solves (0, 1, or 2).

## Responsibility boundaries and overlaps

- The parser and numerical model have distinct responsibilities; neither
  should solve or standardize models.
- `mehrotra.py` owns standard-form conversion and production solve control;
  `linear_system.py` owns only factorizations and Newton solves.
- `pdhg_lp_solver.py` and `pdhg_mixed.py` overlap as PDHG implementations.
  The former has a narrower all-inequality contract; the latter supersedes it
  for mixed E/L/bounded prototypes.  They should remain clearly labelled
  reference/prototype code until one is selected.
- The Mehrotra and PDHG paths both consume `NumericalLP`, but intentionally
  use different formulations: Mehrotra converts to standard form, whereas
  mixed PDHG uses original E/L rows directly.  Solver-specific conversions
  must not leak into `numerical_model.py`.
- Experiment solvers duplicate controlled portions of the production loop by
  design.  They must reuse production interfaces/backends where stated and
  must not become alternate production code accidentally.

## Intentionally out of scope for this prototype

- CUDA/GPU kernels and a GPU execution layer (future layer only).
- MILP, integer variables, QP, conic optimization, SOS, or indicator models.
- A GUI/UI layer (future layer only); current interfaces are Python APIs and
  CLIs/harnesses.
- Replacing the production factorization backend, adding new regularization,
  or combining isolated experiments with scaling changes.
- Treating external solvers as the application solve engine.  They may be
  used only as planned verification or benchmark references.
