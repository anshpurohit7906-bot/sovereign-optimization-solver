# SIH 26119 — Sovereign Optimization Solver

**Indigenous GPU-Accelerated Optimization Solver — Sovereign Alternative to Commercial Optimization Engines**

A from-scratch optimization engine being developed for SIH 26119. The long-term objective is a unified optimization platform covering Linear Programming (LP), Quadratic Programming (QP), Mixed-Integer Linear Programming (MILP), sparse and large-scale optimization, and GPU acceleration where it provides a measurable benefit.

The implementation is independently developed from mathematical foundations. Existing solvers such as HiGHS, PDLP/OR-Tools, and NVIDIA cuOpt may be studied as architectural, numerical, or benchmarking references, but they are not used as the optimization engine.

> **Current milestone:** the numerically reliable LP foundation (Mehrotra IPM + sparse CPU backend + crossover/certification) is established. Canonical sparse QP (`src/qp/`), a native MILP branch-and-bound prototype with verified incumbent heuristic, and an opt-in CUDA/CuPy Mehrotra reduced-Newton backend are implemented. Current work focuses on sparse end-to-end flow, stronger MILP, presolve, and sparse GPU acceleration.

---

## Verified Result (PILOT87)

The independent **validation / experimental** pipeline drives the Mehrotra
IPM terminal basis to a strictly verified optimum on the hard `pilot87.mps`
benchmark:

```text
Mehrotra IPM terminal basis
  → row/column equilibration
  → RRQR basis identification
  → sparse composite Phase-I repair
  → Phase-II simplex
       (driver-side repair/resume + cond cleanup)
  → independent verification on original data
       (primal residual + reduced-cost/dual residual + gap)
  → VERIFIED OPTIMAL
```

That RRQR → Phase I → Phase II → verification workflow is the
**independent validation path** used to verify PILOT87 (and PILOT4). It is
*not* executed for every production solve: the production LP path is the
Mehrotra IPM, and the sparse crossover fallback is entered **only** when the
IPM stalls or lands in the numerical tail *and* the crossover gate is
triggered (see [Architecture](#architecture)).

Standalone Mehrotra on PILOT87 stalls in the numerical tail; the strict
verified optimum below is produced by the **independent validation workflow**
(see [Verified Result](#verified-result-pilot87)), not by the raw IPM iterate.
PILOT87 is therefore **not** unresolved by the complete solver pipeline — it is
verified optimal to objective `301.710347333`.

| Quantity | Value |
|---|---|
| Strict reduced-cost polish | 30 pivots from the terminal Phase-II basis |
| Independently recomputed original objective | `301.710347333` |
| HiGHS reference | `301.710347333` |
| Objective difference | ≈ `1.1e-10` (relative ≈ `3.7e-13`) |
| Primal residual | ≈ `1.6e-11` |
| Basis residual | ≈ `1.6e-11` |
| Complementarity | ≈ `-3.7e-13` |
| Raw reduced-cost minimum | ≈ `-1.4e-14` |
| Classification | `VERIFIED OPTIMAL` |
| Certificate | `artifacts/pilot87/p87_strict_certificate.txt` (strict polish) / `artifacts/pilot87/p87_certificate.txt` (certify) |

Two independent certificates are produced. The **strict certificate**
(`p87_strict_certificate.txt`, populated from `p87_strict_polish.py` stdout by
the orchestrator) is the basis for the objective/reduced-cost numbers above:
after pivoting away every raw negative reduced cost, the *independently
recomputed* original objective is `301.710347333`, agreeing with the HiGHS
reference to `1.1e-10` (relative `3.7e-13`). The **certify certificate**
(`p87_certificate.txt`, written by `tools/certification/p87_certify.py`)
reconstructs the unscaled standard-form and original-LP solution from the
persisted artifacts, checks each KKT condition independently, and (as of the
tightened check) also confirms agreement with the HiGHS reference to within
`1e-4` — well above the observed `~4.6e-5` gap on the pre-polish terminal
basis. Neither trusts any objective reported by the solver. See
[`ARCHITECTURE.md`](ARCHITECTURE.md#verification-and-certification) for the
workflow.

> **Numerical honesty:** a numerical failure is preferable to a falsely reported
> optimum. Claims are backed by reproducible certificates, never by a single
> solver printout.

---

## Repository Layout

```text
src/                  production core: parser, numerical model, Mehrotra IPM,
                      linear system (+ opt-in CUDA/CuPy backend), scaling,
                      crossover, MILP branch-and-bound
src/qp/               canonical sparse convex QP package (Mehrotra IPM on the
                      inequality-form KKT; QPS/SIF reader; KKT certificates)
tests/                production regression + edge-case test suite (incl.
                      test_qp_sparse.py, test_qps_reader.py, test_milp_*,
                      test_gpu_backend.py)
data/                 MPS benchmark inputs (afiro.mps, pilot87.mps, ...)
data/refinery_sas/   provenance + formulation notes for the public SAS/OR mpex06 refinery LP
demo/                demo segments (demo_sih.py, refinery_public.py, etc.)
demo/refinery_public.py public SAS/OR refinery LP demo (continuous LP, NOT MILP)
opticore.py         public CLI: `opticore solve <mps> [--crossover] [--cert-fallback] [--time-limit SECS] [-v]`
pyproject.toml      packaging metadata + console-script entry point (`opticore = "opticore:main"`)
tools/                benchmark harness + certification scripts
  tools/benchmark_netlib.py        Netlib LP benchmark vs HiGHS reference
  tools/benchmark_maros.py         Maros–Mészáros QP benchmark (SIF fetched, never committed)
  tools/benchmark_miplib.py        MIPLIB MILP benchmark harness
  tools/validate_milp_incumbent.py direct B&B validation + independent incumbent check
  tools/certification/            independent KKT certificate + strict-polish
scripts/download_maros_qp.py      fetches QP SIF files into data/qp/
requirements-gpu.txt  optional CUDA/CuPy backend (cupy; CPU-only runs need nothing extra)
experiment/           isolated research: crossover, pdhg, sparse, gpu, ...
results/              generated benchmark/experiment reports (markdown + csv)
artifacts/pilot87/    validated PILOT87 results (npz + certificates)
archive/              historical research, superseded experiments, logs
```

---

## Installation

The core depends only on NumPy and SciPy; tests add `pytest`. No other
third-party packages are required (the HiGHS reference value is obtained via
SciPy's built-in `scipy.optimize.linprog` — it is never the solve engine).

```bash
python -m pip install -r requirements.txt            # runtime
python -m pip install -r requirements-dev.txt        # + pytest for the test suite
```

Run the production test suite with:

```bash
python -m pytest tests/
```

---


## Current Status

The current repository contains a working from-scratch LP optimization core centered around a **Mehrotra predictor-corrector primal-dual interior-point method**.

The project has also explored Revised Simplex and several PDHG/PDLP-style approaches. These are retained as independent algorithmic paths and research experiments rather than being treated as interchangeable production components.

### Implemented

- MPS parsing for currently supported MPS features
- Explicit rejection of unsupported:
  - RANGES
  - SOS
  - QUADOBJ
  - INDICATORS
- Integer-variable support:
  - integer MARKER / INTORG / INTEND constructs
  - binary (BV) and general-integer (LI/UI) bound types
  - mixed-integer models feed the native MILP branch-and-bound
- Numerical LP representation
- Equality (`E`), less-than (`L`), and greater-than (`G`) constraint handling
- Standard-form LP conversion
- Variable-bound transformations:
  - lower bounds
  - upper bounds
  - free variables
  - fixed variables
  - boxed variables
- Row/column equilibration
- Mehrotra predictor-corrector primal-dual interior-point method
- Dense Cholesky linear-system backend
- Sparse CSR + SuperLU Schur-complement linear-system backend
- Opt-in CUDA/CuPy dense-Schur reduced-Newton backend (`src/lp/gpu_linear_system.py`;
  `backend="cpu"` default unchanged, `"gpu"` explicit, `"auto"` GPU-with-CPU-fallback)
- Fraction-to-boundary step control
- Numerical safeguards and finite-value checks
- Residual-based convergence testing
- Scale-aware numerical-tail detection
- Best-trusted-iterate preservation
- Escalating diagonal regularization
- Validated sparse crossover **fallback** (`src/lp/crossover.py`: the sparse
  Phase I crash + sparse Devex Phase II refined-simplex fallback used by
  `solve_lp` when the Mehrotra IPM stalls or lands in the numerical tail and the
  crossover gate passes; no RRQR or external basis needed; the
  `crossover` method label marks the specific Netlib instances where the fallback
  actually replaced the IPM iterate)
- Canonical sparse convex QP package (`src/qp/`: inequality-form Mehrotra IPM,
  sparse SPLU saddle-point backend, independent KKT certificate, QPS/SIF reader;
  legacy dense standard-form slice `src/lp/qp.py` retained for compatibility)
- Native MILP branch-and-bound prototype (`src/lp/branch_bound.py`: LP
  relaxations, node management, pruning, diversified verified rounding
  heuristic for incumbent generation — every incumbent is independently
  re-verified before acceptance)
- Experimental / secondary Revised Simplex implementation
- Experimental PDHG/PDLP research implementations

### Not Yet Implemented

- Full end-to-end sparse LP data flow (numerical model and several
  standard-form transformations still use dense matrices)
- Presolve and advanced reductions
- Parallel linear algebra
- Sparse (as opposed to dense-Schur) GPU acceleration
- Advanced sparse ordering
- Formal infeasibility and unboundedness certificates
- Production-grade algorithm-selection / dispatch layer

---

# Architecture

### Primary LP solver

Mehrotra is the **primary LP solver**. Crossover is **not** always run — it is
a conditional fallback that only triggers on a stalled / numerical-tail IPM
iterate that also passes the crossover gate:

```text
MPS
 │
 ▼
Standard-Form Conversion
 │
 ▼
Scaling / Equilibration
 │
 ▼
Mehrotra Predictor-Corrector IPM
 │
 ├── Dense Cholesky backend
 ├── Sparse CSR + SuperLU Schur backend (production default)
 └── Dense-Schur CUDA/CuPy backend (opt-in, backend="gpu"/"auto")
 │
 ▼
IPM status
 │
 ├── converged ──────────────────────► accept the IPM iterate (crossover NOT run)
 │
 └── stalled / numerical_tail
     AND crossover_fallback=True
     AND best_merit <= CROSSOVER_MERIT_RATIO * tol        (the crossover gate)
     │
     ▼
     sparse crossover fallback
     (sparse Phase I crash → sparse Devex Phase II, no RRQR needed)
     │
     ▼
     independent acceptance checks (rel_p, rel_d, rel_gap all <= tol)
     │
     ├── accepted → iterate replaced, status = "optimal"
     └── rejected / failed → keep the IPM best-trusted iterate, report honestly
 │
 ▼
MehrotraResult
```

In the common case the IPM converges and the returned `MehrotraResult` is the
IPM iterate itself; no simplex work is performed. The `crossover` method label
in `results/benchmark_netlib.md` therefore marks the specific instances where
the fallback actually replaced the iterate.

### Crossover: production fallback vs. independent certification

Two distinct things share the word "crossover" in this repository.

**1. Production fallback** (`src/lp/crossover.py`, used by `solve_lp` when the
gate above passes). Promoted from the validated PILOT4/PILOT87 experimental
pipeline with no algorithmic change, and deliberately **free of RRQR, dense
factorization or any external starting basis**:

```text
sparse_phase1(A, b)      textbook two-phase crash (B = I, no RRQR)
  │
  ▼
sparse_phase2(A, b, c, basis)
                         revised simplex, sparse LU refactorization,
                         iterative refinement, Devex pricing, Bland
                         tie-breaking, condition-aware repair
  │
  ▼
crossover_from_ipm(...)  orchestrator → candidate iterate
  │
  ▼
independent acceptance checks (rel_p, rel_d, rel_gap <= tol)
```

**2. Independent validation / experimental pipeline** (equilibration → RRQR basis
identification → sparse composite Phase-I repair → Phase-II simplex with
driver-side repair/resume and vertex-preserving degenerate-pivot cond cleanup
→ independent verification on the original unscaled data: primal residual
`A x = b`, reduced-cost/dual residual `c_N − Nᵀy ≥ 0`, and primal/dual gap).
This is the path behind the [verified PILOT87 result](#verified-result-pilot87);
it lives in `experiment/crossover/` (`run_pilot87_crossover.py`) and
`tools/certification/` (`p87_strict_polish.py`, `p87_certify.py`) and writes the
artifacts under `artifacts/`. It is **not** part of the per-solve production
path and is not executed for every LP.

Revised Simplex currently consumes the standard-form representation through a separate implementation:

```text
NumericalLP
 │
 ▼
StandardFormLP
 │
 ▼
Revised Simplex
```

PDHG/PDLP research implementations are maintained separately under:

```text
experiment/pdhg/
```

They are not currently part of the primary production LP path.

---

## Quick Start / CLI

The public command is **`opticore`** (module `opticore.py` at the repository
root; console entry point declared in `pyproject.toml`).

```bash
python -m pip install -e .
```

### Solve an MPS file

```bash
opticore solve data/afiro.mps
opticore solve data/afiro.mps --crossover
opticore solve data/pilot4_plain.mps --crossover
```

Every `opticore solve` is a **genuine live solve** by default. `--crossover`
enables the conditional sparse-crossover fallback described in
[Architecture](#architecture); if the IPM converges directly, no crossover is
run and the CLI says so. `-v` / `--verbose` streams the real per-iteration IPM
log and the crossover decision lines from the solver core.

Sample output:

```text
OPTICORE - indigenous LP solver
  input file   : data/afiro.mps
  problem      : AFIRO (objective row: COST)
  variables    : 32
  constraints  : 27
  nonzeros     : 83
  method       : Mehrotra IPM (CPU backend)
  crossover    : disabled
  RESULT SOURCE : LIVE OPTICORE SOLVE
  LIVE SOLVE    : COMPLETED
  status       : optimal
  objective    : -464.753142659
  iterations   : 9
  ...
```

### PILOT87: live solve with a wall-clock limit

```bash
opticore solve data/pilot87.mps --crossover --time-limit 120
```

This is a **genuine live solve** with a wall-clock limit; the certificate is
never consulted. PILOT87 is a hard instance — the standalone IPM stalls in the
numerical tail (see [PILOT87](#pilot87)) — so the live solve typically reaches
the limit and then reports honestly:

```text
RESULT SOURCE : LIVE OPTICORE SOLVE
LIVE SOLVE    : INCOMPLETE (TIME LIMIT of 120s reached ...)
status        : TIME LIMIT
```

The CLI never silently switches to the certificate after a timeout. PILOT87 is **NOT** a fast live benchmark; the limit exists precisely so a demo can bound
the wait.

### PILOT87: explicit certified fast path

```bash
opticore solve data/pilot87.mps --crossover --cert-fallback
```

This is an **explicit opt-in fast path** that reports the stored strict KKT
certificate. It does **not** run the live solve:

```text
RESULT SOURCE : STORED STRICT KKT CERTIFICATE
LIVE SOLVE    : NOT RUN (explicit --cert-fallback fast path)
```

It exists for demonstrations under time pressure, and must never be presented
as a fresh solve. On any instance other than PILOT87 the flag is ignored with a
notice and a normal live solve runs.

### Exit codes

`0` only when the solve reports `optimal`. A missing file, an invalid MPS, a
solver failure, or a `--time-limit` expiry all exit non-zero with a clear
message and no traceback unless `-v` is given. There is no silent fallback to
any other solver.

---

## SIH 26119 Demo

```bash
python demo/demo_sih.py --refinery-public
```

This segment reproduces the **public SAS/OR `mpex06` refinery-planning LP** as a
continuous LP (never an artificial MILP) and exposes the complete live workflow
stage by stage:

```text
public SAS/OR refinery data
  → NumericalLP construction + structural gate
  → live Mehrotra IPM (real per-iteration log)
  → post-solve / crossover status
  → independent feasibility / objective verification
  → crude-2 availability what-if
  → second live re-solve
  → second verification
```

| Item | Value |
|---|---|
| Data source | SAS/OR sample library `mpex06` (transcribed verbatim; provenance in `data/refinery_sas/`) |
| Data policy | **public SAS/OR benchmark data — NOT MRPL operational data** |
| Variables / constraints / nonzeros | `51` / `45` (E=37, L=4, G=4) / `154` |
| Published baseline objective | ≈ `211365.13477` |
| OPTICORE baseline objective | ≈ `211365.13471` |
| Relative difference | ≈ `2.85e-10` (gate: `rel < 1e-6`; the `6e-05` absolute gap is informational only) |
| What-if change | crude-2 availability `30000 → 24000` (a new scenario, **not** compared to the published optimum) |
| What-if objective | ≈ `205304.48801` |

Notes on the honesty of that output:

* both solves print the **actual** IPM iteration log emitted by
  `solve_lp(verbose=True)` (7 iterations each for the base and what-if LPs on
  this instance) — nothing is simulated and there are no added delays;
* `constraints : 45 (E=37, L=4, G=4)` differs from the SAS-published `46/158`
  because exactly **one redundant** fuel-oil equality is omitted — the four
  SAS-written fuel-oil rows sum to zero identically, so keeping the fourth trips
  the standard-form rank guard. Every retained coefficient is verbatim SAS data;
  the reasoning is recorded in `data/refinery_sas/README.md`;
* the post-solve stage reports the real crossover state — this refinery LP does
  not request crossover and the IPM converges directly, so it prints
  `Crossover requested : NO` / `IPM solution used : YES` rather than claiming a
  crossover ran;
* verification is **independent feasibility/objective verification**
  (row-type-aware primal residuals plus an independently recomputed `c @ x`), **NOT** a KKT optimality proof;
* no certificate is used anywhere in this demo.

Other demo modes: `python demo/demo_sih.py` (default MILP + QP segments),
`--milp-only`, `--qp-only`.

---

# Why Mehrotra IPM?

The first development path explored several **Primal-Dual Hybrid Gradient (PDHG)** approaches.

Experiments included:

* preconditioning
* scaling
* restarts
* adaptive step-size mechanisms
* Barzilai-Borwein-style updates
* PDLP-inspired primal weighting
* unified bounded-constraint formulations

These experiments were useful for understanding the numerical behavior of the benchmark problems.

In particular, SC205 exhibited a persistent stationarity limitation under the tested PDHG configurations.

The project therefore pivoted toward a **Mehrotra primal-dual interior-point method** as the current LP foundation.

The earlier PDHG work is retained as an experimental and research path because it remains relevant to future large-scale and GPU-oriented development.

---

# Numerical Robustness

Numerical reliability is treated as a first-class design requirement.

The Mehrotra implementation includes:

* primal and dual residual monitoring
* complementarity monitoring
* relative duality-gap monitoring
* fraction-to-boundary step control
* diagonal regularization
* scale-aware convergence thresholds
* explicit numerical-failure states
* numerical-tail detection
* preservation of the best trusted iterate

The solver distinguishes between states such as:

```text
optimal
numerical_tail
stalled
max_iterations
numerical_failure
```

This is intentional.

A solver reaching numerical difficulty must not silently report an apparently optimal solution.

---

# Linear Algebra & Scalability

The linear-system layer currently contains two backends.

### Dense backend

A dense Cholesky path is retained for general symmetric positive-definite systems and fallback use.

### Sparse backend

When the Mehrotra Newton system has diagonal `H`, the solver can use:

```text
A_E → CSR
H     → diagonal vector
S     = A_E H⁻¹ A_Eᵀ
      → sparse SuperLU factorization
```

The factorization is reused for the predictor and corrector solves within an iteration.

Iterative refinement is also applied to the sparse Schur solve when useful.

### Current limitation

The sparse linear-system backend exists, but the overall LP pipeline is **not yet fully sparse end-to-end**.

The numerical model and several standard-form transformations still use dense matrices.

Therefore the current architecture should not yet be described as a fully scalable sparse LP solver.

---

# PILOT87

PILOT87 is currently the primary large-scale numerical stress case.

**Standalone Mehrotra** reaches iteration 36 before entering a numerical tail
and returns:

```text
status = stalled
```

The important observation is that this is **not treated as an optimal result**.

The diagnosis is associated with severe conditioning in the reduced Schur
complement as the barrier iterations approach the boundary and the diagonal
scaling

```text
H = diag(z / x)
```

develops an extreme dynamic range.

The existing sparse backend removes unnecessary dense work, but it does not, on
its own, solve the underlying conditioning problem for the raw IPM iterate.

**Completing the pipeline resolves PILOT87.** The **standalone Mehrotra IPM
path stalls** on PILOT87 (its extreme dynamic range drives a numerical tail),
so PILOT87 is **not** solved end-to-end by the production per-solve path.
Instead, the strictly verified optimal result for PILOT87 is produced by the
**independent validation workflow** in `experiment/crossover/run_pilot87_crossover.py`
— row/column equilibration → RRQR basis identification → sparse composite
Phase-I repair → Phase-II simplex (driver-side repair/resume and
vertex-preserving degenerate-pivot cond cleanup) → independent verification of
the original unscaled data (primal residual `A x = b`, reduced-cost/dual
residual `c_N − Nᵀy ≥ 0`, and primal/dual gap) (artifacts under
`artifacts/pilot87/`) — which yields an original objective of `301.710347333`
that agrees with the HiGHS reference to within `1.1e-10` (relative ≈
`3.7e-13`) per the strict KKT check recorded in the certificate
(see [Verified Result](#verified-result-pilot87)).

The **production solver's crossover fallback** is a separate path: when the
Mehrotra IPM stalls or hits the numerical tail and the crossover gate passes,
it runs sparse Phase I crash + sparse Devex Phase II followed by the
independent acceptance checks. That production fallback does **not** use RRQR
basis identification — it starts from the all-artificial `B = I` two-phase crash
and does not run strict reduced-cost polish; those RRQR → Phase-I repair →
Phase-II simplex validation steps are reserved for the independent validation
path above.

So the limitation applies to the **standalone Mehrotra IPM path**, not to the
complete solver pipeline. PILOT87 remains an active scalability target for the
standalone path; it is verified optimal through the independent
validation workflow.

---

# Revised Simplex

A from-scratch Revised Simplex implementation is present under:

```text
src/lp/simplex.py
```

It uses:

* explicit basis management
* simplex multipliers
* reduced costs
* Bland-style entering-variable selection
* deterministic Phase-I initialization
* artificial-variable removal
* redundant-row handling

It is currently classified as **experimental / secondary**.

It should not yet be treated as the project's primary LP solver or as production-ready.

Its performance and numerical behavior are being evaluated independently from Mehrotra.

---

# PDHG / PDLP Research Path

The repository contains multiple experimental PDHG/PDLP-style implementations under:

```text
experiment/pdhg/
```

These experiments investigate:

* primal-dual updates
* preconditioning
* scaling
* adaptive step sizes
* Barzilai-Borwein updates
* restart mechanisms
* PDLP-inspired primal weighting
* bounded-constraint formulations

The PDHG path is intentionally separated from the current primary LP implementation.

The experiments are useful for algorithmic comparison and for future large-scale/GPU-oriented work, but current results should not be presented as equivalent to the verified Mehrotra LP core.

---

# GPU Backend (LP/Mehrotra, opt-in)

An optional CUDA/CuPy backend accelerates the Mehrotra reduced-Newton
system.  It is **explicitly opt-in** and never the default:

```python
from lp.mehrotra import solve_lp
r = solve_lp(lp, backend="gpu")    # raises MehrotraError without CUDA/CuPy
r = solve_lp(lp, backend="auto")   # GPU when available, else CPU fallback
r = solve_lp(lp)                   # backend="cpu" (default, unchanged)
```

Install (in addition to `requirements.txt`):

```bash
python -m pip install -r requirements-gpu.txt   # optional (installation compatibility
                                                # note: requirements-gpu.txt pins cupy-cuda12x>=13.0;
                                                # on CUDA 13.x hosts install cupy-cuda13x instead)
```

Verified on real hardware: CuPy 14.2.0 + NVIDIA GeForce RTX 3050 Laptop GPU
(`gpu_available() == True`). The deterministic CPU-vs-GPU LP test
(`tests/test_gpu_backend.py::test_deterministic_lp_cpu_vs_gpu_match`) **passed**.
On this CUDA-capable machine the GPU test module reports 7 passed with 2 expected
skips (the no-GPU fallback/failure-path tests, which only apply without CUDA/CuPy);
on CPU-only machines the 2 skips are instead the GPU-requiring tests.

**Current limitation (documented, intentional):** the GPU path is a
**dense Schur-complement** implementation (`src/lp/gpu_linear_system.py`):
`S = A diag(1/h) A^T` is assembled and Cholesky-factorized on the device
with up to two iterative-refinement corrections.  It is *not* a sparse CUDA
factorization, so for large/sparse Netlib-class models the sparse CPU
backend (SuperLU on the Schur complement) remains the production path.
The GPU backend requires a strictly positive, finite diagonal `H` (always
true inside the Mehrotra iteration).  CuPy is imported lazily; CPU-only
environments run OPTICORE normally without it.

# QP Support (canonical sparse package: `src/qp/`)

`src/qp/` is the **canonical QP implementation**: a sparse-capable convex-QP
Mehrotra predictor-corrector interior point method operating on the native
inequality-form KKT system.

```
src/qp/problem.py         QPProblem: sparse P/q/G/h/A/b/lb/ub + validation,
                          sparse PSD check via eigsh (no densification)
src/qp/solver.py          solve_qp(): Mehrotra predictor-corrector IPM,
                          best-iterate restore, per-iteration history
src/qp/linear_system.py   solve_kkt(): sparse SPLU saddle-point backend
                          [[H+eI, A'], [A, -eI]] + dense fallback
src/qp/verify.py          certificate(): independent KKT certificate
                          (stationarity incl. scaled variant, feasibility,
                          dual sign, complementarity)
src/qp/qps.py             read_qps(): Maros–Mészáros QPS/SIF reader
src/qp/benchmark.py       deterministic sparse QP generator (seed 26119)
                          + NPZ suite writer + benchmark runner
src/qp/io.py              NPZ save/load for QP problems
```

Legacy path: **`src/lp/qp.py` is retained unchanged** as the standard-form
compatibility slice (E/L/G rows + bounds → standard form, dense KKT,
active-set polish). It has its own tests (`tests/test_qp_skeleton.py`).
New callers should prefer `src.qp.solve_qp`; `src/lp/qp.py` will be
migrated or retired in a later, deliberate PR.

Run the QP test suite:

```bash
python -m pytest tests/test_qp_sparse.py tests/test_qps_reader.py
```

Maros–Mészáros benchmark (SIF datasets are fetched, never committed):

```bash
python scripts/download_maros_qp.py     # fetches SIF files into data/qp/
python tools/benchmark_maros.py         # writes reports/maros_qp_benchmark.csv
python tools/benchmark_maros.py --include-generated
```

Current validation: **5/6 Maros–Mészáros instances certify optimal** through the
independent KKT certificate. **QSHIP12S (n=2763) currently reaches
`max_iterations`** after 200 iterations — an honest limitation of the current
QP iteration budget/regularization on this instance, not a regression.

# MILP Support (native branch-and-bound prototype: `src/lp/branch_bound.py`)

A from-scratch branch-and-bound implementation solves the LP relaxation at
each node (production Mehrotra path), with node management, pruning, and a
**diversified verified rounding heuristic** for incumbent generation: a small
bounded set of rounded integer candidates (global down/up plus top-k
floor/ceil flips) is generated from each relaxation, repaired through a
fix-integers → re-solve continuous LP step, and **accepted only after an
independent feasibility re-check** (rows, bounds, integrality, objective).
Global deadline and node limits are respected; statuses (`optimal`,
`node_limit`, `time_limit`) are reported honestly.

```bash
python -m pytest tests/test_milp_skeleton.py tests/test_milp_robustness.py   # 28 MILP tests
python tools/validate_milp_incumbent.py pk1 2000 45      # direct B&B validation + independent check
```

Current validation: `pk1` improves 18.000 → **17.000** via the heuristic;
`mas74` holds **14075.45**; **50v-10 reaches `node_limit` with a verified
feasible incumbent 7002.77 and a substantial remaining gap (bound 2962.88,
rel. gap ≈ 0.577)** — MILP remains a prototype, not a commercial-grade solver.
No cuts, strong branching, warm starts, or parallelism are implemented.

# MPS Support

The parser currently supports the principal sections required by the current LP benchmark set:

```text
NAME
ROWS
COLUMNS
RHS
BOUNDS
ENDATA
```

Supported row types include:

```text
E
L
G
```

Supported bound types include:

```text
LO
UP
FX
FR
MI
PL
BV
LI
UI
```

Integer MARKER / INTORG / INTEND constructs in COLUMNS are supported and mark
the enclosed variables integer; BV, LI, and UI bound types likewise mark
variables binary/integer. Mixed-integer models feed the MILP branch-and-bound.

Unsupported advanced MPS constructs are now **rejected explicitly** rather than silently ignored.

This is particularly important for constructs such as RANGES: the parser must never construct a different mathematical problem without informing the user.

RANGES, SOS, QUADOBJ, and INDICATORS remain rejected (covered by
`tests/test_mps_parser_hardening.py`).

---

# Benchmarking

Benchmarking uses standard LP instances such as:

```text
AFIRO
SC205
ADLITTLE
SHARE2B
BLEND
PILOT87
```

HiGHS and other established optimization software may be used as **reference/oracle implementations for validation and comparison**.

They are not used as the optimization engine.

## Current Benchmark Results

### Production tests

* `pytest tests/`: **167 passed, 2 skipped** (2 skips are hardware-conditional
  GPU tests; on CPU-only machines the GPU-requiring tests skip, on CUDA/CuPy
  machines the no-GPU fallback/failure-path tests skip instead)
* MILP: **28/28 passed** (`tests/test_milp_skeleton.py` + `tests/test_milp_robustness.py`)
* New QP: **17 QP tests** (`tests/test_qp_sparse.py` — 16, `tests/test_qps_reader.py` — 1)
* GPU: **7 passed, 2 expected skips** on CPU-only CI
  (`tests/test_gpu_backend.py` — the 2 skips are the no-GPU fallback/failure-path
  tests on CUDA-capable machines, and the GPU-requiring tests on CPU-only
  machines); on the verified RTX 3050 + CuPy 14.2.0
  machine the deterministic CPU-vs-GPU LP comparison passed
* Netlib LP benchmark: **7/7 verified** (see table below)
* Maros–Mészáros QP: **5/6 certify optimal** (QSHIP12S reaches `max_iterations`)

> The older counts previously listed here (14/14, 19/19, 5/5) described an
> earlier repository state and are superseded by the numbers above.

### Netlib benchmark results

`tools/benchmark_netlib.py` solves every `data/*.mps` instance through the
production sparse path (`load_numeric_mps(sparse=True)` -> `solve_lp`) and
cross-checks each objective against a fresh SciPy HiGHS oracle
(`linprog(method="highs")`).  Report is regenerated to
`results/benchmark_netlib.md`/`.csv`; latest run:

```text
instance       status      solver objective  HiGHS reference   rel obj err     rel_gap
adlittle       optimal     225494.963156     225494.963162     2.95e-11        1.28e-10
afiro          optimal     -464.753142659    -464.753142857     4.26e-10        3.76e-10
blend          optimal     -30.812149660     -30.812149846      5.83e-09        6.39e-09
pilot4_plain   certified   -2581.139259      -2581.139259       4.49e-11        1.76e-15
pilot87        certified   301.710347333     301.710347333      2.42e-13        (strict KKT cert)
sc205          optimal     -52.202061205     -52.202061212      1.27e-10        3.99e-10
share2b        optimal     -415.732240603    -415.732240741     3.31e-10        2.16e-10
```

 PILOT87's strict verified result (objective `301.710347333`,
|delta| = 1.034e-10 vs HiGHS) comes from the **independent validation
workflow**, not from a direct interior-point solve: it is produced by
`experiment/crossover/run_pilot87_crossover.py` — row/column equilibration → RRQR
basis identification → sparse composite Phase-I repair → Phase-II simplex
(driver-side repair/resume and vertex-preserving degenerate-pivot cond cleanup)
→ independent verification of the original unscaled data (primal residual `A x = b`,
reduced-cost/dual residual `c_N − Nᵀy ≥ 0`, and primal/dual gap) — whose
artifacts live under `artifacts/pilot87/` (e.g. `p87_strict_certificate.txt`).
PILOT4's objective is folded from its own independent validation artifact
(`artifacts/pilot4/p4_crossover_certificate.txt`, 3/3 bit-identical
RRQR → repair → Phase II runs, |delta| = 4.5e-11 vs HiGHS).  In short: the
published PILOT87 number is the output of the validation workflow, and the
direct IPM stalls on this instance.

This is distinct from the **production solver's crossover fallback**.  When the
Mehrotra IPM stalls or lands in the numerical tail and the crossover gate
passes, the production path performs sparse Phase I crash followed by sparse
Devex Phase II, then runs the independent acceptance checks — it does **not**
require RRQR basis identification or strict reduced-cost polish.  The RRQR →
Phase-I repair → Phase-II simplex validation workflow is reserved for the
independent validation path and is not part of the per-solve production
crossover fallback.

All 7/7 instances verified.  The canonical benchmark numbers for presentations
should always come from the final frozen repository state.

---

# Validation Strategy

Validation is organized into three levels.

### 1. Unit / component validation

Individual components are tested independently:

* MPS parsing
* scaling
* linear systems
* standard-form conversion
* solver components
* bound transformations

### 2. Edge-case validation

Tests cover cases such as:

* equality constraints
* mixed constraint types
* degeneracy
* badly scaled problems
* variable-bound transformations
* fixed variables
* free variables
* boxed variables
* numerical failure behavior

### 3. Benchmark validation

The solver is tested against standard LP benchmark instances.

Results are compared against independently established reference solutions.

The project follows a strict rule:

> **Never claim a benchmark result that has not been reproduced from the current repository state.**

---

# Repository Structure

```text
sovereign-optimization-solver/
│
├── src/                          # production core (LP + MILP + QP)
│   ├── mps_parser.py             # MPS parser → LPModel (incl. integer MARKER/BV/LI/UI)
│   ├── numerical_model.py        # LPModel → NumericalLP arrays
│   ├── scaling.py                # row/column equilibration
│   ├── constraint_form.py
│   ├── lp/
│   │   ├── linear_system.py       # Newton factorizations / solves (sparse CPU + GPU dispatch)
│   │   ├── gpu_linear_system.py   # opt-in dense-Schur CUDA/CuPy backend
│   │   ├── mehrotra.py            # production IPM solver (backend="cpu"/"gpu"/"auto")
│   │   ├── qp.py                  # legacy dense standard-form QP slice (compatibility)
│   │   ├── branch_bound.py        # native MILP branch-and-bound prototype
│   │   ├── crossover.py           # sparse crossover pipeline
│   │   └── simplex.py             # experimental Revised Simplex
│   └── qp/                       # canonical sparse convex QP package
│       ├── problem.py            # QPProblem (sparse P/q/G/h/A/b/lb/ub, sparse PSD check)
│       ├── solver.py             # solve_qp(): inequality-form Mehrotra IPM
│       ├── linear_system.py      # sparse SPLU saddle-point backend + dense fallback
│       ├── verify.py             # independent KKT certificate
│       ├── qps.py                # Maros–Mészáros QPS/SIF reader
│       ├── benchmark.py          # deterministic QP generator (seed 26119) + runner
│       └── io.py                 # NPZ save/load
│
├── tests/                        # production regression + edge-case suite
│   ├── test_mps_parser_hardening.py
│   ├── test_lp_edge_cases.py
│   ├── test_mehrotra_reporting.py
│   ├── test_qp_skeleton.py        # legacy src/lp/qp.py slice
│   ├── test_qp_sparse.py          # canonical src/qp/ suite (16 tests)
│   ├── test_qps_reader.py         # QPS/SIF reader test
│   ├── test_milp_skeleton.py      # MILP prototype tests
│   ├── test_milp_robustness.py
│   ├── test_gpu_backend.py        # GPU graceful-fallback + CPU-vs-GPU tests
│   ├── run_benchmarks.py
│   └── verify_with_highs.py
│
├── experiment/                   # isolated research (not production deps)
│   ├── crossover/                # RRQR → Phase-I repair → Phase-II validation (PILOT87/PILOT4)
│   ├── pdhg/
│   ├── mcc/
│   ├── regularization/
│   ├── ruiz_scaling/
│   ├── augmented_kkt/
│   └── newton_diagnostics/
│
├── tools/
│   ├── benchmark_netlib.py       # Netlib LP benchmark vs HiGHS reference
│   ├── benchmark_maros.py        # Maros–Mészáros QP benchmark (SIF fetched, never committed)
│   ├── benchmark_miplib.py       # MIPLIB MILP benchmark harness
│   ├── validate_milp_incumbent.py # direct B&B validation + independent incumbent check
│   └── certification/            # independent KKT check + strict polish helpers
│       ├── p87_certify.py
│       ├── p87_strict_polish.py
│       └── README.md
│
├── scripts/
│   └── download_maros_qp.py      # fetches QP SIF files into data/qp/
│
├── artifacts/
│   └── pilot87/                  # validated PILOT87 results (npz + certificates)
│
├── archive/                      # historical research & superseded experiments
│   ├── research/
│   └── root/
│
├── data/                         # benchmark inputs (MPS; QP SIF fetched, never committed)
│   ├── afiro.mps
│   ├── sc205.mps
│   ├── adlittle.mps
│   ├── share2b.mps
│   ├── blend.mps
│   ├── pilot4_plain.mps
│   └── pilot87.mps
│
├── requirements.txt
├── requirements-dev.txt
├── requirements-gpu.txt          # optional (installation compatibility note: pins cupy-cuda12x>=13.0; CUDA 13.x hosts: cupy-cuda13x)
├── ARCHITECTURE.md
└── README.md
```

> The exact repository tree should always be kept synchronized with the actual repository.

---

# Current Limitations

The project is a research/prototype optimization engine rather than a production commercial solver. It is not presented as a replacement for CPLEX, Gurobi, or other commercial solvers.

Current limitations include:

* incomplete end-to-end sparse data flow (numerical model and several
  standard-form transformations still use dense matrices)
* no presolve
* MILP remains a prototype: no cuts, no strong branching, no warm starts, no
  parallelism; hard instances terminate honestly at `node_limit`/`time_limit`
  with a remaining gap (e.g. 50v-10 currently reaches `node_limit` with a
  verified feasible incumbent 7002.77, bound 2962.88, rel. gap ≈ 0.577)
* GPU backend is **dense Schur**, not sparse CUDA factorization; for
  large/sparse models the sparse CPU backend remains the production path
* QSHIP12S (Maros–Mészáros, n=2763) currently reaches `max_iterations`
* no parallel linear algebra
* incomplete infeasibility/unboundedness certification
* Revised Simplex still requires further numerical work
* PDHG/PDLP experiments remain research implementations
* PILOT87 exposes unresolved Schur-complement conditioning limitations in the
  **standalone Mehrotra IPM path** (which stalls in the numerical tail); the
  independent validation workflow resolves PILOT87 to a strictly verified optimum

These limitations are deliberate and documented rather than hidden.

---

# Development Roadmap

The development strategy is staged.

## Phase 1 — Reliable LP Foundation ✅ COMPLETE

* stabilize Mehrotra IPM
* strengthen numerical safeguards
* improve standard-form conversion
* harden MPS parsing
* validate against standard LP benchmarks (Netlib 7/7 verified)
* establish reproducible regression tests (167 passed, 2 hardware-conditional skips)

## Phase 2 — Sparse & Large-Scale LP (CURRENT — in progress)

* preserve sparsity from model ingestion onward (sparse CPU backend done;
  dense transformations remain in the numerical model / standard-form path)
* sparse standard-form construction
* sparse scaling
* improve Newton-system formulations
* investigate augmented-KKT formulations
* improve sparse ordering and factorization
* improve numerical-tail robustness of the **standalone Mehrotra IPM** (PILOT87
  conditioning in the raw IPM path; the independent validation workflow already
  resolves PILOT87 to a strictly verified optimum)
* benchmark memory and runtime scaling

## Phase 3 — Algorithm Portfolio

Develop multiple LP algorithms around a common interface:

```text
                 ┌── Mehrotra IPM
LP Model ────────┼── Revised Simplex
                 └── PDHG / PDLP
```

Algorithm-selection rules will be derived from measured behavior on the project's benchmark suite rather than assumed solely from textbook classifications.

## Phase 4 — QP ✅ COMPLETE (canonical package implemented)

Canonical sparse convex QP package implemented in `src/qp/` (inequality-form
Mehrotra IPM, sparse SPLU saddle-point backend, independent KKT certificate,
QPS/SIF reader; 17 QP tests; Maros–Mészáros 5/6 certify optimal with QSHIP12S
at `max_iterations`). Legacy dense standard-form slice `src/lp/qp.py` retained
for compatibility. Future QP work: iteration-budget/regularization tuning for
hard instances (QSHIP12S), sparse QP-GPU path.

## Phase 5 — MILP ✅ PROTOTYPE (current — strengthening in progress)

Implemented:

* integer-variable parsing (MARKER/INTORG/INTEND, BV/LI/UI bounds)
* native branch-and-bound (`src/lp/branch_bound.py`)
* node management, LP relaxation solving, pruning
* diversified verified rounding heuristic for incumbent generation
  (28 MILP tests passing; pk1 18.000 → 17.000; 50v-10 verified incumbent with
  remaining gap)

Future MILP work:

* presolve and bound tightening
* cutting-plane infrastructure where justified
* stronger branching and node selection
* warm starts
* parallelism

## Phase 6 — GPU Acceleration ✅ INITIAL BACKEND (current — sparse GPU future)

An opt-in dense-Schur CUDA/CuPy backend for the Mehrotra reduced-Newton system
is implemented (`src/lp/gpu_linear_system.py`; verified on RTX 3050 + CuPy
14.2.0 with a passing deterministic CPU-vs-GPU LP test) and validated
against the CPU implementation rather than assumed faster.

Future GPU work:

* sparse matrix-vector operations
* first-order methods
* large-scale iterative linear algebra
* parallel preprocessing
* batched computations

The GPU implementation will be validated against the CPU implementation (no GPU-vs-CPU
performance benchmark has been performed yet) rather than assuming that GPU execution
is automatically faster.

---

# Design Philosophy

The project is being developed around several principles.

### From-scratch implementation

Core optimization algorithms are independently implemented from their mathematical foundations.

### Reference, don't depend

Established solvers may be used for:

* studying algorithms
* validating solutions
* architectural comparison
* performance benchmarking

but not as the underlying optimization engine.

### Numerical honesty

A numerical failure is preferable to a falsely reported optimum.

### Verify before claiming

Repository state is the source of truth.

Every important claim should be reproducible from the current codebase.

### Small, isolated changes

Changes should be:

* narrowly scoped
* testable
* independently verifiable
* integrated only after regression testing

### Profile before optimizing

Performance work should be driven by measurements rather than assumptions.

### Hardware-aware acceleration

GPU acceleration is a means, not an objective. It should only be introduced where the workload and hardware make it beneficial.

---

# Long-Term Goal

The ultimate goal is to build an **independently developed, numerically robust, scalable optimization engine** capable of solving a broad class of mathematical optimization problems:

```text
                    Sovereign Optimization Engine
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
             LP               QP              MILP
             │                │                │
             └────────────────┼────────────────┘
                              │
                    Common Numerical Core
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
          Sparse          Parallel           GPU
```

The objective is not to reproduce an existing solver feature-for-feature.

The objective is to build a **from-scratch optimization platform that satisfies the substantive requirements of SIH 26119 through independently implemented algorithms, rigorous numerical validation, scalable architecture, and measurable engineering progress.**

```
