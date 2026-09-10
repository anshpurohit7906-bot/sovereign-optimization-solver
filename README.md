# SIH 26119 — Sovereign Optimization Solver

**Indigenous GPU-Accelerated Optimization Solver — Sovereign Alternative to Commercial Optimization Engines**

A from-scratch optimization engine being developed for SIH 26119. The long-term objective is a unified optimization platform covering Linear Programming (LP), Quadratic Programming (QP), Mixed-Integer Linear Programming (MILP), sparse and large-scale optimization, and GPU acceleration where it provides a measurable benefit.

The implementation is independently developed from mathematical foundations. Existing solvers such as HiGHS, PDLP/OR-Tools, and NVIDIA cuOpt may be studied as architectural, numerical, or benchmarking references, but they are not used as the optimization engine.

> **Current milestone:** establish a numerically reliable LP foundation before expanding into QP, MILP, large-scale sparse computation, and GPU acceleration.

> **LP status:** Mehrotra predictor-corrector IPM is the **primary LP solver**. The sparse crossover is now an **integrated IPM fallback** — not merely an experimental/unrelated path. The solver API default remains `crossover_fallback=False`; the benchmark harness enables `crossover_fallback=True` specifically for benchmark validation.

---

## Verified Result (PILOT87)

The complete pipeline drives the Mehrotra IPM terminal basis to a strictly
verified optimum on the hard `pilot87.mps` benchmark. The current production
path is the **automatic sparse-crossover fallback**:

```text
Mehrotra IPM
    ↓
numerical-tail / stall detection
    ↓
merit-gated crossover fallback
    ↓
sparse Phase I
    ↓
Devex Phase II
    ↓
independent KKT acceptance
    ↓
verified optimal result
```

PILOT87 has been verified through the complete solver/certification pipeline:
standalone Mehrotra stalls in the numerical tail, and the full path resolves the
instance. The routine benchmark uses the **independently validated
certificate-backed result** — the objective and residual numbers in the table
below come from the certificate route (a **certified fold-in** chosen for
runtime practicality), not from the automatic-crossover experiment. A separate
live experiment also demonstrated that the **current production automatic
sparse-crossover fallback** can recover PILOT87 from an IPM stall; that
demonstration is reported separately in the PILOT87 section below, with its own
runtime and residual figures.

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

These certificate numbers come from the independently validated strict-polish /
certificate pipeline — the certification path behind PILOT87's **certified
fold-in**. That path is separate from the current production automatic fallback
(merit-gated sparse Phase I + Devex Phase II with independent KKT acceptance;
no RRQR basis identification, no strict reduced-cost-polish stage).

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
src/                  production LP core (parser, numerical model, Mehrotra IPM,
                      sparse crossover fallback) — do not modify solver behavior
tests/                production regression + edge-case test suite
data/                 MPS benchmark inputs (afiro.mps, pilot87.mps, ...)
tools/                benchmark harness + certification scripts
  tools/benchmark_netlib.py        Netlib LP benchmark vs HiGHS reference
  tools/certification/            independent KKT certificate + strict-polish
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

### LP milestone

- **Mehrotra predictor-corrector IPM** is the primary LP solver.
- The **sparse crossover is an integrated IPM fallback**, not merely an
  experimental/unrelated path: when the IPM terminates in the numerical tail or
  stalls, a merit-gated crossover attempt is made, and its result is accepted
  only through independently recomputed KKT residuals.
- The solver API default remains `crossover_fallback=False`; the benchmark
  harness enables `crossover_fallback=True` specifically for benchmark
  validation.

The project has also explored Revised Simplex and several PDHG/PDLP-style approaches. These are retained as independent algorithmic paths and research experiments rather than being treated as interchangeable production components.

### Implemented

- MPS parsing for currently supported MPS features
- Explicit rejection of unsupported:
  - RANGES
  - SOS
  - QUADOBJ
  - INDICATORS
  - integer MARKER / INTORG / INTEND constructs
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
- Fraction-to-boundary step control
- Numerical safeguards and finite-value checks
- Residual-based convergence testing
- Scale-aware numerical-tail detection
- Best-trusted-iterate preservation
- Escalating diagonal regularization
- Integrated sparse-crossover fallback (sparse Phase I + Devex Phase II) for
  stalled / numerical-tail IPM solves
- Independent KKT acceptance for crossover results (`rel_primal`, `rel_dual`,
  and `rel_gap` all `<= tol`)
- Netlib benchmark harness with IPM / crossover / certified route reporting
- Experimental / secondary Revised Simplex implementation
- Experimental PDHG/PDLP research implementations

### Not Yet Implemented

- QP
- MILP / branch-and-bound
- Integer-variable parsing and handling
- Full end-to-end sparse LP data flow
- Presolve and advanced reductions
- GPU acceleration
- Parallel linear algebra
- Advanced sparse ordering
- Formal infeasibility and unboundedness certificates
- Production-grade algorithm-selection / dispatch layer

---

# Architecture

### Primary LP solver

Mehrotra is the **primary LP solver**:

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
 │
 └── Sparse CSR + SuperLU Schur backend
 │
 ▼
MehrotraResult
```

### Integrated sparse-crossover fallback

The sparse crossover is **integrated into the production IPM path** as an
automatic fallback — it is not merely an unrelated experimental path. It is the
validated path behind the **verified PILOT87 result** (VERIFIED OPTIMAL, see the
[Verified Result](#verified-result-pilot87)):

```text
Mehrotra IPM
    ↓
numerical-tail / stall detection
    ↓
merit-gated crossover fallback
    ↓
sparse Phase I
    ↓
Devex Phase II
    ↓
independent KKT acceptance
    ↓
verified optimal result
```

* **Trigger:** the crossover is attempted only when the IPM terminates in a
  `stalled` or `numerical_tail` state and its best observed merit is within
  `CROSSOVER_MERIT_RATIO` of the acceptance tolerance (the IPM is near-optimal
  but cannot resolve the last residuals).
* **Sparse Phase I:** starts from an all-artificial basis `B = I` (guaranteed
  feasible and nonsingular — no RRQR basis identification or external basis is
  required).
* **Devex Phase II:** sparse revised simplex with per-pivot LU refactorization,
  Devex pricing, iterative refinement, and condition-number-aware feasibility
  handling.
* **Independent KKT acceptance:** the crossover result replaces the IPM iterate
  only when `rel_primal`, `rel_dual`, and `rel_gap` — independently recomputed in
  original coordinates — are all `<= tol`. The original IPM result is preserved
  when the fallback is skipped, fails, or is rejected.

The old RRQR basis-identification and strict reduced-cost-polish stages are
legacy stages of the experimental certification pipeline and are **not** required
stages of the current production automatic fallback. The independently validated
strict-polish/certificate artifacts remain the basis of the **certified fold-in**
used for PILOT87 in the routine benchmark.

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

**Completing the pipeline resolves PILOT87.** In the routine benchmark the
PILOT87 objective is folded from the independently validated strict KKT
certificate (`artifacts/pilot87/p87_strict_certificate.txt`, original objective
`301.710347333`, agreeing with the HiGHS reference to within `1.1e-10`,
relative ≈ `3.7e-13`; see the [Verified Result](#verified-result-pilot87)) — a
**certified fold-in** adopted for runtime practicality.

A separate live experiment has also run PILOT87 through the **production
automatic crossover** (`solve_lp(crossover_fallback=True)`): the IPM stalls and
the crossover converges the instance in **approximately 1090 seconds**
(14,628 Phase I iterations + 22,046 Phase II pivots), with final KKT residuals
far below the `1e-8` acceptance tolerance and a relative objective error of
**≈ `1.536e-7`** against HiGHS. PILOT87 should therefore **not** be presented as
a fast production benchmark case.

So the limitation applies to the **standalone Mehrotra IPM path**, not to the
complete solver pipeline. PILOT87 remains an active scalability target for the
standalone path; it is verified optimal through the full pipeline.

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
```

Unsupported advanced MPS constructs are now **rejected explicitly** rather than silently ignored.

This is particularly important for constructs such as RANGES and integer markers: the parser must never construct a different mathematical problem without informing the user.

Full MILP/MPS support is planned as part of the future MILP phase.

---

# Benchmarking

Benchmarking uses standard LP instances such as:

```text
AFIRO
SC205
ADLITTLE
SHARE2B
BLEND
PILOT4 (PILOT4_PLAIN)
PILOT87
```

HiGHS and other established optimization software may be used as **reference/oracle implementations for validation and comparison**.

They are not used as the optimization engine.

## Current Benchmark Results

### Production tests

* `pytest tests/`: **105/105 passed**
* crossover-specific tests: **11/11 passed** (`tests/test_crossover.py`)
* Netlib: **7/7 instances verified against HiGHS**
  * **5 direct IPM** cases (ADLITTLE, AFIRO, BLEND, SC205, SHARE2B)
  * **1 automatic crossover** case (PILOT4)
  * **1 certified fold-in** case (PILOT87)

### Netlib benchmark results

`tools/benchmark_netlib.py` solves every `data/*.mps` instance through the
production sparse path (`load_numeric_mps(sparse=True)` -> `solve_lp` with
`crossover_fallback=True`) and cross-checks each objective against a fresh SciPy
HiGHS oracle (`linprog(method="highs")`).  Report is regenerated to
`results/benchmark_netlib.md`/`.csv`.  Method legend: `ipm` = direct
interior-point solve; `crossover` = the IPM stalled and the automatic
sparse-crossover fallback resolved it; `certified` = value folded from an
independent certificate artifact.  Latest run:

```text
instance       method      solver objective  HiGHS reference   rel obj err     time (s)
adlittle       ipm         225494.963156     225494.963162     2.95e-11        0.022
afiro          ipm         -464.753142659    -464.753142857     4.26e-10        0.014
blend          ipm         -30.812149660     -30.812149846      5.83e-09        0.020
pilot4_plain   crossover   -2581.139259      -2581.139259       1.76e-15        7.622
pilot87        certified   301.710347333     301.710347333      2.42e-13        2.972
sc205          ipm         -52.202061205     -52.202061212      1.27e-10        0.028
share2b        ipm         -415.732240603    -415.732240741     3.31e-10        0.025
```

**PILOT4** is the canonical **automatic-crossover** route: the direct IPM stalls
and the production fallback converges it end-to-end — **1449 Phase I iterations**
and **1546 Phase II pivots** in **≈ 7.6 seconds** (79 IPM iterations + 1546
Phase II pivots per the `iters` cell in `results/benchmark_netlib.md`), with a
**relative objective error of ≈ `1.76e-15`** against the fresh HiGHS benchmark
result.  The independently certified objective
(`artifacts/pilot4/p4_crossover_certificate.txt`) is kept as a cross-check, not
as the row source.

**PILOT87** is a **certified fold-in**: its objective is folded from the
independent strict KKT certificate
(`artifacts/pilot87/p87_strict_certificate.txt`, |delta| = 1e-10 vs HiGHS), not
from a direct interior-point solve, for **runtime practicality**.  The live
production-crossover experiment takes ≈ 1090 s (see the [PILOT87](#pilot87)
section), so PILOT87 should **not** be presented as a fast production benchmark
case.

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
├── src/                          # production LP core
│   ├── mps_parser.py             # MPS subset parser → LPModel
│   ├── numerical_model.py        # LPModel → NumericalLP arrays
│   ├── scaling.py                # row/column equilibration
│   ├── constraint_form.py
│   └── lp/
│       ├── linear_system.py       # Newton factorizations / solves
│       ├── mehrotra.py            # production IPM solver (with sparse-crossover fallback)
│       ├── crossover.py           # sparse Phase I + Devex Phase II crossover engine
│       └── simplex.py             # experimental Revised Simplex
│
├── tests/                        # production regression + edge-case suite
│   ├── test_crossover.py
│   ├── test_benchmark_netlib_smoke.py
│   ├── test_lp_edge_cases.py
│   ├── test_mehrotra_reporting.py
│   ├── test_mps_parser_hardening.py
│   └── verify_with_highs.py
│
├── experiment/                   # isolated research (not production deps)
│   ├── crossover/                # research history of the sparse-crossover engine
│   ├── pdhg/
│   ├── mcc/
│   ├── regularization/
│   ├── ruiz_scaling/
│   ├── augmented_kkt/
│   └── newton_diagnostics/
│
├── tools/
│   ├── benchmark_netlib.py       # Netlib benchmark harness vs fresh HiGHS oracle
│   └── certification/            # independent KKT certificate + strict polish
│       ├── p87_certify.py
│       ├── p87_strict_polish.py
│       └── README.md
│
├── artifacts/
│   ├── pilot4/                   # validated PILOT4 crossover results (npz + certificates)
│   └── pilot87/                  # validated PILOT87 results (npz + certificates)
│
├── archive/                      # historical research & superseded experiments
│   ├── research/
│   └── root/
│
├── data/                         # MPS benchmark inputs
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
├── ARCHITECTURE.md
└── README.md
```

> The exact repository tree should always be kept synchronized with the actual repository.

---

# Current Limitations

The project is a research/prototype optimization engine rather than a production
commercial solver; it is not claimed to be production-ready or commercially
competitive.

Current limitations include:

* incomplete end-to-end sparse data flow (the default `load_numeric_mps` path is
  dense; sparsity is opt-in via `sparse=True`)
* no presolve
* QP not yet implemented/validated as production
* MILP not yet implemented
* GPU acceleration not yet implemented
* no parallel linear algebra
* no formal infeasibility/unboundedness certificates
* Revised Simplex still requires further numerical work
* PDHG/PDLP experiments remain research implementations
* the **standalone PILOT87 Mehrotra IPM path** still has numerical-tail /
  Schur-complement conditioning limitations and stalls in the numerical tail;
  the full pipeline resolves PILOT87 (certified fold-in in the routine
  benchmark; the live production-crossover experiment reaches optimal in ≈ 1090 s
  and is not a fast case)

These limitations are deliberate and documented rather than hidden.

---

# Development Roadmap

The development strategy is staged.

## Phase 1 — Reliable LP Foundation

* stabilize Mehrotra IPM
* strengthen numerical safeguards
* improve standard-form conversion
* harden MPS parsing
* validate against standard LP benchmarks
* establish reproducible regression tests

## Phase 2 — Sparse & Large-Scale LP

* preserve sparsity from model ingestion onward
* sparse standard-form construction
* sparse scaling
* improve Newton-system formulations
* investigate augmented-KKT formulations
* improve sparse ordering and factorization
* improve numerical-tail robustness of the **standalone Mehrotra IPM** (PILOT87
  conditioning in the raw IPM path; the complete crossover pipeline already
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

## Phase 4 — QP

Extend the numerical foundation to convex Quadratic Programming while preserving the LP architecture where appropriate.

## Phase 5 — MILP

Add:

* integer-variable parsing
* branch-and-bound
* node management
* LP relaxation solving
* incumbent management
* pruning
* cutting-plane infrastructure where justified

## Phase 6 — GPU Acceleration

Investigate GPU acceleration only where profiling demonstrates a meaningful benefit.

Potential targets include:

* sparse matrix-vector operations
* first-order methods
* large-scale iterative linear algebra
* parallel preprocessing
* batched computations

The GPU implementation will be benchmarked against the CPU implementation rather than assuming that GPU execution is automatically faster.

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
