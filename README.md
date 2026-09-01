# SIH 26119 — Sovereign Optimization Solver

**Indigenous GPU-Accelerated Optimization Solver — Sovereign Alternative to Commercial Optimization Engines**

A from-scratch optimization engine being developed for SIH 26119. The long-term objective is a unified optimization platform covering Linear Programming (LP), Quadratic Programming (QP), Mixed-Integer Linear Programming (MILP), sparse and large-scale optimization, and GPU acceleration where it provides a measurable benefit.

The implementation is independently developed from mathematical foundations. Existing solvers such as HiGHS, PDLP/OR-Tools, and NVIDIA cuOpt may be studied as architectural, numerical, or benchmarking references, but they are not used as the optimization engine.

> **Current milestone:** establish a numerically reliable LP foundation before expanding into QP, MILP, large-scale sparse computation, and GPU acceleration.

---

## Verified Result (PILOT87)

The complete solver pipeline drives the Mehrotra IPM terminal basis to a
strictly verified optimum on the hard `pilot87.mps` benchmark:

```text
Mehrotra IPM
  → RRQR basis identification
  → sparse Phase I
  → sparse Phase II
  → strict reduced-cost polish
  → independent KKT certificate
  → VERIFIED OPTIMAL
```

Standalone Mehrotra on PILOT87 stalls in the numerical tail; it is the
**crossover pipeline, not the raw IPM iterate, that yields the certified
optimum below**. PILOT87 is therefore **not** unresolved by the complete solver
pipeline — it is verified optimal by the full path.

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
src/                  production LP core (parser, numerical model, Mehrotra IPM,
                      linear system, scaling) — do not modify solver behavior
tests/                production regression + edge-case test suite
data/                 MPS benchmark inputs (afiro.mps, pilot87.mps, ...)
experiment/           isolated research: crossover, pdhg, mcc, regularization, ...
tools/certification/  independent KKT certificate + strict-polish scripts
artifacts/pilot87/    validated PILOT87 results (npz + certificates)
archive/              historical research, superseded experiments, logs
docs/                 (future) design documents
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

### Validated crossover (PILOT87 certification)

The crossover is not merely an unrelated experimental path — it is the
validated path that produced the **verified PILOT87 result** (VERIFIED OPTIMAL,
see the [Verified Result](#verified-result-pilot87)). It continues from the
Mehrotra terminal basis:

```text
Mehrotra terminal basis
 │
 ▼
RRQR basis identification
 │
 ▼
Sparse Phase I
 │
 ▼
Sparse Phase II
 │
 ▼
Strict reduced-cost polish
 │
 ▼
Independent KKT certificate
 │
 ▼
VERIFIED OPTIMAL
```

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

**Completing the pipeline resolves PILOT87.** Running the full solver path —
sparse crossover (RRQR basis identification → sparse Phase I → sparse Phase II
→ strict reduced-cost polish) followed by an independent KKT certificate —
produces a **strictly verified optimal** solution for PILOT87, with an original
objective of `301.710347333` that agrees with the HiGHS reference to within
`1.1e-10` (relative ≈ `3.7e-13`) per the strict certificate (see the
[Verified Result](#verified-result-pilot87)).

So the limitation applies to the **standalone Mehrotra IPM path**, not to the
complete solver pipeline. PILOT87 remains an active scalability target for the
standalone path; it is verified optimal through the full crossover pipeline.

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
PILOT87
```

HiGHS and other established optimization software may be used as **reference/oracle implementations for validation and comparison**.

They are not used as the optimization engine.

## Current Benchmark Results

### Production tests

* `pytest tests/`: **14/14 passed**
* edge-case suite: **19/19 passed**
* Netlib: **5/5 passed**
* HiGHS oracle: **19/19 matched**

### Netlib benchmark results

```text
AFIRO
  status                 = optimal
  relative objective error = 8.51e-08

SC205
  status                 = optimal
  relative objective error = 2.55e-08

ADLITTLE
  status                 = optimal
  relative objective error = 4.91e-08

SHARE2B
  status                 = optimal
  relative objective error = 1.54e-08

BLEND
  status                 = optimal
  relative objective error = 5.83e-09
```

These results were reproduced from the current repository state. The canonical
benchmark numbers for presentations should always come from the final frozen
repository state.

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
│       ├── mehrotra.py            # production IPM solver
│       └── simplex.py             # experimental Revised Simplex
│
├── tests/                        # production regression + edge-case suite
│   ├── test_mps_parser_hardening.py
│   ├── test_lp_edge_cases.py
│   ├── test_mehrotra_reporting.py
│   ├── run_benchmarks.py
│   └── verify_with_highs.py
│
├── experiment/                   # isolated research (not production deps)
│   ├── crossover/                # RRQR → sparse Phase I/II → polish pipeline
│   ├── pdhg/
│   ├── mcc/
│   ├── regularization/
│   ├── ruiz_scaling/
│   ├── augmented_kkt/
│   └── newton_diagnostics/
│
├── tools/
│   └── certification/            # independent KKT certificate + strict polish
│       ├── p87_certify.py
│       ├── p87_strict_polish.py
│       └── README.md
│
├── artifacts/
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

The project is a research/prototype optimization engine rather than a production commercial solver.

Current limitations include:

* incomplete end-to-end sparse data flow
* no presolve
* no QP
* no MILP
* no GPU implementation
* no parallel linear algebra
* incomplete infeasibility/unboundedness certification
* Revised Simplex still requires further numerical work
* PDHG/PDLP experiments remain research implementations
* PILOT87 exposes unresolved Schur-complement conditioning limitations in the
  **standalone Mehrotra IPM path** (which stalls in the numerical tail); the
  complete crossover pipeline resolves PILOT87 to a strictly verified optimum

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
