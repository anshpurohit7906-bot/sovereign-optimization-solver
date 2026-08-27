# SIH26119 — Sovereign Optimization Solver

A from-scratch mathematical optimization solver developed for the SIH26119 problem statement:

**Indigenous GPU-Accelerated Optimization Solver — Sovereign Alternative to Commercial Optimization Engines**

The project aims to build a sovereign optimization engine for large-scale industrial applications such as production planning, blending, logistics, power dispatch, refinery scheduling, and supply-chain optimization.

The current implementation focuses on establishing a reliable **Linear Programming (LP) solver core** before extending the architecture to larger-scale and mixed-integer optimization.

---

## Current Status

The project currently contains a working from-scratch LP optimization core based on a **Mehrotra primal-dual interior-point method**.

### Implemented

- MPS file parsing
- Numerical LP representation
- Equality (`E`), less-than (`L`), and greater-than (`G`) constraint handling
- Standard-form LP conversion
- Row/column equilibration
- Mehrotra predictor-corrector primal-dual interior-point method
- Dense reduced Newton-system solver
- Cholesky-based linear-system solution
- Fraction-to-boundary step control
- Numerical safeguards and finite-value checks
- Residual-based convergence testing
- Independent benchmark verification using SciPy/HiGHS
- Experimental PDHG implementations retained for comparison and research history

---

## Solver Architecture

```text
                    MPS Input
                        │
                        ▼
                 ┌─────────────┐
                 │ MPS Parser  │
                 └─────────────┘
                        │
                        ▼
                ┌──────────────┐
                │ Numerical LP │
                │    Model     │
                └──────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Standard-Form LP │
              │   Conversion     │
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Scaling /        │
              │ Equilibration    │
              └──────────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │ Mehrotra Predictor-          │
          │ Corrector Interior-Point     │
          │ Method                       │
          └─────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Reduced Newton   │
              │ System           │
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Solution + KKT   │
              │ Diagnostics      │
              └──────────────────┘
````

The continuous LP is represented internally in standard form as

$$
\min_x c^T x
$$

subject to

$$
Ax=b,\qquad x\ge0.
$$

The primal-dual KKT conditions are

$$
Ax=b,
$$

$$
A^Ty+z=c,
$$

$$
XZe=\mu e.
$$

The Mehrotra method eliminates the bound-multiplier direction and solves a reduced Newton system of the form

$$
\begin{bmatrix}
H & -A^T\\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x\\
\Delta y
\end{bmatrix}
=
\begin{bmatrix}
r_1\\
r_2
\end{bmatrix},
$$

where

$$
H=\operatorname{diag}(z/x).
$$

---

## Constraint Conversion

The current standard-form conversion supports the three basic MPS row types.

### Equality

$$
a_i^T x=b_i
$$

is retained directly.

### Less-than

$$
a_i^T x\le b_i
$$

is converted to

$$
a_i^T x+s_i=b_i,\qquad s_i\ge0.
$$

### Greater-than

$$
a_i^T x\ge b_i
$$

is converted to

$$
a_i^T x-s_i=b_i,\qquad s_i\ge0.
$$

---

## Numerical Scaling

The standard-form problem is equilibrated before the interior-point iterations.

The current scaling transformation is

$$
A_s=RAS,\qquad b_s=Rb,\qquad c_s=Sc,
$$

where \(R\) and \(S\) are row and column scaling factors.

The Newton iterations are performed in scaled coordinates and the final solution and diagnostic quantities are mapped back to the original problem units.

Scaling was particularly important for badly scaled instances such as ADLITTLE.

---

## Current Benchmark Results

The current LP core has been tested on five benchmark problems:

| Benchmark | Status            | Iterations | Relative Primal | Relative Dual | Relative Gap |
| --------- | ----------------- | ---------: | --------------: | ------------: | -----------: |
| AFIRO     | ✅ Optimal         |          8 |        9.92e-12 |      6.78e-12 |     7.55e-08 |
| SC205     | ✅ Optimal         |         10 |        8.61e-11 |      4.34e-12 |     2.67e-08 |
| ADLITTLE  | ✅ Optimal at 1e-7 |         10 |        8.07e-11 |      4.84e-08 |     7.45e-08 |
| SHARE2B   | ✅ Optimal         |         12 |        1.67e-08 |      8.73e-08 |     1.31e-10 |
| BLEND     | ✅ Optimal         |          9 |        1.97e-11 |      1.11e-10 |     6.29e-09 |

The current practical acceptance target is

$$
\text{relative primal residual}\le10^{-7},
$$

$$
\text{relative dual residual}\le10^{-7},
$$

$$
\text{relative gap}\le10^{-7}.
$$

All five current benchmark runs satisfy these criteria.

### Reference objective comparisons

**AFIRO**

Reference:

$$
f^\star\approx-464.7531428571
$$

Solver result:

$$
-464.7531032020
$$

**SC205**

Reference:

$$
f^\star\approx-52.2020612117
$$

Solver result:

$$
-52.2020598569
$$

**ADLITTLE**

Reference:

$$
f^\star\approx225494.9631623802
$$

Solver result:

$$
225494.9747394094
$$

**SHARE2B**

Reference:

$$
f^\star\approx-415.7322407414
$$

Solver result:

$$
-415.7322372130
$$

**BLEND**

Reference:

$$
f^\star\approx-30.8121498458
$$

Solver result:

$$
-30.8121496620
$$

The objectives and KKT diagnostics were independently checked against SciPy's HiGHS implementation.

---

## Why Mehrotra IPM?

An earlier development path explored several **primal-dual hybrid gradient (PDHG)** variants.

The experiments included:

* preconditioning
* scaling
* averaging
* restarts
* adaptive step mechanisms
* Barzilai-Borwein style updates
* PDLP-inspired primal weighting

These experiments were useful for understanding the numerical behavior of the benchmark suite, but SC205 exhibited a persistent stationarity limitation under the tested PDHG configurations.

The project therefore pivoted to a **Mehrotra primal-dual interior-point method**, which provides a more suitable foundation for robust sparse LP solving and later extensions to QP and MILP.

The earlier PDHG work is retained under:

```text
experiment/pdhg/
```

so that the development history and comparative experiments remain available.

---

## Repository Structure

```text
SIH26119/
│
├── data/
│   ├── afiro.mps
│   ├── sc205.mps
│   ├── adlittle.mps
│   ├── share2b.mps
│   └── blend.mps
│
├── src/
│   ├── constraint_form.py
│   ├── mps_parser.py
│   ├── numerical_model.py
│   ├── scaling.py
│   │
│   └── lp/
│       ├── linear_system.py
│       └── mehrotra.py
│
├── experiment/
│   └── pdhg/
│       ├── pdhg_bb.py
│       ├── pdhg_lp_solver.py
│       ├── pdhg_mixed.py
│       ├── pdhg_pdlp_weight.py
│       ├── pdhg_preconditioned.py
│       ├── pdhg_restart.py
│       ├── pdhg_scaled.py
│       └── pdlp_core.py
│
├── verify_afiro.py
├── README.md
└── .gitignore
```

---

## Running the Current Solver

From the repository root:

```text
python src/lp/mehrotra.py
```

The module regression harness currently exercises the internal tiny LP tests and AFIRO.

Individual benchmark instances can be loaded through the numerical model interface and solved using the Mehrotra LP solver.

---

## Current Limitations

The current implementation is a **research/prototype LP solver core**, not yet a production-scale industrial optimization engine.

### Linear algebra

* Dense Newton-system construction
* Dense Cholesky-based factorization
* No sparse matrix factorization yet
* No parallel linear algebra yet

### Problem classes

* LP is currently implemented
* QP is not yet implemented
* MILP is not yet implemented
* Branch-and-bound is not yet implemented
* Cutting planes and mixed-integer heuristics are not yet implemented

### Model support

* Current Mehrotra standard-form conversion assumes variables with bounds

$$
0\le x_j<\infty
$$

* General finite lower/upper bounds are not yet fully transformed
* Additional MPS constructs remain to be added as required

### Optimization infrastructure

* No full presolve pipeline yet
* No advanced sparse ordering yet
* No multicore parallel solver architecture yet
* GPU acceleration has not yet been implemented

---

## Development Roadmap

```text
Current LP Core
      │
      ▼
General MPS / Model Support
      │
      ▼
Presolve
      │
      ▼
Sparse Linear Algebra
      │
      ▼
Larger LP Benchmarks
      │
      ▼
QP Support
      │
      ▼
MILP Branch-and-Bound
      │
      ▼
Cutting Planes / Heuristics
      │
      ▼
Parallelization
      │
      ▼
GPU Acceleration
```

The immediate priority is **scalability**, particularly replacing the current dense Newton linear algebra with a sparse implementation suitable for large sparse optimization models.

---

## Design Philosophy

The project is being developed from the mathematical foundation upward rather than by embedding an existing optimization solver.

The core principles are:

1. Build the optimization algorithms from first principles.
2. Keep the numerical pipeline transparent and inspectable.
3. Validate continuously against recognized benchmark instances.
4. Use independent solvers only as verification oracles, not as the optimization engine itself.
5. Treat numerical stability and scalability as first-class requirements.
6. Keep the architecture modular so that LP, QP, and MILP capabilities can share the same numerical foundation.

---

## Current Milestone

### LP Core Baseline — Completed

The current baseline demonstrates:

$$
\boxed{\text{5/5 benchmark LPs solved at }10^{-7}\text{ tolerance}}
$$

including SC205, which exposed the limitations of the earlier first-order approach.

This baseline now serves as the starting point for the next phase:

$$
\boxed{\text{sparse, scalable optimization}}
$$

````

