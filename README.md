# OptiCore

[![CI](https://github.com/anshpurohit7906-bot/sovereign-optimization-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/anshpurohit7906-bot/sovereign-optimization-solver/actions/workflows/ci.yml)

An indigenous, from-scratch mathematical optimization solver for Linear Programming (LP), convex Quadratic Programming (QP), and Mixed-Integer Linear Programming (MILP) in Python with NumPy and SciPy.

---

## Why OptiCore

OptiCore is built from first mathematical principles without wrapping third-party solving engines (such as HiGHS, OSQP, or SciPy's `linprog`).

The solver emphasizes **numerical transparency and honest status reporting**:
* Solves are performed directly using indigenous interior-point and branch-and-bound algorithms.
* Convergence claims, termination tolerances, and incumbent feasibility are validated against the original, unscaled problem coordinates.
* Stalled, budget-limited, or sub-optimal trajectories exit honestly with non-zero exit codes rather than declaring premature convergence.
* External solvers are referenced solely in test and benchmark harnesses as independent verification oracles.

---

## Quick Install

OptiCore requires Python 3.10 or higher. Install the minimal package directly using `pip`:

```bash
python -m pip install .
```

To install with development dependencies (e.g. `pytest`):

```bash
python -m pip install -e ".[dev]"
```

---

## Quick Usage

OptiCore provides a unified command-line interface with dedicated subcommands:

### Linear Programming (LP)
```bash
opticore solve data/afiro.mps
```

### Quadratic Programming (QP)
```bash
opticore solve-qp data/qp/QSC205.SIF
```

### Mixed-Integer Linear Programming (MILP)
```bash
opticore solve-milp <MPS> --maximize
```

---

## What Is Implemented

* **Linear Programming (LP):**
  * Sparse primal-dual Mehrotra predictor-corrector interior-point method (IPM) in standard equality form.
  * Sparse Schur complement reduction ($S = A H^{-1} A^T$) with SuperLU factorization (`splu` with `permc_spec="MMD_AT_PLUS_A"`).
  * Scale-aware diagonal regularization and monotone-decreasing iterative refinement.
  * Conditional two-phase sparse simplex crossover fallback for stalled or numerical-tail trajectories.
* **Convex Quadratic Programming (QP):**
  * Canonical sparse primal-dual predictor-corrector IPM handling inequality-constrained and bound-constrained convex QPs.
  * Augmented Karush-Kuhn-Tucker (KKT) saddle-point system solved via sparse SuperLU with rank-deficiency regularization.
  * Strict post-solve Karush-Kuhn-Tucker (KKT) certificate verification.
* **Mixed-Integer Linear Programming (MILP):**
  * Native branch-and-bound engine using OptiCore's continuous LP relaxations at each node.
  * Best-bound priority heap search and most-fractional branching with objective tie-breaking.
  * Primal heuristics including variable rounding and continuous LP-repair solves.
  * Independent incumbent feasibility verification checking integrality, bounds, and original constraints.

---

## Benchmark and Validation Results

All reported values represent established repository benchmark measurements.

### LP: Netlib AFIRO
Solved via the sparse Mehrotra predictor-corrector IPM:
* **Iterations:** 8 Mehrotra iterations
* **Runtime:** ~0.016 s
* **Solver Objective:** `-464.753103202022`
* **HiGHS Reference:** `-464.7531428571`
* **Relative Objective Error:** $\approx 8.5 \times 10^{-8}$ (passed the independent primal-dual audit)

### QP: Maros–Mészáros Benchmarks
5 of 6 tested benchmark cases pass OptiCore's independent KKT certificate validation ($\text{rel\_primal} \le 10^{-6}$, $\text{rel\_dual} \le 10^{-6}$, $\text{rel\_gap} \le 10^{-6}$, complementarity $\le 10^{-6}$):

| Problem | $n$ | $m_{\text{eq}}$ | $m_{\text{ineq}}$ | Iterations | Solver Objective | Status | KKT Certificate |
|---|---|---|---|---|---|---|
| **QAFIRO** | 32 | 8 | 19 | 10 | -1.590779e+00 | optimal | PASS |
| **QADLITTLE** | 97 | 15 | 41 | 24 | 4.803189e+05 | optimal | PASS |
| **QSC205** | 203 | 91 | 114 | 12 | -5.809595e-03 | optimal | PASS |
| **QGROW15** | 645 | 300 | 0 | 26 | -1.016936e+08 | optimal | PASS |
| **QPCBOEI1** | 384 | 9 | 431 | 36 | 1.150391e+07 | optimal | PASS |

*Note on QSHIP12S:* `QSHIP12S` ($n=2763, m_{\text{eq}}=1045, m_{\text{ineq}}=106$) is currently a stress and limit case. It terminates at the maximum iteration limit (200 iterations) without achieving target KKT tolerances and is **not** counted among the successful cases.

### MILP: Small Planning Problem & MIPLIB Scalability
* **Small Planning Model (Refinery Fixed-Charge):**
  * **Objective:** `23320.0`
  * **Best Bound:** `23320.0`
  * **Relative Gap:** `0.0`
  * **Nodes Explored:** 3 nodes
  * **Independent Verification:** PASS
* **MIPLIB Scalability Evidence (PK1 & MAS74):**
  * Evaluated under execution time and node limits to demonstrate honest limit behavior.
  * `pk1`: Reaches node/time limits without proving optimality; finds incumbent values (e.g., objective 17.0–14.0 vs HiGHS optimal 11.0) with an open gap (best bound $\approx 2.97$). Optimality is **not** claimed.
  * `mas74`: Reaches time limit with incumbent 14075.45 and an open dual bound. Optimality is **not** claimed.

---

## Test Status

* **Frozen Test Suite Baseline:**
  * **Collected:** 169 tests
  * **Passed:** 167 passed
  * **Skipped:** 2 skipped (GPU backend tests, skipped automatically when CUDA/CuPy hardware is not detected)
  * **Failed:** 0 failed
  * **Environment:** Python 3.13.2, local baseline runtime ~22.37 s (*local reference measurement; not a guarantee across platforms*)
* **Continuous Integration:**
  * Runs on push and pull request via GitHub Actions on standard CPU virtual machines (`ubuntu-latest`, Python 3.11).
  * GPU tests skip automatically in CI when CUDA/CuPy is unavailable.

---

## Known Limitations

1. **No Infeasibility / Unboundedness Certificates:**
   OptiCore LP uses standard Mehrotra predictor-corrector equations without a Homogeneous Self-Dual (HSD) embedding. While Phase-I crossover can identify equality-form infeasibility, the solver does not provide theoretical Farkas/HSD ray certificates.
2. **QSHIP12S Convergence:**
   Large and ill-conditioned QP instances such as Maros–Mészáros `QSHIP12S` do not currently reach target convergence within iteration budgets.
3. **MILP Scalability:**
   Node relaxations are solved via continuous LP cold-starts. The solver lacks dual-simplex warm-starts, a cutting-plane framework, and advanced structural MIP presolve, limiting pure branch-and-bound scalability on large MIPLIB instances.
4. **Coexisting QP Implementations:**
   The canonical sparse inequality-form QP implementation (`src/opticore/qp/`) coexists with a legacy dense equality-form QP module (`src/opticore/lp/qp.py`).
5. **QP Factorization Reuse:**
   `src/opticore/qp/solver.py` re-factors the augmented KKT matrix for both predictor and corrector steps within each iteration; factorization caching remains an open measurement and performance item.
6. **GPU Backend Scope:**
   The optional GPU backend (`--backend gpu` / `--backend auto`) utilizes CuPy for dense Cholesky Schur complement factorizations on CUDA devices. It does not provide general sparse GPU factorizations.

---

## Architecture and Technical Documentation

For complete mathematical derivations, reduced KKT linear systems, scaling procedures, crossover details, and architectural block diagrams, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Project Status

OptiCore is an engineering and research project focused on transparent, from-scratch optimization solver design. It is not intended as a drop-in replacement or performance competitor for mature industrial solvers (such as HiGHS, Gurobi, or CPLEX).
