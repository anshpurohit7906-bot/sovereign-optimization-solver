# OptiCore Architecture

OptiCore is an indigenous, from-scratch mathematical optimization solver implemented in Python using NumPy and SciPy. It provides native solvers for Linear Programming (LP), convex Quadratic Programming (QP), and Mixed-Integer Linear Programming (MILP) without wrapping or delegating to third-party solving engines.

The codebase is organized around numerical transparency: optimization occurs in scaled or normalized internal coordinates, but termination decisions, convergence claims, and incumbent feasibility are validated against the original problem formulation without weakened tolerances.

---

## 1. Project Overview

OptiCore delivers three mathematical programming capabilities:

* **Linear Programming (LP):** A sparse primal-dual Mehrotra predictor-corrector interior-point method (IPM) operating on standard equality form, backed by a sparse SuperLU Schur complement solver with iterative refinement and an indigenous two-phase sparse simplex crossover fallback.
* **Convex Quadratic Programming (QP):** A sparse primal-dual predictor-corrector IPM handling inequality-constrained and bound-constrained convex QPs via an augmented Karush-Kuhn-Tucker (KKT) saddle-point system.
* **Mixed-Integer Linear Programming (MILP):** A native branch-and-bound solver using OptiCore's LP engine for continuous node relaxations, best-bound search, most-fractional branching, rounding and LP-repair heuristics, and independent post-solve incumbent verification.

The repository contains no C/Fortran extension binaries and does not wrap external solvers (such as HiGHS, OSQP, or SciPy's `linprog`) during execution. External solvers are referenced solely in test and benchmark harnesses for oracle validation.

```text
Problem Files (.mps, .qps)
       │
       ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  MPS Parser  │     │  QPS Parser  │     │  MPS Parser  │
 │  (LP Model)  │     │  (QP Problem)│     │ (MILP Model) │
 └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
        │                    │                    │
        ▼                    ▼                    ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │ Standard-Form│     │ Inequality   │     │ Native       │
 │ Conversion & │     │ Formulation  │     │ Branch-&-    │
 │ Sparse Scale │     │              │     │ Bound Engine │
 └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
        │                    │                    │ (Node LPs)
        ▼                    ▼                    │
 ┌──────────────┐     ┌──────────────┐            │
 │ Mehrotra IPM │     │ Mehrotra QP  │            │
 │ (Sparse      │     │ (Augmented   │◄───────────┘
 │  Schur / LU) │     │  KKT / LU)   │
 └──────┬───────┘     └──────┬───────┘
        │ (stalled / tail)   │
        ▼                    │
 ┌──────────────┐            │
 │ Two-Phase    │            │
 │ Crossover    │            │
 └──────┬───────┘            │
        │                    │
        ▼                    ▼
 ┌────────────────────────────────────────────────────────┐
 │        Original-Coordinate Independent Verification     │
 └────────────────────────────────────────────────────────┘
```

---

## 2. LP Architecture

### Production Pipeline

The production LP solving path is sparse end-to-end on the CPU:

1. **MPS Parsing (`src/opticore/mps_parser.py`):** Reads standard MPS format (`NAME`, `ROWS`, `COLUMNS`, `RHS`, `BOUNDS`). Preserves row types (`E`, `L`, `G`, `N`), column coefficients, and variable bounds (`LO`, `UP`, `FX`, `FR`, `BV`, `LI`, `UI`).
2. **Numeric Representation (`src/opticore/numerical_model.py`):** `load_numeric_mps(path, sparse=True)` constructs a `NumericalLP` object. When `sparse=True` is requested, constraint matrix $A$ is assembled directly into Compressed Sparse Row (CSR) format via coordinate triplets, avoiding dense $m \times n$ allocations.
3. **Standard-Form Conversion (`src/opticore/lp/mehrotra.py`):** Transforms the model to standard form:
   $$\min\ c^T x \quad \text{subject to} \quad A x = b,\ x \ge 0$$
   * Lower-bounded variables ($x \ge L$) shift to $x' = x - L$.
   * Upper-bounded variables with free lower bounds ($x \le U$) reflect to $x' = U - x$.
   * Free variables ($-\infty < x < \infty$) split into $x = x^+ - x^-$.
   * Boxed variables ($L \le x \le U$) shift to $x' = x - L \ge 0$ with an added constraint row $x' + s = U - L$.
   * Fixed variables ($x = V$) are eliminated, updating the RHS.
   * Inequality rows append slack columns ($+1$ for $\le$, $-1$ for $\ge$).
   All matrix operations preserve CSR sparsity.
4. **Sparse Scaling (`src/opticore/scaling.py`):** Applies two-sided equilibration:
   $$R = \mathrm{diag}(r_i),\quad r_i = \frac{1}{\max(1, \|A_{i,:}\|_\infty)},\qquad C = \mathrm{diag}(c_j),\quad c_j = \frac{1}{\max(1, \|(RA)_{:,j}\|_\infty)}$$
   Transforming data to $\bar{A} = R A C$, $\bar{b} = R b$, $\bar{c} = C c$.
5. **Mehrotra Predictor-Corrector IPM (`src/opticore/lp/mehrotra.py`):** Iterates in scaled space using positive initial points ($x_0 > 0, z_0 > 0$) derived from least-squares projections.
6. **Schur Complement Factorization (`src/opticore/lp/linear_system.py`):** Solves the reduced Newton system per iteration.
7. **Iterative Refinement:** Stabilizes ill-conditioned linear solves.
8. **Unscaling & Verification:** Solutions are mapped back to original coordinates, where residual bounds are strictly evaluated.

### The Reduced Newton System

The perturbed KKT conditions for standard-form LP are:
$$r_p = A x - b = 0,\qquad r_d = A^T y + z - c = 0,\qquad X Z e = \sigma \mu e$$
where $X = \mathrm{diag}(x)$, $Z = \mathrm{diag}(z)$, and $\mu = x^T z / n$. The full Newton system:
$$\begin{bmatrix} 0 & A^T & I \\ A & 0 & 0 \\ Z & 0 & X \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta z \end{bmatrix} = \begin{bmatrix} -r_d \\ -r_p \\ -r_c \end{bmatrix}$$
Eliminating $\Delta z = -X^{-1} (r_c + Z \Delta x)$ yields the reduced Newton system:
$$\begin{bmatrix} H & -A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix} = \begin{bmatrix} r_d - X^{-1} r_c \\ -r_p \end{bmatrix},\qquad H = \mathrm{diag}\left(\frac{z}{x}\right)$$

Block elimination onto the dual space produces the **Schur complement equation**:
$$S \Delta y = (-r_p) - A H^{-1} (r_d - X^{-1} r_c)$$
where the Schur matrix is:
$$S = A H^{-1} A^T,\qquad H = \mathrm{diag}\left(\frac{z}{x}\right)$$

Once $\Delta y$ is solved, the primal direction is recovered via:
$$\Delta x = H^{-1} (r_d - X^{-1} r_c + A^T \Delta y)$$

### Sparse Factorization and Regularization

Because $x > 0$ and $z > 0$, $H$ is diagonal and positive definite. $H^{-1}$ is computed by elementwise reciprocal divisions with scale-aware regularization:
$$\rho_p = \min\left(\mathrm{base\_reg} \cdot \max(1, \mathrm{mean}(h)),\ \mathrm{MAX\_RHO\_P}\right)$$
with $\mathrm{base\_reg} = 10^{-12}$ and $\mathrm{MAX\_RHO\_P} = 5 \times 10^{-8}$. Capping $\rho_p$ bounds the per-step dual-side perturbation $\rho_p \|\Delta x\|_\infty$ on degenerate problem tails.

$S$ is assembled as a sparse matrix via CSR matrix products and symmetrized:
$$M = \frac{1}{2} (S + S^T) + \mathrm{reg} \cdot I$$
Factorization is performed with SuperLU (`scipy.sparse.linalg.splu`) using column permutation `permc_spec="MMD_AT_PLUS_A"`. If factorization encounters numerical singularity, $\mathrm{reg}$ escalates by factors of 10 (up to 8 times). A single factorization is computed once per IPM iteration and reused across both affine predictor and centering-corrector solves.

### Iterative Refinement

For ill-conditioned Schur systems, SuperLU solves can be rounding-limited. OptiCore executes up to two rounds of iterative refinement in `_schur_refine`:
$$\Delta y^{(k+1)} = \Delta y^{(k)} + S_{\mathrm{reg}}^{-1} \left( b_s - S_{\mathrm{reg}} \Delta y^{(k)} \right)$$
Each correction is accepted **only** if the residual against the exact regularized matrix decreases monotonically:
$$\|b_s - S_{\mathrm{reg}} \Delta y^{(k+1)}\|_\infty < \|b_s - S_{\mathrm{reg}} \Delta y^{(k)}\|_\infty$$
Otherwise, refinement halts and retains the current direction.

---

## 3. LP Crossover

### Clean IPM vs. Conditional Crossover

OptiCore maintains a strict distinction between interior-point convergence and basic simplex solutions:

1. **Normal Clean IPM Termination:** When relative primal residual, dual residual, and duality gap satisfy convergence criteria ($\le 10^{-8}$ default), the interior-point iterate is returned directly. No simplex crossover is required.
2. **Conditional Crossover:** If the IPM reaches an iteration limit, stalls, or terminates in a degenerate numerical tail (`status in ("stalled", "numerical_tail")`), the solver invokes crossover as an optional fallback (`crossover_fallback=True`). The candidate crossover solution is accepted only if its merit satisfies the gate:
   $$\mathrm{merit}_{\mathrm{crossover}} \le \mathrm{CROSSOVER\_MERIT\_RATIO} \cdot \mathrm{merit}_{\mathrm{IPM}}$$
   where $\mathrm{CROSSOVER\_MERIT\_RATIO} = 10^4$.

### Production Crossover Pipeline (`src/opticore/lp/crossover.py`)

Production crossover is an indigenous, cold-start, two-phase sparse simplex implementation operating on standard form $A x = b, x \ge 0$:

* **Phase I (Crash from Artificial Basis):**
  Rows are normalized so $b \ge 0$. An augmented system $[A \quad I] [x \quad a]^T = b$ is initialized from the all-artificial identity basis $B = I$. The objective minimizes $\sum a_i$. When Phase I reaches zero artificial sum ($\le 10^{-7}$), remaining degenerate artificial basic variables are pivoted out using ratio-test steps against original columns. The resulting basis indexes original columns only, providing a guaranteed basic feasible solution without external crash heuristics.
* **Phase II (Sparse Devex Simplex):**
  Solves the Phase-II problem using revised simplex:
  * Basis factorization via sparse SuperLU with per-pivot refactorization.
  * Devex pricing to approximate steepest-edge pivot selection with low computational overhead.
  * Bland tie-breaking on entering and leaving variables activated when consecutive degenerate pivots exceed threshold `MAX_DEGENERATE = 50`.
  * Condition-number-aware feasibility: basic variables slightly negative within $\kappa(B) \cdot \epsilon_{\mathrm{mach}} \cdot \|b\| \cdot \mathrm{SAFETY}$ are soft-clamped to zero without loss of objective consistency.
  * Bases that become genuinely infeasible trigger bounded Phase-I repair restarts (capped at `REPAIR_LIMIT = 30`).

### Status of Experimental RRQR Path

An earlier crossover prototype under `experiment/crossover/` used dense Rank-Revealing QR factorization (`stage1_audit_rrqr.py`) to select an initial basis from the IPM iterate.

On PILOT87 ($m=3608, n=8038$), this dense RRQR step required a 232 MB memory allocation and 21 seconds of compute. The experimental path demonstrated that the cold-start sparse Phase-I crash matches or improves basis quality with **zero dense allocation** and faster end-to-end preparation (265 s vs 559 s). Consequently, RRQR remains an isolated diagnostic experiment under `experiment/` and is **not** part of the production pipeline.

---

## 4. Numerical Validation and Reporting

### Termination Philosophy

OptiCore evaluates convergence against standard relative tolerances in **original standard-form coordinates**:

$$\mathrm{rel\_primal} = \frac{\|A x - b\|_\infty}{1 + \|b\|_\infty} \le \mathrm{tol}$$
$$\mathrm{rel\_dual} = \frac{\|A^T y + z - c\|_\infty}{1 + \|c\|_\infty} \le \mathrm{tol}$$
$$\mathrm{rel\_gap} = \frac{|c^T x - b^T y|}{1 + |c^T x|} \le \mathrm{tol}$$

The solver enforces two operational safeguards:
1. **No Over-Reporting:** Status `optimal` requires all three original-coordinate criteria to pass. If scaled convergence tests pass but unscaled metrics fail due to ill-conditioning, the status is downgraded (e.g., to `numerical_tail`).
2. **Best-Iterate Recovery:** On any non-optimal exit (`stalled`, `max_iterations`, `numerical_tail`), the solver restores the iterate that minimized the original-coordinate merit function:
   $$\mathrm{merit} = \max(\mathrm{rel\_primal},\ \mathrm{rel\_dual},\ \mathrm{rel\_gap})$$

### Independent Validation

The solver does not rely on internal convergence flags to certify correctness:
* **LP Verification (`tools/benchmark_netlib.py`):** Recomputes primal and dual residuals and validates objective values against reference Netlib benchmarks.
* **QP Verification (`src/opticore/qp/verify.py`):** Independently evaluates stationarity, equality violation, inequality slackness, dual multiplier signs, and complementarity.
* **MILP Incumbent Verification (`src/opticore/cli.py`, `tools/validate_milp_incumbent.py`):** Re-checks original row constraints, variable bounds, and integer restrictions using original unscaled model data.

---

## 5. QP Architecture

### Canonical Implementation (`src/opticore/qp/`)

The primary QP implementation handles convex quadratic programs of the form:
$$\min\ \frac{1}{2} x^T P x + q^T x \quad \text{subject to} \quad G x \le h,\quad A x = b,\quad lb \le x \le ub$$
where $P$ is symmetric positive semidefinite (verified via sparse eigenvalue estimation `scipy.sparse.linalg.eigsh` or dense `eigvalsh`).

1. **Problem Formulation (`src/opticore/qp/problem.py`):** Bounds $lb \le x \le ub$ are folded into inequality constraints $G x \le h$. Slack variables $s = h - G x \ge 0$ with dual multipliers $z \ge 0$ represent inequality constraints.
2. **Predictor-Corrector IPM (`src/opticore/qp/solver.py`):**
   Applies a primal-dual Mehrotra algorithm. At each iteration, the solver computes affine predictor directions, calculates centering parameter $\sigma = (\mu_{\mathrm{aff}} / \mu)^3$, and solves the corrected KKT system.
3. **Augmented KKT Saddle-Point System (`src/opticore/qp/linear_system.py`):**
   The KKT conditions linearize into the augmented saddle-point system:
   $$\begin{bmatrix} P + G^T \mathrm{diag}\left(\frac{z}{s}\right) G + \mathrm{reg} \cdot I & A^T \\ A & -\mathrm{dual\_reg} \cdot I \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix} = \begin{bmatrix} r_x \\ -r_p \end{bmatrix}$$
   The $(1,1)$ block is symmetric positive definite for convex $P$, while $-\mathrm{dual\_reg} \cdot I$ prevents exact singularity when $A$ is rank-deficient.
4. **Factorization Backend:**
   In sparse mode, the block matrix $K$ is assembled in CSR format and factorized with SuperLU (`splu`). In dense mode, it falls back to standard dense linear solvers.

### Architectural Note: Predictor-Corrector Factorization

In `src/opticore/qp/solver.py`, the function `_solve_direction` is invoked twice per iteration: once for the affine predictor step and once for the centering-corrector step. Each call invokes `solve_kkt`, which currently reassembles and refactors the augmented matrix $K$ with SuperLU.

Reusing a single factorization across both predictor and corrector steps (analogous to the LP implementation in `src/opticore/lp/linear_system.py`) is an open optimization item. The performance impact of this change remains to be benchmarked and measured.

### Legacy Implementation (`src/opticore/lp/qp.py`)

A separate, earlier QP slice exists under `src/opticore/lp/qp.py`. It converts models to a dense standard equality form $\min \frac{1}{2} x^T Q x + c^T x$ subject to $A x = b, x \ge 0$ and requires finite lower bounds (rejecting free variables).

This module is retained for backward compatibility with earlier tests. It is **not** the canonical sparse QP path and does not support sparse linear algebra or general bounds.

---

## 6. MILP Architecture

### Native Branch-and-Bound (`src/opticore/lp/branch_bound.py`)

OptiCore includes a native branch-and-bound solver designed for mixed-integer linear problems:

* **LP Relaxations:** Continuous node relaxations are solved using OptiCore's production LP solver (`solve_lp`). If an IPM solve stalls, the bounded crossover fallback is invoked. A node relaxation that fails to achieve proven optimality without certifying infeasibility is dropped from the tree to maintain soundness.
* **Node Selection:** Best-bound first, managed via a priority heap ordered by the relaxation objective.
* **Branching Strategy:** Most-fractional variable selection:
  $$j^* = \arg\max_{j \in I} \left( \min(f_j, 1 - f_j) \right),\qquad f_j = x_j - \lfloor x_j \rfloor$$
  Ties are broken by largest absolute objective coefficient $|c_j|$.
* **Strong Branching:** An optional strong branching mechanism (`strong_branch_depth`, `strong_branch_k`) evaluates candidate bound progress on child nodes; it is disabled by default.
* **Pruning Rules:**
  * Prune by infeasibility (certified Phase-I status).
  * Prune by bound: node lower bound $\ge$ incumbent objective $- \epsilon_{\mathrm{gap}}$.
  * Prune by integrality: all integer-marked variables satisfy $|x_j - \mathrm{round}(x_j)| \le \mathrm{int\_tol}$.
* **Rounding and LP Repair Heuristics:**
  At the root node and every `heuristic_freq` nodes, the solver executes an integer-fixing repair heuristic:
  1. Fractional integer variables are rounded to nearest feasible integers.
  2. Rounded integer variables are fixed to constants.
  3. The resulting restricted continuous LP is resolved with the production core.
  4. If feasible, the repaired solution provides an early incumbent.
* **Presolve Redundancy Check:**
  `_drop_redundant_rows` applies a safe mini-presolve removing exact duplicate constraints and linearly dependent equality rows.

### Independent Feasibility Verification

When an incumbent is discovered or reported at termination, it is verified independently via `_verify_milp_incumbent`:
* Row residuals $A_{i,:} x - b_i$ are checked against row types ($= 0$ for `E`, $\le 0$ for `L`, $\ge 0$ for `G`).
* Variable bounds $L_j \le x_j \le U_j$ are verified.
* Integrality $|x_j - \mathrm{round}(x_j)| \le \mathrm{int\_tol}$ is checked for all integer-marked variables.
* Objective value $c^T x$ is independently computed.

### Current Scalability Limitations

The native MILP implementation is an indigenous prototype with known structural limits:
1. **Cold-Start LP Solves:** Every node relaxation solves an LP from scratch via IPM. There is no dual-simplex warm-starting from parent bases.
2. **No Cutting-Plane Framework:** The engine lacks Gomory mixed-integer cuts, Mixed-Integer Rounding (MIR) cuts, knapsack covers, or clique cuts.
3. **Limited Presolve:** The solver performs row-redundancy filtering, but lacks bound tightening, coefficient reduction, probing, or conflict analysis.

---

## 7. Package and CLI Architecture

OptiCore is structured as a standard installable Python package configured via `pyproject.toml`.

### Subcommand Interface (`src/opticore/cli.py`)

The command-line interface provides three primary entry points:

```shell
# Solve continuous LP instance
opticore solve <model.mps> [--crossover] [--verbose]

# Solve convex QP instance
opticore solve-qp <model.qps> [--sparse] [--dense] [--tol TOL]

# Solve mixed-integer LP instance
opticore solve-milp <model.mps> [--maximize] [--node-limit N] [--time-limit S]
```

### Exit Code and Honesty Contract

The CLI enforces strict exit code semantics across all subcommands:
* **Exit code `0`:** Reserved exclusively for **proven optimal** solutions.
* **Exit code `1`:** Returned on solver failure, infeasibility, unproven bounds, limit cut-offs (`node_limit`, `time_limit`, `max_iterations`, `stalled`), or verification failures.

When a solve stops due to limits, the CLI outputs best-bound, incumbent objective, relative gap, and diagnostics, but terminates with exit code 1 to prevent automation scripts from mistaking budget truncations for optimality proofs.

---

## 8. GPU Backend

### Implementation Scope (`src/opticore/lp/gpu_linear_system.py`)

OptiCore provides an optional, opt-in GPU acceleration backend utilizing CUDA and CuPy:

* Enabled by passing `use_gpu=True` to the LP solver or environment.
* Assembles the symmetrized Schur complement $S = A H^{-1} A^T$ in GPU device memory via CuPy array operations.
* Factorizes $S$ using CuPy dense Cholesky decomposition (`cupy.linalg.cholesky`) with iterative refinement on device.

### Architectural Constraints

* **Dense Schur Implementation:** The GPU backend currently materializes a dense Schur complement matrix. It is **not** a sparse CUDA factorization solver.
* **Performance Profile:** For dense or moderately sized constraint matrices, the GPU path offloads Schur assembly and dense solves. For large, highly sparse Netlib problems, the sparse CPU SuperLU backend remains faster and is the production default.
* The GPU backend does not claim general speedups across arbitrary sparse linear programs.

---

## 9. Testing and Benchmark Philosophy

OptiCore prioritizes reproducibility, explicit failure reporting, and independent validation over self-reported metrics:

1. **Deterministic Test Suite:** Over 170 pytest tests cover MPS parsing, boundary conditions, standard-form transformations, linear algebra backends, crossover mechanics, QP optimality conditions, and MILP tree management.
2. **Standard Benchmark Suites:**
   * **Netlib LP:** Evaluated via `tools/benchmark_netlib.py` on standard benchmark models (e.g., `AFIRO`, `ADLITTLE`, `SC205`, `PILOT4`, `PILOT87`).
   * **Maros–Mészáros QP:** Evaluated via `tools/benchmark_maros.py` on canonical convex QP benchmarks.
   * **MIPLIB 2017:** Evaluated via `tools/benchmark_miplib.py` on mixed-integer benchmarks (e.g., `pk1`, `mas74`).
3. **Independent Oracle Certification:**
   Where applicable, solutions are checked against external reference solvers (HiGHS) to confirm mathematical validity to within high numerical precision ($\approx 10^{-10}$ on polished bases).

---

## 10. Known Limitations and Future Work

The following items represent established limitations of the current implementation:

1. **Lack of HSD Infeasibility/Unboundedness Certificates:**
   The LP solver uses standard Mehrotra predictor-corrector equations rather than a Homogeneous Self-Dual (HSD) embedding. While Phase-I crossover can certify infeasibility for standard-form equality models, OptiCore does not currently provide self-dual theoretical certificates for primal/dual infeasibility or ray directions.
2. **QSHIP12S Maros–Mészáros Convergence:**
   On the Maros–Mészáros benchmark set, 5 of 6 tested instances certify optimal. The larger `QSHIP12S` instance ($n=2763$) reaches the iteration limit under default regularization without achieving the target KKT tolerance.
3. **MILP Scalability:**
   Without dual simplex warm-starting and cutting planes, the native MILP solver is limited to small-to-medium combinatorial instances. Large MIPLIB instances require extensive node budgets.
4. **QP Duplication:**
   The codebase contains both the canonical sparse inequality-form QP solver (`src/opticore/qp/`) and the legacy dense equality-form slice (`src/opticore/lp/qp.py`). Consolidating these paths remains future technical cleanup.
5. **QP Factorization Reuse:**
   `src/opticore/qp/solver.py` currently re-factors the augmented KKT matrix for both predictor and corrector steps. Refactoring to solve with multiple right-hand sides against a single factorization will improve QP iteration speed.
6. **Sparse GPU Solvers:**
   Extending the GPU backend from dense Cholesky Schur complements to native sparse GPU linear solvers (e.g., cuDSS or sparse triangular solves) remains future work.
