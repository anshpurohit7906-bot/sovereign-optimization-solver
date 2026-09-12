# SIH 26119 — Requirement Coverage

## Current Project Status

### LP
Status: Implemented and validated

Evidence:
- From-scratch Mehrotra predictor-corrector IPM
- Revised simplex implementation
- Sparse linear-system backend
- Sparse crossover
- Independent LP certification
- Netlib benchmark harness

Validated instances:
- afiro
- adlittle
- blend
- sc205
- share2b
- pilot4
- pilot87

### MILP
Status: In progress

Current state:
- Reference implementation using SciPy/HiGHS exists
- This is NOT our indigenous MILP solver

Required:
- Integer-variable support
- Branch-and-bound
- LP relaxation using our own LP solver
- Incumbent handling
- Node pruning
- MILP tests
- MIPLIB validation

### QP
Status: In progress

Current state:
- QP implementation is being developed

Required:
- Convex QP validation
- Sparse QP validation
- Independent KKT verification
- Benchmark against Maros–Mészáros QP problems
- Numerical robustness testing

### Sparse Computing
Status: Partially implemented

Current state:
- Sparse linear-system backend
- SuperLU-based sparse solves
- Sparse crossover
- Sparse ordering experiments

Remaining:
- Make the complete production path sparse-friendly
- Reduce unnecessary sparse-to-dense conversions

### GPU Acceleration
Status: Experimental

Current state:
- GPU experiments exist using CuPy

Remaining:
- Stable GPU API
- Meaningful CPU vs GPU benchmark
- Determine where GPU acceleration provides measurable benefit

### Multi-Core
Status: Not implemented

Remaining:
- Identify suitable parallel workloads
- Benchmark multi-core execution

### Large-Scale Problems
Status: Partially validated

Current evidence:
- PILOT87: 3608 constraints × 8038 variables
- Approximately 73k nonzeros

Remaining:
- Validate substantially larger sparse problems
- Measure runtime and memory scaling

### Numerical Robustness
Status: Strong

Current state:
- Numerical-tail detection
- Best-iterate handling
- Sparse linear-system regularization
- Iterative refinement
- Independent LP certification
- PILOT4/PILOT87 stress testing

### Benchmarking
Status: Implemented for LP

Current state:
- Netlib benchmark harness
- HiGHS used as an independent reference/oracle
- Objective and numerical results recorded

Remaining:
- MIPLIB benchmarks for MILP
- Maros–Mészáros benchmarks for QP
- Larger-scale LP benchmarks

### API / CLI
Status: Implemented

Current state:
- Python solver API
- CLI benchmark tools
- Optional dashboard/frontend

### From-Scratch Requirement
Status: Strong for LP

Current state:
- Optimization algorithms are implemented in our repository
- HiGHS/SciPy are used as reference/oracle tools, not as the LP algorithm

## Priority Roadmap

1. Indigenous MILP branch-and-bound
2. Validate QP implementation
3. Integrate common verification/benchmarking
4. Improve end-to-end sparse execution
5. Add presolve
6. Test larger-scale problems
7. GPU/multi-core acceleration
8. Packaging and CI