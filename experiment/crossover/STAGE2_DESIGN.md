# STAGE 2 DESIGN — RRQR Basis → Primal-Feasible Basis → Phase II

Production code is frozen and unmodified. Everything here lives under
`experiment/crossover/` and imports production modules read-only.

## Phase A — Existing Simplex interface findings

1. **Basis representation.** A basis is a `list[int]` of `m` column indices
   into the working constraint matrix (`A_norm` for Phase II, the
   `[A_norm | I]` augmentation for Phase I). Row ordering of the basis list
   is significant only through `basis[row]`; the *set* of columns defines the
   basis matrix `B = A[:, basis]`.

2. **How B⁻¹b is computed.** Dense `np.linalg.solve(B, rhs)` inside
   `_solve_basis(B, rhs, *, condition_limit)`. No factorization reuse; every
   call refactors. There is no warm-started inverse/LU anywhere in
   production.

3. **`_solve_basis` validation.** It computes the *true* 2-norm condition
   number `np.linalg.cond(B)` (SVD-based, not a proxy) and raises
   `SimplexError` if `cond > condition_limit` (production default `1e12`) or
   if the value is non-finite. This is the production numerical gate we reuse
   unchanged for every candidate pivot.

4. **Phase I entering/leaving rules.** Phase I minimizes the sum of
   artificials on `[A_norm | I]` starting from the identity basis
   (`_phase_one_setup`). Entering: Bland's rule (first nonbasic with reduced
   cost < −tol). Leaving: min-ratio test with Bland tie-breaking on the
   basic-column index (`_simplex_iterations`).

5. **Can existing Phase I accept an external basis?** No. `solve_simplex`
   always calls `_phase_one_setup`, which hard-codes `basis = [n..n+m)` (the
   artificials). The generic loop `_simplex_iterations(A, b, c, basis, ...)`
   *does* accept a caller-supplied `basis`, but it returns
   `("numerical_failure", "current basis is not primal feasible")` unless
   `B⁻¹b >= -tol` — so it cannot start from the (infeasible) RRQR basis.

6. **Can existing Phase II accept an external feasible basis?** Not through
   the public `solve_simplex` (no basis parameter), but **yes** through the
   private-but-stable `_simplex_iterations`: once a primal-feasible basis of
   `A_norm` exists, calling `_simplex_iterations(A_norm, b_norm, c_min,
   basis, phase=2, ...)` runs a correct Revised Simplex Phase II from it,
   with every basis re-validated by `_solve_basis`. It enforces primal
   feasibility at the start, so the repair step must deliver `x_B >= -tol`.

7. **Reusable pivot operation?** The in-loop pivot is just
   `basis[leaving_row] = entering`; the reusable machinery is `_solve_basis`
   (validation) and `_simplex_iterations` (the Phase II loop). There is no
   standalone "try an exchange" helper, so the feasibility-repair prototype
   implements its own candidate-exchange loop but reuses `_solve_basis` as
   the acceptance gate.

**Interface boundary decision.** No production signature is changed. The
experiment imports `_solve_basis` and `_simplex_iterations` from
`src/lp/simplex.py` read-only, mirrors production tolerances
(`tol = 1e-8`, `condition_limit = 1e12`), and implements repair + Phase II
driving logic locally.


## Phase B–D — Feasibility repair algorithm

Textbook primal Phase-I-from-an-infeasible-basis (piecewise-linear
sum-of-infeasibilities objective; cf. Maros, *Computational Techniques of the
Simplex Method*, ch. 9). Given basis `B`, `x_B = B⁻¹b`, infeasibility

    phi(x_B) = sum_i max(0, -x_B[i]).

One repair pivot, fully deterministic:

1. **Leaving row** `r`: the most negative `x_B[r]` (ties → smallest index).
2. **Row of B⁻¹**: `rho^T = e_r^T B^{-1}` (dense solve against `B.T`).
3. **Candidate entering columns**: nonbasic `j` with
   `alpha_rj = rho^T a_j < -tol_dir`. Along the step `x_B(t) = x_B - t*d`,
   `d = B^{-1} a_j`, such columns *increase* `x_B[r]` at rate `-alpha_rj > 0`.
4. **Ranking (deterministic)**: candidates sorted by `alpha_rj` ascending —
   steepest repair of the selected infeasibility first.
5. **Numerical gate per candidate (in ranked order)**: build the trial basis,
   require (a) full numerical rank via `np.linalg.matrix_rank`, and (b)
   production `_solve_basis(trial_B, b, condition_limit=1e12)` succeeds.
   A candidate that only "increases rank" but fails the gate is rejected;
   the next ranked candidate is tried. Conditioning is treated as continuous.
6. **Step / ratio test.** With the accepted entering column `j`:
   `t1 = -x_B[r] / alpha_rj` (row r reaches 0), and for currently feasible
   rows `i` with `d[i] > tol_dir`, `t2_i = x_B[i] / d[i]`.
   `t = min(t1, min_i t2_i)`.
   - If `t = t1`: row `r` leaves at value 0 — one infeasibility removed.
   - If `t = t2_i`: a feasible row hits 0 and leaves; `x_B[r]` increased
     strictly (by `-alpha_rj * t`) but stays negative. Classic argument:
     along the chosen direction `phi` decreases at rate `-alpha_rj > 0`
     until the first breakpoint, so stepping to the first breakpoint keeps
     `phi` strictly decreasing.
   - Degenerate `t = 0` (a feasible basic at 0 blocks immediately) changes
     the basis without progress; consecutive degenerate pivots trigger a
     Bland-style fallback entering rule (smallest admissible index) to
     break cycles.
7. **Progress / safety checks (Phase D)**:
   - `max_pivots = 20 * m` hard cap;
   - seen-basis set (frozenset hash) → abort `cycle_detected`;
   - phi non-decreasing over a window → abort `no_progress`;
   - no admissible entering column for *any* infeasible row → abort
     `no_admissible_entering` (reported, never silently replaced by cold
     Phase I);
   - every accepted pivot logs: entering, leaving, leaving value, new min
     basic, #negative basics, rank, cond2 estimate, solve residual, and
     production `_solve_basis` acceptance.

## Phase E — Phase II wrapper

`run_phase2(A, b, c_min, basis)` calls production `_simplex_iterations(...,
phase=2)` and maps the result to an original-coordinate objective via
`sf.recover_original` + `sf.c_orig @ x_orig` (the same convention the frozen
production code uses for bound-shift offsets).

## Known limitations

- Dense linear algebra throughout: PILOT87-scale instances need dense SVDs
  per pivot (~O(m³) each); runtimes are reported honestly and PILOT87 is
  attempted only after PILOT4 validates the implementation.
- Bland's rule in Phase II is anti-cycling but can converge slowly; iteration
  counts are reported, not tuned.
