# Crossover feasibility experiment

This is a one-way, isolated probe; it cannot replace a production Mehrotra
result and does not change any source under `src/`.

```text
pilot4_plain.mps -> production parser/numerical model -> StandardFormLP
                 -> production solve_standard_form -> best-trusted x,z
                 -> candidate basis rankings -> B validation and B^-1 b
                 -> existing _simplex_iterations only when x_B >= -tol
```

`solve_standard_form()` already retains and reports its best trusted standard
form iterate on a non-optimal terminal status.  The experiment uses its public
`x_standard` and `z_standard`, so it does not reproduce IPM iterations.

For a standard-form matrix with `m` effective equality rows, every candidate
basis contains exactly `m` columns.  Three simple, deterministic rankings are
tested: large primal values, small dual slacks, and large `|x|/|z|`.  This is
only an identification heuristic.  The experiment invokes Simplex’s existing
`_solve_basis` condition validation and computes `x_B=B^-1b`; if any component
is below `-tol`, it records the failure and deliberately does not repair the
basis or invoke Phase I.  A feasible basis alone may enter existing Phase II.

Out of scope: pivot repair, alternative Phase I, crossover polish, new linear
algebra, changes to IPM/scaling, or automatic selection of a Simplex answer.

## Rank-aware follow-up

`rank_aware_x_over_z_basis()` is a separate PILOT4-only selection probe.  It
scans all columns in descending `|x|/|z|` order, but accepts a column only when
twice-reorthogonalized residual testing finds a numerically independent new
direction.  Lower-ranked columns can therefore replace rejected preferred
columns.  The final square basis is then checked through Simplex’s existing
`_solve_basis` condition-limit check.  It never launches Phase II, evaluates
primal feasibility, repairs a basis, or changes production behavior.

## Terminal-versus-best-gap diagnostic

`terminal_diagnostic.py` uses a temporary Python trace hook while calling the
unchanged production `solve_standard_form`.  It snapshots the solver locals
immediately before a production return path, then applies the same unscaling
map as production.  This preserves two distinct vectors: the public best-gap
result and the actual terminal iterate.  Both feed the identical rank-aware
selection function.  A separate predictor diagnostic reports residuals against
the exact regularized reduced Schur system that was factored and against the
ideal unregularized Newton block equations.  It changes neither backend nor
solver control flow.

## Stage 1 — prior-basis audit + pure RRQR test (`stage1_audit_rrqr.py`)

Diagnostic only; reads production code/data, never modifies it.

Findings (2026-08 freeze state, current production terminal iterate):

- Prior-experiment audit: the naive |x|/z| ranking and the rank-aware greedy
  select genuinely DIFFERENT bases (557 common, 100 unique each on PILOT4).
  The rank-aware "replacement" counter is semantically correct: it counts
  selected columns whose preference-order position >= m, and it equals the
  actual net basis change (100/100) versus the naive set.  The rank-aware
  acceptance threshold (relative residual 1e-10 * ||a_j||) admits
  near-dependence, not just numerical independence: it produced
  sigma_min = 3.4e-12 (rank@1e-10 = 651/657, cond2 = 4.6e15).  The exact
  historical numbers (650/657, cond 4.735e19) are NOT reproducible because
  the terminal IPM iterate — and therefore x/z — changed with the frozen
  production fixes (H-cap, best-iterate policy); only the qualitative
  conclusion reproduces (rank-aware better than naive, still unacceptable;
  naive is singular, sigma_min = 3e-18, cond2 ~ 1e21, LU solve overflows).
- Pure RRQR (`scipy.linalg.qr(..., pivoting=True)`, first m pivots, no IPM
  information) answers the Stage-1 question with conclusion **A**: a
  numerically good full-rank basis exists and RRQR finds it.
  - PILOT4: m=657, rank@1e-10=657, sigma_min=8.57e-2, sigma_max=5.87e4,
    cond2=6.85e5, solve residual 3.8e-16, ACCEPTED by production
    `_solve_basis` (condition limit 1e12).
  - AFIRO/SC205/ADLITTLE/SHARE2B/BLEND: cond2 3.2e1..3.2e4, all accepted.
  - PILOT87: m=3608, rank 3608/3608, sigma_min=2.46e-2, cond2=4.08e4,
    residual 1.7e-16, accepted; dense QR 76 s.
- Part 4 (IPM relation): x_B = B^{-1} b for the RRQR basis is primal
  INFEASIBLE by design (PILOT4: min(x_B) = -3.26e3, 112 negative basics;
  PILOT87: 920 negative basics).  Matrix quality and vertex correspondence
  are separate problems, as intended.

Stage-1 conclusion: the x/z heuristics failed because of basis-selection
quality, not because PILOT4/PILOT87 lack good bases.  Stage 2 should start
from a numerically stable basis (RRQR/LU-selected) and reach a feasible,
then optimal, basis via controlled simplex pivots (Phase-I-style repair),
rather than trying to pick the final basis directly from terminal x/z
ratios.
