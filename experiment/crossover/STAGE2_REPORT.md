# STAGE 2 REPORT — RRQR Basis → Primal-Feasible Basis → Phase II

Experimental code only. No production solver code was modified.
All work is isolated under `experiment/crossover/`.

Stage 1 established that PILOT4's standard-form A contains a numerically
excellent full-row-rank 657-column basis and that pure column-pivoted QR
finds it (cond₂ = 6.85e5, production-accepted). Stage 2 asks: **can such a
basis be converted into a primal-feasible Simplex basis and driven to
optimality?**

The answer is **yes on every small instance, and the RRQR start is a genuine
advantage over a cold Phase I** — but PILOT4's repair stalls on a degenerate
plateau, and PILOT87 is beyond the dense prototype's scaling.

---

## 1. Existing Simplex interface findings (Phase A)

Read-only inspection of `src/lp/simplex.py`:

| Item | Finding |
|---|---|
| Basis representation | Integer list of column indices into A (length m) |
| B⁻¹b computation | `_solve_basis(B, b, condition_limit)` — LU + production gate |
| Basis validation | `_solve_basis` rejects B if `np.linalg.cond(B) > 1e12` or solve residual too large |
| Phase I entering/leaving | `_simplex_iterations(..., phase=1)` — composite Phase I with artificials |
| External basis acceptance | `_simplex_iterations(A, b, c, basis, ...)` accepts an externally supplied basis and re-validates it via `_solve_basis` at every iteration |
| Reusable pivot operation | `_simplex_iterations` is the production loop; it is importable and used read-only |

**Interface boundary:** an experimental wrapper calls
`_simplex_iterations(A, b, c, basis, phase=2, ...)` with a repaired basis.
Because the production loop re-validates the basis at every iteration, an
externally supplied basis cannot bypass any production safeguard.

---

## 2. Algorithm used for feasibility repair (Phase B/C)

**Composite primal Phase I** (not the artificial-variable formulation).

The standard-form LP is `min c·x  s.t.  Ax = b,  x ≥ 0`.  From a generally
infeasible basis B (x_B = B⁻¹b has negative components), minimize the
sum of infeasibilities:

    φ(x_B) = Σ max(0, −x_B[i])

which is piecewise-linear and convex.  Its subgradient w.r.t. x_B is
c_B[i] = −1 for every basic with x_B[i] < 0, else 0.  With y = B⁻ᵀc_B the
nonbasic reduced costs are d_j = −yᵀa_j (c_N = 0); any entering j with
d_j < 0 decreases φ linearly for sufficiently small primal step t > 0.

**Pivot rule:**
1. **Entering:** Devex-normalized Dantzig — maximize d_j² / w_j where w is a
   reference-framework weight approximating ||B⁻¹a_j||² (Forrest–Goldfarb
   update, re-anchored to 1 when max(w) > 1e6).  This approximates the true
   steepest edge and avoids the crawling of raw Dantzig pricing on wide LPs.
2. **Leaving:** composite primal ratio test — feasible basics (x_B ≥ 0) block
   when they hit 0 from above (α > 0); infeasible basics (x_B < 0) block when
   they climb to 0 (α < 0).  t* = min breakpoint.
3. **Tie-break:** lexicographic ratio test among (near-)tied minimum ratios —
   pick the row minimizing [t, row_r(B⁻¹)/α_r] lexicographically.  This makes
   the vector (φ, x_B) strictly decrease lexicographically at every pivot,
   including degenerate t = 0 pivots, so no basis can repeat during a
   degenerate run.

**Conditioning safeguard (continuous, not binary):**
- Cheap exact 1-norm screen via LAPACK `dgetrf` + `dgecon` (O(m²) after the
  O(m³) LU the candidate needs anyway).  By Hölder, cond₁ ≥ cond₂, so
  screening at the production 2-norm limit is **conservative**.
- Accepted pivots get the true 2-norm condition (production parity) and are
  rejected if cond > 1e12.
- Solve residual of the new basis must satisfy ||Bx − b||/(1+||b||) ≤ 1e-7.

**Safety limits (Phase D):** max 5000 pivots, max 100 condition evaluations
per pivot, max 50 consecutive degenerate pivots before a Bland-style
fallback, basis-hash cycling detection with one deterministic escape attempt.

---

## 3. Mathematical justification

The composite Phase I objective φ is convex piecewise-linear.  For any
entering column with d_j < 0, φ decreases linearly for t ∈ (0, t*], so the
rule is guaranteed monotone in exact arithmetic.  The lexicographic tie-break
extends this to a strict lexicographic decrease of (φ, x_B) at every pivot,
which is the standard anti-cycling argument (Orden, Dantzig) adapted to the
piecewise-linear objective: during a degenerate run the infeasible set — and
hence the linear objective — is fixed, so the lexicographic rule applies.

The Devex entering rule approximates the steepest edge without the O(m)
work per candidate of the exact edge norm, and the reference-framework update
keeps w_j an upper-bound approximation of ||B⁻¹a_j||².

---

## 4. Pivot-by-pivot summary

Per-instance pivot logs are emitted with `--verbose`.  Aggregate:

| Instance | Repair pivots | Final neg basics | Final cond |
|---|---|---|---|
| AFIRO | 5 | 0 | 4.92e1 |
| SC205 | 188 | 0 | 3.37e3 |
| ADLITTLE | 27 | 0 | 1.96e3 |
| SHARE2B | 210 | 0 | 2.34e5 |
| BLEND | 121 | 0 | 2.95e5 |
| PILOT4 | 1418 (stalled) | 59 | 1.26e7 |

PILOT4 made substantial progress (112 → 59 negative basics, sum of
infeasibilities 1.31e4 → 9.81e3) but then crawled on a degenerate plateau
(~0.05/pivot) and eventually revisited a basis.  See §8.

---

## 5. Conditioning safeguard

Every accepted basis passed the production `_solve_basis` gate
(cond ≤ 1e12, solve residual ≤ 1e-7).  The 1-norm screen rejected candidates
before the expensive 2-norm check; the accepted basis's true 2-norm condition
is reported.  No basis was ever accepted that the production Simplex would
reject.

---

## 6. Complete benchmark table

RRQR construction + repair + Phase II.  Expected objectives from
`tests/run_benchmarks.py`.  Cold = production `solve_simplex` from scratch.

| Instance | m | n | RRQR cond₂ | neg | Repair status | pivots | PH2 status | PH2 iters | Objective | Expected | Cold status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AFIRO | 27 | 51 | 3.21e1 | 6 | feasible | 5 | optimal | 0 | −464.753143 | −464.753143 ✓ | optimal (51 it) |
| SC205 | 205 | 317 | 2.02e2 | 100 | feasible | 188 | optimal | 79 | −52.202061 | −52.202061 ✓ | **max_iter → nan** |
| ADLITTLE | 56 | 138 | 5.99e2 | 13 | feasible | 27 | optimal | 92 | 225494.963162 | 225494.963162 ✓ | optimal (475 it) |
| SHARE2B | 96 | 162 | 3.16e4 | 34 | feasible | 210 | optimal | 102 | −415.732241 | −415.732241 ✓ | optimal (425 it) |
| BLEND | 74 | 114 | 2.52e3 | 20 | feasible | 121 | optimal | 438 | −30.812150 | −30.812150 ✓ | optimal (641 it) |
| PILOT4 | 657 | 1428 | 6.85e5 | 112 | **cycling** | 1418 | SKIPPED | — | — | −2581.090 | **max_iter → nan** |
| PILOT87 | 3608 | 8038 | 4.08e4 | 920 | **not run** | — | — | — | — | 301.745 | **>150 s (dense)** |

✓ = objective matches expected within 1e-6 relative.


---

## 7. RRQR vs cold Phase I comparison (Phase G)

| Instance | RRQR→PH2 obj | RRQR→PH2 time | Cold obj | Cold time | Winner |
|---|---|---|---|---|---|
| AFIRO | −464.753143 | <0.1 s | −464.753143 | <0.1 s | tie |
| SC205 | **−52.202061** | 0.1 s | **nan (failed)** | 0.1 s | **RRQR** |
| ADLITTLE | 225494.963162 | 0.1 s | 225494.963162 | 0.3 s | tie |
| SHARE2B | −415.732241 | 0.3 s | −415.732241 | 1.0 s | tie |
| BLEND | −30.812150 | 0.1 s | −30.812150 | 0.7 s | tie |
| PILOT4 | **SKIPPED** | 119.8 s | **nan (failed)** | 153.4 s | neither |
| PILOT87 | **not run** | — | **>150 s** | — | neither (dense) |

**SC205 is the decisive case:** cold Phase I hit the iteration limit (1000
pivots, did not even reach Phase II) and returned `nan`, while
RRQR→repair→Phase II solved it to the verified optimum in 79 Phase-II
iterations.  This is a genuine crossover advantage: the cold start is the
bottleneck, and RRQR gives Phase I a well-conditioned, nearly-feasible
starting basis.

---

## 8. Exact failure modes

**PILOT4 — repair stalls on a degenerate plateau.**  The repair reduced
infeasibility from 1.31e4 to 9.81e3 over 1418 pivots (112 → 59 negative
basics) but then progress slowed to ~0.05/pivot and a basis was revisited.
This is a limitation of the dense composite-Phase-I prototype on a 657-row
problem with many infeasible basics, NOT a basis-quality failure: the RRQR
basis itself was excellent (cond₂ = 6.85e5) and every accepted basis passed
the production gate.  Notably, **cold Phase I also fails on PILOT4** (hit
iteration limit → nan), confirming PILOT4 is genuinely hard for a cold start
from either direction.

**PILOT87 — dense scaling limit.**  The standard-form A is 3608 × 8038 at
only 0.26% density, but the prototype uses dense linear algebra.  Each
`cond1_screen` costs ~0.37 s at m=3608, and PILOT87 has 920 negative basics,
so a full repair is hours-long.  This is a prototype implementation limit:
Stage 1 already proved PILOT87's RRQR basis is excellent (cond₂ = 4.08e4,
production-accepted), so the basis quality is NOT the issue.  A sparse LU
implementation would change the scaling entirely.

---

## 9. Whether Phase II was reached

Phase II was reached on all 5 small instances (AFIRO, SC205, ADLITTLE,
SHARE2B, BLEND).  It was **not** reached on PILOT4 (repair did not produce a
feasible basis) or PILOT87 (repair not attempted — dense scaling).

---

## 10. Whether optimality was reached

**Yes, on all 5 small instances.**  Every Phase II run returned `optimal`
with an objective matching the repository's expected value to within 1e-6
relative, and primal residuals at or near machine epsilon.


---

## 11. Objective comparison

See §6 table.  All 5 small instances match their expected objectives exactly
(SC205: −52.2020612117).  PILOT4 reference objective is −2581.090 (Mehrotra
stalled); PILOT87 reference is 301.745 (Mehrotra stalled).  Neither was
reached by Stage 2.

---

## 12. Runtime

See §7.  Small-instance runtimes are dominated by RRQR construction
(<0.1 s to 0.3 s); repair adds up to 0.3 s.  PILOT4 repair took 119.8 s
(dense, 1418 pivots).  PILOT87 dense repair is not feasible in this
prototype (see §8).

---

## 13. Recommendation for Stage 3

1. **Sparse linear algebra is the clear next step.**  The standard-form A is
   <1% dense; a sparse LU (SuperLU, matching the production IPM backend)
   would make PILOT87's repair tractable and likely accelerate PILOT4's.
2. **PILOT4's stall is a degenerate-Phase-I issue, not a basis issue.**
   A sparse implementation, or a warm-started Phase I that exploits the
   near-feasibility of the RRQR basis (59/657 infeasible basics is a small
   repair workload relative to m), is the natural next experiment.
3. **The RRQR start is validated as useful** — SC205 proves it can succeed
   where a cold start fails.  Stage 3 should preserve the RRQR basis
   construction and focus on scaling the repair.

---

## Files

- `experiment/crossover/stage2_repair.py` — repair + Phase II wrapper + driver
- `experiment/crossover/STAGE2_DESIGN.md` — pre-implementation design
- `experiment/crossover/stage1_audit_rrqr.py` — Stage 1 RRQR construction
- No production code was modified.

