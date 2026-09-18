# SIH26119 — Public refinery-planning LP benchmark (SAS/OR `mpex06`)

> Public SAS/OR benchmark only — NOT MRPL operational data.  No plant data,
> no proprietary yields, and no operational records are used anywhere here;
> every coefficient comes from the SAS sample-library page cited below.

This directory records provenance for the public refinery demo
(`demo/refinery_public.py`, wired into `demo/demo_sih.py --refinery-public`).
The SAS data is transcribed verbatim into that module; nothing is downloaded
or vendored here.

## Source

* Official: https://support.sas.com/documentation/onlinedoc/or/ex_code/143/mpex06.html
  (SAS/OR Sample Library `mpex06` — "Refinery Optimization (mpex06)",
  Example 06 from *Mathematical Programming Examples*).
* Worked mirror (same data, OPTMODEL transcription with solution):
  https://sassoftware.github.io/sasoptpy/examples/refinery_optimization.html
* Full citation: `source.txt`.

## Formulation (as published)

Flow network with distillation yields, cracking, octane / vapour-pressure /
fuel-oil-ratio blending, plus capacity/policy limits:

* **Variables (51):** one nonnegative `Flow[i,j]` per SAS arc (42 data arcs +
  5 product→sink arcs) + `CrudeDistilled[crude1, crude2]` (2) +
  `OilCracked[light_oil_cracked, heavy_oil_cracked]` (2).  All continuous —
  the benchmark stays an LP, never an MILP.
* **Constraints (46 as published):** 18 flow-balance equalities (every node
  except source/sink), 12 distillation couplings, 4 cracking couplings,
  2 petrol-octane lower bounds (G), 1 jet-fuel vapour-pressure cap (L),
  4 fuel-oil ratio equalities, crude-total cap (L), naphtha-to-reformer cap
  (L), cracked-oil cap (L), lube-oil range (500–1000), premium≥0.40×regular.
* **Nonzeros (158 as published)**; objective maximises per-unit profit
  (SAS table values ÷ 100): premium 7, regular 6, jet 4, fuel-oil 3.5,
  lube-oil 1.5 on the five product→sink flows.

## OPTICORE solve model (one documented deviation)

The four SAS-written fuel-oil equalities sum to zero identically (each column
appears with `S − Σcoef = 0` where `S = 10+4+3+1 = 18`), so only three are
linearly independent.  Keeping the fourth makes the standard-form matrix rank
deficient and trips the solver's exact-rank guard in
`src/lp/mehrotra.py::to_standard_form` (solver core is NOT modified per the
project rules).  The demo therefore keeps the first three fuel-oil rows
(light/heavy/cracked oil; residuum's share is implied) and drops the redundant
`Blending_fuel_oil[residuum]` row:

* OPTICORE model: 51 variables, **45** constraints (E=37, L=4, G=4),
  **154** nonzeros.  Every retained coefficient is verbatim SAS data.
* `solve_public_refinery` uses the existing `solve_lp(..., maximize=True)`
  Mehrotra API; `verify_public_refinery` performs independent
  feasibility/objective verification only (row-type-aware primal + bounds +
  recomputed objective, tolerance 1e-6, never weakened) — it does NOT prove
  optimality or KKT conditions.  Same philosophy as `verify_qp_kkt` /
  `tools/validate_milp_incumbent.py`.

## Published reference vs OPTICORE

* Published optimum ≈ **211365.13477**, crude1 = 15000, crude2 = 30000.
* OPTICORE: objective ≈ 211365.13471; **absolute difference ≈ 6e-05
  (informational only)**; **relative difference ≈ 3e-10, gate rel < 1e-6** →
  benchmark PASS, with crude1 = 15000.00, crude2 = 30000.00.
* Petrol-blending splits are non-unique (degenerate optimum): OPTICORE, HiGHS
  on the same model, and the SAS-published split all agree on the objective
  and every sink/product quantity but differ on interchangeable petrol arcs —
  expected, not a mismatch.
* What-if (new scenario, NOT compared to the published optimum): crude2
  availability 30000 → 24000, re-solved and independently verified.

Run: `python demo/demo_sih.py --refinery-public`
