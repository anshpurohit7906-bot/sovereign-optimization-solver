# SIH 26119 — 7-Day Milestones

**Demo day: in under 1 week.** This document is the team contract.

---

## Roles

* **MILP engineer** — owns integer MPS parsing (`is_integer`), MIPLIB benchmark
  harness, B&B algorithm upgrades (pseudo-cost, cuts, warm starts).
* **QP engineer** — owns `tools/benchmark_qp.py`, sparse QP, Maros-Meszaros
  validation, QP integer extensions later.
* **You (integrator)** — owns the demo script, benchmark tables, presentation,
  and integration quality gates.

---

## Day-by-day plan

### Day 0 (today) — foundation established

- [x] `src/lp/branch_bound.py` exists and solves knapsack/assignment (proved)
- [x] `src/lp/qp.py` exists and solves 2-var convex QP with KKT cert
- [x] Both test suites pass (105 original + 11 new = 116/116)
- [x] Freeze `src/lp/mehrotra.py` API for the week

### Day 1 — parser + harness skeleton

| MILP engineer | QP engineer | Integrator |
|---|---|---|
| Accept `MARKER 'INTORG'/'INTEND'` in COLUMNS; add `BV`, `LI`, `UI` bound types to MPS parser. Produce `LPModel.is_integer`. | Add 3-5 Maros-Meszaros (QPS) convex instances to `data/`. | Commit `docs/sih26119-coverage.md` (currently untracked). Commit this file. |

**M1 gate (end of Day 2):** MILP engineer solves a knapsack with parser-parsed `is_integer`; QP engineer solves a QPS instance.

### Day 2-3 — benchmark harnesses

| MILP engineer | QP engineer |
|---|---|
| `tools/benchmark_miplib.py`: solve 5 easy MIPLIB2017 (mip1, p0033, bell3a, mas74, misc03); compare to `scipy.optimize.milp` oracle. Report `results/benchmark_miplib.md`. | `tools/benchmark_qp.py`: solve 3-5 small QPS; independent KKT cert. Report `results/benchmark_qp.md`. |

**M2 gate (end of Day 4):** benchmark tables with PASS/FAIL for every instance. **If neither gate is hit by Day 4, the demo leads with whatever exists.**

### Day 4 — integration + demo script

Integrator builds `demo/demo_sih.py` pulling both segments. Dry run. Verify
benchmark tables reproduce in < 2 min total on the demo laptop.

### Day 5 — buffer + polish

Fix integration issues. Record a pre-run of PILOT87 full pipeline as artifact
(20-30 min, do it overnight or off-hours). Prepare the recording as a
slide/fallback.

### Day 6 — dry run on target machine

Full demo on the laptop. Verify `tools/check_demo_env.py` passes. Export
benchmark tables as markdown for slides.

### Day 7 — freeze

No new features. Backup artifact. Confidence check: re-run `pytest tests/`
and `python tools/check_demo_env.py` before leaving for the demo.

---

## What we do NOT build this week

- Presolve (future phase)
- GPU/multi-core integration (future phase)
- Large-scale sparse end-to-end (future phase)
- MIQP / NLP / any extension beyond LP+MILP+QP

---

## Acceptance criteria for the demo

| Segment | Criterion | How verified |
|---|---|---|
| LP foundation | 7/7 Netlib vs HiGHS | `python tools/benchmark_netlib.py` |
| MILP | ≥3 MIPLIB instances solved, status "optimal", obj matches oracle | `python tools/benchmark_miplib.py` |
| QP | ≥3 QPS instances solved, KKT verified | `python tools/benchmark_qp.py` |
| Numeric honesty | Status "optimal" iff proven; certificates for all claims | Cert files in `artifacts/` |

---

## Failure escalation

- If MILP or QP engineer hasn't produced a benchmark table by Day 4:
  - Reduce scope: demo the working solver + show "in-progress" screenshot
  - Do NOT delay demo preparation for a missing benchmark
  - Integrator documents honestly what was achieved
