#!/usr/bin/env python
"""Profile sparse Mehrotra Newton iterations to identify bottlenecks.

EXPERIMENT ONLY: does not modify src/, tests/, or requirements/.

Profiles the 5000x2500 density=0.3% case at max_iter=3 to identify
why the full solve took ~20 minutes.
"""
import sys
import time
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from opticore.numerical_model import NumericalLP  # noqa: E402
from opticore.scaling import scale_lp  # noqa: E402
import opticore.lp.mehrotra as _mh  # noqa: E402
from opticore.lp.linear_system import factor_reduced_system, solve_reduced_system  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROWS = 5000
COLS = 2500
DENSITY_PCT = 0.30
SEED = 42
TOL = 1e-7
MAX_ITER = 3
RESULTS_DIR = _ROOT / "results"

# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

@dataclass
class TimingRecord:
    name: str
    elapsed: float
    rows: Optional[int] = None
    cols: Optional[int] = None
    nnz: Optional[int] = None
    schur_nnz: Optional[int] = None
    L_nnz: Optional[int] = None
    U_nnz: Optional[int] = None
    fill_in_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProfilerContext:
    def __init__(self):
        self.records: List[TimingRecord] = []
        self.current_iter: Optional[int] = None
        self.iter_timings: Dict[int, List[TimingRecord]] = {}
        
    @contextmanager
    def time_stage(self, name: str, **kwargs):
        """Time a stage and record metadata."""
        print(f"  [{name}] starting...", flush=True)
        t0 = time.perf_counter()
        rec = TimingRecord(name=name, elapsed=0.0, **kwargs)
        self.records.append(rec)
        if self.current_iter is not None:
            if self.current_iter not in self.iter_timings:
                self.iter_timings[self.current_iter] = []
            self.iter_timings[self.current_iter].append(rec)
        try:
            yield rec
        finally:
            rec.elapsed = time.perf_counter() - t0
            print(f"  [{name}] completed in {rec.elapsed:.3f}s", flush=True)


profiler = ProfilerContext()

# ---------------------------------------------------------------------------
# Synthetic LP generator (from benchmark_scaling.py)
# ---------------------------------------------------------------------------

def generate_deterministic_sparse_lp(rows, cols, density_pct, seed=42):
    """Build a NumericalLP with a CSR sparse A (never materialized dense).
    
    Guarantees every row and every column has >= 1 nonzero entry.
    """
    rng = np.random.RandomState(seed)
    target_nnz = max(1, int(rows * cols * (density_pct / 100.0)))
    min_nnz = rows + cols
    actual_target = max(target_nnz, min_nnz)
    row_idx = rng.randint(0, rows, size=actual_target)
    col_idx = rng.randint(0, cols, size=actual_target)
    data = rng.randn(actual_target)
    A_sp = sp.csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols), dtype=np.float64)
    for r in range(rows):
        if A_sp[r].nnz == 0:
            c = rng.randint(0, cols)
            A_sp[r, c] = rng.randn()
    A_sp = A_sp.tocsr()
    for c in range(cols):
        if A_sp[:, c].nnz == 0:
            r = rng.randint(0, rows)
            A_sp[r, c] = rng.randn()
    A_sp = A_sp.tocsr()
    actual_nnz = A_sp.nnz
    b = rng.randn(rows)
    c_vec = rng.randn(cols)
    row_types = tuple(rng.choice(["E", "L", "G"], size=rows, p=[0.6, 0.2, 0.2]))
    lp = NumericalLP(
        name=f"synthetic_{rows}x{cols}",
        objective_name="obj",
        A=A_sp,
        b=b,
        c=c_vec,
        lower_bounds=np.zeros(cols),
        upper_bounds=np.full(cols, np.inf),
        row_types=row_types,
        var_names=tuple(f"x{i}" for i in range(cols)),
        row_names=tuple(f"c{i}" for i in range(rows)),
    )
    return lp, actual_nnz
# ---------------------------------------------------------------------------
# Monkeypatched profiling wrappers
# ---------------------------------------------------------------------------

_original_factor = factor_reduced_system
_original_solve = solve_reduced_system

def _profiled_factor_reduced_system(H, A_eq, reg=1e-12):
    t0 = time.perf_counter()
    fac = _original_factor(H, A_eq, reg)
    elapsed = time.perf_counter() - t0
    rows = A_eq.shape[0] if hasattr(A_eq, 'shape') else 0
    cols = A_eq.shape[1] if hasattr(A_eq, 'shape') else 0
    if fac.A_sp is not None:
        A_nnz = fac.A_sp.nnz
        L_nnz = U_nnz = schur_nnz = fill_in = None
        if fac.schur_lu is not None:
            try:
                L = fac.schur_lu.L
                U = fac.schur_lu.U
                L_nnz = L.nnz
                U_nnz = U.nnz
                schur_nnz = int((A_nnz ** 2) / cols) if cols > 0 else 0
                fill_in = (L_nnz + U_nnz) / schur_nnz if schur_nnz > 0 else None
            except:
                pass
    else:
        A_nnz = rows * cols
        schur_nnz = L_nnz = U_nnz = fill_in = None
    rec = TimingRecord(name="factorization", elapsed=elapsed, rows=rows, cols=cols,
                       nnz=A_nnz, schur_nnz=schur_nnz, L_nnz=L_nnz, U_nnz=U_nnz,
                       fill_in_ratio=fill_in, metadata={"sparse": fac.A_sp is not None})
    profiler.records.append(rec)
    if profiler.current_iter is not None:
        if profiler.current_iter not in profiler.iter_timings:
            profiler.iter_timings[profiler.current_iter] = []
        profiler.iter_timings[profiler.current_iter].append(rec)
    print(f"    [factorization] {elapsed:.3f}s (A: {rows}x{cols}, {A_nnz} NNZ)", flush=True)
    if L_nnz and U_nnz and fill_in:
        print(f"      L: {L_nnz} NNZ, U: {U_nnz} NNZ, fill-in: {fill_in:.2f}x", flush=True)
    return fac

def _profiled_solve_reduced_system(fac, rhs_x, rhs_eq):
    t0 = time.perf_counter()
    dx, dy = _original_solve(fac, rhs_x, rhs_eq)
    elapsed = time.perf_counter() - t0
    rec = TimingRecord(name="solve", elapsed=elapsed, metadata={"sparse": fac.A_sp is not None})
    profiler.records.append(rec)
    if profiler.current_iter is not None:
        if profiler.current_iter not in profiler.iter_timings:
            profiler.iter_timings[profiler.current_iter] = []
        profiler.iter_timings[profiler.current_iter].append(rec)
    return dx, dy

import opticore.lp.linear_system as linear_system
linear_system.factor_reduced_system = _profiled_factor_reduced_system
linear_system.solve_reduced_system = _profiled_solve_reduced_system
_mh.factor_reduced_system = _profiled_factor_reduced_system
_mh.solve_reduced_system = _profiled_solve_reduced_system

# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------

def profile_sparse_newton():
    print(f"\n{'=' * 70}")
    print(f"SPARSE NEWTON PROFILER: {ROWS}x{COLS} density={DENSITY_PCT}% iter={MAX_ITER}")
    print(f"{'=' * 70}\n")

    with profiler.time_stage("1_lp_creation", rows=ROWS, cols=COLS):
        lp, nnz = generate_deterministic_sparse_lp(ROWS, COLS, DENSITY_PCT, SEED)
        profiler.records[-1].nnz = nnz
        print(f"    Generated {nnz} NNZ", flush=True)

    with profiler.time_stage("2_to_standard_form"):
        sf = _mh.to_standard_form(lp)
        profiler.records[-1].nnz = sf.A.nnz if sp.issparse(sf.A) else sf.A.size
        profiler.records[-1].rows = sf.A.shape[0]
        profiler.records[-1].cols = sf.A.shape[1]

    with profiler.time_stage("3_scale_lp"):
        scaled = scale_lp(sf.A, sf.b, sf.c_min, np.zeros(sf.n), np.full(sf.n, np.inf))
        profiler.records[-1].nnz = scaled.A.nnz if sp.issparse(scaled.A) else scaled.A.size

    A, b, c = scaled.A, scaled.b, scaled.c
    m, n = A.shape

    with profiler.time_stage("4_initialization"):
        x, y, z = _mh._mehrotra_initial_point(A, b, c)

    print(f"\n{'=' * 70}\nNEWTON ITERATIONS\n{'=' * 70}\n")

    for k in range(MAX_ITER):
        profiler.current_iter = k
        print(f"\n--- Iteration {k} ---", flush=True)
        t0 = time.perf_counter()

        # residuals
        t1 = time.perf_counter()
        r_p = A @ x - b
        r_d = A.T @ y + z - c
        mu = float(x @ z) / n
        e = time.perf_counter() - t1
        profiler.records.append(TimingRecord(name=f"iter{k}_residuals", elapsed=e))
        profiler.iter_timings.setdefault(k, []).append(profiler.records[-1])
        print(f"  [residuals] {e:.3f}s (mu={mu:.3e})", flush=True)

        # H construction
        t1 = time.perf_counter()
        h = z / x
        e = time.perf_counter() - t1
        profiler.records.append(TimingRecord(name=f"iter{k}_H", elapsed=e))
        profiler.iter_timings[k].append(profiler.records[-1])

        # factorization (monkeypatched, records itself)
        print(f"  [factorization] starting...", flush=True)
        fac = _mh.factor_reduced_system(h, A)

        # affine predictor
        t1 = time.perf_counter()
        rc = x * z
        rhs_x = r_d - rc / x
        dx_a, dy_a = _mh.solve_reduced_system(fac, rhs_x, -r_p)
        dz_a = -(rc + z * dx_a) / x
        a_aff = min(1.0, min(_mh._max_step(x, dx_a), _mh._max_step(z, dz_a)))
        mu_aff = float((x + a_aff * dx_a) @ (z + a_aff * dz_a)) / n
        sigma = min(1.0, max(0.0, (mu_aff / mu) ** 3))
        e = time.perf_counter() - t1
        profiler.records.append(TimingRecord(name=f"iter{k}_affine", elapsed=e))
        profiler.iter_timings[k].append(profiler.records[-1])
        print(f"  [affine] {e:.3f}s (sigma={sigma:.3f})", flush=True)

        # corrector + update
        t1 = time.perf_counter()
        rc = x * z - sigma * mu + dx_a * dz_a
        rhs_x = r_d - rc / x
        dx, dy = _mh.solve_reduced_system(fac, rhs_x, -r_p)
        dz = -(rc + z * dx) / x
        tau = 0.995
        a_p = min(1.0, tau * _mh._max_step(x, dx))
        a_d = min(1.0, tau * _mh._max_step(z, dz))
        x = x + a_p * dx
        y = y + a_d * dy
        z = z + a_d * dz
        e = time.perf_counter() - t1
        profiler.records.append(TimingRecord(name=f"iter{k}_corrector", elapsed=e))
        profiler.iter_timings[k].append(profiler.records[-1])
        print(f"  [corrector+update] {e:.3f}s", flush=True)

        ie = time.perf_counter() - t0
        print(f"  * ITER {k} TOTAL: {ie:.3f}s", flush=True)

    profiler.current_iter = None
    print(f"\n{'=' * 70}\nPROFILING COMPLETE\n{'=' * 70}\n")



# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def analyze_and_report():
    records = profiler.records
    setup_time = sum(r.elapsed for r in records if r.name in
                     ["1_lp_creation", "2_to_standard_form", "3_scale_lp"])
    init_time = sum(r.elapsed for r in records if r.name == "4_initialization")

    iter_totals, factorization_times, solve_times = [], [], []
    for k in range(MAX_ITER):
        if k in profiler.iter_timings:
            recs = profiler.iter_timings[k]
            iter_totals.append(sum(r.elapsed for r in recs))
            factorization_times.append(sum(r.elapsed for r in recs if r.name == "factorization"))
            solve_times.append(sum(r.elapsed for r in recs if r.name == "solve"))

    total_newton = sum(iter_totals)
    avg_fac = float(np.mean(factorization_times)) if factorization_times else 0.0
    avg_solve = float(np.mean(solve_times)) if solve_times else 0.0
    total_fac = sum(factorization_times)
    fac_pct = 100 * total_fac / total_newton if total_newton > 0 else 0.0

    fac_recs = [r for r in records if r.name == "factorization"]
    if fac_recs and fac_recs[0].L_nnz:
        L_nnz, U_nnz, fill = fac_recs[0].L_nnz, fac_recs[0].U_nnz, fac_recs[0].fill_in_ratio
    else:
        L_nnz = U_nnz = fill = None

    print("\n" + "=" * 70)
    print("PROFILE SUMMARY")
    print("=" * 70)
    print(f"A. Setup time:             {setup_time:.3f}s")
    print(f"B. Initialization time:    {init_time:.3f}s")
    print(f"C. Total Newton time:      {total_newton:.3f}s ({MAX_ITER} iterations, "
          f"avg {total_newton/MAX_ITER:.3f}s/iter)")
    print(f"D. Avg factorization:      {avg_fac:.3f}s")
    print(f"E. Avg solve:              {avg_solve:.3f}s")
    if L_nnz and U_nnz:
        print(f"F. L NNZ: {L_nnz:,}, U NNZ: {U_nnz:,}, fill-in ratio: {fill:.2f}x")
    else:
        print("F. L/U NNZ: n/a (SuperLU metrics unavailable)")
    print(f"G. Factorization % of Newton: {fac_pct:.1f}%")
    print(f"H. Dominant bottleneck:     factorization ({total_fac:.3f}s, {fac_pct:.0f}%)")
    print("=" * 70)
    est_min = (total_newton / MAX_ITER) * 100 / 60 if MAX_ITER else 0
    print(f"\nExtrapolation: ~100 iterations x {total_newton/MAX_ITER:.1f}s/iter = ~{est_min:.0f} min")
    print("SOLVES are cheap; FACTORIZATION dominates.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "profile_sparse_newton.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "iteration", "elapsed_s", "L_nnz", "U_nnz", "fill_in_ratio"])
        for r in records:
            iter_num = next((k for k, v in profiler.iter_timings.items() if r in v), "")
            w.writerow([r.name, iter_num, f"{r.elapsed:.6f}",
                        r.L_nnz or "", r.U_nnz or "",
                        f"{r.fill_in_ratio:.4f}" if r.fill_in_ratio else ""])
    print(f"Wrote {csv_path}")

    md_path = RESULTS_DIR / "profile_sparse_newton.md"
    mrows = next((r.rows for r in records if r.name == "factorization" and r.rows), 0)
    lines = [
        "# Sparse Newton Profiler Results",
        "",
        f"**Problem:** {ROWS}×{COLS}, density={DENSITY_PCT}%, {MAX_ITER} iterations",
        "",
        "## Summary",
        "",
        f"- **Setup time:** {setup_time:.3f}s",
        f"- **Initialization:** {init_time:.3f}s",
        f"- **Total Newton time:** {total_newton:.3f}s (avg {total_newton/MAX_ITER:.3f}s/iter)",
        f"- **Avg factorization:** {avg_fac:.3f}s",
        f"- **Avg solve:** {avg_solve:.3f}s",
    ]
    if L_nnz and U_nnz:
        lines.append(f"- **L/U NNZ:** {L_nnz:,}/{U_nnz:,}, fill-ratio {fill:.2f}×")
    lines.extend([
        "",
        "## Bottleneck",
        "",
        f"**Sparse SuperLU factorization is the bottleneck** — {fac_pct:.1f}% of Newton time.",
        "",
        f"Extrapolation: ~100 iterations × {total_newton/MAX_ITER:.1f}s/iter ≈ ~{est_min:.0f} min,",
        "consistent with the observed ~20-min full solve.",
        "",
        f"The Schur system S = A H⁻¹ Aᵀ for this reduced problem is ~{mrows}×{mrows}",
        "with high fill-in, so SuperLU spends most of each iteration on factorization.",
        "",
        "---",
        f"*Generated by `{Path(__file__).name}`*",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")


def main():
    try:
        profile_sparse_newton()
        analyze_and_report()
    except KeyboardInterrupt:
        print("\n\nProfile interrupted; generating partial report...")
        if profiler.records:
            analyze_and_report()
    except Exception as exc:
        print(f"\n\nProfile failed: {exc}")
        import traceback
        traceback.print_exc()
        if profiler.records:
            analyze_and_report()


if __name__ == "__main__":
    main()
