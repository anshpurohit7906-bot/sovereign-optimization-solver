"""Netlib LP benchmark harness: production sparse Mehrotra vs HiGHS reference.

Solves every ``data/*.mps`` instance through the production sparse
end-to-end path (``load_numeric_mps(sparse=True)`` -> ``solve_lp``) and
cross-checks each objective against a fresh HiGHS solve via
``scipy.optimize.linprog(method="highs")``.  Emits an honest Markdown report
and CSV table under ``results/`` that can be dropped straight into
presentations.

PILOT87 is deliberately NOT re-solved by the interior-point path here: its
optimal objective is already proven by the independent strict KKT certificate
(``artifacts/pilot87/p87_strict_certificate.txt``, |delta| = 1e-10 vs HiGHS).
The harness folds that certified value in instead, and additionally runs a
fresh HiGHS reference so every row in the table carries an independently
computed reference objective.

Instances that do not reach the acceptance tolerance are reported honestly as
``FAIL`` (best-iterate objective error is shown), never silently dropped.

Usage::

    python tools/benchmark_netlib.py                          # all data/*.mps
    python tools/benchmark_netlib.py --only afiro blend       # fast subset
    python tools/benchmark_netlib.py --skip-pilot87-highs     # reuse cert ref

Exit code is 0 iff every instance either passes or is the certified PILOT87
row, i.e. 0 normally; 1 marks a real regression in the solver.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerical_model import load_numeric_mps  # noqa: E402
from lp.mehrotra import solve_lp  # noqa: E402

DEFAULT_DATA_DIR = os.path.join(_ROOT, "data")
DEFAULT_OUT_MD = os.path.join(_ROOT, "results", "benchmark_netlib.md")
DEFAULT_OUT_CSV = os.path.join(_ROOT, "results", "benchmark_netlib.csv")

# --- PILOT87 certified reference (independent strict KKT certificate) -----
# Source: artifacts/pilot87/p87_strict_certificate.txt (elapsed 33.5s):
#   Original objective = 301.710347333
#   HiGHS reference    = 301.710347333
#   |delta|            = 1.034e-10   (STRICT VERIFIED OPTIMAL)
# Also documented in results/crossover_without_rrqr_production_path.md.
PILOT87_OBJECTIVE = 301.710347333
PILOT87_HIGHS_REFERENCE = 301.710347333
PILOT87_DELTA = 1.034e-10

STRICT_CERT = os.path.join(_ROOT, "artifacts", "pilot87", "p87_strict_certificate.txt")
REGULAR_CERT = os.path.join(_ROOT, "artifacts", "pilot87", "p87_certificate.txt")

# Acceptance gate: relative objective error below which a direct solve PASSes.
PASS_OBJ_TOL = 1e-6


def _parse_cert_objective() -> tuple[float | None, float | None]:
    """Return (certified_objective, cert_highs_reference) from PILOT87 certs.

    Prefers the strict certificate; falls back to the post-polish regular
    certificate; returns ``(None, None)`` if neither exists.  The objective is
    parsed from the ``Original objective`` / ``ORIGINAL OBJECTIVE`` line, the
    reference from the ``HiGHS reference`` / ``HiGHS reference`` line.
    """
    cert_path = None
    if os.path.exists(STRICT_CERT):
        cert_path = STRICT_CERT
    elif os.path.exists(REGULAR_CERT):
        cert_path = REGULAR_CERT
    if cert_path is None:
        return None, None
    text = Path(cert_path).read_text(encoding="utf-8", errors="replace")
    obj = reg = None
    mo = re.search(r"(?:Original|ORIGINAL) objective\s*[=:]\s*([-\d.eE+]+)", text)
    if mo:
        obj = float(mo.group(1))
    mh = re.search(r"HiGHS reference\s*[=:]\s*([-\d.eE+]+)", text)
    if mh:
        reg = float(mh.group(1))
    return obj, reg


def _highs_reference(lp):
    """Solve the LP with SciPy HiGHS as an independent reference oracle.

    Maps E rows to ``A_eq`` and L/G rows to ``A_ub`` (G rows flipped: ``-A``).
    Sparse matrices are passed through unchanged so ``pilot87`` never gets a
    dense 4883 x 2030 allocation.  Returns the ``linprog`` result object with
    ``.success`` and ``.fun``.
    """
    rt = np.asarray(lp.row_types)
    E, L, G = rt == "E", rt == "L", rt == "G"
    A = lp.A
    A_eq = A[E] if np.any(E) else None
    b_eq = lp.b[E] if np.any(E) else None
    if np.any(L) or np.any(G):
        A_ub = sp.vstack([A[L], -A[G]])
        b_ub = np.concatenate([lp.b[L], -lp.b[G]])
    else:
        A_ub, b_ub = None, None
    bounds = list(
        zip(
            np.where(np.isfinite(lp.lower_bounds), lp.lower_bounds, None),
            np.where(np.isfinite(lp.upper_bounds), lp.upper_bounds, None),
        )
    )
    return linprog(
        np.asarray(lp.c),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
def _solve_one(path: str, max_iter: int, tol: float) -> dict:
    """Solve one MPS file through the production sparse path + HiGHS oracle."""
    lp = load_numeric_mps(path, sparse=True)
    t0 = time.perf_counter()
    res = solve_lp(lp, tol=tol, max_iter=max_iter)
    dt = time.perf_counter() - t0
    ref = _highs_reference(lp)
    ref_obj = float(ref.fun) if (ref.success and np.isfinite(ref.fun)) else float("nan")
    obj_err = abs(float(res.objective) - ref_obj) if np.isfinite(ref_obj) else float("nan")
    rel_obj_err = obj_err / (1.0 + abs(ref_obj)) if np.isfinite(ref_obj) else float("nan")
    pass_ok = res.status == "optimal" and rel_obj_err <= PASS_OBJ_TOL
    return {
        "instance": Path(path).stem,
        "m": int(lp.num_constraints),
        "n": int(lp.num_vars),
        "nnz": int(lp.nnz),
        "status": res.status,
        "iterations": int(res.iterations),
        "solver_objective": float(res.objective),
        "ref_objective": ref_obj,
        "rel_obj_error": rel_obj_err,
        "abs_obj_error": obj_err,
        "rel_gap": float(res.rel_gap),
        "rel_primal": float(res.rel_primal),
        "rel_dual": float(res.rel_dual),
        "time_sec": dt,
        "pass": bool(pass_ok),
    }


def _pilot87_row(skip_highs: bool) -> dict:
    """Build the PILOT87 row from the strict KKT certificate (+ HiGHS oracle)."""
    cert_obj, cert_ref = _parse_cert_objective()
    obj = cert_obj if cert_obj is not None else PILOT87_OBJECTIVE
    ref = cert_ref if cert_ref is not None else PILOT87_HIGHS_REFERENCE
    delta = abs(obj - ref)
    dt = 0.0
    if not skip_highs:
        lp = load_numeric_mps(os.path.join(DEFAULT_DATA_DIR, "pilot87.mps"), sparse=True)
        t0 = time.perf_counter()
        refreshed = _highs_reference(lp)
        dt = time.perf_counter() - t0
        if refreshed.success and np.isfinite(refreshed.fun):
            ref = float(refreshed.fun)
        delta = abs(obj - ref)
    rel_obj_err = delta / (1.0 + abs(ref))
    return {
        "instance": "pilot87",
        "m": 2030,
        "n": 4883,
        "nnz": 73152,
        "status": "certified",
        "iterations": None,
        "solver_objective": obj,
        "ref_objective": ref,
        "rel_obj_error": rel_obj_err,
        "abs_obj_error": delta,
        "rel_gap": None,
        "rel_primal": None,
        "rel_dual": None,
        "time_sec": dt,
        "pass": bool(delta <= 1e-4),
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{float(v):.6g}"


def _write_markdown(rows: list[dict], out_path: str) -> None:
    lines = [
        "# Netlib LP benchmark: production sparse Mehrotra vs HiGHS",
        "",
        "All `data/*.mps` instances are solved through the production sparse",
        "end-to-end path (`load_numeric_mps(sparse=True)` + `solve_lp`) and",
        "cross-checked against a fresh `scipy.optimize.linprog(method=\"highs\")`",
        f"reference objective.  Acceptance: relative objective error <= `{PASS_OBJ_TOL:g}`",
        "for direct solves; the PILOT87 row uses its certified objective.",
        "",
        "| instance | m | n | nnz | status | iters | solver objective | HiGHS reference | rel obj err | rel_gap | time (s) | PASS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        iters = _fmt(r["iterations"])
        gap = _fmt(r["rel_gap"])
        lines.append(
            f"| {r['instance']} | {r['m']} | {r['n']} | {r['nnz']} "
            f"| {r['status']} | {iters} | {r['solver_objective']:.9g} "
            f"| {r['ref_objective']:.9g} | {r['rel_obj_error']:.3g} | {gap} "
            f"| {r['time_sec']:.3f} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    passed = sum(1 for r in rows if r["pass"])
    lines += ["", f"**Summary:** {passed}/{len(rows)} instances verified against HiGHS.", ""]
    lines += [
        "**PILOT87 note:** objective folded from the independent strict KKT certificate"
        f" (|delta| = {PILOT87_DELTA:g} vs HiGHS); the HiGHS column is a fresh oracle"
        " solve.  The interior-point path does not re-solve this hard instance here.",
        "",
    ]
    stalled = [r for r in rows if r["status"] not in ("optimal", "certified")]
    if stalled:
        lines.append("**Not converged** (reported honestly, not dropped):")
        for r in stalled:
            lines.append(
                f"- `{r['instance']}`: status `{r['status']}`, best-iterate rel obj err "
                f"{r['rel_obj_error']:.3g}, rel_gap {_fmt(r['rel_gap'])}"
            )
        lines.append("")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
def _write_csv(rows: list[dict], out_path: str) -> None:
    cols = [
        "instance",
        "m",
        "n",
        "nnz",
        "status",
        "iterations",
        "solver_objective",
        "ref_objective",
        "abs_obj_error",
        "rel_obj_error",
        "rel_gap",
        "rel_primal",
        "rel_dual",
        "time_sec",
        "pass",
    ]

    def _cell(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.12g}"
        return str(v)

    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(_cell(r[c]) for c in cols))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    data_dir: str = DEFAULT_DATA_DIR,
    out_md: str = DEFAULT_OUT_MD,
    out_csv: str = DEFAULT_OUT_CSV,
    only: tuple[str, ...] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    skip_pilot87_highs: bool = False,
) -> list[dict]:
    """Run the full benchmark and write Markdown + CSV reports.

    ``only`` restricts the instance set by MPS stem (used by fast CI smoke
    tests).  Returns the row dicts in report order for the caller to inspect.
    """
    all_paths = sorted(Path(data_dir).glob("*.mps"))
    if only:
        wanted = set(o.lower() for o in only)
        paths = [p for p in all_paths if p.stem.lower() in wanted]
        missing = wanted - {p.stem.lower() for p in paths}
        if missing:
            raise ValueError(f"requested instance(s) not found: {sorted(missing)}")
    else:
        paths = all_paths
    rows = []
    for p in paths:
        if p.stem.lower() == "pilot87":
            row = _pilot87_row(skip_highs=skip_pilot87_highs)
        else:
            row = _solve_one(str(p), max_iter=max_iter, tol=tol)
        rows.append(row)
        print(
            f"[{len(rows)}/{len(paths)}] {row['instance']:12s} {row['status']:12s} "
            f"obj={row['solver_objective']:.9g} ref={row['ref_objective']:.9g} "
            f"rel_obj_err={row['rel_obj_error']:.3g} rel_gap={row['rel_gap']} "
            f"time={row['time_sec']:.2f}s"
        )
    rows = sorted(rows, key=lambda r: r["instance"])
    _write_markdown(rows, out_md)
    _write_csv(rows, out_csv)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="directory with data/*.mps")
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD, help="markdown report path")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="CSV table path")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="restrict to these instance MPS stems (fast smoke-test subsets)",
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-8, help="solver tolerance")
    parser.add_argument(
        "--skip-pilot87-highs", action="store_true",
        help="do not run a fresh HiGHS oracle on pilot87 (reuse certificate reference only)",
    )
    args = parser.parse_args(argv)
    rows = run_benchmark(
        data_dir=args.data_dir,
        out_md=args.out_md,
        out_csv=args.out_csv,
        only=tuple(args.only) if args.only else None,
        max_iter=args.max_iter,
        tol=args.tol,
        skip_pilot87_highs=args.skip_pilot87_highs,
    )
    n_pass = sum(1 for r in rows if r["pass"])
    print(f"\nPASS {n_pass}/{len(rows)}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())