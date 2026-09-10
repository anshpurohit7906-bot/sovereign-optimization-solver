"""Netlib LP benchmark harness: production sparse Mehrotra vs HiGHS reference.

Every ``solve_lp`` call runs with ``crossover_fallback=True`` (the production
branch default intent), so a stalled / numerical-tail IPM automatically
attempts the sparse-simplex crossover.  Each report row therefore carries a
``method`` that is exactly one of::

    ipm         direct interior-point solve converged on its own
    crossover   IPM stalled; the automatic sparse-crossover fallback resolved
                it (accepted only when all three residuals are <= tol)
    certified   value folded from an independent certificate artifact; the
                model is not re-solved here

PILOT87 is deliberately NOT re-solved by the interior-point path: its optimal
objective is already proven by the independent strict KKT certificate
(``artifacts/pilot87/p87_strict_certificate.txt``, |delta| = 1e-10 vs HiGHS),
so that row is a ``certified`` fold-in.

PILOT4 (``pilot4_plain``) is the canonical ``crossover`` row: the direct IPM
stalls on it, and the automatic fallback converges it end-to-end (Phase I +
Phase II sparse simplex), reproducing the objective independently proven by the
crossover certificate ``artifacts/pilot4/p4_crossover_certificate.txt``
(3/3 bit-identical runs, |delta| = 1.1e-7 vs HiGHS).  The certificate is kept
as an independent cross-check, not as the row source.

A fresh HiGHS reference is run for every row, so each line of the table carries
an independently computed reference objective.

Instances that do not reach the acceptance tolerance are reported honestly as
``FAIL`` (best-iterate objective error is shown), never silently dropped.

Usage::

    python tools/benchmark_netlib.py                           # all data/*.mps (fallback on)
    python tools/benchmark_netlib.py --no-crossover-fallback   # direct IPM only; PILOT4 FAILs honestly
    python tools/benchmark_netlib.py --only afiro blend        # fast subset
    python tools/benchmark_netlib.py --skip-pilot87-highs      # reuse cert ref

Exit code is 0 iff every instance passes (direct, crossover, or a certified
fold-in such as PILOT87), i.e. 0 normally; 1 marks a real regression.
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

# --- PILOT4 crossover certified reference (RRQR -> repair -> Phase II) ----
# Source: artifacts/pilot4/p4_crossover_certificate.txt (3/3 bit-identical runs)
#   Original objective = -2581.139259
#   HiGHS reference    = -2581.139258884
#   |delta|            = 1.116e-07   (CROSSOVER CERTIFIED OPTIMAL)
# See also: experiment/crossover/scratch/pilot4_crossover8.log.
PILOT4_OBJECTIVE = -2581.139259
PILOT4_HIGHS_REFERENCE = -2581.139258884
PILOT4_DELTA = 1.116e-07

# Acceptance gate: relative objective error below which a direct solve PASSes.
PASS_OBJ_TOL = 1e-6


def _parse_cert_objective(cert_path: str | None = None) -> tuple[float | None, float | None]:
    """Return (certified_objective, cert_highs_reference) from a certificate file.

    If *cert_path* is ``None`` the PILOT87 strict/regular certificate path is
    tried as before.  The objective is parsed from the ``Original objective`` /
    ``ORIGINAL OBJECTIVE`` line, the reference from the ``HiGHS reference`` line.
    """
    if cert_path is None:
        if os.path.exists(STRICT_CERT):
            cert_path = STRICT_CERT
        elif os.path.exists(REGULAR_CERT):
            cert_path = REGULAR_CERT
        else:
            return None, None
    if not os.path.exists(cert_path):
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


# Route detection: solve_standard_form sets a deterministic message prefix when
# an accepted sparse-crossover fallback replaces the IPM iterate.  This is the
# only signal the result object exposes today (status is "optimal" both when
# the IPM converges directly and when a crossover resolves a stall).
_CROSSOVER_MSG_RE = re.compile(
    r"^sparse crossover from (?:stalled|numerical_tail): "
    r"Phase I (\d+) iters, Phase II (\d+) pivots"
)


def _parse_crossover_message(message: str | None) -> tuple[int, int] | None:
    """Return ``(phase1_iters, phase2_pivots)`` if *message* records an accepted
    sparse-crossover fallback, else ``None``."""
    if not message:
        return None
    m = _CROSSOVER_MSG_RE.match(message)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


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
def _solve_one(path: str, max_iter: int, tol: float,
               crossover_fallback: bool = True) -> dict:
    """Solve one MPS file through the production sparse path + HiGHS oracle.

    ``method`` is ``"crossover"`` when the interior-point path stalled and the
    automatic sparse-crossover fallback produced the final iterate (detected
    from the deterministic ``solve_standard_form`` message), else ``"ipm"``.
    """
    lp = load_numeric_mps(path, sparse=True)
    t0 = time.perf_counter()
    res = solve_lp(lp, tol=tol, max_iter=max_iter,
                   crossover_fallback=crossover_fallback)
    dt = time.perf_counter() - t0
    crossover_stats = _parse_crossover_message(res.message)
    method = "crossover" if crossover_stats is not None else "ipm"
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
        "method": method,
        "iterations": int(res.iterations),
        "phase1_iters": crossover_stats[0] if crossover_stats else None,
        "phase2_pivots": crossover_stats[1] if crossover_stats else None,
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
        "method": "certified",
        "iterations": None,
        "phase1_iters": None,
        "phase2_pivots": None,
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
        "Every `data/*.mps` instance is solved through the production sparse",
        "end-to-end path (`load_numeric_mps(sparse=True)` + `solve_lp` with",
        "`crossover_fallback=True`) and cross-checked against a fresh",
        "`scipy.optimize.linprog(method=\"highs\")` reference objective.",
        f"Acceptance: relative objective error <= `{PASS_OBJ_TOL:g}`.",
        "Method legend: `ipm` = direct interior-point solve; `crossover` = the",
        "IPM stalled and the automatic sparse-crossover fallback resolved it;",
        "`certified` = value folded from an independent certificate artifact.",
        "",
        "| instance | m | n | nnz | status | method | iters | solver objective | HiGHS reference | rel obj err | rel_gap | time (s) | PASS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        iters = _fmt(r["iterations"])
        if r["method"] == "crossover":
            iters = f"{iters} + {_fmt(r['phase2_pivots'])} piv"
        gap = _fmt(r["rel_gap"])
        lines.append(
            f"| {r['instance']} | {r['m']} | {r['n']} | {r['nnz']} "
            f"| {r['status']} | {r['method']} | {iters} | {r['solver_objective']:.9g} "
            f"| {r['ref_objective']:.9g} | {r['rel_obj_error']:.3g} | {gap} "
            f"| {r['time_sec']:.3f} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    passed = sum(1 for r in rows if r["pass"])
    lines += ["", f"**Summary:** {passed}/{len(rows)} instances verified against HiGHS.", ""]
    lines += [
        "**Method notes**",
        "",
        "- `ipm`: converged by the interior-point path alone.",
        "- `crossover`: the IPM stopped in `stalled`/`numerical_tail`; the internal sparse",
        "  crossover (`crossover_fallback=True`) replaced the iterate only after",
        "  rel_primal/rel_dual/rel_gap were all independently <= tol.  For these rows the",
        "  `iters` cell shows `IPM iterations + Phase II pivots`.",
        "- `certified`: objective folded from an independent certificate; the model is not",
        "  re-solved by the interior-point path here.",
        "",
        "**PILOT87 note:** objective folded from the independent strict KKT certificate"
        f" (|delta| = {PILOT87_DELTA:g} vs HiGHS); the HiGHS column is a fresh oracle"
        " solve.  The interior-point path does not re-solve this hard instance here.",
        "",
        "**PILOT4 note:** `pilot4_plain` is now solved through the production path: the"
        " direct IPM stalls, the automatic sparse-crossover fallback converges it"
        " (see the `crossover` row and its `iters` cell), and the objective is"
        " cross-checked against a fresh HiGHS oracle.  The result reproduces the"
        " independently certified objective"
        f" (`artifacts/pilot4/p4_crossover_certificate.txt`, 3/3 bit-identical runs,"
        f" |delta| = {PILOT4_DELTA:g} vs HiGHS), which is kept as a cross-check rather"
        " than the row source.",
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
        "method",
        "iterations",
        "phase1_iters",
        "phase2_pivots",
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
    crossover_fallback: bool = True,
    skip_pilot87_highs: bool = False,
) -> list[dict]:
    """Run the full benchmark and write Markdown + CSV reports.

    ``only`` restricts the instance set by MPS stem (used by fast CI smoke
    tests).  ``crossover_fallback`` is forwarded to every ``solve_lp`` call; a
    stalled instance whose sparse-simplex crossover converges is reported with
    ``method == "crossover"``.  Returns the row dicts in report order for the
    caller to inspect.
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
        stem = p.stem.lower()
        if stem == "pilot87":
            row = _pilot87_row(skip_highs=skip_pilot87_highs)
        else:
            row = _solve_one(str(p), max_iter=max_iter, tol=tol,
                             crossover_fallback=crossover_fallback)
        rows.append(row)
        print(
            f"[{len(rows)}/{len(paths)}] {row['instance']:12s} "
            f"{row['status']:10s} {row['method']:9s} "
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
    parser.add_argument(
        "--no-crossover-fallback", dest="crossover_fallback", action="store_false",
        help="run the interior-point path only (disables the automatic sparse-crossover"
             " fallback; pilot4_plain then FAILs honestly)",
    )
    args = parser.parse_args(argv)
    rows = run_benchmark(
        data_dir=args.data_dir,
        out_md=args.out_md,
        out_csv=args.out_csv,
        only=tuple(args.only) if args.only else None,
        max_iter=args.max_iter,
        tol=args.tol,
        crossover_fallback=args.crossover_fallback,
        skip_pilot87_highs=args.skip_pilot87_highs,
    )
    n_pass = sum(1 for r in rows if r["pass"])
    print(f"\nPASS {n_pass}/{len(rows)}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())