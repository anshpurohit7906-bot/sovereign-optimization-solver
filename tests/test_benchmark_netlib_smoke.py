"""Smoke test for the Netlib benchmark harness (fast instances only).

Exercises ``tools/benchmark_netlib.py`` end-to-end on the cheapest
instances (``afiro``, ``blend``) plus the three report routes — direct IPM
(``method == "ipm"``), the automatic sparse-crossover fallback on
``pilot4_plain`` (``method == "crossover"``), and the ``pilot87`` certified
fold-in (``method == "certified"``) — so the harness plumbing — sparse model
loading, ``solve_lp``, the HiGHS reference oracle, crossover-route detection,
and the Markdown/CSV emitters — is covered.  A full all-instance run is
deliberately avoided (PILOT87's fresh HiGHS oracle alone costs ~3s).

The harness is imported directly (no subprocess) so the smoke test stays fast
and portable.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_ROOT, os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_netlib  # noqa: E402

_DATA = os.path.join(_ROOT, "data")


def test_smoke_run_afiro_blend(tmp_path) -> None:
    out_md = os.path.join(tmp_path, "benchmark_netlib.md")
    out_csv = os.path.join(tmp_path, "benchmark_netlib.csv")

    rows = benchmark_netlib.run_benchmark(
        data_dir=_DATA,
        out_md=out_md,
        out_csv=out_csv,
        only=("afiro", "blend"),
        skip_pilot87_highs=True,
    )

    # Both requested instances are present, solved to optimality, and verified.
    assert [r["instance"] for r in rows] == ["afiro", "blend"]
    for r in rows:
        assert r["status"] == "optimal", r
        assert r["method"] == "ipm", r
        assert r["pass"] is True, r
        assert r["rel_obj_error"] <= benchmark_netlib.PASS_OBJ_TOL, r
        assert np.isfinite(r["rel_gap"])
        assert r["m"] > 0 and r["n"] > 0 and r["nnz"] > 0

    # Sanity-check the objective against the known Netlib reference values.
    by_name = {r["instance"]: r for r in rows}
    assert abs(by_name["afiro"]["solver_objective"] - (-464.7531428571)) < 1e-5
    assert abs(by_name["blend"]["solver_objective"] - (-30.8121498458)) < 1e-5

    # Both emitters wrote parseable output with the expected header/rows.
    md_text = open(out_md, encoding="utf-8").read()
    assert "| instance | m | n | nnz | status | method |" in md_text
    for inst in ("afiro", "blend"):
        assert inst in md_text
        assert md_text.index(inst) < md_text.index("**Summary:**")
    csv_text = open(out_csv, encoding="utf-8").read()
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("instance,m,n,nnz,status,method,iterations,phase1_iters")
    assert len(lines) == 3  # header + 2 rows


def test_run_benchmark_verifies_against_highs() -> None:
    """The harness objective and its HiGHS oracle must agree on afiro."""
    from numerical_model import load_numeric_mps  # noqa: E402

    lp = load_numeric_mps(os.path.join(_DATA, "afiro.mps"), sparse=True)
    ref = benchmark_netlib._highs_reference(lp)
    assert ref.success
    # HiGHS oracle reproduces the well-known NETLIB afiro optimum.
    assert abs(float(ref.fun) - (-464.7531428571)) < 1e-9


def test_pilot4_plain_via_automatic_crossover(tmp_path) -> None:
    """pilot4_plain is solved through the production path: the direct IPM
    stalls and the automatic sparse-crossover fallback converges it
    (``method == "crossover"``), reproducing the certified optimum + HiGHS."""
    out_md = os.path.join(tmp_path, "bm_p4.md")
    out_csv = os.path.join(tmp_path, "bm_p4.csv")

    rows = benchmark_netlib.run_benchmark(
        data_dir=_DATA,
        out_md=out_md,
        out_csv=out_csv,
        only=("pilot4_plain",),
        skip_pilot87_highs=True,
    )

    assert len(rows) == 1
    r = rows[0]
    assert r["instance"] == "pilot4_plain"
    assert r["method"] == "crossover"
    assert r["status"] == "optimal"
    assert r["pass"] is True
    assert r["phase1_iters"] and r["phase1_iters"] > 0
    assert r["phase2_pivots"] and r["phase2_pivots"] > 0
    # The production path reproduces the independently certified crossover
    # optimum from artifacts/pilot4/p4_crossover_certificate.txt.
    assert abs(r["solver_objective"] - benchmark_netlib.PILOT4_OBJECTIVE) < 1e-6
    assert r["rel_obj_error"] <= benchmark_netlib.PASS_OBJ_TOL


def test_pilot87_folded_from_strict_certificate(tmp_path) -> None:
    """pilot87 is routed through the strict KKT certificate, not solve_lp."""
    out_md = os.path.join(tmp_path, "bm_p87.md")
    out_csv = os.path.join(tmp_path, "bm_p87.csv")

    rows = benchmark_netlib.run_benchmark(
        data_dir=_DATA,
        out_md=out_md,
        out_csv=out_csv,
        only=("pilot87",),
        skip_pilot87_highs=True,
    )

    assert len(rows) == 1
    r = rows[0]
    assert r["instance"] == "pilot87"
    assert r["method"] == "certified"
    assert r["status"] == "certified"
    assert r["pass"] is True
    assert abs(r["solver_objective"] - benchmark_netlib.PILOT87_OBJECTIVE) < 1e-6