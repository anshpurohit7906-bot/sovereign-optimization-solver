"""Focused CLI tests for the ``opticore solve-milp`` subcommand.

These exercise the public CLI surface end to end on tiny, hand-verifiable MPS
instances written to a temporary directory (no external solver, no committed
fixture needed):

* a general-integer covering model solved to proven optimality (minimize),
* a binary knapsack solved to proven optimality (``--maximize``),
* honest reporting (nonzero exit) when the node/time budget stops the search,
* input validation and clear errors for non-MILP / missing files.

The honesty contract under test: exit status 0 only when the branch-and-bound
search *proves* optimality; the independent incumbent re-check verifies
feasibility only and must pass whenever it is reported.
"""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

from opticore.cli import _parse_args, main  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny MPS fixtures
# ---------------------------------------------------------------------------

_KNAPSACK_MPS = (
    "NAME          KNAPSACK4\n"
    "ROWS\n"
    " N  COST\n"
    " L  CAP\n"
    "COLUMNS\n"
    "    MARKER                 'MARKER'                 'INTORG'\n"
    "    X0        COST      4.0        CAP       3.0\n"
    "    X1        COST      5.0        CAP       4.0\n"
    "    X2        COST      3.0        CAP       2.0\n"
    "    X3        COST      7.0        CAP       6.0\n"
    "    MARKER                 'MARKER'                 'INTEND'\n"
    "RHS\n"
    "    RHS       CAP       8.0\n"
    "BOUNDS\n"
    "    BV BND       X0\n"
    "    BV BND       X1\n"
    "    BV BND       X2\n"
    "    BV BND       X3\n"
    "ENDATA\n"
)

_MINCOVER_MPS = (
    "NAME          MINCOST\n"
    "ROWS\n"
    " N  COST\n"
    " G  DEM\n"
    "COLUMNS\n"
    "    MARKER                 'MARKER'                 'INTORG'\n"
    "    Y0        COST      3.0        DEM       1.0\n"
    "    Y1        COST      2.0        DEM       1.0\n"
    "    Y2        COST      5.0        DEM       1.0\n"
    "    MARKER                 'MARKER'                 'INTEND'\n"
    "RHS\n"
    "    RHS       DEM       4.0\n"
    "BOUNDS\n"
    "    UI BND       Y0        10\n"
    "    UI BND       Y1        10\n"
    "    UI BND       Y2        10\n"
    "ENDATA\n"
)

_CONTINUOUS_MPS = (
    "NAME          CONTONLY\n"
    "ROWS\n"
    " N  COST\n"
    " L  CAP\n"
    "COLUMNS\n"
    "    X0        COST      1.0        CAP       1.0\n"
    "RHS\n"
    "    RHS       CAP       10.0\n"
    "ENDATA\n"
)


def _write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _reported(out: str, label: str) -> str:
    """Return the value printed on the ``  label: value`` report line."""
    prefix = f"  {label}"
    for line in out.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"report line {label!r} not found in output:\n{out}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_parse_args_solve_milp_defaults():
    args = _parse_args(["solve-milp", "some.mps"])
    assert args.command == "solve-milp"
    assert args.mps_file == "some.mps"
    assert args.maximize is False
    assert args.node_limit == 50_000
    assert args.time_limit is None
    assert args.gap_tol == 1e-4
    assert args.int_tol == 1e-6
    assert args.no_crossover_fallback is False
    assert args.no_verify is False
    assert args.verbose is False


def test_parse_args_solve_milp_options():
    args = _parse_args(["solve-milp", "m.mps", "--maximize", "--node-limit", "7",
                        "--time-limit", "2.5", "--gap-tol", "1e-6",
                        "--int-tol", "1e-9", "--no-crossover-fallback",
                        "--no-verify", "-v"])
    assert args.maximize is True
    assert args.node_limit == 7
    assert args.time_limit == 2.5
    assert args.gap_tol == 1e-6
    assert args.int_tol == 1e-9
    assert args.no_crossover_fallback is True
    assert args.no_verify is True
    assert args.verbose is True


# ---------------------------------------------------------------------------
# Proven-optimal solves
# ---------------------------------------------------------------------------

def test_solve_milp_minimize_proven_optimal(tmp_path, capsys):
    """min 3y0 + 2y1 + 5y2 s.t. y0+y1+y2 >= 4, y in [0,10] integer -> 8."""
    path = _write(tmp_path, "mincover.mps", _MINCOVER_MPS)
    code = main(["solve-milp", path])
    out = capsys.readouterr().out

    assert code == 0, out
    assert "OPTICORE - MILP solver (indigenous branch-and-bound)" in out
    assert "variables    : 3 (3 integer, 0 continuous)" in out
    assert _reported(out, "sense        ") == "minimize"
    assert _reported(out, "status       ") == "optimal"
    assert abs(float(_reported(out, "objective    ")) - 8.0) <= 1e-6
    assert abs(float(_reported(out, "best bound   ")) - 8.0) <= 1e-6
    assert abs(float(_reported(out, "relative gap "))) <= 1e-6
    assert _reported(out, "verify pass  ") == "True"


def test_solve_milp_maximize_knapsack_proven_optimal(tmp_path, capsys):
    """max 4x0+5x1+3x2+7x3 s.t. 3x0+4x1+2x2+6x3 <= 8, x binary -> 10."""
    path = _write(tmp_path, "knapsack.mps", _KNAPSACK_MPS)
    code = main(["solve-milp", path, "--maximize"])
    out = capsys.readouterr().out

    assert code == 0, out
    assert "variables    : 4 (4 integer, 0 continuous)" in out
    assert _reported(out, "sense        ") == "maximize"
    assert _reported(out, "status       ") == "optimal"
    assert abs(float(_reported(out, "objective    ")) - 10.0) <= 1e-6
    assert abs(float(_reported(out, "best bound   ")) - 10.0) <= 1e-6
    assert _reported(out, "verify pass  ") == "True"


def test_solve_milp_no_verify_skips_incumbent_check(tmp_path, capsys):
    path = _write(tmp_path, "mincover.mps", _MINCOVER_MPS)
    code = main(["solve-milp", path, "--no-verify"])
    out = capsys.readouterr().out

    assert code == 0, out
    assert "status       : optimal" in out
    assert "verify pass" not in out


# ---------------------------------------------------------------------------
# Honest reporting when the search is cut short
# ---------------------------------------------------------------------------

def test_solve_milp_node_limit_reported_honestly(tmp_path, capsys):
    """A node budget that stops the search must never be reported as optimal."""
    path = _write(tmp_path, "knapsack.mps", _KNAPSACK_MPS)
    code = main(["solve-milp", path, "--maximize", "--node-limit", "1"])
    out = capsys.readouterr().out

    assert code == 1, out
    assert _reported(out, "status       ") == "node_limit"
    assert "status       : optimal" not in out
    assert "detail       : " in out


def test_solve_milp_time_limit_reported_honestly(tmp_path, capsys):
    """An exhausted wall-clock budget is reported as-is with a nonzero exit."""
    path = _write(tmp_path, "knapsack.mps", _KNAPSACK_MPS)
    code = main(["solve-milp", path, "--maximize", "--node-limit", "1",
                 "--time-limit", "1e-9"])
    out = capsys.readouterr().out

    assert code == 1, out
    assert _reported(out, "status       ") == "lp_status:time_limit"
    assert "status       : optimal" not in out


# ---------------------------------------------------------------------------
# Input validation / error paths
# ---------------------------------------------------------------------------

def test_solve_milp_missing_file(capsys):
    code = main(["solve-milp", os.path.join(_ROOT, "data", "definitely_absent.mps")])
    out = capsys.readouterr().out

    assert code == 1
    assert "OPTICORE error: MPS file not found" in out


def test_solve_milp_rejects_continuous_model(tmp_path, capsys):
    path = _write(tmp_path, "continuous.mps", _CONTINUOUS_MPS)
    code = main(["solve-milp", path])
    out = capsys.readouterr().out

    assert code == 1
    assert "no integer variables" in out
    assert "'solve' subcommand" in out


def test_solve_milp_rejects_invalid_node_limit(tmp_path, capsys):
    path = _write(tmp_path, "mincover.mps", _MINCOVER_MPS)
    code = main(["solve-milp", path, "--node-limit", "0"])
    out = capsys.readouterr().out

    assert code == 1
    assert "invalid --node-limit value" in out


def test_solve_milp_rejects_invalid_gap_tol(tmp_path, capsys):
    path = _write(tmp_path, "mincover.mps", _MINCOVER_MPS)
    code = main(["solve-milp", path, "--gap-tol", "0"])
    out = capsys.readouterr().out

    assert code == 1
    assert "invalid --gap-tol value" in out


def test_solve_milp_rejects_invalid_time_limit(tmp_path, capsys):
    path = _write(tmp_path, "mincover.mps", _MINCOVER_MPS)
    code = main(["solve-milp", path, "--time-limit", "-3"])
    out = capsys.readouterr().out

    assert code == 1
    assert "invalid --time-limit value" in out
