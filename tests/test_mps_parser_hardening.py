"""Focused regression tests for MPS parser hardening against silently
unsupported features (RANGES, SOS, QUADOBJ, INDICATORS) and tests for
supported integer variable features (MARKER/INTORG/INTEND, BV, LI, UI bounds).

Each unsupported-construct test asserts that MPSParseError is raised and
that the message names the specific offending construct -- not just that
*some* exception occurs.

Each supported-construct test asserts correct parsing of integer/binary variables.
"""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mps_parser import MPSParser, MPSParseError  # noqa: E402


_BASE_HEADER = """NAME          TESTPROB
ROWS
 N  COST
 L  LIM1
COLUMNS
    X1        COST      1.0        LIM1      1.0
RHS
    RHS       LIM1      10.0
"""


def _parse_text(text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".mps", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        return MPSParser().parse_file(path)
    finally:
        os.unlink(path)


def _expect_rejected(name: str, text: str, must_mention: str) -> bool:
    try:
        _parse_text(text)
    except MPSParseError as exc:
        ok = must_mention.upper() in str(exc).upper()
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name:28s} | rejected with: {exc}")
        return ok
    print(f"[FAIL] {name:28s} | no MPSParseError was raised")
    return False


def _expect_accepted(name: str, text: str, check_fn) -> bool:
    try:
        model = _parse_text(text)
    except MPSParseError as exc:
        print(f"[FAIL] {name:28s} | unexpected rejection: {exc}")
        return False
    ok = check_fn(model)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name:28s} | {check_fn.__name__}")
    return ok


# --- Unsupported constructs (should still be rejected) ---

def test_ranges_rejected() -> None:
    text = _BASE_HEADER + "RANGES\n    RNG       LIM1      4.0\nBOUNDS\nENDATA\n"
    assert _expect_rejected("RANGES rejected", text, "RANGES")


def test_sos_rejected() -> None:
    text = _BASE_HEADER + "SOS\n S1 SET1\n    X1        1\nENDATA\n"
    assert _expect_rejected("SOS rejected", text, "SOS")


def test_quadobj_rejected() -> None:
    text = _BASE_HEADER + "QUADOBJ\n    X1        X1        2.0\nENDATA\n"
    assert _expect_rejected("QUADOBJ rejected", text, "QUADOBJ")


def test_indicators_rejected() -> None:
    text = _BASE_HEADER + "INDICATORS\n IF LIM1      X1        1\nENDATA\n"
    assert _expect_rejected("INDICATORS rejected", text, "INDICATORS")


# --- Supported integer variable constructs ---

def _check_binary_marker(model):
    ok = (
        model.name == "TESTPROB"
        and model.num_vars() == 1
        and model.num_constraints() == 1
        and model.row_types == ["L"]
        and model.rhs == [10.0]
        and model.is_integer[0] == True
        and model.bounds_lb[0] == 0.0
        and model.bounds_ub[0] == 1.0
    )
    return ok


def test_binary_marker_accepted() -> None:
    """Binary variable via MARKER/INTORG/INTEND + BV bounds."""
    text = (
        "NAME          TESTPROB\n"
        "ROWS\n"
        " N  COST\n"
        " L  LIM1\n"
        "COLUMNS\n"
        "    MARKER                 'MARKER'                 'INTORG'\n"
        "    X1        COST      1.0        LIM1      1.0\n"
        "    MARKER                 'MARKER'                 'INTEND'\n"
        "RHS\n"
        "    RHS       LIM1      10.0\n"
        "BOUNDS\n"
        "    BV BND       X1\n"
        "ENDATA\n"
    )
    assert _expect_accepted("Binary marker accepted", text, _check_binary_marker)


def _check_general_integer_marker(model):
    ok = (
        model.name == "TESTPROB"
        and model.num_vars() == 1
        and model.is_integer[0] == True
        and model.bounds_lb[0] == 0.0  # default
        and model.bounds_ub[0] == 10.0  # from UI bound
    )
    return ok


def test_general_integer_marker_accepted() -> None:
    """General integer variable via MARKER/INTORG/INTEND + UI bound."""
    text = (
        "NAME          TESTPROB\n"
        "ROWS\n"
        " N  COST\n"
        " L  LIM1\n"
        "COLUMNS\n"
        "    MARKER                 'MARKER'                 'INTORG'\n"
        "    X1        COST      1.0        LIM1      1.0\n"
        "    MARKER                 'MARKER'                 'INTEND'\n"
        "RHS\n"
        "    RHS       LIM1      10.0\n"
        "BOUNDS\n"
        "    UI BND       X1       10\n"
        "ENDATA\n"
    )
    assert _expect_accepted("General integer marker accepted", text, _check_general_integer_marker)


def _check_binary_bv_bound(model):
    ok = (
        model.is_integer[0] == True
        and model.bounds_lb[0] == 0.0
        and model.bounds_ub[0] == 1.0
    )
    return ok


def test_binary_bv_bound() -> None:
    """Binary variable via BV bound without markers."""
    text = (
        "NAME          TESTPROB\n"
        "ROWS\n"
        " N  COST\n"
        " L  LIM1\n"
        "COLUMNS\n"
        "    X1        COST      1.0        LIM1      1.0\n"
        "RHS\n"
        "    RHS       LIM1      10.0\n"
        "BOUNDS\n"
        "    BV BND       X1\n"
        "ENDATA\n"
    )
    assert _expect_accepted("BV bound accepted", text, _check_binary_bv_bound)


def _check_li_ui_bounds(model):
    ok = (
        model.is_integer[0] == True
        and model.bounds_lb[0] == 2.0  # LI 2
        and model.bounds_ub[0] == 5.0  # UI 5
    )
    return ok


def test_li_ui_bounds() -> None:
    """General integer variable via LI/UI bounds."""
    text = (
        "NAME          TESTPROB\n"
        "ROWS\n"
        " N  COST\n"
        " L  LIM1\n"
        "COLUMNS\n"
        "    X1        COST      1.0        LIM1      1.0\n"
        "RHS\n"
        "    RHS       LIM1      10.0\n"
        "BOUNDS\n"
        "    LI BND       X1       2\n"
        "    UI BND       X1       5\n"
        "ENDATA\n"
    )
    assert _expect_accepted("LI/UI bounds accepted", text, _check_li_ui_bounds)


def _check_continuous_variable(model):
    ok = (
        model.is_integer[0] == False  # continuous
        and model.bounds_lb[0] == 0.0
        and model.bounds_ub[0] == 100.0
    )
    return ok


def test_continuous_variable() -> None:
    """Continuous variable (no markers, no BV/LI/UI bounds)."""
    text = (
        "NAME          TESTPROB\n"
        "ROWS\n"
        " N  COST\n"
        " L  LIM1\n"
        "COLUMNS\n"
        "    X1        COST      1.0        LIM1      1.0\n"
        "RHS\n"
        "    RHS       LIM1      10.0\n"
        "BOUNDS\n"
        "    UP BND       X1       100\n"
        "ENDATA\n"
    )
    assert _expect_accepted("Continuous variable accepted", text, _check_continuous_variable)


def _check_mixed_integer(model):
    ok = (
        model.num_vars() == 3
        and model.is_integer[0] == True   # X1 binary via marker
        and model.is_integer[1] == True   # X2 general int via UI
        and model.is_integer[2] == False  # X3 continuous
        and model.bounds_lb[0] == 0.0 and model.bounds_ub[0] == 1.0  # X1 binary
        and model.bounds_lb[1] == 0.0 and model.bounds_ub[1] == 5.0   # X2 int [0,5]
        and model.bounds_lb[2] == 0.0 and model.bounds_ub[2] == float("inf")  # X3 cont
    )
    return ok


def test_mixed_integer_model() -> None:
    """Mixed-integer model: binary, general integer, continuous."""
    text = (
        "NAME          TESTPROB\n"
        "ROWS\n"
        " N  COST\n"
        " L  LIM1\n"
        "COLUMNS\n"
        "    MARKER                 'MARKER'                 'INTORG'\n"
        "    X1        COST      1.0        LIM1      1.0\n"
        "    X2        COST      2.0        LIM1      1.0\n"
        "    MARKER                 'MARKER'                 'INTEND'\n"
        "    X3        COST      3.0        LIM1      1.0\n"
        "RHS\n"
        "    RHS       LIM1      10.0\n"
        "BOUNDS\n"
        "    BV BND       X1\n"
        "    UI BND       X2       5\n"
        "ENDATA\n"
    )
    assert _expect_accepted("Mixed integer model accepted", text, _check_mixed_integer)


def test_normal_supported_file_still_parses() -> None:
    text = _BASE_HEADER + "BOUNDS\nENDATA\n"
    try:
        model = _parse_text(text)
        ok = (
            model.name == "TESTPROB"
            and model.objective_name == "COST"
            and model.num_vars() == 1
            and model.num_constraints() == 1
            and model.row_types == ["L"]
            and model.rhs == [10.0]
            and model.is_integer[0] == False  # default is continuous
        )
        print(f"[{'PASS' if ok else 'FAIL'}] {'Normal supported file':28s} | "
              f"vars={model.num_vars()} rows={model.num_constraints()} rhs={model.rhs}")
        assert ok, "supported file did not parse as expected"
    except MPSParseError as exc:
        raise AssertionError(f"unexpected rejection: {exc}") from exc


def test_ranges_rejected() -> None:
    text = _BASE_HEADER + "RANGES\n    RNG       LIM1      4.0\nBOUNDS\nENDATA\n"
    assert _expect_rejected("RANGES rejected", text, "RANGES")


def test_sos_rejected() -> None:
    text = _BASE_HEADER + "SOS\n S1 SET1\n    X1        1\nENDATA\n"
    assert _expect_rejected("SOS rejected", text, "SOS")


def test_quadobj_rejected() -> None:
    text = _BASE_HEADER + "QUADOBJ\n    X1        X1        2.0\nENDATA\n"
    assert _expect_rejected("QUADOBJ rejected", text, "QUADOBJ")


def test_indicators_rejected() -> None:
    text = _BASE_HEADER + "INDICATORS\n IF LIM1      X1        1\nENDATA\n"
    assert _expect_rejected("INDICATORS rejected", text, "INDICATORS")


def main() -> int:
    print("=" * 80)
    print("MPS PARSER HARDENING - FOCUSED REGRESSION TESTS")
    print("=" * 80)
    tests = [
        # Unsupported constructs (should be rejected)
        test_ranges_rejected,
        test_sos_rejected,
        test_quadobj_rejected,
        test_indicators_rejected,
        # Supported integer constructs
        test_normal_supported_file_still_parses,
        test_binary_marker_accepted,
        test_general_integer_marker_accepted,
        test_binary_bv_bound,
        test_li_ui_bounds,
        test_continuous_variable,
        test_mixed_integer_model,
    ]
    results = []
    for t in tests:
        try:
            t()
            results.append(True)
        except AssertionError as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            results.append(False)
    passed = sum(results)
    print("=" * 80)
    print(f"SUMMARY: {passed} PASSED, {len(results) - passed} FAILED out of {len(results)} tests")
    print("=" * 80)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())