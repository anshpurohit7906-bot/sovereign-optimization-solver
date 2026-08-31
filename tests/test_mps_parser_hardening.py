"""Focused regression tests for MPS parser hardening against silently
unsupported features (RANGES, SOS, QUADOBJ, INDICATORS, integer MARKER).

Each unsupported-construct test asserts that MPSParseError is raised and
that the message names the specific offending construct -- not just that
*some* exception occurs.
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


def test_integer_marker_rejected() -> None:
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
        "ENDATA\n"
    )
    assert _expect_rejected("MARKER/INTORG/INTEND rejected", text, "MARKER")


def main() -> int:
    print("=" * 80)
    print("MPS PARSER HARDENING - FOCUSED REGRESSION TESTS")
    print("=" * 80)
    tests = [
        test_normal_supported_file_still_parses,
        test_ranges_rejected,
        test_sos_rejected,
        test_quadobj_rejected,
        test_indicators_rejected,
        test_integer_marker_rejected,
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
