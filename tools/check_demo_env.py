"""Demo environment readiness check.

Run this on the demo machine (day-0 setup, day-7 verification, or at the venue
before presenting).  Exit code 0 = ready; 1 = problem detected.

Usage:
    python tools/check_demo_env.py
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "lp"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status:4s}] {name}{suffix}")
    return ok


def main() -> int:
    print("=" * 60)
    print("SIH 26119 — Demo Environment Readiness Check")
    print("=" * 60)
    all_ok = True

    # --- Python version ---
    all_ok &= _check("Python", sys.version_info >= (3, 11),
                      f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # --- NumPy ---
    try:
        import numpy as _np
        all_ok &= _check("NumPy", True, _np.__version__)
    except ImportError:
        all_ok &= _check("NumPy", False, "NOT INSTALLED")

    # --- SciPy ---
    try:
        import scipy as _sp
        all_ok &= _check("SciPy", True, _sp.__version__)
    except ImportError:
        all_ok &= _check("SciPy", False, "NOT INSTALLED (needed for oracle reference)")

    # --- Source modules ---
    try:
        from lp.mehrotra import solve_lp
        all_ok &= _check("solve_lp (production LP core)", True)
    except Exception as e:
        all_ok &= _check("solve_lp (production LP core)", False, str(e)[:60])

    try:
        from lp.branch_bound import solve_milp
        all_ok &= _check("solve_milp (MILP B&B)", True)
    except Exception as e:
        all_ok &= _check("solve_milp (MILP B&B)", False, str(e)[:60])

    try:
        from lp.qp import solve_qp, verify_qp_kkt
        all_ok &= _check("solve_qp + verify_qp_kkt (QP)", True)
    except Exception as e:
        all_ok &= _check("solve_qp + verify_qp_kkt (QP)", False, str(e)[:60])

    # --- Test suite ---
    print()
    print("Running test suite (116 tests)...")
    t0 = time.perf_counter()
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
            cwd=_ROOT, capture_output=True, text=True, timeout=120
        )
        output = result.stdout.strip()
        last_line = output.split("\n")[-1] if output else "no output"
        ok = result.returncode == 0
        all_ok &= _check("pytest tests/", ok, last_line)
    except subprocess.TimeoutExpired:
        all_ok &= _check("pytest tests/", False, "TIMEOUT (>120s)")
    except Exception as e:
        all_ok &= _check("pytest tests/", False, str(e)[:60])

    # --- Quick smoke solve ---
    print()
    t0 = time.perf_counter()
    try:
        from numerical_model import load_numeric_mps
        from lp.mehrotra import solve_lp
        lp = load_numeric_mps(os.path.join(_ROOT, "data", "afiro.mps"), sparse=True)
        res = solve_lp(lp)
        obj_ok = abs(res.objective - (-464.753142659)) < 1e-4
        solve_time = time.perf_counter() - t0
        all_ok &= _check("LP smoke solve (afiro.mps)", obj_ok,
                          f"obj={res.objective:.6f} ({solve_time:.3f}s)")
    except Exception as e:
        all_ok &= _check("LP smoke solve (afiro.mps)", False, str(e)[:60])

    # --- Quick MILP smoke ---
    t0 = time.perf_counter()
    try:
        from numerical_model import NumericalLP
        from lp.branch_bound import solve_milp
        A = np.array([[3., 4., 2., 6.]])
        b = np.array([8.])
        c = np.array([4., 5., 3., 7.])
        lp = NumericalLP("K", "O", A, b, c,
                          np.zeros(4), np.ones(4),
                          ("L",), tuple(f"x{i}" for i in range(4)), ("r0",))
        res = solve_milp(lp, integer_mask=np.ones(4, dtype=bool), maximize=True)
        ok = res.status == "optimal" and abs(res.objective - 10.0) < 1e-4
        solve_time = time.perf_counter() - t0
        all_ok &= _check("MILP smoke (knapsack)", ok,
                          f"obj={res.objective:.2f} ({solve_time:.3f}s)")
    except Exception as e:
        all_ok &= _check("MILP smoke (knapsack)", False, str(e)[:60])

    # --- Quick QP smoke ---
    t0 = time.perf_counter()
    try:
        from lp.qp import NumericalQP, solve_qp
        qp = NumericalQP("T", np.eye(2), np.array([1., -2.]),
                          np.array([[1., 1.], [1., -1.]]), np.array([1., .5]),
                          ("G", "L"), np.zeros(2), np.full(2, np.inf))
        res = solve_qp(qp)
        ok = res.status == "optimal" and abs(res.objective - (-1.5)) < 1e-3
        solve_time = time.perf_counter() - t0
        all_ok &= _check("QP smoke (tiny QP)", ok,
                          f"obj={res.objective:.4f} ({solve_time:.3f}s)")
    except Exception as e:
        all_ok &= _check("QP smoke (tiny QP)", False, str(e)[:60])

    # --- Data files ---
    data_dir = os.path.join(_ROOT, "data")
    data_files = ["afiro.mps", "pilot4_plain.mps", "pilot87.mps"]
    for fn in data_files:
        exists = os.path.exists(os.path.join(data_dir, fn))
        all_ok &= _check(f"data/{fn}", exists)

    # --- Demo scripts ---
    demo_exists = os.path.exists(os.path.join(_ROOT, "demo", "demo_sih.py"))
    all_ok &= _check("demo/demo_sih.py", demo_exists)

    # --- Summary ---
    print()
    print("=" * 60)
    if all_ok:
        print("RESULT: READY FOR DEMO")
    else:
        print("RESULT: ISSUES DETECTED — fix before demo day")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())