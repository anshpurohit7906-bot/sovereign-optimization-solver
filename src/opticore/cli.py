"""OPTICORE CLI: solve an MPS file with the indigenous LP engine.

Public surface (repository root)::

    opticore solve data/afiro.mps --crossover
    opticore solve data/pilot87.mps --crossover
    opticore solve data/pilot87.mps --crossover --cert-fallback
    opticore solve data/pilot87.mps --crossover --time-limit 120
    opticore solve-qp data/qp/real_benchmarks/maros_mesaras/QGROW15.SIF
    opticore solve-milp data/pk1.mps --time-limit 60

Semantics: ``opticore solve <mps>`` ALWAYS means a genuine live solve by
default.  The stored PILOT87 strict certificate is an explicit opt-in fast
path (``--cert-fallback``) for demonstrations under time pressure — it is
NEVER used silently, and NEVER after a timeout.  ``--time-limit SECONDS``
bounds the live solve: on expiry the CLI reports TIME LIMIT / LIVE SOLVE
INCOMPLETE with a nonzero exit.

``opticore solve-milp <mps>`` runs the native branch-and-bound MILP engine
(``opticore.lp.branch_bound.solve_milp``) on an MPS file carrying integer
markers (``MARKER 'INTORG'/'INTEND'``) or BV/LI/UI bounds.  The honesty
contract mirrors the LP core: exit status 0 only when the B&B search
*proves* optimality; ``node_limit`` / ``time_limit`` / ``lp_status:*`` are
reported as-is with a nonzero exit.  A returned incumbent is independently
re-checked (row-type-aware residuals, bounds, integrality, recomputed
``c @ x``) — that check verifies feasibility only, never optimality, and
its tolerances are never weakened.

Only the useful subset of the archived teammate CLI is ported:
``solve <mps> [--crossover] [-v]`` on top of the CURRENT
``load_numeric_mps`` + ``solve_lp`` API.  The old ``--sparse`` flag,
``sovereign-solver`` console name, and getattr-based residual probing are
obsolete and intentionally NOT carried over (sparse end-to-end is the
production default; residuals come from stable MehrotraResult fields).

Error policy: missing file / invalid MPS / solver failure produce a clear
message and a nonzero exit, with NO traceback unless ``-v`` is given.
No silent fallback to any other solver is ever performed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
import traceback

_CROSSOVER_PREFIX = re.compile(r"^sparse crossover from (?:stalled|numerical_tail): ")

# Set when a timed-out live solve leaves the solver worker thread running.
# Interpreter shutdown would otherwise race that thread's writes to stdout
# ("Exception ignored on flushing sys.stdout" -> bogus exit code 120), so
# ``main`` hard-exits with os._exit after flushing our own report.
_HARD_EXIT = False


def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # The CLI now lives inside the ``opticore`` package.  ``__file__`` points
    # into ``src/opticore/``, so the repository root is two parents up
    # from this package's path component and is used only for locating
    # repository-relative asset paths (e.g. stored pilot87 certificate files).
    return os.path.normpath(os.path.join(here, "..", ".."))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="opticore",
        description="OPTICORE indigenous LP solver - solve an MPS file (Mehrotra IPM, CPU).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    solve = sub.add_parser("solve", help="solve one MPS file with the indigenous LP engine")
    solve.add_argument("mps_file", help="path to the .mps file (e.g. data/afiro.mps)")
    solve.add_argument("--crossover", action="store_true",
                       help="allow the automatic sparse-crossover fallback if the IPM stalls")
    solve.add_argument("--cert-fallback", action="store_true",
                       help="explicit opt-in: report pilot87 from the stored strict KKT "
                           "certificate instead of a live solve (fast path for demos "
                           "under time pressure; never used implicitly)")
    solve.add_argument("--no-cert-fallback", action="store_true",
                       help="deprecated no-op kept for command compatibility: the live "
                           "solve is already the default and the certificate is never "
                           "used implicitly")
    solve.add_argument("--time-limit", type=float, default=None, metavar="SECONDS",
                       help="optional wall-clock limit for the live solve; on expiry "
                            "report TIME LIMIT / LIVE SOLVE INCOMPLETE with a nonzero exit "
                            "and NEVER fall back to the certificate")
    solve.add_argument("--verbose", "-v", action="store_true",
                       help="print iteration logs (and tracebacks on failure)")
    qp = sub.add_parser("solve-qp", help="solve a convex QP from a Maros-Mészáros SIF file")
    qp.add_argument("sif_file", help="path to the .SIF file (e.g. data/qp/real_benchmarks/maros_mesaras/QGROW15.SIF)")
    qp.add_argument("--verbose", "-v", action="store_true",
                    help="print iteration logs from the QP solver")
    milp = sub.add_parser(
        "solve-milp",
        help="solve a mixed-integer LP (MPS with integer markers) by native branch-and-bound")
    milp.add_argument("mps_file",
                      help="path to the .mps file with integer markers/bounds "
                           "(e.g. data/pk1.mps)")
    milp.add_argument("--maximize", action="store_true",
                      help="optimize the objective in the maximization sense "
                           "(default: minimization; MPS OBJSENSE is not parsed)")
    milp.add_argument("--node-limit", type=int, default=50_000, metavar="NODES",
                      help="branch-and-bound node budget (default: 50000); on "
                           "exhaustion the solve is reported honestly as node_limit")
    milp.add_argument("--time-limit", type=float, default=None, metavar="SECONDS",
                      help="optional wall-clock limit for the branch-and-bound search; "
                           "on expiry the solve is reported honestly as time_limit")
    milp.add_argument("--gap-tol", type=float, default=1e-4, metavar="GAP",
                      help="relative MIP gap at which the incumbent is accepted as "
                           "proven optimal (default: 1e-4)")
    milp.add_argument("--int-tol", type=float, default=1e-6, metavar="TOL",
                      help="integrality tolerance used for branching/pruning "
                           "(default: 1e-6)")
    milp.add_argument("--no-crossover-fallback", action="store_true",
                      help="disable the bounded sparse-crossover fallback used when a "
                           "node relaxation stalls numerically (default: enabled)")
    milp.add_argument("--no-verify", action="store_true",
                      help="skip the independent incumbent feasibility re-check "
                           "(default: verify)")
    milp.add_argument("--verbose", "-v", action="store_true",
                      help="print branch-and-bound node logs (and tracebacks on failure)")
    return parser.parse_args(argv)


def _fail(message, *, verbose, exc=None) -> int:
    print(f"OPTICORE error: {message}")
    if verbose and exc is not None:
        traceback.print_exception(exc)
    return 1


def _fmt(value, *, precision=9) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if f != f or f in (float("inf"), float("-inf")):
        return "n/a"
    return f"{f:.{precision}f}"


def _fmt_sci(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if f != f or f in (float("inf"), float("-inf")):
        return "n/a"
    return f"{f:.3e}"


def cmd_solve(args) -> int:
    try:
        from opticore.mps_parser import MPSParseError
        from opticore.numerical_model import NumericalModelError, load_numeric_mps
        from opticore.lp.mehrotra import MehrotraError, solve_lp
    except Exception as exc:
        return _fail(f"cannot initialise the OPTICORE engine ({exc})",
                     verbose=args.verbose, exc=exc)

    path = args.mps_file
    if not os.path.isfile(path):
        return _fail(f"MPS file not found: {path}", verbose=args.verbose)

    time_limit = args.time_limit
    if time_limit is not None:
        try:
            time_limit = float(time_limit)
        except (TypeError, ValueError):
            return _fail(f"invalid --time-limit value: {args.time_limit!r} "
                         f"(expected seconds as a number)", verbose=args.verbose)
        if not (time_limit == time_limit and time_limit > 0.0):
            return _fail(f"invalid --time-limit value: {args.time_limit!r} "
                         f"(expected a positive number of seconds)",
                         verbose=args.verbose)

    try:
        t_load = time.perf_counter()
        lp = load_numeric_mps(path, sparse=True)
        load_sec = time.perf_counter() - t_load
    except (MPSParseError, NumericalModelError) as exc:
        return _fail(f"invalid MPS file {path!r}: {exc}", verbose=args.verbose, exc=exc)
    except Exception as exc:
        return _fail(f"could not load MPS file {path!r}: {exc}",
                     verbose=args.verbose, exc=exc)

    method = "Mehrotra IPM (CPU backend)"
    crossover_state = "enabled" if args.crossover else "disabled"
    if args.cert_fallback:
        if _is_pilot87(path, lp):
            return _report_pilot87_certificate(
                path, lp, load_sec, method, crossover_state, verbose=args.verbose)
        print("note: --cert-fallback applies to the stored pilot87 certificate only; "
              "running a genuine live solve instead.")
    try:
        live = _run_live_solve(solve_lp, lp, crossover=args.crossover,
                               verbose=args.verbose, time_limit=time_limit)
    except MehrotraError as exc:
        return _fail(f"solver failure on {path!r}: {exc}", verbose=args.verbose, exc=exc)
    except Exception as exc:
        return _fail(f"solver failure on {path!r}: {exc}", verbose=args.verbose, exc=exc)
    if live["timed_out"]:
        global _HARD_EXIT
        elapsed = live["elapsed"]
        print("OPTICORE - indigenous LP solver")
        print(f"  input file   : {path}")
        print(f"  problem      : {lp.name} (objective row: {lp.objective_name})")
        print(f"  variables    : {lp.num_vars}")
        print(f"  constraints  : {lp.num_constraints}")
        print(f"  nonzeros     : {lp.nnz}")
        print(f"  method       : {method}")
        print(f"  crossover    : {crossover_state}")
        print(f"  RESULT SOURCE : LIVE OPTICORE SOLVE")
        print(f"  LIVE SOLVE    : INCOMPLETE (TIME LIMIT of {time_limit:g}s reached "
              f"after {elapsed:.1f}s; the solver thread was still running and its "
              f"late result is discarded)")
        print(f"  status        : TIME LIMIT")
        print(f"  load time     : {load_sec:.4f} s")
        print(f"  runtime       : {elapsed:.4f} s (limit {time_limit:g}s)")
        print("  detail        : live solve did not finish within the time limit; "
              "the stored certificate was NOT consulted (request it explicitly "
              "with --cert-fallback)")
        sys.stdout.flush()
        _HARD_EXIT = True
        return 1
    result = live["result"]
    runtime = live["elapsed"]

    crossover_used = bool(_CROSSOVER_PREFIX.match(result.message or ""))
    ok = result.status == "optimal"
    print("OPTICORE - indigenous LP solver")
    print(f"  input file   : {path}")
    print(f"  problem      : {lp.name} (objective row: {lp.objective_name})")
    print(f"  variables    : {lp.num_vars}")
    print(f"  constraints  : {lp.num_constraints}")
    print(f"  nonzeros     : {lp.nnz}")
    print(f"  method       : {method}")
    if crossover_used:
        print(f"  crossover    : {crossover_state} (activated: sparse crossover "
              f"replaced the IPM iterate)")
    else:
        print(f"  crossover    : {crossover_state}")
    print(f"  RESULT SOURCE : LIVE OPTICORE SOLVE")
    print(f"  LIVE SOLVE    : COMPLETED")
    print(f"  status       : {result.status}")
    print(f"  objective    : {_fmt(result.objective)}")
    print(f"  iterations   : {result.iterations}")
    print(f"  load time    : {load_sec:.4f} s")
    print(f"  runtime      : {runtime:.4f} s")
    print(f"  primal resid : {_fmt_sci(result.primal_residual)} "
          f"(rel {_fmt_sci(result.rel_primal)})")
    print(f"  dual resid   : {_fmt_sci(result.dual_residual)} "
          f"(rel {_fmt_sci(result.rel_dual)})")
    print(f"  relative gap : {_fmt_sci(result.rel_gap)}")
    if not ok:
        print(f"  detail       : {result.message}")
    return 0 if ok else 1


def cmd_solve_qp(args) -> int:
    try:
        from opticore.qp.qps import read_qps
        from opticore.qp.solver import solve_qp
        from opticore.qp.verify import certificate
    except Exception as exc:
        return _fail(f"cannot initialise the QP engine ({exc})",
                     verbose=args.verbose, exc=exc)

    path = args.sif_file
    if not os.path.isfile(path):
        return _fail(f"SIF file not found: {path}", verbose=args.verbose)

    try:
        problem = read_qps(path)
    except Exception as exc:
        return _fail(f"cannot read QP file {path!r}: {exc}",
                     verbose=args.verbose, exc=exc)

    try:
        t0 = time.perf_counter()
        result = solve_qp(problem, verbose=args.verbose)
        solve_sec = time.perf_counter() - t0
    except Exception as exc:
        return _fail(f"QP solver failure on {path!r}: {exc}",
                     verbose=args.verbose, exc=exc)

    if result.status != "optimal":
        return _fail(f"QP solver did not converge (status={result.status})",
                     verbose=args.verbose)

    try:
        cert = certificate(problem, result.x, result.y, result.z, result.s, tol=1e-6)
    except Exception as exc:
        return _fail(f"certificate verification failed on {path!r}: {exc}",
                     verbose=args.verbose, exc=exc)

    if not cert["ok"]:
        worst = max(
            cert["stationarity"],
            cert["equality_residual"],
            cert["inequality_violation"],
            cert["dual_violation"],
            cert["complementarity"],
        )
        return _fail(f"KKT certificate FAILED on {path!r}: max_violation={worst:.3e}",
                     verbose=args.verbose)

    # ---- report -------------------------------------------------------
    print("OPTICORE - QP solver (indigenous)")
    print(f"  input file   : {path}")
    print(f"  problem      : {problem.name} ({problem.n} variables, "
          f"{problem.m_ineq} inequality rows, {problem.m_eq} equality rows)")
    print(f"  STATUS       : COMPLETED")
    print(f"  objective    : {result.objective:.9f}")
    print(f"  iterations   : {result.iterations}")
    print(f"  runtime      : {solve_sec:.4f} s")
    print(f"  KKT certificate: ok={cert['ok']}")
    print(f"  stationarity : {cert['stationarity']:.3e}")
    print(f"  scaled_statio: {cert['scaled_stationarity']:.3e}")
    print(f"  equality_res : {cert['equality_residual']:.3e}")
    print(f"  inequality_v : {cert['inequality_violation']:.3e}")
    print(f"  dual_violatio: {cert['dual_violation']:.3e}")
    print(f"  complementari: {cert['complementarity']:.3e}")
    print(f"  objective_cert: {cert['objective']:.9f}")
    return 0


def _verify_milp_incumbent(lp, x, integer_mask, *, tol=1e-6) -> dict:
    """Independently re-check a branch-and-bound incumbent from ORIGINAL data.

    Mirrors ``tools/validate_milp_incumbent.py``: row-type-aware primal
    residuals, variable bounds, integrality of the integer variables and an
    independently recomputed ``c @ x``.  It does NOT prove optimality and
    tolerances are never weakened; the caller reports the raw violations.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    resid = (np.asarray(lp.A @ x, dtype=np.float64).ravel()
             - np.asarray(lp.b, dtype=np.float64).ravel())
    row_bad = 0
    worst_row = 0.0
    for i, rt in enumerate(lp.row_types):
        r = float(resid[i])
        if rt == "E":
            v = abs(r)
        elif rt == "L":
            v = max(r, 0.0)
        else:
            v = max(-r, 0.0)
        worst_row = max(worst_row, v)
        if v > tol:
            row_bad += 1
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    bnd_bad = int(np.count_nonzero((x < lb - tol) | (x > ub + tol)))
    mask = (np.asarray(integer_mask, dtype=bool) if integer_mask is not None
            else np.zeros(lp.num_vars, dtype=bool))
    int_bad = int(np.count_nonzero(mask & (np.abs(x - np.round(x)) > tol)))
    return {
        "ok": row_bad == 0 and bnd_bad == 0 and int_bad == 0,
        "rows_violated": row_bad,
        "bounds_violated": bnd_bad,
        "integrality_violated": int_bad,
        "worst_row": worst_row,
        "objective_recomputed": float(np.asarray(lp.c, dtype=np.float64) @ x),
    }


def cmd_solve_milp(args) -> int:
    try:
        import numpy as np

        from opticore.mps_parser import MPSParseError
        from opticore.numerical_model import NumericalModelError, load_numeric_mps
        from opticore.lp.branch_bound import MilpError, solve_milp
    except Exception as exc:
        return _fail(f"cannot initialise the OPTICORE MILP engine ({exc})",
                     verbose=args.verbose, exc=exc)

    path = args.mps_file
    if not os.path.isfile(path):
        return _fail(f"MPS file not found: {path}", verbose=args.verbose)

    time_limit = args.time_limit
    if time_limit is not None:
        try:
            time_limit = float(time_limit)
        except (TypeError, ValueError):
            return _fail(f"invalid --time-limit value: {args.time_limit!r} "
                         f"(expected seconds as a number)", verbose=args.verbose)
        if not (time_limit == time_limit and time_limit > 0.0):
            return _fail(f"invalid --time-limit value: {args.time_limit!r} "
                         f"(expected a positive number of seconds)",
                         verbose=args.verbose)
    node_limit = args.node_limit
    if node_limit is None or node_limit < 1:
        return _fail(f"invalid --node-limit value: {args.node_limit!r} "
                     f"(expected a positive node budget)", verbose=args.verbose)
    gap_tol = args.gap_tol
    if not (gap_tol == gap_tol and gap_tol > 0.0):
        return _fail(f"invalid --gap-tol value: {args.gap_tol!r} "
                     f"(expected a positive relative gap)", verbose=args.verbose)

    try:
        t_load = time.perf_counter()
        lp = load_numeric_mps(path, sparse=True)
        load_sec = time.perf_counter() - t_load
    except (MPSParseError, NumericalModelError) as exc:
        return _fail(f"invalid MPS file {path!r}: {exc}", verbose=args.verbose, exc=exc)
    except Exception as exc:
        return _fail(f"could not load MPS file {path!r}: {exc}",
                     verbose=args.verbose, exc=exc)

    raw_mask = getattr(lp, "is_integer", None)
    integer_mask = (np.asarray(raw_mask, dtype=bool) if raw_mask
                    else np.zeros(lp.num_vars, dtype=bool))
    n_integer = int(np.count_nonzero(integer_mask))
    if n_integer == 0:
        return _fail(f"no integer variables in {path!r}: solve-milp needs MPS integer "
                     f"markers (MARKER/INTORG/INTEND) or BV/LI/UI bounds; use the "
                     f"'solve' subcommand for a continuous LP", verbose=args.verbose)

    try:
        result = solve_milp(
            lp,
            integer_mask=integer_mask,
            maximize=args.maximize,
            int_tol=args.int_tol,
            gap_tol=gap_tol,
            node_limit=node_limit,
            time_limit=time_limit,
            verbose=args.verbose,
            crossover_fallback=not args.no_crossover_fallback,
        )
    except MilpError as exc:
        return _fail(f"MILP solve refused on {path!r}: {exc}",
                     verbose=args.verbose, exc=exc)
    except Exception as exc:
        return _fail(f"MILP solve failed on {path!r}: {exc}",
                     verbose=args.verbose, exc=exc)

    sense = "maximize" if args.maximize else "minimize"
    ok = result.status == "optimal"

    verify = None
    if result.x is not None and not args.no_verify:
        verify = _verify_milp_incumbent(lp, result.x, integer_mask)

    print("OPTICORE - MILP solver (indigenous branch-and-bound)")
    print(f"  input file   : {path}")
    print(f"  problem      : {lp.name} (objective row: {lp.objective_name})")
    print(f"  variables    : {lp.num_vars} ({n_integer} integer, "
          f"{lp.num_vars - n_integer} continuous)")
    print(f"  constraints  : {lp.num_constraints}")
    print(f"  nonzeros     : {lp.nnz}")
    print(f"  sense        : {sense}")
    print("  method       : native branch-and-bound + Mehrotra IPM node relaxations")
    print(f"  status       : {result.status}")
    print(f"  objective    : {_fmt(result.objective)}")
    print(f"  best bound   : {_fmt(result.best_bound)}")
    print(f"  relative gap : {_fmt_sci(result.gap)}")
    print(f"  nodes        : {result.nodes_explored} explored "
          f"(limit {result.node_limit}, {result.nodes_infeasible} pruned infeasible, "
          f"{result.nodes_failed} dropped)")
    print(f"  lp solves    : {result.lp_solves}")
    print(f"  load time    : {load_sec:.4f} s")
    print(f"  runtime      : {result.time_sec:.4f} s")
    if result.heuristic_attempts:
        print(f"  heuristic    : {result.heuristic_successes}/"
              f"{result.heuristic_attempts} attempts found an incumbent "
              f"(best {_fmt(result.heuristic_best_objective)})")
    if result.node_failure_statuses:
        print(f"  failure modes: {', '.join(result.node_failure_statuses[:6])}")
    if verify is not None:
        obj_delta = abs(verify["objective_recomputed"] - (result.objective or 0.0))
        print(f"  verify       : rows_violated={verify['rows_violated']} "
              f"bounds_violated={verify['bounds_violated']} "
              f"integrality_violated={verify['integrality_violated']} "
              f"worst_row={_fmt_sci(verify['worst_row'])}")
        print(f"  verify obj   : recomputed={_fmt(verify['objective_recomputed'])} "
              f"(delta {_fmt_sci(obj_delta)})")
        print(f"  verify pass  : {verify['ok']}")
    if not ok:
        print(f"  detail       : {result.message}")
    if verify is not None and not verify["ok"]:
        return 1
    return 0 if ok else 1


def _run_live_solve(solve_lp, lp, *, crossover: bool, verbose: bool,
                    time_limit=None) -> dict:
    """Run the genuine live solve, optionally bounded by a wall-clock limit.

    The solver core has no time-limit hook (and is NOT modified here), so the
    limit is enforced around it: the solve runs in a daemon worker thread and
    the CLI waits up to ``time_limit`` seconds.  Returns
    ``{"timed_out": bool, "result": ..., "elapsed": float, "error": ...}``;
    a raised solver exception is re-raised in the caller (unless we timed
    out first, in which case the late outcome is discarded and never
    reported).  ``time_limit=None`` waits unconditionally.
    """
    box: dict = {}
    t0 = time.perf_counter()

    def _target():
        try:
            box["result"] = solve_lp(lp, crossover_fallback=crossover,
                                     verbose=verbose, backend="cpu")
        except BaseException as exc:  # noqa: BLE001 - re-raised by caller
            box["error"] = exc
        finally:
            box["done_at"] = time.perf_counter()

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    if time_limit is None:
        worker.join()
        elapsed = time.perf_counter() - t0
        if "error" in box:
            raise box["error"]
        return {"timed_out": False, "result": box.get("result"),
                "elapsed": elapsed}
    worker.join(timeout=time_limit)
    elapsed = time.perf_counter() - t0
    if worker.is_alive():
        # Genuinely still solving: report the timeout, discard the late
        # result, and NEVER consult the certificate implicitly.
        return {"timed_out": True, "result": None, "elapsed": elapsed}
    if "error" in box:
        raise box["error"]
    return {"timed_out": False, "result": box.get("result"), "elapsed": elapsed}


# ---------------------------------------------------------------------------
# pilot87: stored strict-certificate fast path (EXPLICIT OPT-IN ONLY).
#
# Default CLI semantics are a genuine live solve.  This repository's
# benchmark harness folds pilot87 from
# artifacts/pilot87/p87_strict_certificate.txt (see tools/benchmark_netlib.py)
# because a live from-scratch crossover means the full sparse Phase I crash
# (up to 2M iterations) plus Phase II simplex - tens of minutes.  The CLI
# keeps that instant certified answer available for demos under time
# pressure, but ONLY behind --cert-fallback: the normal solve path never
# touches it, and a --time-limit expiry never falls back to it either.
# Nothing here modifies the solver.
# ---------------------------------------------------------------------------
_PILOT87_OBJECTIVE = 301.710347333
_PILOT87_HIGHS_REFERENCE = 301.710347333
_PILOT87_DELTA = 1.034e-10
_PILOT87_CERT_FILES = (
    "artifacts/pilot87/p87_strict_certificate.txt",
    "artifacts/pilot87/p87_certificate.txt",
)


def _is_pilot87(path: str, lp) -> bool:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    name = str(getattr(lp, "name", "") or "").lower()
    return stem == "pilot87" or name == "pilot87"


def _report_pilot87_certificate(path, lp, load_sec, method,
                                crossover_state, *, verbose: bool) -> int:
    root = _project_root()
    cert_rel = next((c for c in _PILOT87_CERT_FILES
                     if os.path.isfile(os.path.join(root, c))), None)
    # Residuals quoted from the stored strict certificate (raw precision):
    # primal ||Ax-b||_inf = 3.638e-11, basis residual = 3.638e-11,
    # E-row residual = 8.251e-12, strong duality gap = -2.331e-12.
    print("OPTICORE - indigenous LP solver")
    print(f"  input file   : {path}")
    print(f"  RESULT SOURCE : STORED STRICT KKT CERTIFICATE")
    print(f"  LIVE SOLVE    : NOT RUN (explicit --cert-fallback fast path)")
    print(f"  problem      : {lp.name} (objective row: {lp.objective_name})")
    print(f"  variables    : {lp.num_vars}")
    print(f"  constraints  : {lp.num_constraints}")
    print(f"  nonzeros     : {lp.nnz}")
    print(f"  method       : {method} + stored strict-certificate result")
    print(f"  crossover    : {crossover_state} (not exercised: this fast path "
          f"reports the stored certificate instead of running the live "
          f"IPM/crossover cycle)")
    print("  status       : optimal (STRICT VERIFIED OPTIMAL per stored certificate)")
    print(f"  objective    : {_fmt(_PILOT87_OBJECTIVE)}")
    print("  iterations   : n/a (certificate result; live Phase I/II not re-run)")
    print(f"  load time    : {load_sec:.4f} s")
    print("  runtime      : n/a (certificate result)")
    print("  primal resid : 3.638e-11 (certificate ||Ax-b||_inf)")
    print("  dual resid   : n/a (certificate: min reduced cost -1.07e-18, "
          "all >= 0 within condition-noise floor)")
    print("  relative gap : 2.331e-12 (certificate |strong duality gap|)")
    print(f"  certificate  : {cert_rel if cert_rel is not None else 'MISSING'}")
    print(f"  HiGHS ref    : {_fmt(_PILOT87_HIGHS_REFERENCE)} "
          f"(|delta| = {_PILOT87_DELTA:.3e})")
    if cert_rel is None:
        print("  detail       : stored pilot87 certificate files not found; "
              "run without --cert-fallback for a genuine live solve")
        return 1
    if verbose:
        print("  note         : live solve intentionally skipped by explicit "
              "--cert-fallback (repository benchmark policy folds pilot87 from "
              "the strict certificate; see tools/benchmark_netlib.py)")
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.command == "solve":
        code = cmd_solve(args)
        if _HARD_EXIT:
            sys.stdout.flush()
            os._exit(code)
        return code
    if args.command == "solve-qp":
        return cmd_solve_qp(args)
    if args.command == "solve-milp":
        return cmd_solve_milp(args)
    return _fail(f"unknown command: {args.command}", verbose=False)


if __name__ == "__main__":
    raise SystemExit(main())
