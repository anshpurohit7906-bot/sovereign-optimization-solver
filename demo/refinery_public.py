"""Public refinery-planning LP benchmark (SAS/OR ``mpex06``) for OPTICORE.

Public SAS/OR benchmark only — NOT MRPL operational data.  No plant data,
no proprietary yields, no operational records are used anywhere here.

Source (transcribed verbatim, no invented coefficients)::

    https://support.sas.com/documentation/onlinedoc/or/ex_code/143/mpex06.html

Worked mirror with the same data and published solution::

    https://sassoftware.github.io/sasoptpy/examples/refinery_optimization.html

The SAS benchmark stays a CONTINUOUS LP (never an artificial MILP).  It is
built directly on the existing ``NumericalLP`` + ``Mehrotra`` API
(``solve_lp(..., maximize=True)``) and checked with OPTICORE's own
independent feasibility/objective verification (row-type-aware residual
checks — primal feasibility plus an independently recomputed ``c @ x``;
it does NOT prove optimality/KKT — same philosophy as ``verify_qp_kkt`` /
``tools/validate_milp_incumbent.py``); tolerances are NOT weakened.

Published reference: 51 variables, 46 constraints, 158 nonzeros, optimal
objective approximately 211365.13477 with crude1 = 15000, crude2 = 30000.

Public adapter surface:

* ``load_public_refinery_data()``
* ``build_public_refinery_lp(...)``
* ``solve_public_refinery(...)``
* ``verify_public_refinery(...)``
"""

from __future__ import annotations

import os
import re
import time

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from opticore.numerical_model import NumericalLP, validate_numeric_lp
from opticore.lp.mehrotra import solve_lp

# ---------------------------------------------------------------------------
# Published reference values (SAS mpex06 / sasoptpy worked solution)
# ---------------------------------------------------------------------------
SOURCE_NAME = "SAS/OR mpex06 — Refinery Optimization"
SOURCE_URL = "https://support.sas.com/documentation/onlinedoc/or/ex_code/143/mpex06.html"
PUBLISHED_OBJECTIVE = 211365.13477
PUBLISHED_CRUDE1 = 15000.0
PUBLISHED_CRUDE2 = 30000.0
# SAS-side reference (as published): 51 variables, 46 constraints, 158 nonzeros.
PUBLISHED_DIMS = {"num_vars": 51, "num_constraints": 46, "nnz": 158}
# OPTICORE solve model: identical except ONE redundant fuel-oil equality is
# omitted (the four SAS-written fuel-oil rows sum to 0 identically, so only
# three are independent; keeping the fourth trips the solver's exact-rank
# guard in to_standard_form).  All SAS coefficients are kept verbatim.
EXPECTED_DIMS = {"num_vars": 51, "num_constraints": 45, "nnz": 154,
                 "row_types": {"E": 37, "L": 4, "G": 4}}

FINAL_PRODUCTS = ["premium_petrol", "regular_petrol", "jet_fuel", "fuel_oil", "lube_oil"]

# Same deterministic detector used by tools/benchmark_netlib.py and the
# OPTICORE CLI: the solver reports an accepted crossover fallback with this
# message prefix (stall detection lives in the solver, not here).
_CROSSOVER_PREFIX = re.compile(r"^sparse crossover from (?:stalled|numerical_tail): ")


def load_public_refinery_data() -> dict:
    """Return the SAS mpex06 data transcribed verbatim from the source page.

    Missing multipliers (``.`` in the SAS datalines) default to 1.0, exactly
    as the OPTMODEL ``arc_mult`` array does (``init 1`` + ``nomiss`` read).
    Profit is stored per-unit (SAS divides the table values by 100).
    """
    arc_data = [
        ("source", "crude1", 6.0), ("source", "crude2", 6.0),
        ("crude1", "light_naphtha", 0.1), ("crude1", "medium_naphtha", 0.2),
        ("crude1", "heavy_naphtha", 0.2), ("crude1", "light_oil", 0.12),
        ("crude1", "heavy_oil", 0.2), ("crude1", "residuum", 0.13),
        ("crude2", "light_naphtha", 0.15), ("crude2", "medium_naphtha", 0.25),
        ("crude2", "heavy_naphtha", 0.18), ("crude2", "light_oil", 0.08),
        ("crude2", "heavy_oil", 0.19), ("crude2", "residuum", 0.12),
        ("light_naphtha", "regular_petrol", 1.0),
        ("light_naphtha", "premium_petrol", 1.0),
        ("medium_naphtha", "regular_petrol", 1.0),
        ("medium_naphtha", "premium_petrol", 1.0),
        ("heavy_naphtha", "regular_petrol", 1.0),
        ("heavy_naphtha", "premium_petrol", 1.0),
        ("light_naphtha", "reformed_gasoline", 0.6),
        ("medium_naphtha", "reformed_gasoline", 0.52),
        ("heavy_naphtha", "reformed_gasoline", 0.45),
        ("light_oil", "jet_fuel", 1.0), ("light_oil", "fuel_oil", 1.0),
        ("heavy_oil", "jet_fuel", 1.0), ("heavy_oil", "fuel_oil", 1.0),
        ("light_oil", "light_oil_cracked", 2.0),
        ("light_oil_cracked", "cracked_oil", 0.68),
        ("light_oil_cracked", "cracked_gasoline", 0.28),
        ("heavy_oil", "heavy_oil_cracked", 2.0),
        ("heavy_oil_cracked", "cracked_oil", 0.75),
        ("heavy_oil_cracked", "cracked_gasoline", 0.2),
        ("cracked_oil", "jet_fuel", 1.0), ("cracked_oil", "fuel_oil", 1.0),
        ("reformed_gasoline", "regular_petrol", 1.0),
        ("reformed_gasoline", "premium_petrol", 1.0),
        ("cracked_gasoline", "regular_petrol", 1.0),
        ("cracked_gasoline", "premium_petrol", 1.0),
        ("residuum", "lube_oil", 0.5),
        ("residuum", "jet_fuel", 1.0), ("residuum", "fuel_oil", 1.0),
    ]
    return {
        "arcs": arc_data,
        "crude_ub": {"crude1": 20000.0, "crude2": 30000.0},
        "octane": {"light_naphtha": 90.0, "medium_naphtha": 80.0,
                   "heavy_naphtha": 70.0, "reformed_gasoline": 115.0,
                   "cracked_gasoline": 105.0},
        "petrol_lb": {"regular_petrol": 84.0, "premium_petrol": 94.0},
        "vapour_pressure": {"light_oil": 1.0, "heavy_oil": 0.6,
                            "cracked_oil": 1.5, "residuum": 0.05},
        "vapour_pressure_ub": 1.0,
        "fuel_oil_coef": {"light_oil": 10.0, "cracked_oil": 4.0,
                          "heavy_oil": 3.0, "residuum": 1.0},
        "profit": {"premium_petrol": 7.0, "regular_petrol": 6.0,
                   "jet_fuel": 4.0, "fuel_oil": 3.5, "lube_oil": 1.5},
        "crude_total_ub": 45000.0,
        "naphtha_ub": 10000.0,
        "cracked_oil_ub": 8000.0,
        "lube_oil_lb": 500.0,
        "lube_oil_ub": 1000.0,
        "premium_ratio": 0.40,
    }

_BALANCE_NODES = [
    "crude1", "crude2", "light_naphtha", "medium_naphtha",
    "heavy_naphtha", "light_oil", "heavy_oil", "residuum",
    "light_oil_cracked", "heavy_oil_cracked", "cracked_oil",
    "cracked_gasoline", "reformed_gasoline", "regular_petrol",
    "premium_petrol", "jet_fuel", "fuel_oil", "lube_oil",
]


def _all_arcs(data):
    sink = [(p, "sink", 1.0) for p in FINAL_PRODUCTS]
    return list(data["arcs"]) + sink


def build_public_refinery_lp(data=None, crude2_ub=None):
    """Build the SAS mpex06 LP as a ``NumericalLP`` (continuous, MAX sense).

    Parameters
    ----------
    data : dict, optional
        Output of :func:`load_public_refinery_data` (loaded when omitted).
    crude2_ub : float, optional
        Override of the crude2 distillation upper bound (what-if channel).
        When None the published 30000.0 is used.
    """
    if data is None:
        data = load_public_refinery_data()
    arcs = _all_arcs(data)
    n_flow = len(arcs)
    IDX_CRUDE1 = n_flow
    IDX_CRUDE2 = n_flow + 1
    IDX_CRACK_LO = n_flow + 2
    IDX_CRACK_HO = n_flow + 3
    n = n_flow + 4
    flow_of = {(i, j): k for k, (i, j, _) in enumerate(arcs)}
    out_of: dict = {}
    in_to: dict = {}
    for k, (i, j, mm) in enumerate(arcs):
        out_of.setdefault(i, []).append(k)
        in_to.setdefault(j, []).append(k)

    row_names: list = []
    row_types: list = []
    rhs: list = []
    acc: dict = {}

    def add_row(name, rt, bval):
        row_names.append(name)
        row_types.append(rt)
        rhs.append(float(bval))
        return len(row_names) - 1

    def put(vi, r, v):
        if v != 0.0:
            acc[(int(vi), int(r))] = acc.get((int(vi), int(r)), 0.0) + float(v)

    # --- Flow balance: out - mult*in = 0 (E) for every node but source/sink.
    for node in _BALANCE_NODES:
        r = add_row(f"Flow_balance[{node}]", "E", 0.0)
        for k in out_of.get(node, []):
            put(k, r, 1.0)
        for k in in_to.get(node, []):
            put(k, r, -arcs[k][2])

    # --- Distillation coupling: Flow[crude,*] = CrudeDistilled[crude] (E).
    # NOTE: source->crude arcs are NOT product flows (multiplier 6 there is a
    # SAS data placeholder, never a constraint coefficient), so only the six
    # crude->intermediate arcs per crude are coupled.
    for crude, vidx in (("crude1", IDX_CRUDE1), ("crude2", IDX_CRUDE2)):
        for k in out_of.get(crude, []):
            i, j, _mm = arcs[k]
            if j == "sink":
                continue
            r = add_row(f"Distillation[{i},{j}]", "E", 0.0)
            put(k, r, 1.0)
            put(vidx, r, -1.0)

    # --- Cracking coupling: Flow[cracked,*] = OilCracked[cracked] (E).
    for cracked, vidx in (("light_oil_cracked", IDX_CRACK_LO),
                          ("heavy_oil_cracked", IDX_CRACK_HO)):
        for k in out_of.get(cracked, []):
            i, j, _mm = arcs[k]
            r = add_row(f"Cracking[{i},{j}]", "E", 0.0)
            put(k, r, 1.0)
            put(vidx, r, -1.0)

    # --- Petrol octane blending (G): sum (oct-mult-lb*mult) F >= 0.
    for petrol in ("regular_petrol", "premium_petrol"):
        r = add_row(f"Blending_petrol[{petrol}]", "G", 0.0)
        for k in in_to.get(petrol, []):
            i, _j, mm = arcs[k]
            put(k, r, (data["octane"][i] - data["petrol_lb"][petrol]) * mm)

    # --- Jet-fuel vapour-pressure blending (L): sum (vp-ub)*mult F <= 0.
    r = add_row("Blending_jet_fuel", "L", 0.0)
    for k in in_to.get("jet_fuel", []):
        i, _j, mm = arcs[k]
        put(k, r, (data["vapour_pressure"][i] - data["vapour_pressure_ub"]) * mm)

    # --- Fuel-oil ratio: S*F[i] - coef[i]*sum(F) = 0 (E), one row per inlet.
    # NOTE: the four SAS-written equalities sum to 0 identically (each column
    # appears with S - sum(coef) = 0), so only three are independent; the
    # redundant fourth would make the standard-form constraint matrix rank
    # deficient and trip the solver's exact-rank guard.  Keep the first three
    # (light/heavy/cracked oil); residuum's share is then implied.
    fuel_in = list(in_to.get("fuel_oil", []))
    fsum = sum(data["fuel_oil_coef"][arcs[k][0]] for k in fuel_in)
    for k_fix in fuel_in[:3]:
        i_fix = arcs[k_fix][0]
        r = add_row(f"Blending_fuel_oil[{i_fix}]", "E", 0.0)
        for k in fuel_in:
            coef = (fsum if k == k_fix else 0.0) - data["fuel_oil_coef"][i_fix]
            put(k, r, coef)

    # --- Capacity / policy rows.
    r = add_row("Crude_total_ub", "L", data["crude_total_ub"])
    put(IDX_CRUDE1, r, 1.0)
    put(IDX_CRUDE2, r, 1.0)

    r = add_row("Naphtha_ub", "L", data["naphtha_ub"])
    for k, (i, j, _mm) in enumerate(arcs):
        if j == "reformed_gasoline" and "naphtha" in i:
            put(k, r, 1.0)

    r = add_row("Cracked_oil_ub", "L", data["cracked_oil_ub"])
    for k, (i, j, _mm) in enumerate(arcs):
        if j == "cracked_oil":
            put(k, r, 1.0)

    lube_vi = flow_of[("lube_oil", "sink")]
    r = add_row("Lube_oil_lb", "G", data["lube_oil_lb"])
    put(lube_vi, r, 1.0)

    r = add_row("Premium_ratio", "G", 0.0)
    put(flow_of[("premium_petrol", "sink")], r, 1.0)
    put(flow_of[("regular_petrol", "sink")], r, -data["premium_ratio"])

    m = len(row_names)
    A = np.zeros((m, n), dtype=np.float64)
    for (vi, rr), v in acc.items():
        A[rr, vi] = v
    b = np.asarray(rhs, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    for p in FINAL_PRODUCTS:
        c[flow_of[(p, "sink")]] = data["profit"][p]
    lb = np.zeros(n, dtype=np.float64)
    ub = np.full(n, np.inf, dtype=np.float64)
    ub[IDX_CRUDE1] = data["crude_ub"]["crude1"]
    ub[IDX_CRUDE2] = data["crude_ub"]["crude2"] if crude2_ub is None else float(crude2_ub)
    # Lube-oil range: lower bound is the Lube_oil_lb row; upper bound is a box.
    ub[lube_vi] = data["lube_oil_ub"]

    var_names = [f"Flow[{i},{j}]" for (i, j, _mm) in arcs]
    var_names += ["CrudeDistilled[crude1]", "CrudeDistilled[crude2]",
                  "OilCracked[light_oil_cracked]", "OilCracked[heavy_oil_cracked]"]
    lp = NumericalLP(
        name="REFINERY_PUBLIC_SAS_MPEX06",
        objective_name="TOTAL_PROFIT",
        A=A, b=b, c=c,
        lower_bounds=lb, upper_bounds=ub,
        row_types=tuple(row_types),
        var_names=tuple(var_names),
        row_names=tuple(row_names),
    )
    meta = {"flow_of": flow_of, "IDX_CRUDE1": IDX_CRUDE1, "IDX_CRUDE2": IDX_CRUDE2,
            "IDX_CRACK_LO": IDX_CRACK_LO, "IDX_CRACK_HO": IDX_CRACK_HO,
            "crude2_ub": float(ub[IDX_CRUDE2]), "data": data}
    return lp, meta


def verify_public_refinery(lp, x, tol=1e-6) -> dict:
    """Independently verify primal feasibility + objective of a candidate.

    Row-type-aware primal check (E/L/G + box bounds) plus an independently
    recomputed objective ``c @ x``.  This is independent
    feasibility/objective verification — it does NOT prove optimality or
    KKT conditions.  PASS only when all checks <= ``tol``.
    """
    A = np.asarray(lp.A, dtype=np.float64)
    b = np.asarray(lp.b, dtype=np.float64)
    c = np.asarray(lp.c, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    resid = A @ x - b
    worst = 0.0
    bad_rows: list = []
    for i, rt in enumerate(lp.row_types):
        r = float(resid[i])
        if rt == "E":
            viol = abs(r)
        elif rt == "L":
            viol = max(r, 0.0)
        else:
            viol = max(-r, 0.0)
        worst = max(worst, viol)
        if viol > tol:
            bad_rows.append((lp.row_names[i], viol))
    lb_v = float(np.max(np.maximum(0.0, lb - x))) if x.size else 0.0
    ub_v = float(np.max(np.maximum(0.0, x - ub))) if x.size else 0.0
    ok = (not bad_rows) and lb_v <= tol and ub_v <= tol
    return {"max_row_violation": worst, "bad_rows": bad_rows,
            "lb_violation": lb_v, "ub_violation": ub_v,
            "objective": float(c @ x), "PASS": bool(ok)}


def solve_public_refinery(lp, meta, **kwargs) -> dict:
    """Solve with the existing Mehrotra API (MAX sense) + verify."""
    t0 = time.perf_counter()
    res = solve_lp(lp, maximize=True, **kwargs)
    dt = time.perf_counter() - t0
    verification = verify_public_refinery(lp, res.x)
    return {"result": res, "time_sec": dt, "verify": verification,
            "objective": float(np.asarray(lp.c) @ np.asarray(res.x)),
            "meta": meta, "lp": lp}


def _key_flows():
    return [("light_oil", "light_oil_cracked"),
            ("heavy_oil", "heavy_oil_cracked"),
            ("heavy_naphtha", "reformed_gasoline"),
            ("reformed_gasoline", "premium_petrol"),
            ("cracked_gasoline", "regular_petrol"),
            ("residuum", "lube_oil"), ("residuum", "jet_fuel"),
            ("cracked_oil", "jet_fuel")]


def _print_summary(lp, meta, solved, title, published=True):
    res = solved["result"]
    x = np.asarray(res.x, dtype=np.float64)
    fo = meta["flow_of"]
    print(f"  --- {title} ---")
    print(f"  solver status : {res.status}")
    print(f"  objective     : {solved['objective']:.5f}")
    print(f"  solve time    : {solved['time_sec']:.3f}s (iters={res.iterations})")
    print(f"  crude1        : {x[meta['IDX_CRUDE1']]:.2f}")
    print(f"  crude2        : {x[meta['IDX_CRUDE2']]:.2f} (ub={meta['crude2_ub']:.0f})")
    print(f"  cracked lo/ho : {x[meta['IDX_CRACK_LO']]:.2f} / "
          f"{x[meta['IDX_CRACK_HO']]:.2f}")
    print("  products to sink:")
    for p in FINAL_PRODUCTS:
        print(f"    {p:15s}: {x[fo[(p, 'sink')]]:.2f}")
    print("  selected process flows:")
    for a, bb in _key_flows():
        print(f"    {a:22s} -> {bb:18s}: {x[fo[(a, bb)]]:.2f}")
    v = solved["verify"]
    print(f"  feasibility/objective verification: "
          f"{'PASS' if v['PASS'] else 'FAIL'} "
          f"(rows={v['max_row_violation']:.2e}, lb={v['lb_violation']:.2e}, "
          f"ub={v['ub_violation']:.2e})")
    for name, viol in v["bad_rows"][:5]:
        print(f"    violated: {name} ({viol:.2e})")
    if published:
        diff = abs(solved["objective"] - PUBLISHED_OBJECTIVE)
        # Gate is on the RELATIVE difference (absolute scale ~2e5, so the
        # ~6e-05 absolute gap is reported for information only).
        rel = diff / (1.0 + abs(PUBLISHED_OBJECTIVE))
        ok = res.status in ("optimal", "numerical_tail") and v["PASS"] and rel < 1e-6
        print(f"  published opt : {PUBLISHED_OBJECTIVE:.5f}")
        print(f"  abs diff      : {diff:.4e} (informational only)")
        print(f"  rel diff      : {rel:.2e} (gate: rel < 1e-6)")
        print(f"  benchmark     : {'PASS' if ok else 'FAIL'}")
        return ok
    return bool(res.status in ("optimal", "numerical_tail") and v["PASS"])


def _banner():
    print("=" * 70)
    print("OPTICORE - PUBLIC REFINERY PLANNING")
    print("SAS/OR mpex06 benchmark")
    print("Public benchmark data - NOT MRPL operational data")
    print("=" * 70)
    print("End-to-end workflow: public data -> model construction -> live")
    print("Mehrotra IPM (real iteration log) -> verification -> what-if re-solve.")
    print("Nothing below is simulated: the solver output is printed by")
    print("solve_lp(verbose=True) from the unchanged solver core.")


def _stage(number, title):
    print()
    print("=" * 70)
    print(f"STAGE {number}: {title}")
    print("=" * 70)


def _ipm_outcome(solved):
    """Report the genuine IPM outcome fields (no re-derived solver logic)."""
    res = solved["result"]
    print(f"  status            : {res.status}")
    print(f"  objective         : {solved['objective']:.5f}")
    print(f"  iterations        : {res.iterations}")
    print(f"  solve time        : {solved['time_sec']:.4f}s")
    print(f"  primal residual   : {res.primal_residual:.3e} "
          f"(rel {res.rel_primal:.3e})")
    print(f"  dual residual     : {res.dual_residual:.3e} "
          f"(rel {res.rel_dual:.3e})")
    print(f"  complementarity   : {res.complementarity:.3e}")
    print(f"  relative gap      : {res.rel_gap:.3e}")


def run_public_refinery_demo(verbose_solver=True) -> bool:
    """Entry point used by ``demo/demo_sih.py --refinery-public``.

    Presentation only: the staged banners wrap the existing real operations
    (public data load, ``build_public_refinery_lp`` validation, the genuine
    ``solve_lp(..., verbose=True)`` IPM runs, ``verify_public_refinery`` and
    the what-if re-solve).  No solver logic is reimplemented here and no
    progress is simulated.
    """
    _banner()

    # ---------------------------------------------------------------- data
    _stage(1, "PUBLIC REFINERY DATA (SAS/OR mpex06)")
    print(f"  benchmark      : {SOURCE_NAME}")
    print(f"  source         : {SOURCE_URL}")
    print("  provenance     : data/refinery_sas/source.txt + README.md")
    print("  data policy    : public SAS/OR sample-library data transcribed")
    print("                   verbatim - NOT MRPL operational data")
    data = load_public_refinery_data()
    print(f"  process arcs   : {len(data['arcs'])} (+5 product->sink arcs)")
    print(f"  crude limits   : crude1 <= {data['crude_ub']['crude1']:.0f}, "
          f"crude2 <= {data['crude_ub']['crude2']:.0f}")
    print(f"  published ref  : objective {PUBLISHED_OBJECTIVE:.5f}, "
          f"crude1={PUBLISHED_CRUDE1:.0f}, crude2={PUBLISHED_CRUDE2:.0f}")

    # ------------------------------------------------------------- model
    _stage(2, "MODEL CONSTRUCTION (NumericalLP, continuous LP)")
    t_build = time.perf_counter()
    lp, meta = build_public_refinery_lp(data)
    build_sec = time.perf_counter() - t_build
    validate_numeric_lp(lp, expected={"num_vars": 51, "num_constraints": 45,
                                      "nnz": 154,
                                      "row_types": {"E": 37, "L": 4, "G": 4}})
    print(f"  variables      : {lp.num_vars}")
    print(f"  constraints : {lp.num_constraints} "
          f"(E={lp.row_types.count('E')}, L={lp.row_types.count('L')}, "
          f"G={lp.row_types.count('G')})")
    print(f"  nonzeros       : {lp.nnz}")
    print(f"  objective      : maximize TotalProfit "
          f"(row '{lp.objective_name}')")
    print(f"  build time     : {build_sec:.4f}s")
    print("  structural gate: PASS (dimensions / row types / nnz validated)")

    # -------------------------------------------------------- live IPM run
    _stage(3, "OPTICORE MEHROTRA IPM - LIVE SOLVE (CPU backend)")
    print(f"  problem        : {lp.name}")
    print("  iteration log  : printed live by the solver core "
          "(solve_lp(verbose=True))")
    print("-" * 70)
    base = solve_public_refinery(lp, meta, verbose=verbose_solver)
    print("-" * 70)

    # -------------------------------------------- IPM solution + crossover
    _stage(4, "OPTIMAL INTERIOR-POINT SOLUTION")
    _ipm_outcome(base)
    _stage(5, "POST-SOLVE / CROSSOVER STATUS")
    res = base["result"]
    crossover_used = bool(_CROSSOVER_PREFIX.match(res.message or ""))
    if crossover_used:
        print("Crossover requested : YES")
        print("IPM solution used   : NO")
        print("Reason              : IPM stalled; the sparse crossover fallback "
              "replaced the iterate")
    else:
        print("Crossover requested : NO")
        print("IPM solution used   : YES")
        print("Reason              : IPM converged directly; no crossover required")
    print(f"Solver message      : {res.message}")

    # -------------------------------------------------------- verification
    _stage(6, "FEASIBILITY / OBJECTIVE VERIFICATION (independent)")
    print("  method         : row-type-aware primal residuals (E/L/G + box "
          "bounds) plus")
    print("                   an independently recomputed c@x; NOT a KKT proof")
    ok_base = _print_summary(lp, meta, base, "BASE CASE (published data)")

    # ------------------------------------------------------------ what-if
    _stage(7, "WHAT-IF SCENARIO")
    print("  change         : crude2 availability 30000 -> 24000 "
          "(a new scenario)")
    print("  baseline data  : unchanged (a separate LP object is built)")
    lp2, meta2 = build_public_refinery_lp(data, crude2_ub=24000.0)
    print(f"  what-if model  : {lp2.num_vars} variables, "
          f"{lp2.num_constraints} constraints, {lp2.nnz} nonzeros "
          f"(crude2 <= {meta2['crude2_ub']:.0f})")
    print("  note           : NOT compared against the published SAS optimum")

    # ---------------------------------------------------------- re-solve
    _stage(8, "RE-SOLVE (OPTICORE MEHROTRA IPM - LIVE)")
    print("  iteration log  : printed live by the solver core "
          "(solve_lp(verbose=True))")
    print("-" * 70)
    alt = solve_public_refinery(lp2, meta2, verbose=verbose_solver)
    print("-" * 70)
    _ipm_outcome(alt)

    # ----------------------------------------- verification + comparison
    _stage(9, "WHAT-IF VERIFICATION + ALLOCATION COMPARISON")
    ok_alt = _print_summary(lp2, meta2, alt, "WHAT-IF (crude2_ub=24000)",
                            published=False)
    xb = np.asarray(base["result"].x)
    xa = np.asarray(alt["result"].x)
    print("  change vs base case:")
    print(f"    delta objective : {alt['objective'] - base['objective']:+.2f}")
    print(f"    delta crude1    : "
          f"{xa[meta2['IDX_CRUDE1']] - xb[meta['IDX_CRUDE1']]:+.2f}")
    print(f"    delta crude2    : "
          f"{xa[meta2['IDX_CRUDE2']] - xb[meta['IDX_CRUDE2']]:+.2f}")
    print()
    ok = bool(ok_base and ok_alt)
    print(f"Public-refinery segment: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_public_refinery_demo() else 1)
