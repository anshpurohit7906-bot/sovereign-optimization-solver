"""STRESS TEST: Condition-gate replacement validation.

Validates whether replacing dense cond2 with the sparse Hager/Higham
1-norm condition estimator changes the Phase II gate behavior when the
condition estimate actually controls eff_tol.

Parts:
  1. Real-basis gate parameters (with actual eff_tol, not short-circuited)
  2. Threshold sweep: probe min_xB around both thresholds
  3. Adversarial b_norm scaling (1, 1e3, 1e6, 1e9, 1e12)
  4. Estimator error characterization
  5. Acceptance criterion
  6. Small exact matrices (diagonal, triangular, random, near-singular)
"""
from __future__ import annotations
import os, sys, csv
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_HERE, "..", "crossover")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS_DIR = os.path.join(_ROOT, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "stress_condition_gate.csv")
MD_PATH = os.path.join(RESULTS_DIR, "stress_condition_gate.md")
CAPTURE_CACHE_DIR = os.path.join(_ROOT, "scratch", "cond_est_captures")

TOL = 1e-7
SAFETY_FACTOR = 50.0
EPS = np.finfo(np.float64).eps


# ======================================================================
# Core: estimator + gate
# ======================================================================

def hager_higham_inv_norm1(lu, n, itmax=5):
    """Estimate ||B^{-1}||_1 via Hager/Higham power iteration."""
    x = np.ones(n, dtype=np.float64) / n
    n_solves = 0
    for _it in range(itmax):
        try:
            z = lu.solve(x)
        except Exception:
            return np.nan, n_solves
        n_solves += 1
        est = np.sum(np.abs(z))
        if est < 1e-300:
            return 0.0, n_solves
        j = int(np.argmax(np.abs(z)))
        e_j = np.zeros(n, dtype=np.float64)
        e_j[j] = 1.0
        try:
            w = lu.solve(e_j, trans='T')
        except Exception:
            return est, n_solves + 1
        n_solves += 1
        x_new = np.sign(w)
        if not np.any(x_new):
            x_new = np.ones(n, dtype=np.float64)
        if _it > 0 and np.array_equal(x_new, x):
            break
        x = x_new
    try:
        z = lu.solve(x)
        n_solves += 1
        est = np.sum(np.abs(z))
    except Exception:
        pass
    return est, n_solves


def norm1_sparse(B):
    """Compute ||B||_1 from sparse B."""
    return float(sp.linalg.norm(B, ord=1))


def sparse_cond1_estimate(B, lu_override=None):
    """cond1_est = ||B||_1 * est||B^{-1}||_1, fully sparse."""
    n = B.shape[0]
    norm_B1 = norm1_sparse(B)
    lu = lu_override if lu_override is not None else splu(B)
    est_inv, ns = hager_higham_inv_norm1(lu, n)
    c1 = norm_B1 * est_inv if np.isfinite(est_inv) else np.nan
    return c1, norm_B1, est_inv, ns, lu


def dense_cond2_reference(B):
    """Exact cond2 via dense SVD -- REFERENCE ONLY."""
    return float(np.linalg.cond(B.toarray()))


def try_factorize(B):
    """Try sparse LU factorization with fallback."""
    try:
        return splu(B)
    except RuntimeError:
        pass
    try:
        return splu(B, permc_spec='MMD_AT_PLUS_A')
    except RuntimeError:
        pass
    return None


# ======================================================================
# Gate logic -- exact replica of p87_phase2_v2.py
# ======================================================================

def gate_decision(cond_val, b_norm, min_xB, tol=TOL):
    """Phase II condition-gate logic.

    Returns: (decision, eff_tol, raw_tol)
      decision: 'no_action' | 'soft_clamp' | 'repair'
    """
    if min_xB >= -tol:
        return "no_action", tol, tol
    if not np.isfinite(cond_val) or cond_val <= 0:
        raw_tol = tol * 1000.0
    else:
        raw_tol = cond_val * EPS * b_norm * SAFETY_FACTOR
    eff_tol = max(raw_tol, tol)
    if min_xB >= -eff_tol:
        return "soft_clamp", eff_tol, raw_tol
    return "repair", eff_tol, raw_tol


# ======================================================================
# Cache loading
# ======================================================================

def load_captures(model_name):
    """Load cached captured bases; returns list of dicts or None."""
    model_dir = os.path.join(CAPTURE_CACHE_DIR, model_name)
    if not os.path.isdir(model_dir):
        return None
    b_path = os.path.join(model_dir, "_b.npy")
    b_ref = np.load(b_path) if os.path.exists(b_path) else np.zeros(0)
    captures = []
    meta_files = sorted(f for f in os.listdir(model_dir)
                        if f.startswith("cap_") and f.endswith("_meta.npz"))
    for mf in meta_files:
        meta = np.load(os.path.join(model_dir, mf))
        it = int(mf.replace("cap_", "").replace("_meta.npz", ""))
        B = sp.load_npz(os.path.join(model_dir, f"cap_{it}_B.npz"))
        captures.append({
            "iteration": it, "B": B,
            "x_basic": np.asarray(meta["x_basic"], float),
            "fraction": float(meta.get("fraction", 0)),
            "min_xB": float(meta.get("min_xB", meta["x_basic"].min())),
            "_b_ref": b_ref,
        })
    return captures


# ======================================================================
# PART 1: Real-basis gate parameters
# ======================================================================

def part1_real_basis_params(captures, model_name):
    """Compute gate parameters for each captured basis."""
    b = captures[0]["_b_ref"]
    b_norm = float(np.max(np.abs(b)))
    rows = []
    for cap in captures:
        B, it, min_xB = cap["B"], cap["iteration"], cap["min_xB"]
        lu = try_factorize(B)
        if lu is None:
            rows.append({"model": model_name, "iteration": it,
                         "min_xB": min_xB, "b_norm": b_norm,
                         "cond2_exact": np.nan, "cond1_est": np.nan,
                         "raw_tol_exact": np.nan, "raw_tol_est": np.nan,
                         "eff_tol_exact": np.nan, "eff_tol_est": np.nan})
            continue
        cond2 = dense_cond2_reference(B)
        cond1, _, _, _, _ = sparse_cond1_estimate(B, lu_override=lu)
        re = (cond2 * EPS * b_norm * SAFETY_FACTOR
              if np.isfinite(cond2) and cond2 > 0 else TOL * 1000)
        rs = (cond1 * EPS * b_norm * SAFETY_FACTOR
              if np.isfinite(cond1) and cond1 > 0 else TOL * 1000)
        rows.append({"model": model_name, "iteration": it,
                      "min_xB": min_xB, "b_norm": b_norm,
                      "cond2_exact": cond2, "cond1_est": cond1,
                      "raw_tol_exact": re, "raw_tol_est": rs,
                      "eff_tol_exact": max(re, TOL),
                      "eff_tol_est": max(rs, TOL)})
        print(f"  {model_name} it={it:5d}  c2={cond2:.3e}  "
              f"c1={cond1:.3e}  raw_e={re:.3e}  raw_s={rs:.3e}",
              flush=True)
    return rows


# ======================================================================
# PART 2: Gate-threshold sweep
# ======================================================================

def part2_threshold_sweep(real_rows):
    """Sweep synthetic min_xB around both thresholds for every basis."""
    RANK = {"no_action": 0, "soft_clamp": 1, "repair": 2}
    sweep_rows = []
    for r in real_rows:
        if not (np.isfinite(r["cond2_exact"]) and np.isfinite(r["cond1_est"])):
            continue
        model, it = r["model"], r["iteration"]
        bn = r["b_norm"]
        c2, c1 = r["cond2_exact"], r["cond1_est"]
        ee, es = r["eff_tol_exact"], r["eff_tol_est"]

        pts = {"0": 0.0, "-tol": -TOL, "-1.1*tol": -1.1 * TOL,
               "-0.1*ee": -0.1 * ee, "-ee": -ee,
               "-1.1*ee": -1.1 * ee, "-10*ee": -10 * ee,
               "-0.1*es": -0.1 * es, "-es": -es,
               "-1.1*es": -1.1 * es, "-10*es": -10 * es}
        for lbl, xB in pts.items():
            de, _, _ = gate_decision(c2, bn, xB)
            ds, _, _ = gate_decision(c1, bn, xB)
            diff = RANK[ds] - RANK[de]
            cons = ("MORE_conservative" if diff > 0
                    else "LESS_conservative" if diff < 0 else "SAME")
            sweep_rows.append({"model": model, "iteration": it,
                               "test_point": lbl, "min_xB": xB,
                               "decision_exact": de, "decision_est": ds,
                               "match": de == ds,
                               "conservativeness": cons})
    return sweep_rows


# ======================================================================
# PART 3: Adversarial b_norm scaling
# ======================================================================

def part3_adversarial_scaling(real_rows):
    """Evaluate gate under several b_norm scales."""
    SCALES = [1, 1e3, 1e6, 1e9, 1e12]
    RANK = {"no_action": 0, "soft_clamp": 1, "repair": 2}
    rows = []
    for r in real_rows:
        if not (np.isfinite(r["cond2_exact"]) and np.isfinite(r["cond1_est"])):
            continue
        model, it = r["model"], r["iteration"]
        c2, c1 = r["cond2_exact"], r["cond1_est"]
        bbase = r["b_norm"]
        for sc in SCALES:
            bn = bbase * sc
            re_ = c2 * EPS * bn * SAFETY_FACTOR
            rs_ = c1 * EPS * bn * SAFETY_FACTOR
            ee, es = max(re_, TOL), max(rs_, TOL)
            ratio = ee / es if es > 0 else np.nan
            for probe, probe_xB in [("mid", -min(ee, es) * 0.5),
                                     ("boundary", -ee)]:
                de, _, _ = gate_decision(c2, bn, probe_xB)
                ds, _, _ = gate_decision(c1, bn, probe_xB)
                diff = RANK[ds] - RANK[de]
                cons = ("MORE_conservative" if diff > 0
                        else "LESS_conservative" if diff < 0
                        else "SAME")
                rows.append({"model": model, "iteration": it,
                             "scale": sc, "b_norm": bn,
                             "eff_tol_exact": ee, "eff_tol_est": es,
                             "ratio_eff_tol": ratio,
                             "probe": probe, "probe_xB": probe_xB,
                             "decision_exact": de, "decision_est": ds,
                             "match": de == ds,
                             "conservativeness": cons})
    return rows


# ======================================================================
# PART 4: Estimator error characterization
# ======================================================================

def part4_error_characterization(real_rows):
    """Report cond1/cond2 and raw_tol ratios."""
    rows = []
    for r in real_rows:
        c2, c1 = r["cond2_exact"], r["cond1_est"]
        re_, rs_ = r["raw_tol_exact"], r["raw_tol_est"]
        cr = c1/c2 if (np.isfinite(c2) and c2 > 0 and np.isfinite(c1)) else np.nan
        er = rs_/re_ if (np.isfinite(re_) and re_ > 0 and np.isfinite(rs_)) else np.nan
        rows.append({"model": r["model"], "iteration": r["iteration"],
                      "cond2_exact": c2, "cond1_est": c1,
                      "cond_ratio": cr, "raw_tol_exact": re_,
                      "raw_tol_est": rs_, "eff_ratio": er})
    return rows


# ======================================================================
# PART 6: Small exact validation matrices
# ======================================================================

def part6_small_matrices():
    """Test with matrices of known condition numbers."""
    rng = np.random.RandomState(42)
    cases = []

    def _test(Bd, name):
        B = sp.csc_matrix(Bd)
        n = B.shape[0]
        c2 = float(np.linalg.cond(Bd))
        lu = try_factorize(B)
        if lu is None:
            cases.append({"name": name, "n": n, "cond2_exact": c2,
                          "cond1_est": np.nan, "ratio": np.nan,
                          "match_soft": False, "match_repair": False})
            return
        c1, _, _, _, _ = sparse_cond1_estimate(B, lu_override=lu)
        ratio = c1/c2 if c2 > 0 else np.nan
        bn = 1.0
        re_ = c2 * EPS * bn * SAFETY_FACTOR
        rs_ = c1 * EPS * bn * SAFETY_FACTOR
        ee, es = max(re_, TOL), max(rs_, TOL)
        ps, pr = -min(ee, es)*0.5, -max(ee, es)*2.0
        de_s, _, _ = gate_decision(c2, bn, ps)
        ds_s, _, _ = gate_decision(c1, bn, ps)
        de_r, _, _ = gate_decision(c2, bn, pr)
        ds_r, _, _ = gate_decision(c1, bn, pr)
        cases.append({"name": name, "n": n, "cond2_exact": c2,
                       "cond1_est": c1, "ratio": ratio,
                       "dec_exact_soft": de_s, "dec_est_soft": ds_s,
                       "dec_exact_repair": de_r, "dec_est_repair": ds_r,
                       "match_soft": de_s == ds_s,
                       "match_repair": de_r == ds_r})

    n = 50
    # Diagonal
    for sc in [1, 10, 1e3, 1e6, 1e9, 1e12]:
        d = np.ones(n) * sc; d[0] = 1.0
        _test(np.diag(d), f"diag_1e{int(np.log10(sc))}" if sc > 1 else "diag_1")
    # Upper-triangular
    for k in [1, 3, 6, 9, 12]:
        T = np.eye(n)*10.0**k + np.triu(np.abs(rng.randn(n,n))*0.1, 1)
        _test(T, f"tri_1e{k}")
    # Random
    for s in [0.1, 1.0, 10.0, 100.0]:
        _test(rng.randn(n,n)*s + np.eye(n)*10.0, f"rand_s{s}")
    # Hilbert-like near-singular
    for exp in [4, 6, 8, 10, 12]:
        H = np.array([[1.0/(i+j+1.0) for j in range(n)] for i in range(n)])
        H[0,:] *= 10.0**exp
        _test(H, f"hilbert_1e{exp}")
    # Near-identity
    for exp in [4, 8, 12]:
        I = np.eye(n); I[-1,-1] = 10.0**(-exp)
        _test(I, f"nearsing_1e-{exp}")
    return cases


# ======================================================================
# Output: CSV
# ======================================================================

def write_csv(real_rows, sweep_rows, scaling_rows, char_rows, small_cases):
    """Write all results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["=== PART 1: Real-basis gate parameters ==="])
        w.writerow(["model","iteration","min_xB","b_norm","cond2_exact",
                     "cond1_est","raw_tol_exact","raw_tol_est",
                     "eff_tol_exact","eff_tol_est"])
        for r in real_rows:
            w.writerow([r["model"],r["iteration"],
                        f"{r['min_xB']:.6e}",f"{r['b_norm']:.6e}",
                        f"{r['cond2_exact']:.6e}",f"{r['cond1_est']:.6e}",
                        f"{r['raw_tol_exact']:.6e}",f"{r['raw_tol_est']:.6e}",
                        f"{r['eff_tol_exact']:.6e}",f"{r['eff_tol_est']:.6e}"])

        w.writerow([])
        w.writerow(["=== PART 2: Gate-threshold sweep ==="])
        w.writerow(["model","iteration","test_point","min_xB",
                     "decision_exact","decision_est","match","conservativeness"])
        for r in sweep_rows:
            w.writerow([r["model"],r["iteration"],r["test_point"],
                        f"{r['min_xB']:.6e}",r["decision_exact"],
                        r["decision_est"],r["match"],r["conservativeness"]])

        w.writerow([])
        w.writerow(["=== PART 3: Adversarial b_norm scaling ==="])
        w.writerow(["model","iteration","scale","b_norm","eff_tol_exact",
                     "eff_tol_est","ratio_eff_tol","probe","probe_xB",
                     "decision_exact","decision_est","match","conservativeness"])
        for r in scaling_rows:
            w.writerow([r["model"],r["iteration"],r["scale"],
                        f"{r['b_norm']:.6e}",f"{r['eff_tol_exact']:.6e}",
                        f"{r['eff_tol_est']:.6e}",f"{r['ratio_eff_tol']:.6f}",
                        r["probe"],f"{r['probe_xB']:.6e}",
                        r["decision_exact"],r["decision_est"],
                        r["match"],r["conservativeness"]])

        w.writerow([])
        w.writerow(["=== PART 4: Estimator error characterization ==="])
        w.writerow(["model","iteration","cond2_exact","cond1_est","cond_ratio",
                     "raw_tol_exact","raw_tol_est","eff_ratio"])
        for r in char_rows:
            w.writerow([r["model"],r["iteration"],
                        f"{r['cond2_exact']:.6e}",f"{r['cond1_est']:.6e}",
                        f"{r['cond_ratio']:.6f}",
                        f"{r['raw_tol_exact']:.6e}",f"{r['raw_tol_est']:.6e}",
                        f"{r['eff_ratio']:.6f}"])

        w.writerow([])
        w.writerow(["=== PART 6: Small-matrix validation ==="])
        w.writerow(["name","n","cond2_exact","cond1_est","ratio",
                     "dec_exact_soft","dec_est_soft","match_soft",
                     "dec_exact_repair","dec_est_repair","match_repair"])
        for c in small_cases:
            c1 = f"{c['cond1_est']:.6e}" if np.isfinite(c.get("cond1_est",np.nan)) else "nan"
            rt = f"{c['ratio']:.6f}" if np.isfinite(c.get("ratio",np.nan)) else "nan"
            w.writerow([c["name"],c["n"],f"{c['cond2_exact']:.6e}",c1,rt,
                        c.get("dec_exact_soft","nan"),c.get("dec_est_soft","nan"),
                        c.get("match_soft","nan"),
                        c.get("dec_exact_repair","nan"),c.get("dec_est_repair","nan"),
                        c.get("match_repair","nan")])
    print(f"CSV saved: {CSV_PATH}", flush=True)


def write_md(real_rows, sweep_rows, scaling_rows, char_rows, small_cases):
    """Write Markdown report with acceptance assessment."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    L = []  # lines buffer
    ap = L.append

    all_sweep = sweep_rows + scaling_rows
    total = len(all_sweep)
    same = sum(1 for r in all_sweep if r["conservativeness"] == "SAME")
    mc = sum(1 for r in all_sweep if r["conservativeness"] == "MORE_conservative")
    lc = sum(1 for r in all_sweep if r["conservativeness"] == "LESS_conservative")
    s_ms = sum(1 for c in small_cases if c.get("match_soft"))
    s_mr = sum(1 for c in small_cases if c.get("match_repair"))
    s_n = len(small_cases)
    cr = [r["cond_ratio"] for r in char_rows if np.isfinite(r["cond_ratio"])]
    er = [r["eff_ratio"] for r in char_rows if np.isfinite(r["eff_ratio"])]
    less_cc = [r for r in all_sweep if r["conservativeness"] == "LESS_conservative"]
    mism = [r for r in all_sweep if not r["match"]]

    ap("# Stress Test: Condition-Gate Replacement\n")
    ap("**Goal:** Validate sparse cond1 estimator preserves gate safety.\n")
    ap("## Gate Logic\n```")
    ap("if min_xB >= -tol:    -> no_action")
    ap("elif min_xB >= -eff_tol: -> soft_clamp")
    ap("else:                    -> repair")
    ap(f"eff_tol = max(kappa*eps*||b||_inf*{SAFETY_FACTOR}, {TOL})")
    ap("```\n")

    # Part 1
    ap("## Part 1: Real-Basis Gate Parameters\n")
    ap("| model | iter | min_xB | cond2 | cond1 | raw_exact | raw_est | eff_exact | eff_est |")
    ap("|-------|------|--------|-------|-------|-----------|---------|-----------|---------|")
    for r in real_rows:
        ap(f"| {r['model']} | {r['iteration']} | {r['min_xB']:.3e} | "
           f"{r['cond2_exact']:.3e} | {r['cond1_est']:.3e} | "
           f"{r['raw_tol_exact']:.3e} | {r['raw_tol_est']:.3e} | "
           f"{r['eff_tol_exact']:.3e} | {r['eff_tol_est']:.3e} |")
    ap("")

    # Part 2
    ap("## Part 2: Gate-Threshold Sweep\n")
    ap("| model | iter | test_point | min_xB | exact | est | match | conservatism |")
    ap("|-------|------|------------|--------|-------|-----|-------|--------------|")
    for r in sweep_rows:
        ms = "Y" if r["match"] else "N"
        ap(f"| {r['model']} | {r['iteration']} | {r['test_point']} | "
           f"{r['min_xB']:.3e} | {r['decision_exact']} | "
           f"{r['decision_est']} | {ms} | {r['conservativeness']} |")
    ap("")

    # Part 3
    ap("## Part 3: Adversarial b_norm Scaling\n")
    ap("| model | iter | scale | eff_exact | eff_est | ratio | exact | est | match | conservatism |")
    ap("|-------|------|-------|-----------|---------|-------|-------|-----|-------|--------------|")
    for r in scaling_rows:
        ms = "Y" if r["match"] else "N"
        ap(f"| {r['model']} | {r['iteration']} | {r['scale']:.0e} | "
           f"{r['eff_tol_exact']:.3e} | {r['eff_tol_est']:.3e} | "
           f"{r['ratio_eff_tol']:.4f} | {r['decision_exact']} | "
           f"{r['decision_est']} | {ms} | {r['conservativeness']} |")
    ap("")

    # Part 4
    ap("## Part 4: Estimator Error Characterization\n")
    ap("| model | iter | cond2 | cond1 | ratio | raw_exact | raw_est | eff_ratio |")
    ap("|-------|------|-------|-------|-------|-----------|---------|----------|")
    for r in char_rows:
        ap(f"| {r['model']} | {r['iteration']} | {r['cond2_exact']:.3e} | "
           f"{r['cond1_est']:.3e} | {r['cond_ratio']:.3f} | "
           f"{r['raw_tol_exact']:.3e} | {r['raw_tol_est']:.3e} | "
           f"{r['eff_ratio']:.3f} |")
    ap("")
    if cr:
        ap(f"- cond1/cond2: [{min(cr):.1f}, {max(cr):.1f}] "
           f"(mean {np.mean(cr):.1f})")
    if er:
        ap(f"- eff_ratio: [{min(er):.4f}, {max(er):.4f}] "
           f"(mean {np.mean(er):.4f})")
    ap("")

    # Part 5: Acceptance
    ap("## Part 5: Acceptance Criterion\n")
    ap(f"- Total points: {total}")
    ap(f"- SAME: {same} ({100*same/total:.1f}%)")
    ap(f"- MORE conservative: {mc} ({100*mc/total:.1f}%)")
    ap(f"- LESS conservative: {lc} ({100*lc/total:.1f}%)\n")
    if less_cc:
        ap("### Less-conservative decisions\n")
        ap("| model | iter | probe | min_xB | exact | est |")
        ap("|-------|------|-------|--------|-------|-----|")
        for r in less_cc:
            tp = r.get("test_point", f"s={r.get('scale','?')}")
            ap(f"| {r['model']} | {r['iteration']} | {tp} | "
               f"{r['min_xB']:.3e} | {r['decision_exact']} | "
               f"{r['decision_est']} |")
        ap("")
    else:
        ap("**No less-conservative decisions observed.**\n")
    if mism:
        ap(f"**{len(mism)} mismatches** out of {total}.\n")
    else:
        ap(f"**Zero mismatches** out of {total}.\n")

    # Part 6
    ap("## Part 6: Small-Matrix Validation\n")
    ap("| name | cond2 | cond1 | ratio | soft | repair |")
    ap("|------|-------|-------|-------|------|--------|")
    for c in small_cases:
        cv = c.get("cond1_est", np.nan)
        cs_ = f"{cv:.3e}" if np.isfinite(cv) else "nan"
        rv = c.get("ratio", np.nan)
        rs_ = f"{rv:.3f}" if np.isfinite(rv) else "nan"
        ms = "Y" if c.get("match_soft") else "N"
        mr = "Y" if c.get("match_repair") else "N"
        ap(f"| {c['name']} | {c['cond2_exact']:.3e} | {cs_} | "
           f"{rs_} | {ms} | {mr} |")
    ap(f"\n- Soft matches: {s_ms}/{s_n}")
    ap(f"- Repair matches: {s_mr}/{s_n}\n")

    # Summary
    ap("## Summary\n")
    if er:
        ap(f"- eff_tol ratio: [{min(er):.4f}, {max(er):.4f}]")
    ap("\n### Final Safety Assessment\n")
    if less_cc:
        ap("**NOT SAFE YET** -- less-conservative decisions observed.")
    elif mc > 0:
        ap("**A. SAFE TO INTEGRATE**\n")
        ap(f"{mc} MORE conservative, {same} SAME.")
    else:
        ap("**A. SAFE TO INTEGRATE** -- all matched.")
    ap("\n---\n*Generated by `experiment/sparse/stress_condition_gate.py`*")

    with open(MD_PATH, "w") as fobj:
        fobj.write("\n".join(L) + "\n")
    print(f"Markdown saved: {MD_PATH}", flush=True)


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70, flush=True)
    print("  STRESS TEST: CONDITION-GATE REPLACEMENT", flush=True)
    print("=" * 70, flush=True)

    all_real_rows = []
    models = ["pilot4_plain", "pilot87"]
    for model in models:
        captures = load_captures(model)
        if captures is None:
            print(f"  [{model}] No cached captures — skipping", flush=True)
            continue
        print(f"\n--- {model}: {len(captures)} captures ---", flush=True)
        print("  PART 1: Real-basis gate parameters", flush=True)
        rows = part1_real_basis_params(captures, model)
        all_real_rows.extend(rows)

    if not all_real_rows:
        print("\nERROR: No captures. Run compare_condition_estimators.py first.",
              flush=True)
        return

    print(f"\n--- PART 2: Threshold sweep ({len(all_real_rows)} bases) ---",
          flush=True)
    sweep = part2_threshold_sweep(all_real_rows)
    nm = sum(1 for r in sweep if not r["match"])
    print(f"  Points: {len(sweep)}, mismatches: {nm}", flush=True)

    print(f"\n--- PART 3: Adversarial scaling ---", flush=True)
    scaling = part3_adversarial_scaling(all_real_rows)
    nsm = sum(1 for r in scaling if not r["match"])
    print(f"  Points: {len(scaling)}, mismatches: {nsm}", flush=True)

    print(f"\n--- PART 4: Error characterization ---", flush=True)
    char = part4_error_characterization(all_real_rows)
    for r in char:
        print(f"  {r['model']} it={r['iteration']:5d}  "
              f"cr={r['cond_ratio']:.3f}  er={r['eff_ratio']:.3f}",
              flush=True)

    print(f"\n--- PART 6: Small matrices ---", flush=True)
    small = part6_small_matrices()
    nm6 = sum(1 for c in small
              if not (c.get("match_soft") and c.get("match_repair")))
    print(f"  Cases: {len(small)}, mismatches: {nm6}", flush=True)
    for c in small:
        if not (c.get("match_soft") and c.get("match_repair")):
            print(f"  MISMATCH: {c['name']}  soft={c.get('dec_exact_soft')}"
                  f" vs {c.get('dec_est_soft')}  "
                  f"repair={c.get('dec_exact_repair')}"
                  f" vs {c.get('dec_est_repair')}", flush=True)

    print(f"\n--- Writing outputs ---", flush=True)
    write_csv(all_real_rows, sweep, scaling, char, small)
    write_md(all_real_rows, sweep, scaling, char, small)

    print(f"\n{'='*70}", flush=True)
    print("  DONE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
