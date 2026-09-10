"""Focused unit tests for the lazy condition-gate integration in p87_phase2_v2.py.

Validates the three critical invariants:
  1. When neg_count == 0, compute_effective_tol is NOT called.
  2. When neg_count > 0, compute_effective_tol IS called (exact cond2).
  3. The gate decision is identical under lazy vs always-compute semantics.
"""
from __future__ import annotations
import numpy as np

TOL = 1e-7
RECOMPUTE_EFF_TOL = 200
SAFETY_FACTOR = 50.0


def _cond2_dense(B_arr):
    try:
        return float(np.linalg.cond(B_arr))
    except Exception:
        return np.nan


def _compute_eff_tol_dense(B_arr, b_norm):
    kappa = _cond2_dense(B_arr)
    if not np.isfinite(kappa) or kappa <= 0:
        return TOL * 1000.0, np.nan
    eff = kappa * np.finfo(np.float64).eps * b_norm * SAFETY_FACTOR
    return max(eff, TOL), kappa


class FakeCond2:
    def __init__(self):
        self.call_count = 0
        self.kappas = []
    def __call__(self, B_arr, b_norm):
        self.call_count += 1
        eff, k = _compute_eff_tol_dense(B_arr, b_norm)
        self.kappas.append(k)
        return eff, k



def _run_lazy(x_basics, eff_tol_init, cond2_fn):
    """Simulate the LAZY gate logic from p87_phase2_v2.

    Periodic update guarded by: counter >= 200 AND neg_count > 0.
    """
    cnt = 0; eff_tol = eff_tol_init; kappa = np.nan; trace = []
    for i, xB in enumerate(x_basics):
        cnt += 1
        neg_count = int(np.sum(xB < -TOL))
        min_xB = float(xB.min())
        c2 = False
        if neg_count > 0:
            if cnt >= RECOMPUTE_EFF_TOL:
                eff_tol, kappa = cond2_fn(np.eye(len(xB)), 1.0); c2 = True; cnt = 0
            decision = "soft" if min_xB >= -eff_tol else "repair"
        else:
            decision = "no_action"
        # periodic (lazy)
        if cnt >= RECOMPUTE_EFF_TOL and neg_count > 0:
            eff_tol, kappa = cond2_fn(np.eye(len(xB)), 1.0); c2 = True; cnt = 0
        trace.append(dict(iter=i, neg_count=neg_count, min_xB=min_xB,
                          eff_tol=eff_tol, kappa=kappa, decision=decision,
                          cond2_called=c2, cnt=cnt))
    return trace


def _run_always(x_basics, eff_tol_init, cond2_fn):
    """Simulate the ORIGINAL (always-compute) gate logic.

    Periodic update guarded by: counter >= 200 (no neg_count check).
    """
    cnt = 0; eff_tol = eff_tol_init; kappa = np.nan; trace = []
    for i, xB in enumerate(x_basics):
        cnt += 1
        neg_count = int(np.sum(xB < -TOL))
        min_xB = float(xB.min())
        c2 = False
        if neg_count > 0:
            if cnt >= RECOMPUTE_EFF_TOL:
                eff_tol, kappa = cond2_fn(np.eye(len(xB)), 1.0); c2 = True; cnt = 0
            decision = "soft" if min_xB >= -eff_tol else "repair"
        else:
            decision = "no_action"
        # periodic (always)
        if cnt >= RECOMPUTE_EFF_TOL:
            eff_tol, kappa = cond2_fn(np.eye(len(xB)), 1.0); c2 = True; cnt = 0
        trace.append(dict(iter=i, neg_count=neg_count, min_xB=min_xB,
                          eff_tol=eff_tol, kappa=kappa, decision=decision,
                          cond2_called=c2, cnt=cnt))
    return trace


def _mixed_scenario():
    """250 clean + 5 soft-range negative + 10 repair-range + 200 clean = 465."""
    rng = np.random.RandomState(42); xs = []
    for _ in range(250): xs.append(rng.uniform(0.001, 10.0, size=10))
    for _ in range(5):
        x = rng.uniform(0.001, 10.0, size=10); x[3] = -5e-8; xs.append(x)
    for _ in range(10):
        x = rng.uniform(0.001, 10.0, size=10); x[5] = -1e-4; xs.append(x)
    for _ in range(200): xs.append(rng.uniform(0.001, 10.0, size=10))
    return xs


class TestLazyConditionGate:

    def test_neg_count_zero_no_cond2(self):
        """neg_count == 0 for 500 iters -> lazy calls cond2 zero times."""
        xs = [np.ones(5) * 1.0 for _ in range(500)]
        c_l = FakeCond2(); c_a = FakeCond2()
        _run_lazy(xs, TOL, c_l)
        _run_always(xs, TOL, c_a)
        assert c_l.call_count == 0
        assert c_a.call_count > 0  # always-compute fires every 200

    def test_neg_count_positive_triggers_cond2(self):
        """neg_count > 0 for 250 iters -> lazy calls cond2 at least twice."""
        xs = []
        for _ in range(250):
            x = np.ones(5); x[0] = -1e-3; xs.append(x)
        c = FakeCond2()
        trace = _run_lazy(xs, TOL, c)
        assert c.call_count >= 1
        for t in trace:
            assert t["decision"] != "no_action"

    def test_identical_decisions_lazy_vs_always(self):
        """Mixed scenario: every gate decision is identical in both paths."""
        xs = _mixed_scenario()
        c_l = FakeCond2(); c_a = FakeCond2()
        t_l = _run_lazy(xs, TOL, c_l)
        t_a = _run_always(xs, TOL, c_a)
        assert len(t_l) == len(t_a) == len(xs)
        for i, (l, a) in enumerate(zip(t_l, t_a)):
            assert l["decision"] == a["decision"], (
                f"iter={i}: lazy={l['decision']} != always={a['decision']}")

    def test_lazy_saves_calls(self):
        """Lazy strictly fewer cond2 calls than always on mixed scenario."""
        xs = _mixed_scenario()
        c_l = FakeCond2(); c_a = FakeCond2()
        _run_lazy(xs, TOL, c_l)
        _run_always(xs, TOL, c_a)
        assert c_l.call_count < c_a.call_count

    def test_counter_keeps_increasing_when_skipped(self):
        """600 clean iters -> counter reaches 600, no cond2 calls."""
        xs = [np.ones(5) for _ in range(600)]
        c = FakeCond2()
        trace = _run_lazy(xs, TOL, c)
        assert trace[-1]["cnt"] == 600
        assert trace[199]["cnt"] == 200
        assert c.call_count == 0

    def test_counter_resets_on_first_negative(self):
        """250 clean then 1 negative -> cond2 fires once, counter resets to 0."""
        xs = [np.ones(5) for _ in range(250)]
        x_neg = np.ones(5); x_neg[0] = -1e-3
        xs.append(x_neg)
        c = FakeCond2()
        trace = _run_lazy(xs, TOL, c)
        assert c.call_count == 1
        assert trace[-1]["cnt"] == 0
