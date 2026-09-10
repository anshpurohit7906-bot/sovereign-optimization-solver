# -*- coding: utf-8 -*-
"""Focused experiment test: CURRENT vs LAZY condition-gate control flow
produces an identical Phase II trajectory on a small LP.

This is the same check performed by experiment/sparse
/validate_lazy_condition_gate_ab.py but kept fast so it can run under pytest.
It instantiates its own tiny solver that mirrors the exact PILOT87 Phase II
math (cond2 dense condition-number based effective tolerance, devex pricing,
sparse LU, iterative refinement) and swaps ONLY the condition-gate control flow.

Lazy semantics under test:
    if min_xB >= -tol:  skip compute_effective_tol, decision = no_action
    else:               call the SAME compute_effective_tol + soft_clamp/repair
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.normpath(os.path.join(_HERE, "..", "experiment", "sparse"))
for _p in (_EXP,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from validate_lazy_condition_gate_ab import (  # noqa: E402
    phase2_ab,
    compare_runs,
)

SMALL_A = np.array([[2.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 2.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0]])
SMALL_B = np.array([4.0, 5.0, 3.0])
SMALL_C = np.array([3.0, 2.0, 0.0, 0.0, 0.0])
SMALL_BASIS = [0, 1, 4]  # feasible, not optimal -> forces pivots


def test_lazy_gate_small_lp_identical_trajectory():
    cur = phase2_ab(sp.csc_matrix(SMALL_A), SMALL_B, SMALL_C, SMALL_BASIS,
                    mode="current", full_trace=True)
    lazy = phase2_ab(sp.csc_matrix(SMALL_A), SMALL_B, SMALL_C, SMALL_BASIS,
                     mode="lazy", full_trace=True)
    comp = compare_runs("SMALL", cur, lazy)
    assert comp["match"], comp["mismatches"]
    assert cur[0] == lazy[0]           # same final status
    assert cur[3] == lazy[3]           # same iteration count
    assert np.isclose(cur[1], lazy[1], rtol=1e-9, atol=1e-9)
    assert sorted(cur[2]) == sorted(lazy[2])  # same final basis (as multiset)


def test_lazy_gate_small_lp_condition_calls_identical():
    """LAZY must never compute MORE than CURRENT, and both use the exact same
    cond2 routine whenever it does fire (same count for this small problem)."""
    cur = phase2_ab(sp.csc_matrix(SMALL_A), SMALL_B, SMALL_C, SMALL_BASIS,
                    mode="current", full_trace=True)
    lazy = phase2_ab(sp.csc_matrix(SMALL_A), SMALL_B, SMALL_C, SMALL_BASIS,
                     mode="lazy", full_trace=True)
    assert lazy[4] <= cur[4]           # skipped calls are never additional
    assert lazy[4] == cur[4]           # equality for the small feasible case