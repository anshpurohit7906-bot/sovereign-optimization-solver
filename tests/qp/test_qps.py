import os, sys
from pathlib import Path
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from qp.qps import read_qps


def test_qps_reader_minimal(tmp_path):
    text = """NAME TEST
ROWS
 E EQ
 L LIM
 N COST
COLUMNS
 X1 EQ 1 LIM 1
 X1 COST -2
 X2 EQ 1 LIM -1
 X2 COST -4
RHS
 RHS1 EQ 3 LIM 2
BOUNDS
 LO BND X1 0
 LO BND X2 0
QUADOBJ
 X1 X1 2
 X1 X2 1
 X2 X2 2
ENDATA
"""
    f = tmp_path / "TEST.SIF"
    f.write_text(text)
    p = read_qps(f)
    assert p.n == 2
    assert p.m_eq == 1
    assert p.m_ineq == 1
    assert np.allclose(p.q, [-2., -4.])
    assert np.allclose(p.P.toarray(), [[2., 1.], [1., 2.]])
