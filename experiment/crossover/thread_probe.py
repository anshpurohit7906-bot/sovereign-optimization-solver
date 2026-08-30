"""Thread-count probe: does BLAS threading change the RRQR/repair pivot path?"""
import sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
sys.path.insert(0, 'src'); sys.path.insert(0, 'src/lp'); sys.path.insert(0, 'experiment/crossover')
from numerical_model import load_numeric_mps
from mehrotra import to_standard_form
from stage1_audit_rrqr import rrqr_basis
from _proto_sparse import repair

sf = to_standard_form(load_numeric_mps('data/pilot4_plain.mps'))
A = sp.csc_matrix(sf.A); b = np.asarray(sf.b, float)
piv, basis0 = rrqr_basis(A.toarray())
basis, steps, feas = repair(A, b, list(basis0), verbose=False)
basis_sig = hash(tuple(basis)) & 0xffffffff
rrqr_sig = hash(tuple(basis0.tolist())) & 0xffffffff
print(f'PROBE rrqr_sig={rrqr_sig:x} steps={steps} feas={feas} basis_sig={basis_sig:x}')
