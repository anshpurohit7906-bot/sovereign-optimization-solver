"""NPZ serialization for QP benchmark instances."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from .problem import QPProblem

def save_qp(problem, path):
    data={'q':problem.q,'h':problem.h,'b':problem.b,'lb':problem.lb,'ub':problem.ub}
    P=problem.P if sp.issparse(problem.P) else sp.csr_matrix(problem.P)
    sp.save_npz(path+'_P.npz',P)
    if problem.G is not None: sp.save_npz(path+'_G.npz',problem.G if sp.issparse(problem.G) else sp.csr_matrix(problem.G))
    if problem.A is not None: sp.save_npz(path+'_A.npz',problem.A if sp.issparse(problem.A) else sp.csr_matrix(problem.A))
    np.savez_compressed(path+'_data.npz',**{k:v for k,v in data.items() if v is not None})

def load_qp(path):
    d=np.load(path+'_data.npz',allow_pickle=False)
    G=sp.load_npz(path+'_G.npz') if __import__('os').path.exists(path+'_G.npz') else None
    A=sp.load_npz(path+'_A.npz') if __import__('os').path.exists(path+'_A.npz') else None
    def get(k): return d[k] if k in d else None
    return QPProblem(sp.load_npz(path+'_P.npz'),get('q'),G,get('h'),A,get('b'),get('lb'),get('ub'))
