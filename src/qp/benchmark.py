"""Deterministic sparse QP benchmark generator and runner."""
from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import scipy.sparse as sp
from .problem import QPProblem
from .solver import solve_qp


def generate_sparse_qp(n=1000, m_ineq=None, m_eq=None, density=0.005, seed=26119, name=None):
    rng=np.random.default_rng(seed); m_ineq=m_ineq if m_ineq is not None else max(10,n//4); m_eq=m_eq if m_eq is not None else max(2,n//20)
    # Diagonal positive definite Hessian keeps the large benchmark genuinely sparse.
    P=sp.diags(rng.uniform(0.5,2.0,n),format="csr")
    G=sp.random(m_ineq,n,density=density,random_state=rng,data_rvs=lambda k:rng.uniform(-1,1,k),format='csr')
    # Start equality rows with an identity block so the KKT equality block is full row rank.
    A=sp.random(m_eq,n,density=density,random_state=rng,data_rvs=lambda k:rng.uniform(-1,1,k),format='lil')
    for i in range(m_eq):
        A[i, i] = 1.0
    A=A.tocsr()
    x0=rng.uniform(0.1,1.0,n)
    h=np.asarray(G@x0).ravel()+rng.uniform(1.0,2.0,m_ineq)
    b=np.asarray(A@x0).ravel()
    q=-np.asarray(P@x0).ravel()+0.01*rng.standard_normal(n)
    return QPProblem(P,q,G,h,A,b,lb=np.zeros(n),name=name or f'sparse_qp_{n}')


def make_suite(out='data/qp/generated'):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    specs=[(50,20,5,0.08),(200,60,10,0.03),(500,120,20,0.01),(1000,250,40,0.005),(2000,500,80,0.0025),(5000,1250,200,0.001)]
    manifest=[]
    for n,mi,me,d in specs:
        p=generate_sparse_qp(n,mi,me,d,seed=26119+n,name=f'qp_{n}')
        # NPZ stores sparse components without requiring a large text dataset.
        fn=out/f'{p.name}.npz'
        sp.save_npz(str(fn.with_name(fn.stem+'_P.npz')),p.P)
        sp.save_npz(str(fn.with_name(fn.stem+'_G.npz')),p.G)
        sp.save_npz(str(fn.with_name(fn.stem+'_A.npz')),p.A)
        np.savez_compressed(out/f'{p.name}_data.npz',q=p.q,h=p.h,b=p.b,lb=p.lb)
        manifest.append({'name':p.name,'n':n,'m_ineq':mi,'m_eq':me,'density':d,'seed':26119+n})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def run_benchmark(sizes=(50,200,500,1000), max_iterations=80):
    rows=[]
    for n in sizes:
        p=generate_sparse_qp(n,seed=26119+n)
        t=time.perf_counter(); r=solve_qp(p,max_iterations=max_iterations,tol=1e-6,sparse=True); elapsed=time.perf_counter()-t
        rows.append({'name':p.name,'n':n,'m_ineq':p.m_ineq,'m_eq':p.m_eq,'status':r.status,'iterations':r.iterations,'objective':r.objective,'primal_residual':r.primal_residual,'dual_residual':r.dual_residual,'complementarity':r.complementarity,'runtime_seconds':elapsed})
    return rows
