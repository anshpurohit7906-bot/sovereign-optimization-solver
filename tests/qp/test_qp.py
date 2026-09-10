import os,sys
import numpy as np
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0,os.path.join(ROOT,'src'))
from qp import QPProblem, solve_qp, certificate

def test_unconstrained():
    P=2*np.eye(2); q=np.array([-4.,-6.]); p=QPProblem(P,q)
    r=solve_qp(p,tol=1e-8)
    assert r.status=='optimal'; assert np.allclose(r.x,[2,3],atol=1e-5)

def test_inequality():
    P=2*np.eye(2); q=np.array([-4.,-6.]); G=np.array([[1.,1.]]); h=np.array([4.])
    p=QPProblem(P,q,G,h); r=solve_qp(p,tol=1e-7)
    assert r.status=='optimal'; assert np.allclose(r.x,[1.5,2.5],atol=1e-4)

def test_equality():
    P=2*np.eye(2); q=np.array([-4.,-6.]); A=np.array([[1.,1.]]); b=np.array([4.])
    p=QPProblem(P,q,A=A,b=b); r=solve_qp(p,tol=1e-7)
    assert r.status=='optimal'; assert np.allclose(r.x,[1.5,2.5],atol=1e-4)

def test_bounds():
    P=2*np.eye(2); q=np.array([-4.,-6.]); p=QPProblem(P,q,lb=np.array([0.,0.]),ub=np.array([1.,3.]))
    r=solve_qp(p,tol=1e-7)
    assert r.status=='optimal'; assert np.allclose(r.x,[1,3],atol=5e-4)

def test_certificate():
    p=QPProblem(2*np.eye(2),[-4,-6],G=[[1,1]],h=[4]); r=solve_qp(p,tol=1e-7)
    c=certificate(p,r.x,r.y,r.z,tol=1e-4); assert c['ok']
