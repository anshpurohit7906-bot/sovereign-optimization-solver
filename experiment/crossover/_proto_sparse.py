"""Sparse composite-Phase-I repair prototype — measure PILOT87 cost."""
import sys, time, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, 'src'); sys.path.insert(0, 'src/lp')
sys.path.insert(0, '.'); sys.path.insert(0, 'experiment/crossover')
from numerical_model import load_numeric_mps
from mehrotra import to_standard_form
from stage1_audit_rrqr import rrqr_basis


def repair(A, b, basis0, max_steps=2000, verbose=False):
    """Sparse composite Phase I: drive x_B >= 0. Returns (basis, steps, feasible)."""
    A_dense = np.asarray(A.todense()) if sp.issparse(A) else np.asarray(A, float)
    b = np.asarray(b, float)
    m, n = A_dense.shape
    basis = list(basis0)
    nb_set = set(range(n)) - set(basis)
    piv_tol = 1e-9

    for step in range(max_steps):
        B_dense = A_dense[:, basis]
        try:
            lu = splu(sp.csc_matrix(B_dense))
        except RuntimeError:
            return basis, step, False
        xb = lu.solve(b)
        infeas = xb < -1e-7
        if not infeas.any():
            return basis, step, True

        cb = np.where(infeas, -1.0, 0.0)
        y = lu.solve(cb, trans='T')
        d = np.array(-(A_dense[:, sorted(nb_set)].T @ y)).ravel()
        nb_list = sorted(nb_set)
        cand_idx = [k for k in range(len(nb_list)) if d[k] < -1e-9]
        if not cand_idx:
            return basis, step, False
        picked = False
        for k in sorted(cand_idx, key=lambda kk: d[kk])[:50]:
            q = nb_list[k]
            aq = A_dense[:, q]
            alpha = lu.solve(np.asarray(aq).ravel())
            amax = float(np.max(np.abs(alpha)))
            if amax < piv_tol:
                continue
            mask = alpha > piv_tol * max(1.0, amax)
            if not mask.any():
                continue
            ratios = np.full(m, np.inf)
            ratios[mask] = xb[mask] / alpha[mask]
            r = int(np.argmin(ratios))
            if alpha[r] <= piv_tol * max(1.0, amax):
                continue
            old = basis[r]; basis[r] = q
            nb_set.discard(q); nb_set.add(old)
            picked = True
            if verbose:
                print(f'  step {step}: enter={q} leave_row={r} xb={xb[r]:.3e} '
                      f'alpha={alpha[r]:.3e} neg={int(infeas.sum())}')
            break
        if not picked:
            return basis, step, False
    # max_steps exhausted without confirming feasibility
    return basis, max_steps, False


def run():
    for name in ['pilot4_plain', 'pilot87']:
        sf = to_standard_form(load_numeric_mps(f'data/{name}.mps'))
        A = sp.csc_matrix(sf.A); b = np.asarray(sf.b, float)
        piv, basis0 = rrqr_basis(A.toarray())
        t0 = time.perf_counter()
        basis, steps, feas = repair(A, b, list(basis0), verbose=(name=='pilot4_plain'))
        dt = time.perf_counter() - t0
        m = A.shape[0]
        B = A[:, basis]; lu = splu(B.tocsc()); xb = lu.solve(b)
        neg = int((xb < -1e-7).sum())
        print(f'{name}: m={m} steps={steps} t={dt:.2f}s feas={feas} neg={neg} min_xb={xb.min():.3e}')


if __name__ == '__main__':
    run()