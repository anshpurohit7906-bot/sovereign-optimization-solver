"""
PDHG (Chambolle-Pock Primal-Dual Hybrid Gradient) solver for linear programs
of the form:

    minimize    c^T x
    subject to  A x <= b
                x >= 0

This is the same algorithm family used by Google's PDLP and FICO Xpress's
GPU-accelerated solver (as of Xpress 9.8) -- it is chosen specifically
because, unlike simplex, it consists entirely of matrix-vector products and
elementwise clipping, which parallelize cleanly on a GPU.

Correctness is checked against scipy.optimize.linprog, used here strictly
as a validation oracle -- not as the solving engine.
"""

import time
import numpy as np
from scipy.optimize import linprog


def spectral_norm_power_iteration(A, n_iter=200, tol=1e-12, seed=0):
    """
    Estimate the largest singular value (spectral norm) of A via power
    iteration on A^T A. Needed to pick step sizes that guarantee PDHG
    convergence: we require tau * sigma * ||A||_2^2 < 1.
    """
    n = A.shape[1]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)

    sigma_prev = 0.0
    for _ in range(n_iter):
        u = A @ v
        v = A.T @ u
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-300:
            break
        v /= norm_v
        sigma = np.linalg.norm(A @ v)
        if abs(sigma - sigma_prev) < tol:
            break
        sigma_prev = sigma
    return sigma


def pdhg_lp(c, A, b, max_iter=50000, tol=1e-9, check_every=500, verbose=False):
    """
    Solve  min c^T x  s.t.  A x <= b, x >= 0  using PDHG.

    Returns (x, y, iterations_used, converged_bool).

    Convergence is judged on three quantities, all of which should go to
    zero at an optimal, feasible solution:
      - duality gap:      |primal_objective - dual_objective|
      - primal infeasibility: how much Ax exceeds b
      - dual infeasibility:   how much c + A^T y goes negative
    """
    m, n = A.shape
    L = spectral_norm_power_iteration(A)
    tau = sigma = 0.9 / L  # 0.9 safety margin below the 1/L convergence bound

    x = np.zeros(n)
    y = np.zeros(m)
    converged = False
    k = 0

    for k in range(max_iter):
        x_prev = x.copy()

        # Primal step: gradient descent on c^T x + y^T A x, projected onto x >= 0
        x = np.maximum(x - tau * (c + A.T @ y), 0.0)

        # Extrapolation -- this is what makes it Chambolle-Pock rather than
        # plain alternating gradient descent, and it's required for convergence.
        x_bar = 2 * x - x_prev

        # Dual step: gradient ascent on -y^T(Ax - b), projected onto y >= 0
        y = np.maximum(y + sigma * (A @ x_bar - b), 0.0)

        if k % check_every == 0 or k == max_iter - 1:
            primal_obj = c @ x
            dual_obj = -b @ y
            gap = abs(primal_obj - dual_obj)
            primal_infeas = np.linalg.norm(np.maximum(A @ x - b, 0.0))
            dual_infeas = np.linalg.norm(np.maximum(-(c + A.T @ y), 0.0))

            if verbose:
                print(f"iter {k:6d}  primal={primal_obj: .6f}  dual={dual_obj: .6f}  "
                      f"gap={gap:.2e}  primal_infeas={primal_infeas:.2e}  "
                      f"dual_infeas={dual_infeas:.2e}")

            if gap < tol and primal_infeas < tol and dual_infeas < tol:
                converged = True
                break

    return x, y, k, converged


if __name__ == "__main__":
    print("=== Toy LP (2 variables, verified exactly against scipy) ===")
    c = np.array([-1.0, -2.0])
    A = np.array([[1.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([4.0, 6.0])

    x, y, iters, ok = pdhg_lp(c, A, b, verbose=True)
    print(f"\nPDHG:   x = {x}   objective = {c @ x:.6f}   iters = {iters}   converged = {ok}")

    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
    print(f"scipy:  x = {res.x}   objective = {res.fun:.6f}")

    print("\n=== Larger random LP (20 vars, 30 constraints) ===")
    print("(Demonstrates the honest limitation: vanilla PDHG converges slowly")
    print(" without restarts/preconditioning -- this is the real frontier,")
    print(" not a bug, and is what production GPU solvers spend most of their")
    print(" engineering effort on.)\n")

    rng = np.random.default_rng(42)
    n, m = 20, 30
    A2 = rng.uniform(0.1, 1.0, size=(m, n))
    x_feas = rng.uniform(0.5, 2.0, size=n)
    b2 = A2 @ x_feas + rng.uniform(0.1, 1.0, size=m)
    c2 = -rng.uniform(0.5, 2.0, size=n)

    t0 = time.time()
    x2, y2, iters2, ok2 = pdhg_lp(c2, A2, b2, max_iter=100000, tol=1e-7)
    t1 = time.time()
    res2 = linprog(c2, A_ub=A2, b_ub=b2, bounds=(0, None))

    print(f"PDHG:   obj = {c2 @ x2:.6f}   iters = {iters2}   converged = {ok2}   time = {t1 - t0:.3f}s")
    print(f"scipy:  obj = {res2.fun:.6f}")
    print(f"objective gap: {abs(c2 @ x2 - res2.fun):.2e}")
