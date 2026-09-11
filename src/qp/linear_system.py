"""Dense and sparse KKT linear algebra for the QP interior-point method."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.linalg import solve as dense_solve


class QPLinearSystemError(RuntimeError):
    pass


def solve_kkt(H, A, rhs, *, sparse=False, reg=1e-10):
    """Solve [H A.T; A 0][dx;dy]=rhs without densifying sparse systems."""
    n = H.shape[0]
    meq = 0 if A is None else A.shape[0]

    if sparse:
        Hs = H if sp.issparse(H) else sp.csc_matrix(H)
        Hs = Hs.tocsc() + reg * sp.eye(n, format="csc")
        if meq == 0:
            try:
                sol = splu(Hs).solve(np.asarray(rhs[:n], dtype=float))
                return sol, np.empty(0)
            except Exception as e:
                raise QPLinearSystemError(f"sparse KKT solve failed: {e}") from e

        As = A if sp.issparse(A) else sp.csc_matrix(A)
        K = sp.bmat([[Hs, As.T], [As, None]], format="csc")
        try:
            sol = splu(K).solve(np.asarray(rhs, dtype=float))
            return sol[:n], sol[n:]
        except Exception as e:
            raise QPLinearSystemError(f"sparse KKT solve failed: {e}") from e

    # Dense path is explicit and may densify only because the caller requested it.
    Hd = H.toarray() if sp.issparse(H) else np.asarray(H, dtype=float)
    if meq == 0:
        try:
            return dense_solve(Hd + reg * np.eye(n), rhs[:n]), np.empty(0)
        except Exception as e:
            raise QPLinearSystemError(str(e)) from e

    Ad = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=float)
    K = np.block([[Hd + reg * np.eye(n), Ad.T],
                  [Ad, np.zeros((meq, meq))]])
    try:
        sol = dense_solve(K, rhs)
        return sol[:n], sol[n:]
    except Exception as e:
        raise QPLinearSystemError(f"dense KKT solve failed: {e}") from e
