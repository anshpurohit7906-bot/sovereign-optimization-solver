from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve as dense_solve
from scipy.sparse.linalg import splu


class QPLinearSystemError(RuntimeError):
    """Raised when the QP KKT linear system cannot be solved."""


def solve_kkt(
    H,
    A,
    rhs,
    *,
    sparse=False,
    reg=1e-10,
):
    """
    Solve the equality-constrained KKT system

        [ H   A.T ] [dx] = [rhs_x]
        [ A    0  ] [dy]   [rhs_y]

    For sparse systems, a small regularization is also applied
    to the equality-dual block:

        [ H + reg*I    A.T       ]
        [ A           -reg*I     ]

    This improves numerical stability for large or rank-deficient
    sparse QP benchmark instances.

    Returns
    -------
    dx : ndarray
        Primal direction.
    dy : ndarray
        Equality-dual direction.
    """

    H = H
    rhs = np.asarray(rhs, dtype=float)

    n = H.shape[0]
    meq = 0 if A is None else A.shape[0]

    if rhs.shape[0] != n + meq:
        raise QPLinearSystemError(
            f"KKT RHS has length {rhs.shape[0]}, "
            f"expected {n + meq}"
        )

    # ---------------------------------------------------------
    # Sparse path
    # ---------------------------------------------------------
    if sparse:
        try:
            # Keep H sparse from beginning to end.
            if sp.issparse(H):
                Hs = H.tocsc()
            else:
                Hs = sp.csc_matrix(H)

            # Regularize primal Hessian block.
            Hs = Hs + reg * sp.eye(n, format="csc")

            rhs = np.asarray(rhs, dtype=float)

            # -------------------------------------------------
            # No equality constraints
            # -------------------------------------------------
            if meq == 0:
                lu = splu(Hs)
                dx = lu.solve(rhs)
                dy = np.empty(0, dtype=float)

                if not np.all(np.isfinite(dx)):
                    raise QPLinearSystemError(
                        "sparse KKT solve produced non-finite values"
                    )

                return dx, dy

            # -------------------------------------------------
            # Equality-constrained sparse KKT system
            #
            # [ H + reg*I      A.T       ]
            # [ A             -reg*I     ]
            #
            # The -reg*I term prevents an exactly singular
            # equality-dual block when A is rank deficient.
            # -------------------------------------------------
            if sp.issparse(A):
                As = A.tocsc()
            else:
                As = sp.csc_matrix(A)

            dual_reg = max(reg, 1e-12)

            zero_dual = -dual_reg * sp.eye(
                meq,
                format="csc",
            )

            K = sp.bmat(
                [
                    [Hs, As.T],
                    [As, zero_dual],
                ],
                format="csc",
            )

            # Eliminate explicit zeros that can interfere with
            # sparse factorization.
            K.eliminate_zeros()

            lu = splu(K)

            sol = lu.solve(rhs)

            if not np.all(np.isfinite(sol)):
                raise QPLinearSystemError(
                    "sparse KKT solve produced non-finite values"
                )

            dx = sol[:n]
            dy = sol[n:]

            return dx, dy

        except QPLinearSystemError:
            raise

        except Exception as e:
            raise QPLinearSystemError(
                f"sparse KKT solve failed: {e}"
            ) from e

    # ---------------------------------------------------------
    # Dense path
    # ---------------------------------------------------------
    try:
        if meq == 0:
            H_dense = (
                H.toarray()
                if sp.issparse(H)
                else np.asarray(H, dtype=float)
            )

            H_dense = H_dense + reg * np.eye(n)

            dx = dense_solve(
                H_dense,
                rhs,
                assume_a="gen",
            )

            dy = np.empty(0, dtype=float)

            if not np.all(np.isfinite(dx)):
                raise QPLinearSystemError(
                    "dense KKT solve produced non-finite values"
                )

            return dx, dy

        H_dense = (
            H.toarray()
            if sp.issparse(H)
            else np.asarray(H, dtype=float)
        )

        A_dense = (
            A.toarray()
            if sp.issparse(A)
            else np.asarray(A, dtype=float)
        )

        H_dense = H_dense + reg * np.eye(n)

        K = np.block(
            [
                [H_dense, A_dense.T],
                [A_dense, np.zeros((meq, meq))],
            ]
        )

        sol = dense_solve(
            K,
            rhs,
            assume_a="gen",
        )

        if not np.all(np.isfinite(sol)):
            raise QPLinearSystemError(
                "dense KKT solve produced non-finite values"
            )

        dx = sol[:n]
        dy = sol[n:]

        return dx, dy

    except QPLinearSystemError:
        raise

    except Exception as e:
        raise QPLinearSystemError(
            f"dense KKT solve failed: {e}"
        ) from e