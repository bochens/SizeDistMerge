import numpy as np
from scipy.optimize import nnls

def _finite_diff_matrix(n: int, order: int) -> np.ndarray:
    """
    Build the regularization operator L for Tikhonov/Twomey.
    order=0: Identity (n x n)
    order=1: First-difference (n-1 x n) with rows [..., -1, +1, ...]
    order=2: Second-difference (n-2 x n) with rows [..., 1, -2, 1, ...]
    """
    if order == 0:
        L = np.eye(n, dtype=float)
    elif order == 1:
        L = np.zeros((n - 1, n), dtype=float)
        for i in range(n - 1):
            L[i, i] = -1.0
            L[i, i + 1] = +1.0
    elif order == 2:
        L = np.zeros((n - 2, n), dtype=float)
        for i in range(n - 2):
            L[i, i]     = +1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = +1.0
    else:
        raise ValueError("order must be 0, 1, or 2")
    return L


def twomey_inversion(
    K: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-2,
    *,
    order: int = 2,
    w: np.ndarray | None = None,
    nonneg: bool = True
) -> tuple[np.ndarray, dict]:
    """
    Twomey/Tikhonov inversion: solve y ≈ K n with smoothness penalty λ ||L n||^2.

    Args
    ----
    K : (m, n) measurement kernel (rows = instrument bins, cols = fine bins)
    y : (m,)   observed instrument-bin numbers (or weighted measurements)
    lam : float  regularization strength λ ≥ 0
    order : int  0, 1, or 2 (zeroth/first/second-order smoothing)
    w : (m,) optional weights for data misfit; if None, all ones
    nonneg : bool  if True, enforce n ≥ 0 via NNLS on augmented system

    Returns
    -------
    n : (n,) solution for dN/dlogD on the fine grid
    info : dict with simple diagnostics
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float).ravel()
    m, n = K.shape
    if y.size != m:
        raise ValueError("Shapes mismatch: y must have length m = K.shape[0]")

    # Data weighting
    if w is None:
        Wsqrt = np.ones(m, float)
    else:
        Wsqrt = np.sqrt(np.asarray(w, float).ravel())
        if Wsqrt.size != m:
            raise ValueError("w must have length m")
    A = (Wsqrt[:, None] * K)
    b = (Wsqrt * y)

    # Regularization operator
    if lam > 0.0:
        L = _finite_diff_matrix(n, order)
        A_aug = np.vstack([A, np.sqrt(lam) * L])
        b_aug = np.concatenate([b, np.zeros(L.shape[0], float)])
    else:
        A_aug = A
        b_aug = b

    # Solve
    if nonneg:
        n_sol, rnorm = nnls(A_aug, b_aug)
        res = b_aug - A_aug @ n_sol
        info = {"method": "nnls", "rnorm": float(rnorm), "res_norm": float(np.linalg.norm(res))}
    else:
        # Unconstrained Tikhonov solve (least squares on augmented system)
        n_sol, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
        res = b_aug - A_aug @ n_sol
        info = {"method": "lstsq", "res_norm": float(np.linalg.norm(res))}

    return n_sol, info