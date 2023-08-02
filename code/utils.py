import numpy as np
import scipy.sparse as sparse


def shift(X: np.ndarray, n: int, shift_right: bool = True):
    dims = X.ndim
    if not shift_right:
        n = dims - n
    shift = n % dims
    shift_indices = np.arange(shift)
    return np.moveaxis(X, shift_indices - shift, shift_indices)


# helper functions
def fold(X, dim, i):
    dim = np.roll(dim, 1 - i)
    X = np.reshape(X, dim)
    X = shift(X, i - 1)
    return X


def unfold(X, dim, i):
    X = shift(X, i - 1, shift_right=False)
    X = X.reshape(((dim[(i - 1) % len(dim)]), -1))
    return X


def factor(A, rho):
    m, n = A.shape
    if m >= n:  # if skinny
        L = np.linalg.cholesky(A.T @ A + rho * sparse.eye(n))
    else:  # if fat
        L = np.linalg.cholesky(sparse.eye(m) + 1 / rho * A @ A.T)
    # force Python to recognize the upper/lower triangular structure
    L = np.array(L)
    U = L.T
    return L, U


def prox_l1(v, lambda_):
    """The proximal operator of the l1 norm.

    prox_l1(v, lambda_) is the proximal operator of the l1 norm with parameter lambda_.
    """
    return np.maximum(0, v - lambda_) - np.maximum(0, -v - lambda_)


def shrinkage_penalty(v, lambda_):
    """Shrinkage penalty, somewhat simulating prox l2 norm.

    shrinkage(v, lambda_) is shrinkage penalty simulating prox l2 norm with parameter lambda_.
    """
    return v / (1 + lambda_)


def prox_nuclear(v, lambda_):
    """Evaluates the proximal operator of the nuclear norm at v
    (i.e., the singular value thresholding operator).
    """
    U, S, V = np.linalg.svd(v, full_matrices=False)
    S = np.diag(S)
    return U @ np.diag(prox_l1(np.diag(S), lambda_)) @ V


def center_data(
    X,
    y,
    fit_intercept,
    copy=True,
):
    """Center data."""
    if copy:
        X = X.copy(order="K")

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        X_offset = np.average(X, axis=0)
        X_offset = X_offset.astype(X.dtype, copy=False)
        X -= X_offset
        y_offset = np.average(y, axis=0)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset
