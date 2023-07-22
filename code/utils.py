"""
This module provides a collection of mathematical and data manipulation
functions, useful in multilinear algebra and machine learning preprocessing.

Functions:
    shift: Changes the axes of the array.
    fold: Collapses all but one of the dimensions of an array to perform
          operations on the data.
    unfold: Reverts the array back to its original shape after performing
            operations using fold.
    factor: Performs Cholesky decomposition on the input matrix.
    prox_l1: Evaluates the proximal operator of the l1 norm.
    shrinkage_penalty: Applies shrinkage penalty on an array.
    prox_nuclear: Evaluates the proximal operator of the nuclear norm (i.e.,
                  the singular value thresholding operator).
    center_data: Preprocesses the data by centering it around the mean.
"""

import numpy as np
import scipy.sparse as sparse


def shift(X: np.ndarray, n: int, shift_right: bool = True):
    """Changes the axes of the array.

    Args:
        X (np.ndarray): Input array.
        n (int): Number of positions to shift.
        shift_right (bool, optional): Direction of shift. True for right, False for left. Defaults to True.

    Returns:
        np.ndarray: Array after shifting axes.
    """
    dims = X.ndim
    if not shift_right:
        n = dims - n
    shift = n % dims
    shift_indices = np.arange(shift)
    return np.moveaxis(X, shift_indices - shift, shift_indices)


# helper functions
def fold(X: np.ndarray, dim: tuple, i: int):
    """
    Collapses all but one of the dimensions of an array to perform operations
    on the data.

    Args:
        X (np.ndarray): The input tensor.
        dim (list/tuple): Dimensions to be preserved.
        i (int): Index of dimension to be preserved.

    Returns:
        np.ndarray: Folded array with one dimension preserved.
    """
    dim = np.roll(dim, 1 - i)
    X = np.reshape(X, dim)
    X = shift(X, i - 1)
    return X


def unfold(X: np.ndarray, dim: tuple, i: int):
    """Reverts the array back to its original shape after performing operations
    using fold.

    Args:
        X (np.ndarray): The folded array.
        dim (list/tuple): The original dimensions of the array.
        i (int): Index of dimension to be preserved during fold.

    Returns:
        np.ndarray: Unfolded array in its original shape.
    """
    X = shift(X, i - 1, shift_right=False)
    X = X.reshape(((dim[(i - 1) % len(dim)]), -1))
    return X


def factor(A: np.ndarray, rho: float):
    """Performs Cholesky decomposition on the input matrix.

    Args:
        A (np.ndarray): The input matrix for decomposition.
        rho (float): A regularization parameter.

    Returns:
        tuple: The lower triangular matrix and the upper triangular matrix
               obtained from Cholesky decomposition.
    """
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
    """Center data. Based on scikit-learn _preprocess_data."""
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
