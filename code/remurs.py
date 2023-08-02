import numpy as np
from sklearn.preprocessing import LabelBinarizer

import utils


def _center_data(
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


def _remurs_regression(
    tX: np.ndarray,
    y: np.ndarray,
    alpha: float,
    beta: float,
    epsilon: float = 1e-4,
    max_iter: int = 1000,
):
    # modify to make work
    # TODO: find a better general solution

    if y.shape[0] != tX.shape[-1]:
        tX = tX.T

    # initialise other variables
    lambda_ = 1
    rho = 1 / lambda_
    N = tX.ndim - 1
    size_tX = tX.shape
    dim = size_tX[:N]
    X = utils.unfold(tX, size_tX, N + 1)
    Xty = X.T @ y
    num_features = X.shape[1]
    num_samples = X.shape[0]
    tV = np.array([np.zeros(dim) for _ in range(N)])
    tB = np.array([np.zeros(dim) for _ in range(N)])

    tW = np.zeros(dim)  # tensor W
    tU = np.zeros(dim)  # tensor U
    tA = np.zeros(dim)  # tensor A

    L, U = utils.factor(X, rho)

    error_list = []

    # REMURS algorithm
    more_count = 0
    less_count = 0
    for _ in range(max_iter):
        # update tU: quadratic proximal operator
        q = Xty + np.reshape(rho * (tW.flatten() - tA.flatten()), Xty.shape)

        if num_samples >= num_features:
            u = np.linalg.inv(U) @ (np.linalg.inv(L) @ q)
            more_count += 1
        else:
            less_count += 1
            u = lambda_ * (
                q - lambda_ * (X.T @ (np.linalg.inv(U) @ (np.linalg.inv(L) @ (X @ q))))
            )

        tU = u.reshape(dim)

        for n in range(N):
            tV[n] = utils.fold(
                utils.prox_nuclear(
                    utils.unfold(tW - tB[n], dim, n),
                    alpha / rho / N,
                ),
                dim,
                n,
            )

        # update tW: l1-norm proximal operator
        last_tW = tW.copy()

        tSum_uv = tU.copy()
        tSum_ab = tA.copy()

        for n in range(N):
            tSum_uv += tV[n]
            tSum_ab += tB[n]

        tW = utils.prox_l1((tSum_uv + tSum_ab) / (N + 1), beta / (N + 1) / rho)

        # update tA
        tA = tA + tU - tW

        # update tB_n
        for n in range(N):
            tB[n] += tV[n] - tW

        error = np.linalg.norm(tW.ravel() - last_tW.ravel()) / np.linalg.norm(
            tW.ravel()
        )

        error_list.append(error)

        if error < epsilon or np.isnan(error):
            break

    return tW


class RemursClassifier:
    def __init__(
        self, alpha=1.0, beta=1.0, fit_intercept=True, max_iter=1000, tol=1e-4
    ):
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = None
        self.coef_ = None
        self.binarizer = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y):
        # preprocess
        Y = self.binarizer.fit_transform(y.reshape(-1, 1))
        X, Y, X_offset, Y_offset = _center_data(X, Y, self.fit_intercept)

        # flatten coef for ease of prediction calculation
        self.coef_ = _remurs_regression(
            X, Y, self.alpha, self.beta, self.tol, self.max_iter
        ).flatten()

        if self.fit_intercept:
            self.intercept_ = Y_offset - np.dot(X_offset.flatten(), self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def decision_function(self, X):
        self.check_is_fitted()
        # flatten each sample in X for ease of calculation
        flat_X = X.reshape(X.shape[0], -1)
        return np.dot(flat_X, self.coef_) + self.intercept_

    def predict(self, X, certainty=False):
        scores = self.decision_function(X)
        labels = self.binarizer.inverse_transform(scores > 0)
        if certainty:
            return zip(labels, scores)
        return labels

    def check_is_fitted(self):
        if self.coef_ is None:
            raise ValueError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
