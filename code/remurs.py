"""
This module provides functionality to train a multilinear classifier using the 
Regularized Multilinear Regression and Selection (Remurs) method.

Original MATLAB code: https://uk.mathworks.com/matlabcentral/fileexchange/71204
Remurs paper: https://ojs.aaai.org/index.php/AAAI/article/view/10871

Classes:
    RemursClassifier: Multilinear classifier using Remurs method.

Functions:
    center_data: Preprocesses data by centering it.
    remurs_regression: Regularized multilinear regression and selection algorithm implementation.
"""

import numpy as np
import utils
from sklearn.preprocessing import LabelBinarizer


def remurs_regression(
    tX: np.ndarray,
    y: np.ndarray,
    alpha: float,
    beta: float,
    epsilon: float = 1e-4,
    max_iter: int = 1000,
    flatten_input: bool = False,
):
    """Regularized multilinear regression and selection."""
    # used in old implementation
    if flatten_input:
        tX = tX.reshape((y.shape[0], -1))

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
        self,
        alpha=1.0,
        beta=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        flatten_input=False,
    ):
        """Multilinear classifier using Remurs method."""
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.flatten_input = flatten_input
        self.class_weight = None
        self.coef_ = None
        self.binarizer = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y):
        """Fit the model given training data X labels."""
        # preprocess
        Y = self.binarizer.fit_transform(y.reshape(-1, 1))
        X, y, X_offset, Y_offset = center_data(X, Y, self.fit_intercept)

        # flatten coef for ease of prediction calculation
        self.coef_ = remurs_regression(
            X,
            y,
            alpha=self.alpha,
            beta=self.beta,
            epsilon=self.tol,
            max_iter=self.max_iter,
            flatten_input=self.flatten_input,
        ).flatten()

        if self.fit_intercept:
            self.intercept_ = Y_offset - np.dot(X_offset.flatten(), self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def decision_function(self, X):
        """Decide where the point lies on the fit."""
        self.check_is_fitted()
        # flatten each sample in X for ease of calculation
        flat_X = X.reshape(X.shape[0], -1)
        return np.dot(flat_X, self.coef_) + self.intercept_

    def predict(self, X, certainty=False):
        """Predict labels for inputs. Optionally return certainty values."""
        scores = self.decision_function(X)
        labels = self.binarizer.inverse_transform(scores > 0)
        if certainty:
            return zip(labels, scores)
        return labels

    def check_is_fitted(self):
        """Check that the model has been fitted."""
        if self.coef_ is None:
            raise ValueError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
