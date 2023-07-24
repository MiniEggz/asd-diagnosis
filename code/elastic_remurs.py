from abc import ABCMeta, abstractmethod

import numbers
import numpy as np
import scipy.sparse as sparse
from sklearn.linear_model._base import (
    LinearClassifierMixin,
    LinearModel,
)
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils._array_api import get_namespace
import scipy.sparse as sp
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.validation import FLOAT_DTYPES, _check_y, check_X_y
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_column_scale

from sklearn.preprocessing._data import _is_constant_feature
from sklearn.preprocessing import LabelBinarizer

from sklearn.utils import column_or_1d, compute_sample_weight
from sklearn.utils.validation import check_is_fitted, _check_sample_weight


# helper methods
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

def prox_l2(v, lambda_):
    """The proximal operator of the l2 norm.
    
    prox_l2(v, lambda_) is the proximal operator of the l1 norm with parameter lambda_
    """
    return v / (1 + 2 * lambda_)


def prox_nuclear(v, lambda_):
    """Evaluates the proximal operator of the nuclear norm at v
    (i.e., the singular value thresholding operator).
    """
    U, S, V = np.linalg.svd(v, full_matrices=False)
    S = np.diag(S)
    return U @ np.diag(prox_l1(np.diag(S), lambda_)) @ V


def _preprocess_data(
    X,
    y,
    fit_intercept,
    normalize=False,
    copy=True,
    sample_weight=None,
    check_input=True,
):
    """Center and scale data.

    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output

        X = (X - X_offset) / X_scale

    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    fit_intercept=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).

    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype

    Returns
    -------
    X_out : {ndarray, sparse matrix} of shape (n_samples, n_features)
        If copy=True a copy of the input X is triggered, otherwise operations are
        inplace.
        If input X is dense, then X_out is centered.
        If normalize is True, then X_out is rescaled (dense and sparse case)
    y_out : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
        Centered version of y. Likely performed inplace on input y.
    X_offset : ndarray of shape (n_features,)
        The mean per column of input X.
    y_offset : float or ndarray of shape (n_features,)
    X_scale : ndarray of shape (n_features,)
        The standard deviation per column of input X.
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES, ensure_2d=False, allow_nd=True)
    elif copy:
        if sp.issparse(X):
            X = X.copy()
        else:
            X = X.copy(order="K")

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0, weights=sample_weight)
        else:
            if normalize:
                X_offset, X_var, _ = _incremental_mean_and_var(
                    X,
                    last_mean=0.0,
                    last_variance=0.0,
                    last_sample_count=0.0,
                    sample_weight=sample_weight,
                )
            else:
                X_offset = np.average(X, axis=0, weights=sample_weight)

            X_offset = X_offset.astype(X.dtype, copy=False)
            X -= X_offset

        if normalize:
            X_var = X_var.astype(X.dtype, copy=False)
            # Detect constant features on the computed variance, before taking
            # the np.sqrt. Otherwise constant features cannot be detected with
            # sample weights.
            constant_mask = _is_constant_feature(X_var, X_offset, X.shape[0])
            if sample_weight is None:
                X_var *= X.shape[0]
            else:
                X_var *= sample_weight.sum()
            X_scale = np.sqrt(X_var, out=X_var)
            X_scale[constant_mask] = 1.0
            if sp.issparse(X):
                inplace_column_scale(X, 1.0 / X_scale)
            else:
                X /= X_scale
        else:
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale

class _BaseElasticRemurs(LinearModel, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        solver="auto",
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
    
    def fit(self, X, y, sample_weight=None):

        # TODO: might want to try with the preprocessing step in BaseRidge removed
        # when X is sparse we only remove offset from y
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X.reshape((X.shape[0], -1)),
            y,
            self.fit_intercept,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        self.coef_ = _elastic_remurs_regression(
            X,
            y,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            epsilon=self.tol,
            max_iter=self.max_iter,
        )

        self._set_intercept(X_offset, y_offset, X_scale)

        return self
    
    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features), default='no validation'
            The input samples.
            If `'no_validation'`, no validation is performed on `X`. This is
            useful for meta-estimator which can delegate input validation to
            their underlying estimator(s). In that case `y` must be passed and
            the only accepted `check_params` are `multi_output` and
            `y_numeric`.

        y : array-like of shape (n_samples,), default='no_validation'
            The targets.

            - If `None`, `check_array` is called on `X`. If the estimator's
              requires_y tag is True, then an error will be raised.
            - If `'no_validation'`, `check_array` is called on `X` and the
              estimator's requires_y tag is ignored. This is a default
              placeholder and is never meant to be explicitly set. In that case
              `X` must be passed.
            - Otherwise, only `y` with `_check_y` or both `X` and `y` are
              checked with either `check_array` or `check_X_y` depending on
              `validate_separately`.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.

        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.

            `estimator=self` is automatically added to these dicts to generate
            more informative error message in case of invalid input data.

        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

            `estimator=self` is automatically added to these params to generate
            more informative error message in case of invalid input data.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if both `X` and `y` are
            validated.
        """
        self._check_feature_names(X, reset=reset)

        if y is None and self._get_tags()["requires_y"]:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )

        no_val_X = isinstance(X, str) and X == "no_validation"
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"

        default_check_params = {"estimator": self}
        check_params = {**default_check_params, **check_params}

        if no_val_X and no_val_y:
            raise ValueError("Validation should be done on X, y or both.")
        elif not no_val_X and no_val_y:
            X = check_array(X, ensure_2d=False, allow_nd=True, input_name="X", **check_params)
            out = X
        elif no_val_X and not no_val_y:
            y = _check_y(y, **check_params)
            out = y
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                if "estimator" not in check_X_params:
                    check_X_params = {**default_check_params, **check_X_params}
                X = check_array(X, ensure_2d=False, allow_nd=True, input_name="X", **check_X_params)
                if "estimator" not in check_y_params:
                    check_y_params = {**default_check_params, **check_y_params}
                y = check_array(y, input_name="y", **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if not no_val_X and check_params.get("ensure_2d", True):
            self._check_n_features(X, reset=reset)

        return out



class _RemursClassifierMixin(LinearClassifierMixin):
    
    def _prepare_data(self, X, y, sample_weight, solver):
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith("multilabel"):
            y = column_or_1d(y, warn=True)
        
        sample_weight = _check_sample_weight(sample_weight, X.reshape((X.shape[0], -1)), dtype=X.dtype)
        if self.class_weight:
            sample_weight = sample_weight * compute_sample_weight(self.class_weight, y)
        return X, y, sample_weight, Y

    def predict(self, X):
        check_is_fitted(self, attributes=["_label_binarizer"])
        if self._label_binarizer.y_type_.startswith("multilabel"):
            # Threshold such that the negative label is -1 and positive label
            # is 1 to use the inverse transform of the label binarizer fitted
            # during fit
            if len(X.shape) < 2:
                X = X.reshape(X.shape + (1,))
            scores = 2 * (self.decision_function(X) > 0) - 1
            return self._label_binarizer.inverse_transform(scores)
        return super().predict(X)
    
    def decision_function(self, X):
        check_is_fitted(self)
        xp, _ = get_namespace(X)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        coef = self.coef_.flatten()
        scores = safe_sparse_dot(X.reshape((X.shape[0], coef.shape[0])), coef, dense_output=True) + self.intercept_
        try:
            return xp.reshape(scores, -1) if scores.shape[1] == 1 else scores
        except IndexError:
            return scores
            
    
    @property
    def classes_(self):
        """Classes labels."""
        return self._label_binarizer.classes_

    def _more_tags(self):
        return {"multilabel": True}


class ElasticRemursClassifier(_RemursClassifierMixin, _BaseElasticRemurs):

    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        class_weight=None,
        solver="auto",
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
        )
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight, Y = self._prepare_data(X, y, sample_weight, self.solver)

        super().fit(X, Y, sample_weight=sample_weight)
        return self

def _elastic_remurs_regression(
    tX: np.ndarray,
    y: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
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
    X = unfold(tX, size_tX, N + 1)
    Xty = X.T @ y
    num_features = X.shape[1]
    num_samples = X.shape[0]
    tV = np.array([np.zeros(dim) for _ in range(N)])
    tB = np.array([np.zeros(dim) for _ in range(N)])

    tW = np.zeros(dim)  # tensor W
    tU = np.zeros(dim)  # tensor U
    tA = np.zeros(dim)  # tensor A

    L, U = factor(X, rho)

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
            tV[n] = fold(
                prox_nuclear(
                    unfold(tW - tB[n], dim, n),
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

        tW = prox_l1((tSum_uv + tSum_ab) / (N + 1), beta / (N + 1) / rho) + prox_l2((tSum_uv + tSum_ab) / (N + 1), gamma / (N + 1) / rho)

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

