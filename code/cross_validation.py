import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


# cross validation pipeline for multi-site data
def leave_one_out(x, y, covariates, estimator):
    """Leave one site out cross validation.

    Params:
        x: data for all sites.
        y: labels for all sites.
        covariates: all sites in the dataset.
        estimator: classifier used to estimate labels for test set.
    Returns:
        pd.DataFrame: results from leave on site out cross validation.
    """
    results = {"Target": [], "Num_samples": [], "Accuracy": []}
    unique_sites = np.unique(covariates)
    n_covariates = len(unique_sites)
    enc = OneHotEncoder(handle_unknown="ignore")
    covariate_mat = enc.fit_transform(covariates.reshape(-1, 1)).toarray()

    for tgt in unique_sites:
        idx_tgt = np.where(covariates == tgt)
        idx_src = np.where(covariates != tgt)
        x_tgt = x[idx_tgt]
        x_src = x[idx_src]
        y_tgt = y[idx_tgt]
        y_src = y[idx_src]

        # train
        estimator.fit(x_src, y_src)

        # test
        y_pred = estimator.predict(x_tgt)

        results["Accuracy"].append(accuracy_score(y_tgt, y_pred))
        results["Target"].append(tgt)
        results["Num_samples"].append(x_tgt.shape[0])

    # mean
    mean_acc = sum(
        [
            results["Num_samples"][i] * results["Accuracy"][i]
            for i in range(n_covariates)
        ]
    )
    mean_acc /= x.shape[0]

    # std
    squared_diffs = [
        results["Num_samples"][i] * (results["Accuracy"][i] - mean_acc) ** 2
        for i in range(n_covariates)
    ]

    variance = sum(squared_diffs) / x.shape[0]
    std = np.sqrt(variance)

    # append to results table
    results["Target"].append("Average")
    results["Num_samples"].append(x.shape[0])
    results["Accuracy"].append(mean_acc)

    results["Target"].append("Std")
    results["Num_samples"].append(x.shape[0])
    results["Accuracy"].append(std)

    return pd.DataFrame(results)


# cross validation pipeline for multi-site data
def k_folds(x, y, covariates, estimator, k=10):
    """Leave one site out cross validation.

    Params:
        x: data for all sites.
        y: labels for all sites.
        covariates: all sites in the dataset.
        estimator: classifier used to estimate labels for test set.
        k: number of folds.
    Returns:
        pd.DataFrame: results from leave on site out cross validation.
    """
    num_samples = len(covariates)
    idx_total = range(num_samples)

    results = {"Fold": [], "Num_samples": [], "Accuracy": []}

    num_test = int(len(idx_total) / k)

    for test_fold in range(k):
        start_idx_test = test_fold * num_test
        end_idx_test = (test_fold + 1) * num_test
        # handle edge case where too many for k equal folds
        if test_fold == k - 1:
            end_idx_test = num_samples
        idx_test = idx_total[start_idx_test:end_idx_test]
        idx_train = np.setdiff1d(idx_total, idx_test)
        x_test = x[idx_test]
        x_train = x[idx_train]
        y_test = y[idx_test]
        y_train = y[idx_train]

        estimator.fit(x_train, y_train)

        y_pred = estimator.predict(x_test)
        results["Accuracy"].append(accuracy_score(y_test, y_pred))
        results["Fold"].append(test_fold)
        results["Num_samples"].append(x_test.shape[0])

    # calc mean
    mean_acc = sum(
        [results["Num_samples"][i] * results["Accuracy"][i] for i in range(k)]
    )
    mean_acc /= x.shape[0]

    # calc std
    squared_diffs = [
        results["Num_samples"][i] * (results["Accuracy"][i] - mean_acc) ** 2
        for i in range(k)
    ]

    variance = sum(squared_diffs) / x.shape[0]
    std = np.sqrt(variance)

    # add avg accuracy
    results["Fold"].append("Average")
    results["Num_samples"].append(x.shape[0])
    results["Accuracy"].append(mean_acc)

    # add std
    results["Fold"].append("Std")
    results["Num_samples"].append(x.shape[0])
    results["Accuracy"].append(std)

    return pd.DataFrame(results)
