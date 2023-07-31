import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from config import get_cfg_defaults
from results_handling import Best
from datetime import datetime

# config
# cfg_path = "configs/tutorial.yaml"  # Path to `.yaml` config file

cfg = get_cfg_defaults()
# cfg.merge_from_file(cfg_path)
cfg.freeze()
print("CONFIG:")
print(cfg)

# data preparation
print("Preparing data...")
data_start_time = time.time()
root_dir = cfg.DATASET.ROOT
pipeline = cfg.DATASET.PIPELINE  # fmri pre-processing pipeline
atlas = cfg.DATASET.ATLAS
site_ids = cfg.DATASET.SITE_IDS
abide = fetch_abide_pcp(
    data_dir=root_dir,
    pipeline=pipeline,
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=atlas,
    SITE_ID=site_ids,
    quality_checked=False,
    verbose=0,
)
download_time = time.time() - data_start_time
print(f"Time taken: {download_time // 60}m {download_time % 60}s")

# read phenotypic data
print("Reading phenotypic data...")
pheno_file = os.path.join(
    cfg.DATASET.ROOT, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
)
pheno_info = pd.read_csv(pheno_file, index_col=0)

# read timeseries from file
print("Reading timeseries...")
data_dir = os.path.join(root_dir, "ABIDE_pcp/%s/filt_noglobal" % pipeline)
use_idx = []
time_series = []
for i in pheno_info.index:
    data_file_name = "%s_%s.1D" % (pheno_info.loc[i, "FILE_ID"], atlas)
    data_path = os.path.join(data_dir, data_file_name)
    if os.path.exists(data_path):
        time_series.append(np.loadtxt(data_path, skiprows=0))
        use_idx.append(i)

pheno = pheno_info.loc[use_idx, ["SITE_ID", "DX_GROUP"]].reset_index(drop=True)

# extracting brain networks features
print("Extracting brain networks features...")
if cfg.MODEL.CONNECTIVITY == "tp":
    conn_measure = ConnectivityMeasure(kind="correlation")
    time_series = conn_measure.fit_transform(time_series)
    conn_measure = ConnectivityMeasure(kind="tangent", vectorize=cfg.MODEL.VECTORIZE)
elif cfg.MODEL.CONNECTIVITY == "correlation":
    conn_measure = ConnectivityMeasure(
        kind="correlation", vectorize=cfg.MODEL.VECTORIZE
    )
else:
    raise ValueError(
        "Connectivity measure config option invalid. Please use 'correlation' or 'tp'."
    )
# correlation_measure = ConnectivityMeasure(kind="correlation", vectorize=cfg.MODEL.VECTORIZE)
brain_networks = conn_measure.fit_transform(time_series)

print(f"Number of samples: {brain_networks.shape[0]}")
print(f"Shape of brain networks: {brain_networks.shape}")


# cross validation pipeline for multi-site data
def cross_validation(x, y, covariates, estimator, domain_adaptation=False):
    results = {"Target": [], "Num_samples": [], "Accuracy": []}
    unique_covariates = np.unique(covariates)
    n_covariates = len(unique_covariates)
    enc = OneHotEncoder(handle_unknown="ignore")
    covariate_mat = enc.fit_transform(covariates.reshape(-1, 1)).toarray()

    for tgt in unique_covariates:
        idx_tgt = np.where(covariates == tgt)
        idx_src = np.where(covariates != tgt)
        x_tgt = brain_networks[idx_tgt]
        x_src = brain_networks[idx_src]
        y_tgt = y[idx_tgt]
        y_src = y[idx_src]

        if domain_adaptation:
            estimator.fit(
                np.concatenate((x_src, x_tgt)),
                y_src,
                np.concatenate((covariate_mat[idx_src], covariate_mat[idx_tgt])),
            )
        else:
            estimator.fit(x_src, y_src)

        y_pred = estimator.predict(x_tgt)
        results["Accuracy"].append(accuracy_score(y_tgt, y_pred))
        results["Target"].append(tgt)
        results["Num_samples"].append(x_tgt.shape[0])

    mean_acc = sum(
        [
            results["Num_samples"][i] * results["Accuracy"][i]
            for i in range(n_covariates)
        ]
    )
    mean_acc /= x.shape[0]

    # calculate squared differences for std
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
def k_fold_cross_validation(x, y, covariates, estimator, k=10):
    num_samples = len(covariates)
    idx_total = range(num_samples)

    results = {"Fold": [], "Num_samples": [], "Accuracy": []}

    num_test = int(len(idx_total) / k)

    for test_fold in range(k):
        start_idx_test = test_fold * num_test
        end_idx_test = (test_fold + 1) * num_test
        if test_fold == k - 1:
            end_idx_test = num_samples
        idx_test = idx_total[start_idx_test:end_idx_test]
        idx_train = np.setdiff1d(idx_total, idx_test)
        x_test = brain_networks[idx_test]
        x_train = brain_networks[idx_train]
        y_test = y[idx_test]
        y_train = y[idx_train]

        estimator.fit(x_train, y_train)

        y_pred = estimator.predict(x_test)
        results["Accuracy"].append(accuracy_score(y_test, y_pred))
        results["Fold"].append(test_fold)
        results["Num_samples"].append(x_test.shape[0])

    # TODO: handle if the folds don't match... do a final fold with remaining, or validate number of folds

    mean_acc = sum(
        [results["Num_samples"][i] * results["Accuracy"][i] for i in range(k)]
    )
    mean_acc /= x.shape[0]

    # calculate squared differences
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


# baseline with ridge classifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

from elastic_remurs import ElasticRemursClassifier
from remurs import RemursClassifier

print("Running cross-validation...")
start_time = time.time()
alpha_range = cfg.MODEL.ALPHA_RANGE
beta_range = cfg.MODEL.BETA_RANGE
gamma_range = cfg.MODEL.GAMMA_RANGE
estimator_name = cfg.MODEL.ESTIMATOR
test_method = cfg.MODEL.TEST_METHOD

if test_method == "loo":
    test_nums = ["loo"]
elif test_method == "k_folds":
    test_nums = range(cfg.MODEL.NUM_TESTS)
else:
    raise ValueError(f"Test method '{test_method}' is invalid.")

best = Best()
# set up results dataframe
results_dict = {"alpha": [], "beta": [], "gamma": [], "avg_score": []}


# can be pulled out into separate function -- to set alpha, beta, gamma
NO_BETA = ["ridge", "logistic_l1", "logistic_l2", "mpca"]
GAMMA_NEEDED = ["elastic_remurs"]

if estimator_name in NO_BETA:
    beta_range = [""]
elif estimator_name == "logistic_elastic" or estimator_name == "elastic_net":
    beta_range = cfg.MODEL.RATIO

if estimator_name not in GAMMA_NEEDED:
    gamma_range = [""]

if estimator_name == "mpca":
    alpha_range = [""]


# running the experiment can also be pulled out into a separate function
for alpha_val in alpha_range:
    for beta_val in beta_range:
        for gamma_val in gamma_range:
            print(f"Alpha: {alpha_val}, Beta: {beta_val}, Gamma: {gamma_val}")

            if estimator_name == "remurs":
                estimator = RemursClassifier(
                    alpha=alpha_val, beta=beta_val, fit_intercept=True
                )
            elif estimator_name == "elastic_remurs":
                estimator = ElasticRemursClassifier(
                    alpha=alpha_val, beta=beta_val, gamma=gamma_val
                )
            elif estimator_name == "ridge":
                estimator = RidgeClassifier(alpha=alpha_val)
            elif estimator_name == "svm":
                estimator = SVC(C=alpha_val, gamma=beta_val)
            elif estimator_name == "logistic_l1":
                estimator = LogisticRegression(
                    penalty="l1", C=alpha_val, solver="liblinear", max_iter=1000
                )
            elif estimator_name == "logistic_l2":
                estimator = LogisticRegression(
                    penalty="l2", C=alpha_val, solver="liblinear", max_iter=1000
                )
            elif estimator_name == "logistic_elastic":
                estimator = LogisticRegression(
                    penalty="elasticnet",
                    C=alpha_val,
                    l1_ratio=beta_val,
                    solver="saga",
                    max_iter=1000,
                )
            # estimator = Lasso(alpha=alpha_val)
            # estimator = LogisticRegression(penalty="l2", C=alpha_val, solver="saga", max_iter=1000)

            # can make validation method more abstract
            if test_method == "k_folds":
                res_df = k_fold_cross_validation(
                    brain_networks,
                    pheno["DX_GROUP"].values,
                    pheno["SITE_ID"].values,
                    estimator,
                )
            elif test_method == "loo":
                res_df = cross_validation(
                    brain_networks,
                    pheno["DX_GROUP"].values,
                    pheno["SITE_ID"].values,
                    estimator,
                )

            # displaying can be separate function
            print("Displaying findings...")
            print(f"Alpha: {alpha_val}, Beta: {beta_val}")
            print(res_df)
            if test_method == "loo":
                average_score = res_df[res_df["Target"] == "Average"][
                    "Accuracy"
                ].values[0]
            elif test_method == "k_folds":
                average_score = res_df[res_df["Fold"] == "Average"]["Accuracy"].values[
                    0
                ]
            elif test_method == "n_k_folds":
                average_score = res_df[res_df["fold"] == "average"]["accuracy"].values[
                    0
                ]
            results_dict["alpha"].append(alpha_val)
            results_dict["beta"].append(beta_val)
            results_dict["gamma"].append(gamma_val)
            results_dict["avg_score"].append(average_score)
            if average_score > best.accuracy:
                best.alpha = alpha_val
                best.beta = beta_val
                best.gamma = gamma_val
                best.accuracy = average_score
            print(best)

# saving results can be a separate method
results_df = pd.DataFrame(results_dict)

# results in the form "results/{k_folds/loo}_{regression method}_{test num}_{OPTIONAL quality checked}_{sites}"
if cfg.MODEL.VECTORIZE:
    vectorized_string = "vectorized"
else:
    vectorized_string = "tensor"

if not os.path.exists(cfg.OUTPUT.RESULTS_DIR):
    os.mkdir(cfg.OUTPUT.RESULTS_DIR)

results_df.to_csv(
    f"{cfg.OUTPUT.RESULTS_DIR}/{estimator_name}-{datetime.now().isoformat()}.csv"
)

print("Finished.")
time_taken = time.time() - start_time
print(f"Time taken: {time_taken // 60}m {time_taken % 60}s")

# want to try more alphas around the 100 mark, lower beta values
# need to try on bigger datasets

# need to try on single site data too to see how remurs performs against other methods on that...
# would likely show whether the issue is the cross domain adaptation#
