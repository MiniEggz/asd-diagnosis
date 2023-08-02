import os
import time

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp

from config import get_cfg_defaults
from results_handling import Best
from datetime import datetime

from cross_validation import leave_one_out, k_folds
from pipeline_utils import save_results, set_target_key

# baseline with ridge classifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

from elastic_remurs import ElasticRemursClassifier
from remurs import RemursClassifier

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
    
target_key = set_target_key(test_method)

# running the experiment can also be pulled out into a separate function
for alpha_val in alpha_range:
    for beta_val in beta_range:
        for gamma_val in gamma_range:
            print(f"Alpha: {alpha_val}, Beta: {beta_val}, Gamma: {gamma_val}")

            # init estimator
            if estimator_name == "remurs":
                estimator = RemursClassifier(
                    alpha=alpha_val, beta=beta_val
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

            # run cross validaiton
            if test_method == "k_folds":
                res_df = k_folds(
                    brain_networks,
                    pheno["DX_GROUP"].values,
                    pheno["SITE_ID"].values,
                    estimator,
                )
            elif test_method == "loo":
                res_df = leave_one_out(
                    brain_networks,
                    pheno["DX_GROUP"].values,
                    pheno["SITE_ID"].values,
                    estimator,
                )

            print("Displaying findings...")
            print(f"Alpha: {alpha_val}, Beta: {beta_val}, Gamma: {gamma_val}")
            print(res_df)

            average_score = res_df[res_df[target_key] == "Average"] ["Accuracy"].values[0]

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

save_results(results_df, estimator_name, cfg.OUTPUT.RESULTS_DIR)

print("Finished.")
time_taken = time.time() - start_time
print(f"Time taken: {time_taken // 60}m {time_taken % 60}s")

# want to try more alphas around the 100 mark, lower beta values
# need to try on bigger datasets

# need to try on single site data too to see how remurs performs against other methods on that...
# would likely show whether the issue is the cross domain adaptation#
