"""Pipeline for testing classification model performance on ABIDE dataset"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from config import get_cfg_defaults
from cross_validation import k_folds, leave_one_out
from elastic_remurs import ElasticRemursClassifier
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
from old_remurs import OldRemursClassifier
from pipeline_utils import save_results, set_target_key
from remurs import RemursClassifier
from results_handling import Best

# baseline with ridge classifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

VALID_ESTIMATOR_NAMES = [
    "remurs",
    "old_remurs",
    "elastic_remurs",
    "ridge",
    "svm",
    "logistic_l1",
    "logistic_l2",
    "logistic_elastic",
]
NO_BETA = ["ridge", "logistic_l1", "logistic_l2", "mpca"]
GAMMA_NEEDED = ["elastic_remurs"]


class AbideClassificationPipeline:
    def __init__(self):
        # load config
        self.cfg = get_cfg_defaults()
        self.cfg.freeze()

        # load abide data
        self.download_abide()
        pheno_info = self.read_pheno_info()
        self.brain_networks = self.extract_brain_networks(pheno_info)
        self.pheno = pheno_info.loc[self.use_idx, ["SITE_ID", "DX_GROUP"]].reset_index(
            drop=True
        )

    def download_abide(self):
        # add option for reload
        start_time = time.time()

        # site ids - need to handle if none
        site_ids = self.cfg.DATASET.SITE_IDS

        fetch_abide_pcp(
            data_dir=self.cfg.DATASET.ROOT,
            pipeline=self.cfg.DATASET.PIPELINE,
            band_pass_filtering=True,
            global_signal_regression=False,
            derivatives=self.cfg.DATASET.ATLAS,
            SITE_ID=site_ids,
            quality_checked=False,
            verbose=0,
        )
        self.abide_download_time = time.time() - start_time

    def read_pheno_info(self):
        pheno_file = os.path.join(
            self.cfg.DATASET.ROOT, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
        )
        return pd.read_csv(pheno_file, index_col=0)

    def extract_brain_networks(self, pheno_info):
        data_dir = os.path.join(
            self.cfg.DATASET.ROOT,
            f"ABIDE_pcp/{self.cfg.DATASET.PIPELINE}/filt_noglobal",
        )
        atlas = self.cfg.DATASET.ATLAS

        data_files = [
            os.path.join(data_dir, f"{pheno_info.loc[i, 'FILE_ID']}_{atlas}.1D")
            for i in pheno_info.index
        ]
        time_series = [
            np.loadtxt(data_path, skiprows=0)
            for data_path in data_files
            if os.path.exists(data_path)
        ]
        self.use_idx = [
            i for i, data_path in enumerate(data_files) if os.path.exists(data_path)
        ]  # You don't seem to be using this, though.

        if self.cfg.MODEL.CONNECTIVITY not in ["tp", "correlation"]:
            raise ValueError(
                "Connectivity measure config option invalid. Please use 'correlation' or 'tp'."
            )

        if self.cfg.MODEL.CONNECTIVITY == "tp":
            conn_measure = ConnectivityMeasure(kind="correlation")
            time_series = conn_measure.fit_transform(time_series)
            conn_measure = ConnectivityMeasure(
                kind="tangent", vectorize=self.cfg.MODEL.VECTORIZE
            )
        else:
            conn_measure = ConnectivityMeasure(
                kind="correlation", vectorize=self.cfg.MODEL.VECTORIZE
            )

        return conn_measure.fit_transform(time_series)

    def get_estimator(self, estimator_name, alpha, beta, gamma):
        if estimator_name == "remurs":
            return RemursClassifier(alpha=alpha, beta=beta)
        elif estimator_name == "elastic_remurs":
            return ElasticRemursClassifier(
                alpha=alpha, beta=beta, gamma=gamma
            )
        elif estimator_name == "ridge":
            return RidgeClassifier(alpha=alpha)
        elif estimator_name == "svm":
            return SVC(C=alpha, gamma=beta)
        elif estimator_name == "logistic_l1":
            return LogisticRegression(
                penalty="l1", C=alpha, solver="liblinear", max_iter=1000
            )
        elif estimator_name == "logistic_l2":
            return LogisticRegression(
                penalty="l2", C=alpha, solver="liblinear", max_iter=1000
            )
        elif estimator_name == "logistic_elastic":
            return LogisticRegression(
                penalty="elasticnet",
                C=alpha,
                l1_ratio=beta,
                solver="saga",
                max_iter=1000,
            )
        else:
            raise ValueError(f"{estimator_name} is an invalid estimator.")

    def run(self, estimator_name=None, display=True):
        start_time = time.time()
        print("Running cross validation...")

        alpha_range = self.cfg.MODEL.ALPHA_RANGE
        beta_range = self.cfg.MODEL.BETA_RANGE
        gamma_range = self.cfg.MODEL.BETA_RANGE
        if estimator_name is None:
            estimator_name = self.cfg.MODEL.ESTIMATOR

        test_method = self.cfg.MODEL.TEST_METHOD
        if test_method == "loo":
            test_nums = ["loo"]
        elif test_method == "k_folds":
            test_nums = range(self.cfg.MODEL.NUM_TESTS)
        else:
            raise ValueError(f"Test method '{test_method}' is invalid.")

        # set up for results
        best = Best()
        results_dict = {"alpha": [], "beta": [], "gamma": [], "avg_score": []}

        if estimator_name in NO_BETA:
            beta_range = [np.nan]
        elif estimator_name == "logistic_elastic" or estimator_name == "elastic_net":
            beta_range = cfg.MODEL.RATIO

        if estimator_name not in GAMMA_NEEDED:
            gamma_range = [np.nan]

        target_key = set_target_key(test_method)

        # running the experiment can also be pulled out into a separate function
        for alpha_val in alpha_range:
            for beta_val in beta_range:
                for gamma_val in gamma_range:
                    estimator = self.get_estimator(
                        estimator_name, alpha_val, beta_val, gamma_val
                    )

                    # run cross validaiton
                    if test_method == "k_folds":
                        res_df = k_folds(
                            self.brain_networks,
                            self.pheno["DX_GROUP"].values,
                            self.pheno["SITE_ID"].values,
                            estimator,
                        )
                    elif test_method == "loo":
                        res_df = leave_one_out(
                            self.brain_networks,
                            self.pheno["DX_GROUP"].values,
                            self.pheno["SITE_ID"].values,
                            estimator,
                        )
                    else:
                        raise ValueError(f"{test_method} is an invalid test method.")

                    if display:
                        print("Displaying findings...")
                        print(
                            f"Alpha: {alpha_val}, Beta: {beta_val}, Gamma: {gamma_val}"
                        )
                        print(res_df)

                    average_score = res_df[res_df[target_key] == "Average"][
                        "Accuracy"
                    ].values[0]

                    results_dict["alpha"].append(alpha_val)
                    results_dict["beta"].append(beta_val)
                    results_dict["gamma"].append(gamma_val)
                    results_dict["avg_score"].append(average_score)

                    if average_score > best.accuracy:
                        best.alpha = alpha_val
                        best.beta = beta_val
                        best.gamma = gamma_val
                        best.accuracy = average_score

                    if display:
                        print(best)

        # saving results can be a separate method
        self.results_df = pd.DataFrame(results_dict)
        save_results(self.results_df, estimator_name, self.cfg.OUTPUT.RESULTS_DIR)

        self.cv_runtime = time.time() - start_time
        print("Finished.")
        print(f"Time taken: {self.cv_runtime // 60}m {self.cv_runtime % 60}s")
