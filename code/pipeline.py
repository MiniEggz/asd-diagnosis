"""
Pipeline for Testing Classification Model Performance on ABIDE Dataset.

This module provides a pipeline to facilitate the evaluation of classification 
models on the ABIDE dataset. It handles data fetching, preprocessing, and 
the execution of cross-validation tests with various classifiers.

Primary Features:
- Data fetching and preprocessing for the ABIDE dataset.
- Supports various estimators including Remurs, Elastic Remurs, Ridge, SVM, 
  and different types of Logistic Regressions.
- Offers k-fold and leave-one-out cross-validation methods.
- Results are saved to a specified output directory.

References:
This pipeline is adapted from the example provided at:
    https://github.com/pykale/pykale/tree/main/examples/multisite_neuroimg_adapt

Dependencies:
- Requires nilearn for neuroimage data handling.
- Uses sklearn for some classification models.

Classes:
- AbideClassificationPipeline: the main pipeline class that orchestrates the 
  testing and evaluation process.

Usage:
To use the pipeline, create an instance of the AbideClassificationPipeline class 
and call its run method, specifying the desired classifier and other parameters
in the config.py file.
"""
import os
import time

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp

# baseline with ridge classifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

from config import get_cfg_defaults
from cross_validation import k_folds, leave_one_out
from elastic_remurs import ElasticRemursClassifier
from pipeline_utils import save_results, set_target_key
from remurs import RemursClassifier
from results_handling import Best

VALID_ESTIMATOR_NAMES = [
    "remurs",
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
    """
    A pipeline for processing and classifying brain network data from the ABIDE dataset.

    This class facilitates the workflow for data downloading, preprocessing, and
    cross-validation of various classifiers on the ABIDE dataset. It is designed for
    flexibility in choosing different estimators and cross-validation methods.

    Attributes:
        verbose (bool): If True, the pipeline will print progress updates.
                cfg (CfgNode): configuration for the pipeline.
                abide_download_time (float): Time taken to load/download ABIDE dataset.
                brain_networks (np.ndarray): preprocessed brain network data from the ABIDE dataset.
                pheno (pd.DataFrame): phenotypic data from the ABIDE dataset.

    Methods:
        download_abide: Downloads the ABIDE dataset.
        read_pheno_info: Reads the phenotypic information of the ABIDE dataset.
        extract_brain_networks: Extracts brain networks from the dataset.
        get_estimator: Returns an estimator based on specified parameters.
        run: Executes the classification and cross-validation pipeline.
    """

    def __init__(self, verbose=False):
        """Initialise ABIDE classification pipeline.

        Args:
                verbose (bool): whether to print progress updates or not.
        """
        self.verbose = verbose

        # load config
        if self.verbose:
            print("Loading config.")
        self.cfg = get_cfg_defaults()
        self.cfg.freeze()

        # load abide data
        if self.verbose:
            print("(Down)Loading ABIDE dataset.")
        self.download_abide()

        # preprocess
        if self.verbose:
            print("Carrying out preprocessing steps.")
        pheno_info = self.read_pheno_info()
        self.brain_networks = self.extract_brain_networks(pheno_info)
        self.pheno = pheno_info.loc[self.use_idx, ["SITE_ID", "DX_GROUP"]].reset_index(
            drop=True
        )

    def download_abide(self):
        """Downloads the ABIDE dataset.

        Uses the nilearn library to fetch the ABIDE dataset based on
        parameters in config.
        """
        # add option for reload
        start_time = time.time()

        # site ids - need to handle if none
        site_ids = self.cfg.DATASET.SITE_IDS

        # use arg dict for compactness with site ids
        abide_args = {
            "data_dir": self.cfg.DATASET.ROOT,
            "pipeline": self.cfg.DATASET.PIPELINE,
            "band_pass_filtering": True,
            "global_signal_regression": False,
            "derivatives": self.cfg.DATASET.ATLAS,
            "quality_checked": self.cfg.DATASET.QC,
            "verbose": 0,
        }
        if site_ids is not None:
            abide_args["SITE_ID"] = site_ids
        if site_ids is None:
            kwargs = {}
        else:
            kwargs = {"SITE_ID": site_ids}

        fetch_abide_pcp(**abide_args)

        self.abide_download_time = time.time() - start_time

    def read_pheno_info(self):
        """Reads the phenotypic information of the ABIDE dataset.

        Returns:
            pd.DataFrame: DataFrame containing phenotypic information.
        """
        pheno_file = os.path.join(
            self.cfg.DATASET.ROOT, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
        )
        return pd.read_csv(pheno_file, index_col=0)

    def extract_brain_networks(self, pheno_info):
        """Extracts brain networks based on specified atlas and other configs.

        Args:
            pheno_info (pd.DataFrame): DataFrame containing phenotypic information.

        Returns:
            np.ndarray: Array containing extracted brain networks.
        """
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
        ]

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
        """Get estimator object for given name and alpha/beta/gamma.

        Args:
            estimator_name (str): Name of the estimator.
            alpha (float): Alpha for the estimator.
            beta (float): Beta for the estimator (not all estimators use this).
            gamma (float): Parameter for the estimator (specific to some estimators).

        Returns:
            Estimator object compliant with core functionalities of sklearn estimators.
        """
        if estimator_name == "remurs":
            return RemursClassifier(alpha=alpha, beta=beta, flatten_input=False)
        elif estimator_name == "elastic_remurs":
            return ElasticRemursClassifier(alpha=alpha, beta=beta, gamma=gamma, flatten_input=False)
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
        """Executes the classification and cross-validation pipeline.

        Runs the pipeline after all initialisation and setup is complete.

        Args:
            estimator_name (str, optional): The name of the estimator. If None,
            will use the default from the configuration.

            display (bool): Whether to print the results. Default is True.
        """
        start_time = time.time()
        print("Running cross validation...")

        alpha_range = self.cfg.MODEL.ALPHA_RANGE
        beta_range = self.cfg.MODEL.BETA_RANGE
        gamma_range = self.cfg.MODEL.GAMMA_RANGE
        if estimator_name is None:
            estimator_name = self.cfg.MODEL.ESTIMATOR

        VALID_TEST_METHODS = ["loo", "k_folds"]
        test_method = self.cfg.MODEL.TEST_METHOD

        if test_method not in VALID_TEST_METHODS:
            raise ValueError(f"Test method '{test_method}' is invalid.")

        # set up for results
        best = Best()
        results_dict = {"alpha": [], "beta": [], "gamma": [], "avg_score": []}

        if estimator_name in NO_BETA:
            beta_range = [np.nan]
        elif estimator_name == "logistic_elastic" or estimator_name == "elastic_net":
            beta_range = self.cfg.MODEL.RATIO

        if estimator_name not in GAMMA_NEEDED:
            gamma_range = [np.nan]

        target_key = set_target_key(test_method)

        # potentially move to separate function
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

        self.results_df = pd.DataFrame(results_dict)
        save_results(self.results_df, estimator_name, self.cfg.OUTPUT.RESULTS_DIR)

        self.cv_runtime = time.time() - start_time
