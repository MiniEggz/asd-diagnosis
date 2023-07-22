"""
This module provides helper functions for the pipeline in main.py.

The main functions are:
  - set_target_key: Determines the target key based on the specified test
                    method.
  - save_results: Saves the result of an estimator to a CSV file.
"""
import os
from datetime import datetime

import pandas as pd


def set_target_key(test_method):
    """Determines the target key based on the specified test method.

    Args:
        test_method (str): Method of testing. Expected values are 'loo' or any
                           other string.

    Returns:
        str: Returns "Target" if test_method is 'loo', otherwise returns "Fold".
    """
    if test_method == "loo":
        return "Target"
    else:
        return "Fold"


def save_results(results_df, estimator_name, results_dir):
    """Save the result of an estimator to a CSV file.

    Args:
        results_df (pd.DataFrame): Dataframe containing the results.
        estimator_name (str): Name of the estimator.
        results_dir (str): Directory where the results will be saved.
    """

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    results_df.to_csv(
        f"{results_dir}/{estimator_name}-{datetime.now().isoformat()}.csv"
    )
