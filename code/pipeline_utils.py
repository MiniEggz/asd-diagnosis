import pandas as pd
import os
from datetime import datetime

def set_target_key(test_method):
    if test_method == "loo":
        return "Target" 
    else:
        return "Fold"

def save_results(results_df, estimator_name, results_dir):
    """Save results to csv file."""
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    results_df.to_csv(
        f"{results_dir}/{estimator_name}-{datetime.now().isoformat()}.csv"
    )