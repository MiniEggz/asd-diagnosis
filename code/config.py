"""
Configs for classification on resting-state fMRI of ABIDE.
"""

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "../data"
_C.DATASET.PIPELINE = "cpac"  # options: {‘cpac’, ‘css’, ‘dparsf’, ‘niak’}
_C.DATASET.ATLAS = "rois_cc200"
# options: {rois_aal, rois_cc200, rois_cc400, rois_dosenbach160, rois_ez, rois_ho, rois_tt}
_C.DATASET.SITE_IDS = None
# [
#    "NYU",
#    "UM_1",
#    "UCLA_1",
#    "USM",
# ]  # list of site ids to use, if None, use all sites
_C.DATASET.QC = True  # whether the data has been quality checked
# ---------------------------------------------------------------------------- #
# Machine learning pipeline
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.KERNEL = "rbf"
_C.MODEL.ALPHA_RANGE = [
    500,
]
_C.MODEL.BETA_RANGE = [
    1000,
]
_C.MODEL.GAMMA_RANGE = [1000]
_C.MODEL.RATIO = [
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
]
_C.MODEL.ESTIMATOR = "elastic_remurs"  # {"remurs", "elastic_remurs", "ridge", "svm", "logistic_l1", "logistic_l2", "logistic_elastic"}
# Vectorize should be set to true for all options other than remurs and elastic remurs
_C.MODEL.VECTORIZE = False
_C.MODEL.CONNECTIVITY = "correlation" # {"tp", "correlation"}
_C.MODEL.TEST_METHOD = "loo"  # {"loo", "k_folds"}
# only apply when TEST_METHOD is "k_folds"
_C.MODEL.NUM_TESTS = 2
_C.MODEL.NUM_FOLDS = 10
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "./outputs"  # output_dir
_C.OUTPUT.RESULTS_DIR = "results"  # don't add final /


def get_cfg_defaults():
    return _C.clone()
