"""
Default configurations for classification on resting-state fMRI of ABIDE
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
_C.DATASET.SITE_IDS = ['NYU', "UM_1", "UCLA_1", "USM"] # list of site ids to use, if None, use all sites
# ---------------------------------------------------------------------------- #
# Machine learning pipeline
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.KERNEL = "rbf"
# _C.MODEL.ALPHA_RANGE = [
#     # 10**-3,
#     # 5 * 10**-3,
#     # 10**-2,
#     # 5 * 10**-2,
#     # 10**-1,
#     # 5 * 10**-1,
#     # 10**0,
#     5 * 10**0,
#     10**1,
#     5 * 10**1,
#     7.5 * 10**1,
#     10**2,
#     5 * 10**2,
#     10 ** 3,
#     5 * 10**3,
# ]
_C.MODEL.ALPHA_RANGE = [0.001, 0, 1, 500, 1000]
# _C.MODEL.BETA_RANGE = [
#     1 * 10**-4,
#     5 * 10**-4,
#     10**-3,
#     5 * 10**-3,
#     10**-2,
#     5 * 10**-2,
#     10**-1,
#     5 * 10**-1,
#     10**0,
#     5 * 10**0,
#     10**1,
#     5 * 10**1,
#     10**2,
#     5 * 10**2,
# ]
# _C.MODEL.BETA_RANGE = [0.0001, 0.001, 0.01, 0.1]
_C.MODEL.BETA_RANGE = [0.1, 5, 100, 1000]
_C.MODEL.GAMMA_RANGE = [0.1, 1, 100, 1000]
_C.MODEL.RATIO = [
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
]
_C.MODEL.ESTIMATOR = "elastic_remurs" # {"remurs", "elastic_remurs", "ridge", "svm", "logistic_l1", "logistic_l2", "logistic_elastic"}
_C.MODEL.VECTORIZE = False
_C.MODEL.CONNECTIVITY = "tp"
_C.MODEL.TEST_METHOD = "k_folds"
_C.MODEL.TESTCODE = "_tangent_pearson_k"
_C.MODEL.NUM_TESTS = 2
_C.MODEL.NUM_FOLDS = 5
_C.MODEL.ALPHA = 500.0
_C.MODEL.BETA = 0.005
_C.MODEL.LAMBDA_ = 1.0
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "./outputs"  # output_dir
_C.OUTPUT.RESULTS_DIR = "results/non_quality_checked_all_sites" # don't add final /
#_C.OUTPUT.RESULTS_DIR = "results/quality_checked_all_sites"


def get_cfg_defaults():
    return _C.clone()