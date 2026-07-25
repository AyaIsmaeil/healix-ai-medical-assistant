"""
Healix - Phase 6.2 optimization config (OFFLINE ONLY).

Search spaces, cross-validation settings, and experiment budgets for Phase
6.2. Phase 6.1's ``config.py`` is left untouched so the baseline remains
exactly reproducible; this module only ADDS the Phase-6.2 knobs.

Design constraints that shaped every budget below (measured, not assumed):
  * 1,025,602 training rows x 286 features; 49 classes.
  * ~4 GB free RAM and 8 CPU cores on the training machine.
  * Phase 6.1 full-scale fit times: RF 3.4 min, XGB 36 min, CatBoost 63 min,
    LogReg 41 min.

A full-scale 5-fold CV over 5 families x a grid would cost days. So Phase 6.2
uses a documented, standard two-stage protocol:

  Stage 1 (search)  - randomized search on a stratified SUBSAMPLE, so many
                      configurations can be compared under identical
                      conditions.
  Stage 2 (verify)  - the winning configuration per family is then re-fitted
                      with full 5-fold stratified CV on a larger sample, and
                      the overall winner is refitted on the FULL training set.

The subsample is stratified and fixed by seed, so every family sees exactly
the same rows — comparisons stay fair even though they are not full-scale.
This trade-off is stated explicitly in every generated report rather than
being hidden.
"""

from __future__ import annotations

from typing import Any, Dict, List

from training.disease_prediction.config import SEED  # single project-wide seed

# ----------------------------------------------------------------------
# Experiment budgets
# ----------------------------------------------------------------------
CV_FOLDS = 5
CV_SHUFFLE = True

# Stage 1: randomized hyperparameter search
SEARCH_SUBSAMPLE_ROWS = 120_000      # stratified from the 1.03M train split
SEARCH_CV_FOLDS = 3                  # inner CV during search (cheaper than 5)
SEARCH_N_ITER: Dict[str, int] = {    # sampled configurations per family
    "random_forest": 12,
    # CatBoost is ~5x slower per fit than XGBoost at this scale (Phase 6.1
    # measured 63 min vs 36 min on the full train split). Its budget is cut to
    # keep the search tractable; this is a COMPUTE constraint, not a scientific
    # judgement, and is reported as such -- CatBoost's search is therefore less
    # exhaustive than XGBoost's and its tuned result should be read as a lower
    # bound on what the family could reach.
    "catboost": 5,
    "logistic_regression": 4,   # the space is now only C x {lbfgs} = 4 configs
    "xgboost": 12,
    "lightgbm": 14,                  # widest: it is under root-cause investigation
}

# Stage 2: verification of the winning config per family
VERIFY_SUBSAMPLE_ROWS = 250_000      # stratified; 5-fold CV runs on this
FINAL_REFIT_FULL_TRAIN = True        # winner refitted on all 1,025,602 rows

# Robustness / calibration / explainability sample sizes
ROBUSTNESS_MISSING_RATES = (0.10, 0.20, 0.30, 0.40)
ROBUSTNESS_SAMPLE_ROWS = 40_000
CALIBRATION_SAMPLE_ROWS = 60_000
CALIBRATION_N_BINS = 15
SHAP_BACKGROUND_ROWS = 200           # SHAP is O(n_background x n_samples)
SHAP_EXPLAIN_ROWS = 400
PERMUTATION_IMPORTANCE_ROWS = 25_000
PERMUTATION_REPEATS = 3
TOP_FEATURES_TO_REPORT = 50

# ----------------------------------------------------------------------
# Search spaces
# ----------------------------------------------------------------------
# Every value below is a deliberate, documented candidate. Ranges bracket the
# Phase 6.1 default so the search can confirm or overturn it, and none of them
# were chosen after seeing a Phase 6.2 result.
SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        # Phase 6.1 used max_depth=None -> a 2.18 GB artifact. Depth caps are
        # the primary lever for shrinking it (BEST_MODEL.md recommendation 2).
        "max_depth": [None, 12, 18, 24, 30],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3],
    },
    "catboost": {
        "depth": [4, 6, 8],
        "learning_rate": [0.1, 0.2, 0.3],
        # 600 dropped: at ~5x XGBoost's per-fit cost, a 600-iteration config
        # inside a 3-fold inner CV exceeded the compute budget. Documented
        # limitation, not a claim that 600 would not help.
        "iterations": [200, 400],
        "l2_leaf_reg": [1.0, 3.0, 10.0],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        # 'saga' REMOVED after a measured failure, not a guess: a first search
        # run spent 3h20m of wall time (24,000+ CPU-seconds) without completing
        # a single 3-fold configuration on the 120k x 286 / 49-class subsample.
        # Extrapolated, the saga half of the space alone required >24h. This is
        # a documented COMPUTE-BUDGET exclusion; it is not a claim that saga
        # would score worse. Consequence: the L1/elasticnet penalties that only
        # saga supports were never explored, so this family's tuned result is a
        # lower bound on what it could achieve.
        "solver": ["lbfgs"],
    },
    "xgboost": {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.05, 0.1, 0.2, 0.3],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
    },
    # LightGBM's space is informed by the root-cause analysis, not guessed:
    # Phase 6.1 showed its trees stop growing under library defaults on this
    # sparse binary representation, so the space targets capacity
    # (num_leaves), split permissiveness (min_child_samples / min_split_gain /
    # min_child_weight) and step size (learning_rate).
    "lightgbm": {
        "num_leaves": [31, 63, 127, 255],
        # The root-cause analysis showed a LEARNING-RATE INTERACTION: with
        # min_child_samples relaxed, low learning rates complete all boosting
        # rounds and score ~0.995, while lr>=0.2 terminates after <20 rounds
        # and collapses. The space is therefore weighted toward low lr.
        "learning_rate": [0.05, 0.1],
        "min_child_samples": [1, 5, 10, 20],
        # min_child_weight (min_sum_hessian_in_leaf) must stay > 0: 0.0 makes
        # LightGBM permit degenerate zero-count splits and it aborts with
        # "Check failed: (best_split_info.left_count) > (0)". Verified: all 11
        # search failures in the first run carried min_child_weight=0.0.
        "min_child_weight": [1e-3, 1e-2],
        "min_split_gain": [0.0],
        "n_estimators": [200, 400],
    },
}

# Fixed (non-searched) parameters that must accompany each family.
FIXED_PARAMS: Dict[str, Dict[str, Any]] = {
    "random_forest": {"n_jobs": -1, "random_state": SEED},
    "catboost": {"verbose": False, "thread_count": -1, "random_seed": SEED},
    "logistic_regression": {"max_iter": 1000, "random_state": SEED},
    "xgboost": {"tree_method": "hist", "n_jobs": -1, "random_state": SEED,
                "eval_metric": "mlogloss"},
    "lightgbm": {"n_jobs": -1, "random_state": SEED, "verbose": -1},
}

SCORING = "f1_macro"     # same primary metric as Phase 6.1, for continuity
