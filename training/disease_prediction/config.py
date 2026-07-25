"""
Healix - Disease Prediction training config (Phase 6.1).

Every hyperparameter below is a FIXED default, chosen once before looking at
any validation metric. None of it was searched, tuned, or selected based on
performance. The only deviations from a library's literal default are:

  * ``N_ESTIMATORS = 200`` applied uniformly to every tree-ensemble family
    (RandomForest, XGBoost, LightGBM, CatBoost) instead of each library's own
    default (100 / 100 / 100 / 1000 respectively). This single constant was
    fixed BEFORE training to bound baseline runtime on 1.03M rows — it is a
    practicality adjustment, not hyperparameter optimization (no search, no
    comparison of candidate values, no validation-driven selection).
  * ``LogisticRegression(max_iter=1000)`` — raised from sklearn's default 100
    because multinomial logistic regression on ~1M rows / 49 classes needs
    more iterations to reach the solver's convergence tolerance; this affects
    whether the optimizer *finishes*, not what it optimizes for.

Hyperparameter search, cross-validation, and threshold calibration are
explicitly out of scope for Phase 6.1 (see docs/research/MODEL_TRAINING_REPORT.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent

DATASET_ROOT = REPO_ROOT / "app" / "data" / "processed" / "ddxplus"
EXPECTED_SCHEMA_VERSION = "assessment-features-v2"
EXPECTED_ONTOLOGY_VERSION = "healix-ontology-v1.0.0"

OUTPUT_DIR = PACKAGE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORTS_DIR = OUTPUT_DIR / "reports"

DOCS_RESEARCH_DIR = REPO_ROOT / "docs" / "research"

# Single seed reused from the Phase-5 Dataset Builder (docs/research
# /DATASET_BUILDER_DESIGN.md §8.3) so the whole pipeline — dataset build and
# model training — is anchored to one documented, reproducible value.
SEED = 20260721

# Columns present in every split's Parquet file that are labels/QC metadata,
# never features. The feature matrix is everything in feature_order (286
# columns from feature_dictionary.json), nothing more, nothing less.
LABEL_AND_QC_COLUMNS = (
    "y_disease", "y_disease_icd10", "y_urgency_prior", "y_specialty",
    "y_differential_ids", "y_differential_probs", "differential_length",
    "differential_top_id", "differential_top_prob", "truth_in_differential",
    "content_hash", "duplicate_group_id", "leakage_flag", "row_index",
)

TARGET_COLUMN = "y_disease"
LEAKAGE_FLAG_COLUMN = "leakage_flag"

# Model families in the order they are trained and reported.
MODEL_FAMILIES = ("logistic_regression", "random_forest", "xgboost",
                  "lightgbm", "catboost")

N_ESTIMATORS = 200   # fixed uniformly across every tree ensemble — see module docstring

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        # sklearn defaults kept as-is except max_iter (see module docstring).
        "solver": "lbfgs",
        "max_iter": 1000,
        "n_jobs": None,          # lbfgs multinomial does not parallelize via n_jobs
    },
    "random_forest": {
        # sklearn defaults kept as-is except n_estimators (see module docstring).
        "n_estimators": N_ESTIMATORS,
        "n_jobs": -1,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": None,
    },
    "xgboost": {
        # XGBoost defaults kept as-is (max_depth=6, learning_rate=0.3, ...)
        # except n_estimators (see module docstring).
        "n_estimators": N_ESTIMATORS,
        "tree_method": "hist",     # histogram method: memory-efficient at this scale
        "n_jobs": -1,
        "eval_metric": "mlogloss",
    },
    "lightgbm": {
        # LightGBM defaults kept as-is (num_leaves=31, learning_rate=0.1, ...)
        # except n_estimators (see module docstring).
        "n_estimators": N_ESTIMATORS,
        "n_jobs": -1,
        "verbose": -1,
    },
    "catboost": {
        # CatBoost's own default (iterations=1000) is replaced by the same
        # uniform N_ESTIMATORS as every other tree ensemble (see docstring).
        "iterations": N_ESTIMATORS,
        "verbose": False,
        "thread_count": -1,
    },
}

# Evaluation
TOP_K_VALUES = (3, 5)
PRIMARY_SELECTION_METRIC = "f1_macro"          # on the leakage-free validation subset
PRIMARY_SELECTION_SPLIT = "validation_clean"   # see docs/research/CROSS_SPLIT_LEAKAGE_REPORT.md


def resolve_dataset_dir(explicit: Optional[Path] = None) -> Path:
    """Locate the single Phase-5 dataset directory under ``DATASET_ROOT``.

    Verifies the manifest's schema/ontology versions match what this training
    package was built against, so a schema drift fails loudly instead of
    silently training on the wrong feature contract.
    """
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir()) \
            if DATASET_ROOT.is_dir() else []

    if not candidates:
        raise FileNotFoundError(
            f"No dataset found under {DATASET_ROOT}. Run the Phase-5 "
            f"Dataset Builder (app.ml.dataset_builder) first.")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple dataset versions found under {DATASET_ROOT}: "
            f"{[c.name for c in candidates]}. Pass an explicit path.")

    dataset_dir = candidates[0]
    metadata = json.loads((dataset_dir / "metadata" / "metadata.json")
                         .read_text(encoding="utf-8"))
    if metadata["feature_schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema mismatch: dataset was built with "
            f"{metadata['feature_schema_version']!r}, training expects "
            f"{EXPECTED_SCHEMA_VERSION!r}.")
    if metadata["ontology_version"] != EXPECTED_ONTOLOGY_VERSION:
        raise RuntimeError(
            f"Ontology mismatch: dataset was built with "
            f"{metadata['ontology_version']!r}, training expects "
            f"{EXPECTED_ONTOLOGY_VERSION!r}.")
    return dataset_dir
