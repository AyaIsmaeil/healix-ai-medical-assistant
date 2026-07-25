"""
Healix - Disease Prediction model factory (Phase 6.1).

Builds one UNFITTED estimator per model family. Every hyperparameter comes
from ``config.MODEL_DEFAULTS`` (fixed once, documented, not tuned — see
config.py's module docstring).

Two families need help with missing values because Feature Schema v2 is
null-honest (tri-state booleans, null-propagating one-hot) and roughly a
quarter of the descriptor columns are null for any given row:

  * Logistic Regression cannot accept NaN at all -> wrapped in a Pipeline
    with a constant-0 imputer (documented explicitly as a baseline-only
    simplification, see ModelSpec.imputation_note).
  * Random Forest (this sklearn version), XGBoost, LightGBM, and CatBoost
    all handle NaN natively and receive the raw null-preserving matrix
    unchanged — exactly what the schema's design intends the missingness
    signal to be used for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from training.disease_prediction.config import LOGS_DIR, MODEL_DEFAULTS, SEED


@dataclass
class ModelSpec:
    name: str
    display_name: str
    build: Callable[[], Any]
    handles_native_nan: bool
    imputation_note: Optional[str] = None


def _logistic_regression() -> Pipeline:
    params = dict(MODEL_DEFAULTS["logistic_regression"])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("classifier", LogisticRegression(random_state=SEED, **params)),
    ])


def _random_forest() -> RandomForestClassifier:
    params = dict(MODEL_DEFAULTS["random_forest"])
    return RandomForestClassifier(random_state=SEED, **params)


def _xgboost():
    from xgboost import XGBClassifier
    params = dict(MODEL_DEFAULTS["xgboost"])
    return XGBClassifier(random_state=SEED, **params)


def _lightgbm():
    from lightgbm import LGBMClassifier
    params = dict(MODEL_DEFAULTS["lightgbm"])
    return LGBMClassifier(random_state=SEED, **params)


def _catboost():
    from catboost import CatBoostClassifier
    params = dict(MODEL_DEFAULTS["catboost"])
    # CatBoost writes its own training logs to CWD (catboost_info/) unless
    # told otherwise — keep repo root clean by redirecting into outputs/logs.
    train_dir = str(LOGS_DIR / "catboost_info")
    return CatBoostClassifier(random_seed=SEED, train_dir=train_dir, **params)


def model_registry() -> dict:
    """Ordered mapping of model-family name -> ModelSpec (unfitted builders)."""
    return {
        "logistic_regression": ModelSpec(
            name="logistic_regression", display_name="Logistic Regression",
            build=_logistic_regression, handles_native_nan=False,
            imputation_note=(
                "Cannot accept NaN. Missing values are filled with a "
                "constant 0 before fitting — a baseline-only simplification "
                "(0 is the majority class for almost every tri-state "
                "boolean/one-hot column), not a claim that 'unknown' truly "
                "means 'absent'. Tree-ensemble models below receive the raw "
                "null-preserving matrix instead."),
        ),
        "random_forest": ModelSpec(
            name="random_forest", display_name="Random Forest",
            build=_random_forest, handles_native_nan=True,
        ),
        "xgboost": ModelSpec(
            name="xgboost", display_name="XGBoost",
            build=_xgboost, handles_native_nan=True,
        ),
        "lightgbm": ModelSpec(
            name="lightgbm", display_name="LightGBM",
            build=_lightgbm, handles_native_nan=True,
        ),
        "catboost": ModelSpec(
            name="catboost", display_name="CatBoost",
            build=_catboost, handles_native_nan=True,
        ),
    }
