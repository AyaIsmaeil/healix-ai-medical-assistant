"""Tests for DDXPlus class-imbalance handling in Phase 6.1 training."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from training.disease_prediction import config
from training.disease_prediction.train import _class_balancing_meta, _fit_one_model
from training.disease_prediction.utils import balanced_sample_weights


def test_config_enables_balanced_class_weights():
    assert config.USE_BALANCED_CLASS_WEIGHTS is True
    assert config.MODEL_DEFAULTS["logistic_regression"]["class_weight"] == "balanced"
    assert config.MODEL_DEFAULTS["random_forest"]["class_weight"] == "balanced"
    assert "xgboost" in config.SAMPLE_WEIGHT_AT_FIT_MODELS


def test_balanced_sample_weights_upweights_rare_class():
    y = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    weights = balanced_sample_weights(y)

    assert weights.shape == y.shape
    assert weights[4] > weights[0]
    assert np.isclose(weights[0], weights[1])


def test_class_balancing_meta_sklearn_vs_boosting():
    y = np.zeros(10, dtype=np.int64)

    lr_meta = _class_balancing_meta("logistic_regression", y)
    xgb_meta = _class_balancing_meta("xgboost", y)

    assert lr_meta["enabled"] is True
    assert lr_meta["method"] == "class_weight_balanced"
    assert xgb_meta["enabled"] is True
    assert xgb_meta["method"] == "sample_weight_balanced"


def test_fit_one_model_passes_sample_weight_to_xgboost():
    spec = MagicMock()
    spec.display_name = "XGBoost"
    model = MagicMock()
    spec.build.return_value = model

    X = np.zeros((6, 3), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    logger = MagicMock()

    with patch("training.disease_prediction.train.joblib.dump"):
        with patch("training.disease_prediction.train.directory_size_bytes", return_value=1000):
            result = _fit_one_model("xgboost", spec, X, y, logger)

    model.fit.assert_called_once()
    _, kwargs = model.fit.call_args
    assert "sample_weight" in kwargs
    assert kwargs["sample_weight"].shape == y.shape
    assert result["class_balancing"]["method"] == "sample_weight_balanced"
