"""
Healix - Disease Prediction evaluation metrics (Phase 6.1).

Every metric requested by the phase brief, plus the operational figures
(training time, inference time, model size, memory) needed to compare model
families honestly. No metric here is used to tune anything — Phase 6.1 does
not perform hyperparameter search, cross-validation, or threshold
calibration; metrics are computed once, after training, for comparison only.

Class imbalance is severe (49 classes, ~247:1 in the source corpus per
docs/research/DDXPLUS_EDA.md), so **macro-F1 is the primary comparison
metric**, never bare accuracy (docs/research/DATASET_BUILDER_DESIGN.md §7.7).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             top_k_accuracy_score)

from training.disease_prediction.config import TOP_K_VALUES


def align_proba(proba: np.ndarray, model_classes: np.ndarray,
                n_classes: int) -> np.ndarray:
    """Reorder an estimator's predict_proba columns into canonical class
    order [0..n_classes-1], filling 0.0 for any class the estimator's
    ``classes_`` happens not to report (defensive; in this dataset every
    class appears in every training run, but the reorder must never
    silently misalign columns if that assumption is ever violated)."""
    aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for col, cls in enumerate(model_classes):
        aligned[:, int(cls)] = proba[:, col]
    return aligned


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           proba: Optional[np.ndarray],
                           n_classes: int) -> Dict[str, Any]:
    labels = list(range(n_classes))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = \
        precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    # "F1 Score" (the brief's plain bullet, distinct from macro/weighted) is
    # reported here as the micro average, which for single-label multiclass
    # classification is mathematically identical to accuracy — documented
    # explicitly so the three F1 figures are never confused with each other.
    _, _, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0)

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision_macro), 6),
        "precision_weighted": round(float(precision_weighted), 6),
        "recall_macro": round(float(recall_macro), 6),
        "recall_weighted": round(float(recall_weighted), 6),
        "f1_score_micro": round(float(f1_micro), 6),
        "f1_macro": round(float(f1_macro), 6),
        "f1_weighted": round(float(f1_weighted), 6),
        "n_samples": int(len(y_true)),
        "n_classes_present_in_split": int(len(np.unique(y_true))),
    }

    if proba is not None:
        for k in TOP_K_VALUES:
            try:
                metrics[f"top_{k}_accuracy"] = round(float(
                    top_k_accuracy_score(y_true, proba, k=k, labels=labels)), 6)
            except ValueError as exc:  # noqa: BLE001 - degrade, never crash a report
                metrics[f"top_{k}_accuracy"] = None
                metrics[f"top_{k}_accuracy_error"] = str(exc)

        try:
            metrics["roc_auc_ovr_macro"] = round(float(
                roc_auc_score(y_true, proba, multi_class="ovr",
                              average="macro", labels=labels)), 6)
            metrics["roc_auc_ovr_weighted"] = round(float(
                roc_auc_score(y_true, proba, multi_class="ovr",
                              average="weighted", labels=labels)), 6)
        except ValueError as exc:  # noqa: BLE001 - e.g. a class absent from y_true
            metrics["roc_auc_ovr_macro"] = None
            metrics["roc_auc_ovr_weighted"] = None
            metrics["roc_auc_error"] = str(exc)

    return metrics


def confusion_matrix_full(y_true: np.ndarray, y_pred: np.ndarray,
                          n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray,
                     class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Per-class precision/recall/F1/support — needed to see whether the
    rare tail of the 247:1 imbalance is actually being learned."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    return {
        class_names[i]: {
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
