"""
Multi-label evaluation metrics for symptom extraction.

Multi-label tasks need label-aware metrics. We report:
  - subset_accuracy: exact match of the full label set (strict, informative
    but harsh - one wrong label fails the whole row).
  - micro F1/precision/recall: aggregate over all label decisions (reflects
    overall token-level quality, dominated by frequent labels).
  - macro F1/precision/recall: unweighted mean over labels (reflects how well
    we do on RARE labels too - the honest number under class imbalance).
  - weighted F1/precision/recall: mean over labels weighted by support
    (per-label positive count) - a middle ground between micro and macro.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DEFAULT_THRESHOLD = 0.5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def multilabel_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Compute multi-label metrics from raw logits and binary labels."""
    probs = _sigmoid(np.asarray(logits, dtype="float64"))
    preds = (probs >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)

    metrics = {"subset_accuracy": float(accuracy_score(labels, preds))}
    # Precision / Recall / F1 in all three averaging modes (thesis-ready grid).
    for average in ("micro", "macro", "weighted"):
        metrics[f"precision_{average}"] = float(
            precision_score(labels, preds, average=average, zero_division=0)
        )
        metrics[f"recall_{average}"] = float(
            recall_score(labels, preds, average=average, zero_division=0)
        )
        metrics[f"f1_{average}"] = float(
            f1_score(labels, preds, average=average, zero_division=0)
        )
    return metrics


def compute_metrics(eval_pred) -> dict:
    """Adapter for transformers.Trainer (receives an EvalPrediction)."""
    logits = getattr(eval_pred, "predictions", None)
    labels = getattr(eval_pred, "label_ids", None)
    if logits is None:  # tuple fallback
        logits, labels = eval_pred[0], eval_pred[1]
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return multilabel_metrics(logits, labels)
