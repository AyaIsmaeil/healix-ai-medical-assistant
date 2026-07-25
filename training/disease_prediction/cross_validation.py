"""
Healix - Phase 6.2 Stratified K-Fold Cross Validation (OFFLINE ONLY).

Phase 6.1 ranked five model families on a SINGLE train/validation split and
explicitly flagged the Random-Forest-vs-CatBoost gap (0.99615 vs 0.99594,
a difference of 0.0002) as "within plausible single-split noise ... should be
treated as a statistical tie". This module exists to settle that question
with measured variance instead of a point estimate.

Protocol
--------
* StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) so every fold
  preserves the 49-class distribution (the corpus is ~247:1 imbalanced).
* Folds are drawn from a stratified subsample of the training split
  (VERIFY_SUBSAMPLE_ROWS) because full-scale 5-fold CV across five families
  would take days on this machine — the subsample is fixed by seed and
  IDENTICAL for every family, so the comparison stays fair. This is stated
  in the generated report, never hidden.
* Per fold we record accuracy, macro-F1, weighted-F1, fit seconds, predict
  seconds, and peak process RSS; the report carries mean AND standard
  deviation for each, which is the entire point of the exercise.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from training.disease_prediction.config import SEED
from training.disease_prediction.config_optimization import (CV_FOLDS,
                                                             CV_SHUFFLE)
from training.disease_prediction.utils import PeakMemorySampler, Timer


def stratified_subsample(y: np.ndarray, n_rows: int, seed: int = SEED) -> np.ndarray:
    """Indices of a class-proportional subsample.

    Guarantees at least one row per present class (so no class silently
    vanishes from a fold), then fills the remainder proportionally.
    """
    rng = np.random.default_rng(seed)
    n_rows = min(n_rows, len(y))
    classes, counts = np.unique(y, return_counts=True)

    chosen: List[np.ndarray] = []
    for cls in classes:                       # guarantee 1 per class
        idx = np.flatnonzero(y == cls)
        chosen.append(rng.choice(idx, size=1, replace=False))
    guaranteed = np.concatenate(chosen)

    remaining = np.setdiff1d(np.arange(len(y)), guaranteed, assume_unique=False)
    n_more = max(0, n_rows - len(guaranteed))
    if n_more and len(remaining):
        # proportional draw without replacement
        take = rng.choice(remaining, size=min(n_more, len(remaining)),
                          replace=False)
        out = np.concatenate([guaranteed, take])
    else:
        out = guaranteed
    out.sort()
    return out


def cross_validate_model(build_fn: Callable[[], Any], X: np.ndarray,
                         y: np.ndarray, n_splits: int = CV_FOLDS,
                         seed: int = SEED,
                         logger=None) -> Dict[str, Any]:
    """Run stratified K-fold CV for one estimator factory.

    ``build_fn`` must return a FRESH unfitted estimator on every call, so no
    state leaks between folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=CV_SHUFFLE,
                          random_state=seed)
    folds: List[Dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        model = build_fn()
        with PeakMemorySampler() as mem, Timer() as t_fit:
            model.fit(X_tr, y_tr)
        with Timer() as t_pred:
            pred = model.predict(X_te)

        rec = {
            "fold": fold_i,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": round(float(accuracy_score(y_te, pred)), 6),
            "f1_macro": round(float(f1_score(y_te, pred, average="macro",
                                             zero_division=0)), 6),
            "f1_weighted": round(float(f1_score(y_te, pred, average="weighted",
                                                zero_division=0)), 6),
            "fit_seconds": t_fit.elapsed,
            "predict_seconds": t_pred.elapsed,
            "predict_seconds_per_1000": round(
                (t_pred.elapsed / max(len(test_idx), 1)) * 1000, 6),
            "peak_memory_mb": mem.peak_mb,
        }
        folds.append(rec)
        if logger:
            logger.info("  fold %d/%d acc=%.5f f1_macro=%.5f fit=%.1fs",
                        fold_i, n_splits, rec["accuracy"], rec["f1_macro"],
                        rec["fit_seconds"])
        del model

    def agg(key: str) -> Dict[str, float]:
        vals = np.array([f[key] for f in folds], dtype=float)
        return {"mean": round(float(vals.mean()), 6),
                "std": round(float(vals.std(ddof=1)), 6),
                "min": round(float(vals.min()), 6),
                "max": round(float(vals.max()), 6)}

    return {
        "n_splits": n_splits, "seed": seed, "folds": folds,
        "accuracy": agg("accuracy"),
        "f1_macro": agg("f1_macro"),
        "f1_weighted": agg("f1_weighted"),
        "fit_seconds": agg("fit_seconds"),
        "predict_seconds_per_1000": agg("predict_seconds_per_1000"),
        "peak_memory_mb": agg("peak_memory_mb"),
    }


def paired_fold_comparison(cv_a: Dict[str, Any], cv_b: Dict[str, Any],
                           name_a: str, name_b: str,
                           metric: str = "f1_macro") -> Dict[str, Any]:
    """Compare two families fold-by-fold on identical splits.

    Because both CV runs use the same StratifiedKFold seed, fold i contains
    exactly the same rows for both models, so the per-fold differences are
    PAIRED. That makes 'how many folds did A beat B' and the mean difference
    +/- its own std a far more honest read than comparing two independent
    means. No significance test is claimed here beyond reporting the paired
    differences and whether the gap exceeds the observed fold-to-fold noise.
    """
    a = np.array([f[metric] for f in cv_a["folds"]], dtype=float)
    b = np.array([f[metric] for f in cv_b["folds"]], dtype=float)
    diff = a - b
    pooled_std = float(np.std(np.concatenate([a, b]), ddof=1))
    return {
        "metric": metric, "model_a": name_a, "model_b": name_b,
        "mean_a": round(float(a.mean()), 6), "mean_b": round(float(b.mean()), 6),
        "mean_difference_a_minus_b": round(float(diff.mean()), 6),
        "std_of_difference": round(float(diff.std(ddof=1)), 6),
        "folds_a_wins": int((diff > 0).sum()),
        "folds_b_wins": int((diff < 0).sum()),
        "folds_tied": int((diff == 0).sum()),
        "pooled_fold_std": round(pooled_std, 6),
        "difference_exceeds_pooled_noise": bool(abs(diff.mean()) > pooled_std),
    }
