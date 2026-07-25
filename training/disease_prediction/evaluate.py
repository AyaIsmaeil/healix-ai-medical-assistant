"""
Healix - Disease Prediction evaluation (Phase 6.1).

Two uses:
  1. Imported by train.py to evaluate a freshly-fitted model against the
     validation split (both the full split and the leakage-free subset).
  2. Run standalone to re-evaluate an already-saved model against any split
     (typically the TEST split, exactly once, after a best model has been
     selected on validation — never used to pick between models).

Reporting both the full split and the leakage-free subset is the mitigation
adopted in docs/research/CROSS_SPLIT_LEAKAGE_REPORT.md §5.1: leaked rows are
never dropped from the dataset, but they must never silently inflate a
reported metric either. The leakage-free subset is the PRIMARY number.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from training.disease_prediction import config
from training.disease_prediction.dataset import (DatasetBundle,
                                                  DiseaseLabelEncoder,
                                                  load_bundle,
                                                  load_feature_order)
from training.disease_prediction.metrics import (align_proba,
                                                  classification_metrics,
                                                  confusion_matrix_full,
                                                  per_class_report)
from training.disease_prediction.utils import Timer, write_json


def _predict_with_timing(model: Any, X: np.ndarray) -> Dict[str, Any]:
    with Timer() as t:
        pred = model.predict(X)
    proba = None
    proba_seconds = None
    if hasattr(model, "predict_proba"):
        with Timer() as tp:
            proba = model.predict_proba(X)
        proba_seconds = tp.elapsed
    n = max(len(X), 1)
    return {
        "y_pred": pred,
        "proba": proba,
        "predict_seconds_total": t.elapsed,
        "predict_seconds_per_1000_rows": round((t.elapsed / n) * 1000, 6),
        "predict_proba_seconds_total": proba_seconds,
    }


def evaluate_split(model: Any, bundle: DatasetBundle, class_names,
                   split_label: str) -> Dict[str, Any]:
    """Evaluate on the FULL split, then again on the leakage-free subset."""
    n_classes = len(class_names)
    timing = _predict_with_timing(model, bundle.X)
    y_pred, proba = timing["y_pred"], timing["proba"]

    if proba is not None and hasattr(model, "classes_"):
        proba = align_proba(proba, np.asarray(model.classes_), n_classes)

    full_metrics = classification_metrics(bundle.y, y_pred, proba, n_classes)
    full_metrics["timing"] = {
        "predict_seconds_total": timing["predict_seconds_total"],
        "predict_seconds_per_1000_rows": timing["predict_seconds_per_1000_rows"],
    }

    clean_mask = bundle.clean_mask
    n_clean = int(clean_mask.sum())
    n_leaked = int((~clean_mask).sum())
    if n_clean > 0:
        clean_proba = proba[clean_mask] if proba is not None else None
        clean_metrics = classification_metrics(
            bundle.y[clean_mask], y_pred[clean_mask], clean_proba, n_classes)
    else:
        clean_metrics = None

    return {
        "split": split_label,
        "rows_total": int(bundle.n_rows),
        "rows_leaked_flagged": n_leaked,
        "rows_clean": n_clean,
        "leaked_pct": round(100 * n_leaked / bundle.n_rows, 4) if bundle.n_rows else 0.0,
        "metrics_full_split": full_metrics,
        "metrics_leakage_free_subset": clean_metrics,
        "note": ("metrics_leakage_free_subset is the PRIMARY, headline result "
                "(docs/research/CROSS_SPLIT_LEAKAGE_REPORT.md); "
                "metrics_full_split is reported for comparability only."),
        "confusion_matrix_full_split": confusion_matrix_full(
            bundle.y, y_pred, n_classes).tolist(),
        "per_class_full_split": per_class_report(bundle.y, y_pred, class_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved disease-prediction model")
    parser.add_argument("--model", required=True,
                        help="model family name (e.g. xgboost) or path to a .joblib file")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = config.resolve_dataset_dir()
    encoder = DiseaseLabelEncoder.from_label_dictionary(dataset_dir)
    feature_order = load_feature_order(dataset_dir)

    model_path = Path(args.model)
    if not model_path.exists():
        model_path = config.MODELS_DIR / f"{args.model}.joblib"
    model = joblib.load(model_path)

    bundle = load_bundle(dataset_dir, args.split, encoder, feature_order,
                        limit=args.limit)
    report = evaluate_split(model, bundle, encoder.classes_, args.split)

    out_path = config.REPORTS_DIR / f"evaluate_{model_path.stem}_{args.split}.json"
    write_json(out_path, report)
    print(f"wrote {out_path}")
    m = report["metrics_leakage_free_subset"] or report["metrics_full_split"]
    print(f"accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
          f"f1_weighted={m['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
