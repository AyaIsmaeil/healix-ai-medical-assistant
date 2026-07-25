"""
Healix - Disease Prediction offline inference demo (Phase 6.1).

OFFLINE ONLY. This script is a standalone demonstration of loading a saved
model and producing top-k disease predictions. It is NOT wired into the
FastAPI runtime, NOT called by any route, and NOT a replacement for
``RuleBasedDiseasePredictor`` — the Assessment Engine is completely
untouched by this package.

Two ways to use it:
  1. ``--split test --n 5``           - predict on N real rows from a split
                                         (demonstrates the full loop end to end)
  2. ``--features path/to/row.json``  - predict on a single hand-built feature
                                         dict, keyed by the 286 v2 column names
                                         (any missing key is treated as null,
                                         exactly like the runtime encoder would)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from training.disease_prediction import config
from training.disease_prediction.dataset import (DiseaseLabelEncoder,
                                                  load_feature_order,
                                                  load_split)
from training.disease_prediction.metrics import align_proba


def load_model(model_name_or_path: str):
    path = Path(model_name_or_path)
    if not path.exists():
        path = config.MODELS_DIR / f"{model_name_or_path}.joblib"
    return joblib.load(path), path


def predict_rows(model, X: np.ndarray, encoder: DiseaseLabelEncoder,
                 top_k: int = 5) -> List[List[Dict[str, Any]]]:
    proba = model.predict_proba(X)
    if hasattr(model, "classes_"):
        proba = align_proba(proba, np.asarray(model.classes_), encoder.n_classes)

    results = []
    for row in proba:
        order = np.argsort(row)[::-1][:top_k]
        results.append([
            {"disease_id": encoder.classes_[i], "probability": round(float(row[i]), 6)}
            for i in order
        ])
    return results


def features_from_json(path: Path, feature_order: List[str]) -> np.ndarray:
    payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    row = [payload.get(name) for name in feature_order]
    return np.array([[np.nan if v is None else v for v in row]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline disease-prediction demo (NOT wired into the runtime)")
    parser.add_argument("--model", default=None,
                        help="model family name or path; defaults to outputs/models/best_model.json")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--n", type=int, default=5, help="rows to sample from --split")
    parser.add_argument("--features", type=str, default=None,
                        help="path to a JSON file of {feature_name: value}")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset_dir = config.resolve_dataset_dir()
    feature_order = load_feature_order(dataset_dir)
    encoder = DiseaseLabelEncoder.from_label_dictionary(dataset_dir)

    model_ref = args.model
    if model_ref is None:
        best = json.loads((config.MODELS_DIR / "best_model.json").read_text(encoding="utf-8"))
        model_ref = best["model"]
        print(f"(no --model given; using best_model.json -> {model_ref})")
    model, model_path = load_model(model_ref)
    print(f"loaded {model_path}")

    if args.features:
        X = features_from_json(Path(args.features), feature_order)
        truth = [None]
    else:
        df = load_split(dataset_dir, args.split, feature_order, limit=args.n)
        X = df[feature_order].to_numpy(dtype=np.float32, na_value=np.nan)
        truth = df["y_disease"].tolist() if "y_disease" in df.columns else [None] * len(df)

    predictions = predict_rows(model, X, encoder, top_k=args.top_k)
    for i, (preds, true_label) in enumerate(zip(predictions, truth)):
        print(f"\nrow {i} (true={true_label}):")
        for rank, p in enumerate(preds, start=1):
            marker = " <-- correct" if p["disease_id"] == true_label else ""
            print(f"  {rank}. {p['disease_id']}  p={p['probability']:.4f}{marker}")


if __name__ == "__main__":
    main()
