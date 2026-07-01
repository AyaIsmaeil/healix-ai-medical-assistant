"""
Phase 1 - Step 5: Evaluate a trained model on the held-out test set.

Loads a fine-tuned model, runs inference over app/data/processed/test.csv, and
writes both AGGREGATE metrics (JSON) and PER-LABEL metrics (CSV) for the thesis.

The test set is used ONLY here - it was never seen during training/validation,
so these numbers are the honest, reportable performance.

Run from the repository root:
    python training/evaluate.py --model-dir ml_models/healix_marbert_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset import SymptomDataset, load_labels, load_split
from training.metrics import multilabel_metrics

DEFAULT_MODEL = REPO_ROOT / "ml_models" / "healix_marbert_v2"
DEFAULT_PROCESSED = REPO_ROOT / "app" / "data" / "processed"
DEFAULT_REPORTS = REPO_ROOT / "training" / "reports"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override decision threshold (default: read from metadata, else 0.5).")
    return p.parse_args()


def _read_metadata(model_dir: Path) -> dict:
    meta_path = model_dir / "metadata.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def main() -> None:
    args = parse_args()

    # Fail early with a clear message if the weights are missing.
    if not any((args.model_dir / w).exists() for w in WEIGHT_FILES):
        raise SystemExit(
            f"No model weights found in {args.model_dir}\n"
            f"Expected one of: {', '.join(WEIGHT_FILES)}.\n"
            f"Copy the trained weights file back into that folder (it is the large\n"
            f"~650 MB file produced on Colab), then re-run."
        )

    metadata = _read_metadata(args.model_dir)
    max_length = int(metadata.get("max_length", 64))
    threshold = args.threshold if args.threshold is not None else float(metadata.get("threshold", 0.5))

    # Canonical label order from the processed split; verify the model agrees.
    labels = load_labels(args.processed_dir / "labels.json")
    model_labels = metadata.get("symptom_columns")
    if model_labels is not None and model_labels != labels:
        raise SystemExit(
            "Label mismatch between model metadata and processed/labels.json.\n"
            "The model and the test set must share the exact same label order."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | threshold: {threshold} | max_length: {max_length}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir)).to(device)
    model.eval()

    test_texts, test_y = load_split(args.processed_dir / "test.csv", labels)
    dataset = SymptomDataset(test_texts, test_y, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    print(f"Test samples: {len(dataset):,} | labels: {len(labels)}")

    # Inference -> collect raw logits.
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch_labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch_labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0).astype(int)
    y_pred = (1.0 / (1.0 + np.exp(-logits)) >= threshold).astype(int)

    # 1. Aggregate metrics.
    aggregate = multilabel_metrics(logits, y_true, threshold=threshold)
    aggregate.update({
        "n_test_samples": int(len(dataset)),
        "n_labels": len(labels),
        "threshold": threshold,
        "model_dir": str(args.model_dir),
    })

    # 2. Per-label metrics.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_label = pd.DataFrame({
        "label": labels,
        "support": support.astype(int),
        "precision": precision.round(4),
        "recall": recall.round(4),
        "f1": f1.round(4),
    }).sort_values("f1", ascending=False).reset_index(drop=True)

    # 3. Persist reports.
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "test_metrics.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    per_label.to_csv(args.report_dir / "test_per_label.csv", index=False, encoding="utf-8-sig")

    # 4. Console summary.
    print("\n" + "=" * 56)
    print("AGGREGATE TEST METRICS")
    print("=" * 56)
    print(f"{'subset_accuracy':<22}{aggregate['subset_accuracy']:.4f}")
    for avg in ("micro", "macro", "weighted"):
        print(f"{'precision_'+avg:<22}{aggregate['precision_'+avg]:.4f}"
              f"   {'recall_'+avg:<18}{aggregate['recall_'+avg]:.4f}"
              f"   {'f1_'+avg:<14}{aggregate['f1_'+avg]:.4f}")
    print("\nTop 5 labels by F1:")
    print(per_label.head(5).to_string(index=False))
    print("\nBottom 5 labels by F1:")
    print(per_label.tail(5).to_string(index=False))
    print(f"\nReports written to {args.report_dir.relative_to(REPO_ROOT)}/"
          f" (test_metrics.json, test_per_label.csv)")


if __name__ == "__main__":
    main()
