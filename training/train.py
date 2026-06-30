"""
Phase 1 - Fine-tune MARBERTv2 for multi-label Arabic symptom extraction.

Designed to run on Google Colab with a CUDA GPU. Fully standalone: imports
nothing from the FastAPI app and reads the normalized splits from
app/data/processed/.

Run from the repository root (after running training/preprocess.py):
    python training/train.py
    python training/train.py --epochs 4 --batch-size 32   # override defaults

The trained model is written to a NEW directory (default
ml_models/marbert_symptoms_v2) and never overwrites an existing model unless
--force is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Make `training` importable when run as a script (python training/train.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset import SymptomDataset, load_labels, load_split
from training.report_callback import MetricsReportCallback
from training.trainer import build_trainer, build_training_args, save_artifacts

BASE_MODEL = "UBC-NLP/MARBERT"
DEFAULT_PROCESSED = REPO_ROOT / "app" / "data" / "processed"
DEFAULT_OUTPUT = REPO_ROOT / "ml_models" / "marbert_symptoms_v2"
DEFAULT_RESULTS = REPO_ROOT / "training" / "_results"
DEFAULT_REPORTS = REPO_ROOT / "training" / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune MARBERTv2 for symptom extraction.")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--force", action="store_true", help="Overwrite output dir if it exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir.exists() and not args.force:
        raise SystemExit(
            f"Output dir already exists: {args.output_dir}\n"
            f"Refusing to overwrite an existing model. Use --force or pick another --output-dir."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = torch.cuda.is_available()
    print(f"Device: {device} | fp16: {fp16}")

    # 1. Labels and data.
    labels = load_labels(args.processed_dir / "labels.json")
    train_texts, train_y = load_split(args.processed_dir / "train.csv", labels)
    val_texts, val_y = load_split(args.processed_dir / "val.csv", labels)
    print(f"Labels: {len(labels)} | train: {len(train_texts):,} | val: {len(val_texts):,}")

    # 2. Tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        problem_type="multi_label_classification",
    )

    train_ds = SymptomDataset(train_texts, train_y, tokenizer, args.max_length)
    val_ds = SymptomDataset(val_texts, val_y, tokenizer, args.max_length)

    # 3. Train.
    training_args = build_training_args(
        output_dir=str(args.results_dir),
        epochs=args.epochs,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=fp16,
    )
    trainer = build_trainer(model, training_args, train_ds, val_ds, tokenizer, args.patience)
    # Save eval loss + all metrics to JSON/CSV after every evaluation (thesis).
    trainer.add_callback(MetricsReportCallback(args.report_dir))
    trainer.train()

    # 4. Validation metrics (test set is reserved for Step 5 - evaluate.py).
    val_metrics = trainer.evaluate()
    print("\nValidation metrics (best model):")
    for key, value in sorted(val_metrics.items()):
        if key.startswith("eval_"):
            print(f"  {key[5:]:18s} {value:.4f}" if isinstance(value, float) else f"  {key[5:]}: {value}")

    # 5. Save artifacts in a service-loadable layout.
    train_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "max_length": args.max_length,
    }
    out = save_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        labels=labels,
        max_length=args.max_length,
        threshold=args.threshold,
        base_model=BASE_MODEL,
        train_config=train_config,
    )
    print(f"\nSaved model -> {out}")


if __name__ == "__main__":
    main()
