"""
Trainer wiring for MARBERTv2 fine-tuning.

Keeps train.py thin: this module builds the TrainingArguments, the Trainer, and
saves the final artifacts (model + tokenizer + metadata.json + labels.json) in a
format the FastAPI service can load directly.

A small compatibility shim picks `eval_strategy` vs `evaluation_strategy`
depending on the installed transformers version (the key was renamed), so the
same code runs on a pinned local version and on Colab's newer transformers.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from training.metrics import compute_metrics


def build_training_args(
    output_dir: str,
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    fp16: bool,
) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=fp16,
        report_to="none",
    )
    kwargs[eval_key] = "epoch"
    return TrainingArguments(**kwargs)


def build_trainer(model, args, train_dataset, eval_dataset, tokenizer, patience: int = 2) -> Trainer:
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )


def save_artifacts(
    model,
    tokenizer,
    output_dir: str | Path,
    labels: List[str],
    max_length: int,
    threshold: float,
    base_model: str,
    train_config: dict,
) -> Path:
    """Persist the model in a service-loadable layout and return its path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    metadata = {
        "model_name": base_model,
        "symptom_columns": labels,
        "threshold": threshold,
        "max_length": max_length,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "num_labels": len(labels),
        "config": train_config,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out
