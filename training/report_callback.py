"""
Training report callback.

A transformers TrainerCallback that records validation loss and every
evaluation metric after each evaluation step, then writes a cumulative report
to JSON and CSV. Intended to provide reproducible tables/curves for the
graduation thesis.

The report is rewritten in full after every evaluation, so the files are always
complete even if training is interrupted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from transformers import TrainerCallback


class MetricsReportCallback(TrainerCallback):
    """Persist eval loss + metrics to JSON and CSV after each evaluation."""

    def __init__(
        self,
        report_dir: str | Path,
        json_name: str = "training_report.json",
        csv_name: str = "training_report.csv",
    ):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.report_dir / json_name
        self.csv_path = self.report_dir / csv_name
        self.rows: list[dict] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: D401
        if not metrics:
            return

        row: dict = {
            "epoch": round(float(state.epoch), 4) if state.epoch is not None else None,
            "step": int(state.global_step),
        }

        # Most recent training loss logged before this evaluation (for the
        # train-vs-validation loss curve in the thesis).
        for entry in reversed(state.log_history):
            if "loss" in entry:
                row["train_loss"] = float(entry["loss"])
                break
        row.setdefault("train_loss", None)

        # Validation loss + all metrics (strip the 'eval_' prefix Trainer adds).
        # eval_loss is renamed to val_loss so it is unambiguous next to train_loss.
        for key, value in metrics.items():
            if key == "eval_loss":
                clean = "val_loss"
            elif key.startswith("eval_"):
                clean = key[len("eval_"):]
            else:
                clean = key
            row[clean] = float(value) if isinstance(value, (int, float)) else value

        self.rows.append(row)
        self._flush()

    def _flush(self) -> None:
        self.json_path.write_text(
            json.dumps(self.rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Stable, union-of-keys header (order preserved across rows).
        fieldnames: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        # utf-8-sig so Excel opens the file cleanly for the thesis.
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
