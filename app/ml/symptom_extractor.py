"""
MARBERT symptom extractor (inference).

Loads a fine-tuned MARBERT multi-label model and extracts symptoms from Arabic
text. The pipeline mirrors training exactly:
    normalize (app.utils.arabic) -> tokenize (same max_length) -> sigmoid ->
    threshold.

The model, tokenizer, labels, threshold, and max_length are all read from the
model directory (weights + labels.json + metadata.json), so the extractor is
self-contained and stays consistent with whatever was trained.

This class does no I/O beyond loading and is framework-agnostic: it does not
import FastAPI and can be used by any module (Rule Engine, Disease Prediction,
Triage) or a script.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.utils.arabic import normalize_arabic

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_LENGTH = 64


class SymptomExtractor:
    """Loads a trained MARBERT model once and extracts symptoms from text."""

    def __init__(
        self,
        model_dir: str | Path,
        threshold: float | None = None,
        max_length: int | None = None,
        device: str | None = None,
    ):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        metadata = self._read_json(self.model_dir / "metadata.json") or {}

        # Labels: prefer labels.json, fall back to metadata's symptom_columns.
        self.labels: List[str] = (
            self._read_json(self.model_dir / "labels.json")
            or metadata.get("symptom_columns")
        )
        if not self.labels:
            raise ValueError(f"No labels found in {self.model_dir} (labels.json / metadata.json).")

        # Threshold and max_length: explicit arg > metadata > default.
        self.threshold = threshold if threshold is not None else float(
            metadata.get("threshold", DEFAULT_THRESHOLD)
        )
        self.max_length = max_length if max_length is not None else int(
            metadata.get("max_length", DEFAULT_MAX_LENGTH)
        )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        # Fail loudly if the head size and the label list disagree.
        if self.model.config.num_labels != len(self.labels):
            raise ValueError(
                f"Model has {self.model.config.num_labels} outputs but "
                f"{len(self.labels)} labels were loaded — they must match."
            )

        logger.info(
            "SymptomExtractor loaded: %d labels | threshold %.2f | max_length %d | device %s",
            len(self.labels), self.threshold, self.max_length, self.device,
        )

    @staticmethod
    def _read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @torch.no_grad()
    def extract(self, text: str) -> List[str]:
        """Return the list of detected symptoms, most confident first.

        Returns an empty list for empty input or when nothing crosses the
        threshold (e.g. a greeting with no symptoms).
        """
        normalized = normalize_arabic(text)
        if not normalized:
            return []

        inputs = self.tokenizer(
            normalized,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits)[0].cpu().tolist()

        detected = [
            (label, score)
            for label, score in zip(self.labels, probs)
            if score >= self.threshold
        ]
        detected.sort(key=lambda pair: pair[1], reverse=True)
        return [label for label, _ in detected]
