"""
Prediction service - the reusable entry point for symptom extraction.

This is the stable interface the rest of the system depends on. Future modules
(Rule Engine, Disease Prediction, Dynamic Questions, Triage) and the future API
endpoint call `get_prediction_service()` and use `extract_symptoms()` /
`detect()` — they never touch the model or tokenizer directly. That keeps the
model an implementation detail we can swap without changing callers.

The model is loaded exactly once per process via a cached singleton
(`get_prediction_service`). Call it during application startup to load eagerly.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List

from app.config import SYMPTOM_MODEL_DIR
from app.ml.symptom_extractor import SymptomExtractor

logger = logging.getLogger(__name__)


class PredictionService:
    """Thin, framework-agnostic wrapper around the symptom extractor."""

    def __init__(self, extractor: SymptomExtractor):
        self._extractor = extractor

    @classmethod
    def load(cls, model_dir: str = SYMPTOM_MODEL_DIR) -> "PredictionService":
        """Load the model and return a ready service."""
        logger.info("Loading symptom-extraction model from %s", model_dir)
        service = cls(SymptomExtractor(model_dir))
        logger.info("Symptom-extraction model ready.")
        return service

    def detect(self, text: str) -> List[str]:
        """Return just the list of detected symptom names (for internal callers)."""
        return self._extractor.extract(text)

    def extract_symptoms(self, text: str) -> Dict[str, List[str]]:
        """Return the public response contract: {"detected_symptoms": [...]}."""
        return {"detected_symptoms": self.detect(text)}

    @property
    def labels(self) -> List[str]:
        return self._extractor.labels


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Return the process-wide singleton, loading the model on first call."""
    return PredictionService.load()
