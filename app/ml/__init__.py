"""Machine-learning inference layer for Healix.

Holds model-loading and prediction code (as opposed to `app/services`, which
holds business orchestration). Each future model — disease predictor, triage
classifier — gets its own module here next to `symptom_extractor`.
"""

from app.ml.symptom_extractor import SymptomExtractor

__all__ = ["SymptomExtractor"]
