"""
Healix - LLM Symptom Extractor
يستدعي الـLLM لاستخراج الأعراض من رسائل المريض — فشل ناعم (قائمة فارغة).
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from app.domain.conversation import Symptom
from app.domain.ports import LLMProvider
from app.exceptions import FeatureExtractionError
from app.parsing.symptom_extraction_parser import parse_symptoms
from app.prompts.symptom_extraction_builder import SymptomExtractionPromptBuilder

logger = logging.getLogger(__name__)


class LLMSymptomExtractor:
    """يستدعي الـLLM لاستخراج الأعراض، مع تدهور لطيف عند الفشل."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: SymptomExtractionPromptBuilder,
    ) -> None:
        self._provider = provider
        self._prompts = prompt_builder

    def extract(
        self,
        raw_messages: Sequence[str],
        known_symptoms: Sequence[str] | None = None,
    ) -> List[Symptom]:
        if not raw_messages:
            return []

        system_prompt = self._prompts.system_prompt()
        user_prompt = self._prompts.extraction_prompt(
            list(raw_messages),
            list(known_symptoms or []),
        )

        try:
            completion = self._provider.generate(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("فشل استدعاء LLM استخراج الأعراض: %s", exc)
            return []

        try:
            return parse_symptoms(completion.text, self._prompts.concepts)
        except FeatureExtractionError as exc:
            logger.warning("فشل تحليل مخرجات استخراج الأعراض: %s", exc)
            return []
