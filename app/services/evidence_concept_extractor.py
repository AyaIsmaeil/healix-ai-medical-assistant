"""
Healix - Evidence Concept Extractor
يستدعي الـLLM لاختيار أيّ مفاهيم أدلة DDXPlus (من قائمة مغلقة) ينطبق على
رسائل المريض — يغذّي ``SymptomEvidenceEncoder`` بمفاهيم جاهزة بدل مطابقة
نصّية هشّة. بنفس نمط ``LLMFeatureExtractor``: فشل ناعم (قائمة فارغة) لا
ينهار التقييم كاملاً.
"""

from __future__ import annotations

import logging
from typing import List

from app.domain.ports import LLMProvider
from app.exceptions import FeatureExtractionError
from app.parsing.evidence_concept_extraction_parser import parse_evidence_concepts
from app.prompts.evidence_concept_extraction_builder import (
    EvidenceConceptExtractionPromptBuilder,
)

logger = logging.getLogger(__name__)


class EvidenceConceptExtractor:
    """يستدعي الـLLM لاختيار مفاهيم الأدلة المنطبقة، مع تدهور لطيف عند الفشل."""

    def __init__(self, provider: LLMProvider, prompt_builder: EvidenceConceptExtractionPromptBuilder):
        self._provider = provider
        self._prompts = prompt_builder

    def extract(self, raw_messages: List[str]) -> List[str]:
        if not raw_messages:
            return []

        system_prompt = self._prompts.system_prompt()
        user_prompt = self._prompts.extraction_prompt(raw_messages)

        try:
            completion = self._provider.generate(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - فشل ناعم، لا ينهار التقييم كاملاً
            logger.warning("فشل استدعاء LLM اختيار مفاهيم الأدلة: %s", exc)
            return []

        try:
            return parse_evidence_concepts(completion.text, self._prompts.concepts)
        except FeatureExtractionError as exc:
            logger.warning("فشل تحليل مخرجات اختيار مفاهيم الأدلة: %s", exc)
            return []
