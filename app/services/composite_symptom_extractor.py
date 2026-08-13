"""
Healix - Composite Symptom Extractor
مسار هجين: LLM (enum مغلق + evidence_source) + استخراج قاعدي حتمي — اتحاد النتائج.
"""

from __future__ import annotations

from typing import List, Sequence

from app.domain.conversation import Symptom
from app.services.llm_symptom_extractor import LLMSymptomExtractor
from app.services.rule_based_symptom_extractor import RuleBasedSymptomExtractor


class CompositeSymptomExtractor:
    """يجمع الأعراض من LLM والقواعد الحتمية — LLM يُفضَّل عند التعارض."""

    def __init__(
        self,
        llm_extractor: LLMSymptomExtractor,
        rule_extractor: RuleBasedSymptomExtractor,
    ) -> None:
        self._llm = llm_extractor
        self._rule = rule_extractor

    def extract(
        self,
        raw_messages: Sequence[str],
        known_symptoms: Sequence[str] | None = None,
    ) -> List[Symptom]:
        llm_symptoms = self._llm.extract(raw_messages, known_symptoms)
        rule_symptoms = self._rule.extract(raw_messages)

        merged: dict[tuple[str, bool], Symptom] = {}
        for symptom in rule_symptoms:
            key = (symptom.text, symptom.negated)
            merged[key] = symptom

        for symptom in llm_symptoms:
            key = (symptom.text, symptom.negated)
            existing = merged.get(key)
            if existing is None or symptom.confidence > existing.confidence:
                merged[key] = symptom

        return list(merged.values())
