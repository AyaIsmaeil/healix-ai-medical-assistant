"""
Healix - Composite Evidence Concept Extractor
مسار هجين: LLM (enum مغلق) + استخراج قاعدي حتمي — اتحاد النتائج بلا تكرار.

يضمن عمل ML حتى مع ``LLM_PROVIDER=mock`` أو فشل شبكة OpenRouter، طالما
وردت الأعراض في ``raw_messages`` أو ``symptoms``.
"""

from __future__ import annotations

from typing import List, Sequence

from app.services.evidence_concept_extractor import EvidenceConceptExtractor
from app.services.rule_based_evidence_concept_extractor import (
    RuleBasedEvidenceConceptExtractor,
)


class CompositeEvidenceConceptExtractor:
    """يجمع مفاهيم الأدلة من LLM والقواعد الحتمية."""

    def __init__(
        self,
        llm_extractor: EvidenceConceptExtractor,
        rule_extractor: RuleBasedEvidenceConceptExtractor,
    ) -> None:
        self._llm = llm_extractor
        self._rule = rule_extractor

    def extract(
        self,
        raw_messages: List[str],
        symptom_names: Sequence[str] | None = None,
    ) -> List[str]:
        llm_concepts = self._llm.extract(raw_messages)
        rule_concepts = self._rule.extract(raw_messages, symptom_names)
        return sorted(set(llm_concepts) | set(rule_concepts))
