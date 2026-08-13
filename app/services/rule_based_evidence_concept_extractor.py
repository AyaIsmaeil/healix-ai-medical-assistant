"""
Healix - Rule-Based Evidence Concept Extractor
استخراج حتمي لمفاهيم أدلة DDXPlus من نص المريض وأسماء الأعراض المُستخلَصة —
احتياطي موثوق عند غياب/فشل الـLLM، ويُكمّل مخرجاته في المسار الهجين.

يستخدم ``symptom_evidence_map.yaml`` (نفس مصدر ``SymptomEvidenceEncoder``):
مطابقة جزئية بعد ``normalize_arabic``، بلا تخمين خارج القاموس المغلق.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from app.domain.text_preprocessing import normalize_arabic

_NEGATION_PATTERN = re.compile(
    r"(?:^|\s)(?:لا|ما|ليس|بدون|بلا|من\s+غير|مافي|ما\s+في)\s",
    re.UNICODE,
)


class RuleBasedEvidenceConceptExtractor:
    """يختار مفاهيم الأدلة المنطبقة حتمياً من نص عربي و/أو أسماء أعراض."""

    def __init__(self, mappings: Sequence[Dict[str, Any]]) -> None:
        self._entries: List[tuple[str, tuple[str, ...]]] = [
            (entry["concept"], tuple(entry.get("names", ())))
            for entry in mappings
            if entry.get("concept") and entry.get("names")
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleBasedEvidenceConceptExtractor":
        return cls(mappings=data.get("mappings", ()))

    def extract(
        self,
        raw_messages: Sequence[str],
        symptom_names: Sequence[str] | None = None,
    ) -> List[str]:
        """يُعيد قائمة مفاهيم فريدة مرتّبة أبجدياً (حتمية)."""
        text = normalize_arabic(" ".join(str(message) for message in raw_messages))
        matched: set[str] = set()

        for concept, keywords in self._entries:
            if self._concept_in_symptoms(keywords, symptom_names or ()):
                matched.add(concept)
                continue
            if self._concept_in_text(text, keywords):
                matched.add(concept)

        return sorted(matched)

    @staticmethod
    def _concept_in_symptoms(
        keywords: tuple[str, ...],
        symptom_names: Sequence[str],
    ) -> bool:
        for name in symptom_names:
            normalized = normalize_arabic(name)
            if not normalized:
                continue
            for keyword in keywords:
                if normalize_arabic(keyword) in normalized:
                    return True
        return False

    @staticmethod
    def _concept_in_text(text: str, keywords: tuple[str, ...]) -> bool:
        for keyword in keywords:
            normalized_keyword = normalize_arabic(keyword)
            if not normalized_keyword:
                continue
            start = 0
            while True:
                index = text.find(normalized_keyword, start)
                if index == -1:
                    break
                prefix = text[:index]
                if not _NEGATION_PATTERN.search(prefix + " "):
                    return True
                start = index + len(normalized_keyword)
        return False
