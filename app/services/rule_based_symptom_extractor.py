"""
Healix - Rule-Based Symptom Extractor
استخراج حتمي للأعراض من نص المريض — احتياطي موثوق عند غياب/فشل الـLLM.

يستخدم ``symptom_evidence_map.yaml`` (مصدر DDXPlus): مطابقة جزئية بعد
``normalize_arabic``، بلا تخمين خارج القاموس المغلق.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from app.domain.conversation import Symptom
from app.domain.text_preprocessing import normalize_arabic

_NEGATION_PATTERN = re.compile(
    r"(?:^|\s)(?:لا|ما|ليس|بدون|بلا|من\s+غير|مافي|ما\s+في)\s",
    re.UNICODE,
)


class RuleBasedSymptomExtractor:
    """يستخرج أعراضاً حتمياً من نص عربي عبر قاموس symptom_evidence_map."""

    def __init__(self, mappings: Sequence[Dict[str, Any]]) -> None:
        self._entries: List[tuple[str, str, tuple[str, ...]]] = [
            (entry["concept"], entry["names"][0], tuple(entry.get("names", ())))
            for entry in mappings
            if entry.get("concept") and entry.get("names")
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleBasedSymptomExtractor":
        return cls(mappings=data.get("mappings", ()))

    def extract(self, raw_messages: Sequence[str]) -> List[Symptom]:
        text = normalize_arabic(" ".join(str(m) for m in raw_messages))
        if not text:
            return []

        found: Dict[tuple[str, bool], Symptom] = {}
        for concept, canonical, keywords in self._entries:
            for keyword in sorted(keywords, key=len, reverse=True):
                norm_kw = normalize_arabic(keyword)
                if not norm_kw:
                    continue
                start = 0
                while True:
                    index = text.find(norm_kw, start)
                    if index == -1:
                        break
                    prefix = text[:index]
                    negated = bool(_NEGATION_PATTERN.search(prefix + " "))
                    key = (concept, negated)
                    if key not in found:
                        found[key] = Symptom(
                            text=canonical,
                            negated=negated,
                            confidence=0.85,
                            evidence=keyword,
                        )
                    start = index + len(norm_kw)

        return list(found.values())
