"""
Healix - Symptom Extraction Parser
تحليل مخرجات SymptomExtractor (JSON) إلى قائمة Symptom.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from app.domain.conversation import Symptom
from app.exceptions import FeatureExtractionError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

SYMPTOM_EXTRACTION_JSON_NUDGE = (
    'ردّك السابق لم يطابق الشكل المطلوب. أعد JSON فقط: {"symptoms": []} '
    "أو قائمة أعراض بحقول text, concept, negated, confidence, evidence."
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise FeatureExtractionError("مخرجات استخراج الأعراض فارغة.")

    candidate = text.strip()
    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise FeatureExtractionError("لم يُعثر على JSON بمخرجات استخراج الأعراض.")

    try:
        data = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise FeatureExtractionError(f"JSON غير صالح: {exc}") from exc

    return data if isinstance(data, dict) else {}


def validate_symptom_extraction_shape(text: str) -> None:
    """بوّابة صارمة لإعادة المحاولة داخل مزوّد الـLLM."""
    data = _extract_json_object(text)
    if "symptoms" not in data or not isinstance(data["symptoms"], list):
        raise FeatureExtractionError("مخرجات استخراج الأعراض تفتقد 'symptoms' كقائمة.")


def parse_symptoms(text: str, known_concepts: List[str]) -> List[Symptom]:
    """تحليل متساهل: يُسقِط أي عرض بمفهوم غير معروف."""
    data = _extract_json_object(text)
    raw = data.get("symptoms")
    if not isinstance(raw, list):
        return []

    known = set(known_concepts)
    symptoms: List[Symptom] = []
    seen: set[tuple[str, bool]] = set()

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        concept = entry.get("concept")
        text_val = entry.get("text")
        if not isinstance(concept, str) or concept not in known:
            continue
        if not isinstance(text_val, str) or not text_val.strip():
            continue

        negated = entry.get("negated", False)
        if not isinstance(negated, bool):
            negated = False

        confidence = entry.get("confidence", 1.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = 1.0
        confidence = max(0.0, min(1.0, float(confidence)))

        evidence = entry.get("evidence")
        if evidence is not None and not isinstance(evidence, str):
            evidence = None
        evidence = evidence.strip() if isinstance(evidence, str) and evidence.strip() else None

        key = (concept, negated)
        if key in seen:
            continue
        seen.add(key)

        # text = Arabic display; concept stored in text via canonical name from mapping
        symptoms.append(Symptom(
            text=text_val.strip(),
            negated=negated,
            confidence=confidence,
            evidence=evidence,
        ))

    return symptoms
