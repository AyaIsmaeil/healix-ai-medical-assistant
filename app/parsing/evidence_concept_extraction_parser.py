"""
Healix - Evidence Concept Extraction Parser
تحويل مخرجات EvidenceConceptExtractor (نص JSON) إلى قائمة مفاهيم.

بنفس بنية ``assessment_extraction_parser``: بوّابة صارمة لإعادة المحاولة
(``validate_evidence_concepts_shape``) + تحليل متساهل كخطّ دفاع أخير
(``parse_evidence_concepts``) يُسقِط أي مفهوم خارج القائمة المعروفة بدل
رفع خطأ — النموذج مُقيَّد أصلاً بـenum، فهذا تحقّق دفاعي إضافي فقط.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from app.exceptions import FeatureExtractionError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

EVIDENCE_CONCEPT_JSON_NUDGE = (
    'ردّك السابق لم يطابق الشكل المطلوب. أعد JSON فقط: {"concepts": []} '
    "أو قائمة تحوي فقط مفاهيم من القائمة المُعطاة."
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise FeatureExtractionError("مخرجات LLM اختيار مفاهيم الأدلة فارغة.")

    candidate = text.strip()
    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise FeatureExtractionError("لم يُعثر على كائن JSON بمخرجات اختيار مفاهيم الأدلة.")

    try:
        data = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise FeatureExtractionError(f"JSON غير صالح باختيار مفاهيم الأدلة: {exc}") from exc

    return data if isinstance(data, dict) else {}


def validate_evidence_concepts_shape(text: str) -> None:
    """بوّابة صارمة لإعادة المحاولة داخل مزوّد الـLLM (response_validator)."""
    data = _extract_json_object(text)
    if "concepts" not in data or not isinstance(data["concepts"], list):
        raise FeatureExtractionError("مخرجات اختيار مفاهيم الأدلة تفتقد 'concepts' كقائمة.")


def parse_evidence_concepts(text: str, known_concepts: List[str]) -> List[str]:
    """تحليل متساهل: يُسقِط أي مفهوم غير معروف أو مكرَّر بصمت بدل رفع خطأ."""
    data = _extract_json_object(text)
    raw = data.get("concepts")
    if not isinstance(raw, list):
        return []

    known = set(known_concepts)
    seen: List[str] = []
    for item in raw:
        if isinstance(item, str) and item in known and item not in seen:
            seen.append(item)
    return seen
