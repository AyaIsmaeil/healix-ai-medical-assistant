"""
عقد JSON لاستخراج المعرفة الطبية من ملخصات PubMed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.exceptions import HealixError

TRIAGE_LEVELS = ("Emergency", "Urgent", "Routine")

KNOWLEDGE_EXTRACTION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "medical_knowledge_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "disease_name": {"type": "string"},
            "associated_symptoms": {
                "type": "array",
                "items": {"type": "string"},
            },
            "medical_specialty": {"type": "string"},
            "triage_level": {
                "type": "string",
                "enum": list(TRIAGE_LEVELS),
            },
            "clinical_summary": {"type": "string"},
            "red_flags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "disease_name",
            "associated_symptoms",
            "medical_specialty",
            "triage_level",
            "clinical_summary",
            "red_flags",
        ],
        "additionalProperties": False,
    },
}

KNOWLEDGE_JSON_NUDGE = (
    "أعد JSON صالحاً فقط بالمفاتيح المطلوبة: disease_name, associated_symptoms, "
    "medical_specialty, triage_level (Emergency|Urgent|Routine), clinical_summary, red_flags. "
    "لا نص خارج JSON."
)


class ExtractedMedicalKnowledge(BaseModel):
    """حقول مستخرجة من ملخّص بحث علمي."""

    disease_name: str
    associated_symptoms: List[str] = Field(default_factory=list)
    medical_specialty: str
    triage_level: Literal["Emergency", "Urgent", "Routine"]
    clinical_summary: str = ""
    red_flags: List[str] = Field(default_factory=list)

    # بيانات المصدر (تُضاف بعد الاستخراج، ليست جزءاً من مخرجات LLM)
    source_pmid: Optional[str] = None
    source_title: Optional[str] = None
    source_journal: Optional[str] = None
    source_year: Optional[str] = None
    source_query_disease: Optional[str] = None
    extraction_mode: Optional[str] = None

    @field_validator("associated_symptoms", "red_flags", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []


class KnowledgeExtractionError(HealixError):
    """فشل التحقّق من مخرجات استخراج المعرفة."""


def validate_knowledge_extraction_shape(raw_json: str) -> None:
    """بوّابة تحقّق لإعادة المحاولة داخل مزوّد LLM (response_validator)."""
    validate_knowledge_extraction_json(raw_json)


def validate_knowledge_extraction_json(raw_json: str) -> ExtractedMedicalKnowledge:
    """يتحقّق من شكل JSON ويُعيد نموذج Pydantic."""
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise KnowledgeExtractionError(f"JSON غير صالح: {exc}") from exc

    if not isinstance(payload, dict):
        raise KnowledgeExtractionError("المخرج ليس كائن JSON.")

    try:
        return ExtractedMedicalKnowledge.model_validate(payload)
    except Exception as exc:
        raise KnowledgeExtractionError(f"عقد الاستخراج غير مطابق: {exc}") from exc
