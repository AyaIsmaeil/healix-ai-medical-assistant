"""
Healix - Symptom Extraction Prompt Builder
بناء تعليمات الـLLM لاستخراج الأعراض من نصّ المريض العربي.

مصدر الحقيقة الوحيد للمفاهيم: ``symptom_evidence_map.yaml`` (مُولَّد من
DDXPlus ``release_evidences.json`` + ``symptom_ontology.json``). كل مفهوم
يُحقَن مع ``evidence_source`` — نصّ السؤال الإنجليزي الرسمي من DDXPlus،
لا ترجمة اجترادية.

العقد: enum مغلق للمفاهيم (``concept``) + نصّ عربي حرّ (``text``) + شاهد
حرفي (``evidence``) + نفي (``negated``) + ثقة (``confidence``).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

LABEL_PATIENT_MESSAGES = "PATIENT_MESSAGES"
LABEL_KNOWN_SYMPTOMS = "KNOWN_SYMPTOMS"
LABEL_CONCEPT_CATALOG = "CONCEPT_CATALOG"


class SymptomExtractionPromptBuilder:
    """يبني تعليمات استخراج الأعراض من كامل رسائل المريض."""

    version = "symptom-extract-v1"

    def __init__(self, mappings: Sequence[Dict[str, Any]]) -> None:
        self._mappings = list(mappings)
        self._concepts = [m["concept"] for m in self._mappings if m.get("concept")]

    @property
    def concepts(self) -> List[str]:
        return list(self._concepts)

    def catalog(self) -> List[Dict[str, Any]]:
        """كتalog المفاهيم للحقن في البرومبت — مصدر طبي موثَّق لكل مدخلة."""
        return [
            {
                "concept": entry["concept"],
                "arabic_synonyms": entry.get("names", []),
                "evidence_source": entry.get("evidence_source", ""),
                "ddxplus_codes": entry.get("evidence_codes", []),
            }
            for entry in self._mappings
        ]

    def schema(self) -> Dict[str, Any]:
        symptom_item = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "concept": {"type": "string", "enum": self._concepts},
                "negated": {"type": "boolean"},
                "confidence": {"type": "number"},
                "evidence": {"type": ["string", "null"]},
            },
            "required": ["text", "concept", "negated", "confidence", "evidence"],
            "additionalProperties": False,
        }
        return {
            "name": "symptom_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "symptoms": {"type": "array", "items": symptom_item},
                },
                "required": ["symptoms"],
                "additionalProperties": False,
            },
        }

    def system_prompt(self) -> str:
        return (
            "أنت مساعد استخراج أعراض طبية من نصّ عربي. مهمتك استخراج الأعراض "
            "التي ذكرها المريض صراحةً كحالة حالية، وربط كل عرض بمفهوم من "
            "CONCEPT_CATALOG فقط.\n\n"
            "CONCEPT_CATALOG يحوي مفاهيم DDXPlus الرسمية — كل مدخلة تتضمّن:\n"
            "  - concept: اسم المفهوم (يُستخدم في JSON)\n"
            "  - arabic_synonyms: مرادفات عربية معروفة\n"
            "  - evidence_source: نصّ السؤال الإنجليزي الرسمي من DDXPlus "
            "(مرجع طبي — لا تُشخّص، استخدمه للتأكّد من مطابقة المفهوم)\n"
            "  - ddxplus_codes: رموز الأدلة E_*\n\n"
            "قواعد الاستخراج:\n"
            "1. استخرج من PATIENT_MESSAGES كاملةً.\n"
            "2. لا تختلق عرضاً لم يُذكر. لا تستنتج مفهوماً خارج القائمة.\n"
            "3. الأعراض المنفية (مثلاً \"ما عندي حرارة\"): negated=true.\n"
            "4. confidence بين 0 و1 حسب وضوح ذكر العرض.\n"
            "5. evidence: انسخ المقطع الحرفي من كلام المريض — لا إعادة صياغة. "
            "إن استنتجت العرض بلا نصّ حرفي فاجعل evidence=null.\n"
            "6. text: صياغة عربية مختصرة للعرض (من مرادفات المفهوم أو كلام المريض).\n"
            "7. concept: المفهوم المطابق من CONCEPT_CATALOG فقط.\n\n"
            "ممنوع: أي تشخيص، احتمالات أمراض، تخصّص، أو درجة خطورة.\n\n"
            'أعد JSON فقط: {"symptoms": [{"text": "...", "concept": "...", '
            '"negated": false, "confidence": 0.9, "evidence": "..."}]}'
        )

    def extraction_prompt(
        self,
        raw_messages: List[str],
        known_symptoms: List[str] | None = None,
    ) -> str:
        def line(label: str, value) -> str:
            return f"{label}: {json.dumps(value, ensure_ascii=False)}"

        block = "\n".join([
            line(LABEL_PATIENT_MESSAGES, raw_messages),
            line(LABEL_KNOWN_SYMPTOMS, known_symptoms or []),
            line(LABEL_CONCEPT_CATALOG, self.catalog()),
        ])
        return (
            "الحالة الحالية:\n"
            f"{block}\n\n"
            "استخرج الأعراض المنطبقة من الرسائل، واربط كل عرض بمفهوم من "
            "CONCEPT_CATALOG، وأعد JSON فقط."
        )
