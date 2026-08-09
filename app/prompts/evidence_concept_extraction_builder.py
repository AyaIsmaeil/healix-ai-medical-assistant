"""
Healix - Evidence Concept Extraction Prompt Builder
بناء تعليمات الـLLM لاختيار مفاهيم أدلة DDXPlus المنطبقة من نص المريض.

قرار تصميم (بُني على بحث، لا اختراع): الأبحاث المنشورة عن الترميز السريري
بالـLLM تُظهر أنّ النماذج أوثق كـ"مُتحقِّق يختار من مجموعة مرشَّحة مغلقة" لا
كـ"مولِّد حرّ" — لذلك المخطّط أدناه يُقيَّد بـenum (قائمة المفاهيم المعروفة
فقط)، بدل ترك النموذج يكتب نصاً حرّاً يُطابَق لاحقاً بكود بايثون (الأسلوب
الهشّ الذي حلّ محلّه هذا الملف).

بنفس أسلوب ``AssessmentExtractionPromptBuilder``: نظام + تعليمات دور، تسمية
JSON واضحة، عقد إخراج ثابت الشكل (enum من قائمة مفاهيم مُحقَنة عند البناء —
مصدرها الوحيد ``symptom_evidence_map.yaml`` عبر ``SymptomEvidenceEncoder.concepts``،
لا تكرار للقائمة هنا).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

LABEL_PATIENT_MESSAGES = "PATIENT_MESSAGES"
LABEL_KNOWN_CONCEPTS = "KNOWN_CONCEPTS"


class EvidenceConceptExtractionPromptBuilder:
    """يبني تعليمات اختيار مفاهيم الأدلة المنطبقة من كامل رسائل المريض."""

    version = "evidence-concept-extract-v1"

    def __init__(self, concepts: List[str]) -> None:
        self._concepts = list(concepts)

    @property
    def concepts(self) -> List[str]:
        return list(self._concepts)

    def schema(self) -> Dict[str, Any]:
        """مخطّط JSON صارم: مصفوفة عناصرها enum مغلق — لا توليد نصّ حرّ."""
        return {
            "name": "evidence_concept_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "items": {"type": "string", "enum": self._concepts},
                    },
                },
                "required": ["concepts"],
                "additionalProperties": False,
            },
        }

    def system_prompt(self) -> str:
        concepts_list = "، ".join(self._concepts)
        return (
            "أنت مساعد استخراج بيانات طبية من نصّ عربي. مهمتك اختيار أيّ "
            f"من المفاهيم التالية فقط ذكرها المريض صراحةً كعرَض حالي (غير "
            f"منفيّ): {concepts_list}.\n\n"
            "لا تستنتج مفهوماً لم يُذكر، ولا تُدرج عرَضاً نفاه المريض صراحةً "
            "(مثلاً \"ما عندي حرارة\" لا تعني اختيار fever). لا تخترع مفهوماً "
            "خارج القائمة أعلاه — أي عرَض آخر يُتجاهَل (يبقى غير مغطّى بهذا "
            "الإصدار المصغَّر).\n\n"
            'أعد ردّك بصيغة JSON فقط: {"concepts": [<قائمة من القائمة أعلاه>]}. '
            "قائمة فارغة إن لم ينطبق أي مفهوم. لا نصّ خارج كائن الـJSON."
        )

    def extraction_prompt(self, raw_messages: List[str]) -> str:
        def line(label: str, value) -> str:
            return f"{label}: {json.dumps(value, ensure_ascii=False)}"

        block = "\n".join([
            line(LABEL_PATIENT_MESSAGES, raw_messages),
            line(LABEL_KNOWN_CONCEPTS, self._concepts),
        ])
        return (
            "الحالة الحالية:\n"
            f"{block}\n\n"
            "اختر المفاهيم المنطبقة فقط من القائمة أعلاه، وأعد JSON فقط."
        )
