"""
Healix - Assessment Feature Extraction Prompt Builder
بناء تعليمات الـLLM لاستخراج الحقول التي عجز عنها المستخرج القاعدي.

بنفس أسلوب ``InterviewPromptBuilder``: نظام + تعليمات دور، تسميات JSON
واضحة، عقد إخراج ثابت الشكل (نفس 6 مفاتيح دائماً؛ غير المطلوب منها يبقى
null). هذا العقد مقصود أن يكون بسيطاً وغير مُحسَّن بعد (كما هو مطلوب
بمرحلة ٣.١) — التحسين والتوسّع لاحقاً بمراحل قادمة.

ملاحظة تصميم: العقد ثابت الشكل دائماً (لا JSON schema ديناميكي لكل طلب)
تفادياً لتعقيد أوضاع الإخراج المنظَّم لدى مزوّدي الـLLM الحاليين.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

LABEL_PATIENT_MESSAGES = "PATIENT_MESSAGES"
LABEL_UNRESOLVED_FIELDS = "UNRESOLVED_FIELDS"

FIELD_KEYS = ("age", "gender", "smoking", "temperature", "duration", "severity")

# مخطط JSON Schema صارم يطابق FIELD_KEYS تماماً — يُمرَّر لمزوّد الـLLM
# (response_schema بـQwenOpenRouterProvider) بدل عقد المقابلة الافتراضي.
# بنفس شكل _INTERVIEW_SCHEMA (name/strict/schema) حسب اتفاقية OpenRouter.
EXTRACTION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "assessment_feature_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "age": {"type": ["integer", "null"]},
            "gender": {"type": ["string", "null"], "enum": ["male", "female", None]},
            "smoking": {"type": ["boolean", "null"]},
            "temperature": {"type": ["number", "null"]},
            "duration": {"type": ["string", "null"]},
            "severity": {"type": ["integer", "null"]},
        },
        "required": list(FIELD_KEYS),
        "additionalProperties": False,
    },
}


class AssessmentExtractionPromptBuilder:
    """يبني تعليمات استخراج الحقول الستة الصريحة من كامل رسائل المريض."""

    version = "assessment-extract-v1"

    def system_prompt(self) -> str:
        return (
            "أنت مساعد استخراج بيانات طبية من نصّ عربي. مهمتك استخراج قيم "
            "محدّدة فقط مما ذكره المريض صراحة أو ضمناً — لا تُشخّص ولا تستنتج "
            "معلومات لم تُذكر إطلاقاً.\n\n"
            "تتلقّى PATIENT_MESSAGES (كل رسائل المريض) وUNRESOLVED_FIELDS "
            "(الحقول التي لم تُحلّ بعد بطريقة قاعدية). استخرج فقط الحقول "
            "المذكورة بـUNRESOLVED_FIELDS من محتوى الرسائل.\n\n"
            "أعد ردّك بصيغة JSON فقط، بالمفاتيح الستة التالية دائماً "
            "(اجعل أي حقل لم يُذكر أو ليس ضمن UNRESOLVED_FIELDS يساوي null):\n"
            '{"age": <رقم أو null>, "gender": <"male"|"female"|null>, '
            '"smoking": <true|false|null>, "temperature": <رقم أو null>, '
            '"duration": <نص أو null>, "severity": <رقم من 0 إلى 10 أو null>}\n\n'
            "لا تختلق قيماً غير مذكورة. لا نصّ خارج كائن الـJSON."
        )

    def extraction_prompt(self, raw_messages: List[str], unresolved_fields: List[str]) -> str:
        def line(label: str, value) -> str:
            return f"{label}: {json.dumps(value, ensure_ascii=False)}"

        block = "\n".join([
            line(LABEL_PATIENT_MESSAGES, raw_messages),
            line(LABEL_UNRESOLVED_FIELDS, unresolved_fields),
        ])
        return (
            "الحالة الحالية:\n"
            f"{block}\n\n"
            "استخرج الحقول المطلوبة فقط من الرسائل أعلاه، وأعد JSON فقط."
        )
