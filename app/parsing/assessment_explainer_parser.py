"""
Healix - Assessment Explainer Parser (Phase 3.8)
تحويل مخرجات ``AssessmentExplainer`` (نص JSON) إلى ``AssessmentExplanation``
مع تحقّق صارم — يرفض أي JSON مشوّه أو ناقص المفاتيح.

يوفّر طبقتَي دفاع بنفس نمط ``assessment_extraction_parser``:

1. ``validate_explanation_shape`` — بوّابة صارمة تُمرَّر كـ``response_validator``
   لمثيل ``QwenOpenRouterProvider`` الخاص بالتفسير (انظر ``main.py``). تفرض
   وجود المفاتيح الأربعة نصوصاً غير فارغة، فتُشغِّل آلية إعادة المحاولة +
   التلقين المدمجة بالمزوّد إن أعاد النموذج شكلاً خاطئاً.
2. ``parse_assessment_explanation`` — تحليل صارم أيضاً (خط الدفاع الذي
   تستهلكه الخدمة): يرفض المشوّه/الناقص برفع ``AssessmentExplanationError``.
   الخدمة (AssessmentExplainer) تلتقط هذا الخطأ وتتدهور بلطف إلى تفسير حتمي
   بديل — فلا يتسرّب للراوت ولا يُسقط التقييم المحسوب سلفاً.

استخراج الكائن من النص (إزالة أسوار ```json```) منطق محلي مستقل عمداً — لا
يُستورد من ``parsing.interview_parser`` (ملف مجمَّد) تفادياً للاقتران.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from app.domain.explanation import AssessmentExplanation
from app.exceptions import AssessmentExplanationError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_REQUIRED_FIELDS = ("summary", "medical_reasoning", "recommendation", "disclaimer")

# نص التصحيح المُرسَل لمزوّد الـLLM عند فشل validate_explanation_shape —
# يُمرَّر كـresponse_format_hint عند بناء المزوّد (بدل نصّ تصحيح المقابلة).
EXPLANATION_JSON_NUDGE = (
    "ردّك السابق لم يطابق الشكل المطلوب. أعد JSON فقط بالمفاتيح الأربعة "
    "التالية دائماً (كلّها نصوص عربية غير فارغة):\n"
    '{"summary": "...", "medical_reasoning": "...", '
    '"recommendation": "...", "disclaimer": "..."}'
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise AssessmentExplanationError("مخرجات LLM التفسير فارغة.")

    candidate = text.strip()
    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise AssessmentExplanationError("لم يُعثر على كائن JSON بمخرجات التفسير.")

    try:
        data = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AssessmentExplanationError(f"JSON غير صالح بمخرجات التفسير: {exc}") from exc

    if not isinstance(data, dict):
        raise AssessmentExplanationError("المتوقَّع كائن JSON وليس نوعاً آخر.")
    return data


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_explanation_shape(text: str) -> None:
    """بوّابة صارمة لإعادة المحاولة داخل مزوّد الـLLM (response_validator).

    تفرض وجود المفاتيح الأربعة نصوصاً غير فارغة — أي رد لا يطابق هذا الشكل
    يُرفض فيُعاد المحاولة مع نص تصحيح موجَّه، بدل قبوله بصمت.
    """
    data = _extract_json_object(text)
    missing = [key for key in _REQUIRED_FIELDS if not _non_empty_str(data.get(key))]
    if missing:
        raise AssessmentExplanationError(
            f"مخرجات التفسير ناقصة/فارغة المفاتيح: {missing}"
        )


def parse_assessment_explanation(text: str) -> AssessmentExplanation:
    """تحليل صارم: يرفض المشوّه أو ناقص المفاتيح برفع ``AssessmentExplanationError``.

    يُعيد ``AssessmentExplanation`` بالنصوص الأربعة (مُشذَّبة) عند النجاح فقط.
    """
    data = _extract_json_object(text)

    missing = [key for key in _REQUIRED_FIELDS if not _non_empty_str(data.get(key))]
    if missing:
        raise AssessmentExplanationError(
            f"مخرجات التفسير ناقصة/فارغة المفاتيح: {missing}"
        )

    return AssessmentExplanation(
        summary=data["summary"].strip(),
        medical_reasoning=data["medical_reasoning"].strip(),
        recommendation=data["recommendation"].strip(),
        disclaimer=data["disclaimer"].strip(),
    )
