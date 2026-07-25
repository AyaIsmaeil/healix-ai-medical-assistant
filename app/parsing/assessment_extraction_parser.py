"""
Healix - Assessment Feature Extraction Parser
تحويل مخرجات LLMFeatureExtractor (نص JSON) إلى ``ExtractionResult``.

يوفّر هذا الملف طبقتَي دفاع متكاملتين لا متعارضتين:

1. ``validate_extraction_shape`` — بوّابة صارمة تُمرَّر كـ``response_validator``
   لمثيل ``QwenOpenRouterProvider`` الخاص بمحرك التقييم (انظر ``main.py``).
   تفرض وجود المفاتيح الستة كاملة، فتُشغِّل آلية إعادة المحاولة + التلقين
   المدمجة بالمزوّد إن أعاد النموذج شكلاً خاطئاً (كعقد المقابلة) — يعطي
   النموذج فرصة تصحيح نفسه قبل الاستسلام.
2. ``parse_extraction_result`` — متساهل عمداً (خط دفاع أخير): يبحث عن
   المفاتيح الستة بأي كائن JSON يصله دون فرض بنية صارمة، فحتى لو استُنفدت
   محاولات الخطوة (1) ووصل شكل غير متوقّع، لا ينهار — تُحسم الحقول
   الناقصة "غير محلولة" بدل رفع خطأ.

استخراج الكائن من النص (إزالة أسوار ```json```) منطق محلي مستقل عمداً —
لا يُستورد من ``parsing.interview_parser`` (ملف مجمَّد) لتفادي الاقتران
بتفصيل تنفيذي خاص قد يتغيّر شكله لاحقاً هناك.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from app.domain.feature_extraction_rules import FIELDS as _RULE_FIELDS
from app.domain.feature_extraction_rules import RuleExtractionResult
from app.exceptions import FeatureExtractionError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# نص التصحيح المُرسَل لمزوّد الـLLM عند فشل validate_extraction_shape —
# يُمرَّر كـresponse_format_hint عند بناء المزوّد (بدل نصّ تصحيح المقابلة).
EXTRACTION_JSON_NUDGE = (
    "ردّك السابق لم يطابق الشكل المطلوب. أعد JSON فقط بالمفاتيح الستة "
    "التالية دائماً (استخدم null لأي حقل غير مذكور):\n"
    '{"age": null, "gender": null, "smoking": null, "temperature": null, '
    '"duration": null, "severity": null}'
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise FeatureExtractionError("مخرجات LLM الاستخراج فارغة.")

    candidate = text.strip()
    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise FeatureExtractionError("لم يُعثر على كائن JSON بمخرجات استخراج الميزات.")

    try:
        data = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise FeatureExtractionError(f"JSON غير صالح باستخراج الميزات: {exc}") from exc

    return data if isinstance(data, dict) else {}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):  # bool هو subclass من int — استبعاده صراحة
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def validate_extraction_shape(text: str) -> None:
    """بوّابة صارمة لإعادة المحاولة داخل مزوّد الـLLM (response_validator).

    تفرض وجود المفاتيح الستة كاملة (حتى لو قيمتها null) — أي رد لا يطابق
    هذا الشكل (كعقد المقابلة عند تعارض غير متوقّع) يُرفض فيُعاد المحاولة
    مع نص تصحيح موجَّه، بدل قبوله بصمت. هذه بوّابة صرف — لا تُستخدم لبناء
    ``RuleExtractionResult`` الفعلي؛ ``parse_extraction_result`` المتساهل هو
    المسؤول عن ذلك بعد نجاح هذا التحقّق (أو استنفاد المحاولات).
    """
    data = _extract_json_object(text)
    missing = [key for key in _RULE_FIELDS if key not in data]
    if missing:
        raise FeatureExtractionError(
            f"مخرجات استخراج الميزات ناقصة المفاتيح: {missing}"
        )


def parse_extraction_result(text: str) -> RuleExtractionResult:
    """تحليل متساهل: أي مفتاح غير موجود أو من نوع خاطئ يبقى None بلا خطأ.

    يرفع ``FeatureExtractionError`` فقط عند تعذّر تحليل نص JSON من الأساس.
    """
    data = _extract_json_object(text)

    return RuleExtractionResult(
        age=_as_int(data.get("age")),
        gender=_as_str(data.get("gender")),
        smoking=_as_bool(data.get("smoking")),
        temperature=_as_float(data.get("temperature")),
        duration_text=_as_str(data.get("duration")),
        duration_days=None,  # الـLLM يُرجع نصّاً حرّاً؛ لا تحويل رقمي هنا بمرحلة ٣.١
        severity=_as_int(data.get("severity")),
    )
