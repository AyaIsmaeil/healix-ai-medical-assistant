"""
Healix - Clinical Interview Output Parser
تحويل مخرجات الـ LLM (نص JSON) إلى كائنات المجال مع تحقّق صارم.

عقدان مدعومان — الثاني توسعة متوافقة خلفياً للأول:

1. عقد القرار (الأصلي، يبقى مقبولاً كما هو):
       {"finished": false, "next_slot": "...", "question": "..."}
       {"finished": true}

2. عقد المقابلة السريرية الموحّد (استخراج + قرار في استدعاء واحد):
       {"chief_complaint": ..., "symptoms": [...], "severity": ...,
        "duration": ..., "body_location": ..., "medications": [...],
        "allergies": [...], "chronic_conditions": [...],
        "family_history": [...], "missing_fields": [...],
        "finished": false, "next_slot": "...", "question": "..."}

``parse_interview_decision`` يقرأ حقول القرار فقط من أيٍّ من الشكلين، فيظلّ
صالحاً دون تعديل. ``parse_interview_turn`` يقرأ العقد الموحّد كاملاً، ويتحمّل
غياب حقول الاستخراج (يعيد سجلاً فارغاً) حتى لا ينكسر أي مزوّد قديم.

يتسامح المحلّل مع أسوار الشيفرة (```json) أو نصّ محيط بسيط، لكنه يرفض أي شكل
غير صالح — الحقول الطبية لا تُخمَّن أبداً عند الغموض.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from app.domain.clinical_record import ClinicalRecord
from app.domain.conversation import InterviewDecision, InterviewTurnOutput, Symptom
from app.exceptions import InterviewParsingError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# نص التصحيح المُرسَل لمزوّد الـLLM عند مخرجات غير مطابقة للعقد الموحّد —
# يُمرَّر كـresponse_format_hint عند بناء المزوّد. لا يحتوي أي بيانات مريض.
INTERVIEW_JSON_NUDGE = (
    "ردّك السابق لم يكن JSON صالحاً بالعقد المطلوب. أعد JSON واحداً فقط، دون "
    'أي نصّ خارجه، بالشكل: {"chief_complaint": null, "symptoms": [], '
    '"severity": null, "duration": null, "body_location": null, '
    '"medications": [], "allergies": [], "chronic_conditions": [], '
    '"family_history": [], "missing_fields": [], "finished": false, '
    '"next_slot": "...", "question": "..."}'
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """استخراج أول كائن JSON من النص (يزيل الأسوار والنص المحيط)."""
    if not text or not text.strip():
        raise InterviewParsingError("مخرجات الـ LLM فارغة.")

    candidate = text.strip()

    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    # الرجوع إلى أول '{' وآخر '}' في حال وجود نصّ محيط.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise InterviewParsingError("لم يُعثر على كائن JSON في مخرجات الـ LLM.")

    snippet = candidate[start : end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise InterviewParsingError(f"JSON غير صالح: {exc}") from exc

    if not isinstance(data, dict):
        raise InterviewParsingError("المتوقَّع كائن JSON وليس نوعاً آخر.")
    return data


def parse_interview_decision(text: str) -> InterviewDecision:
    """تحليل مخرجات الـ LLM إلى قرار مقابلة مُتحقَّق منه."""
    data = _extract_json_object(text)

    if "finished" not in data or not isinstance(data["finished"], bool):
        raise InterviewParsingError("الحقل 'finished' (boolean) مطلوب.")

    if data["finished"]:
        return InterviewDecision(finished=True)

    next_slot = data.get("next_slot")
    question = data.get("question")

    if not isinstance(next_slot, str) or not next_slot.strip():
        raise InterviewParsingError("الحقل 'next_slot' مطلوب عند finished=false.")
    if not isinstance(question, str) or not question.strip():
        raise InterviewParsingError("الحقل 'question' مطلوب عند finished=false.")

    return InterviewDecision(
        finished=False,
        next_slot=next_slot.strip(),
        question=question.strip(),
    )


# ----------------------------------------------------------------------
# العقد الموحّد: استخراج + قرار
# ----------------------------------------------------------------------
def _opt_str(data: Dict[str, Any], key: str):
    """قراءة حقل مفرد اختياري: نص غير فارغ أو None (لا تخمين)."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise InterviewParsingError(f"الحقل '{key}' يجب أن يكون نصاً أو null.")
    trimmed = value.strip()
    return trimmed or None


def _str_list(data: Dict[str, Any], key: str) -> List[str]:
    """قراءة قائمة نصوص اختيارية (الغياب = فارغة؛ النوع الخاطئ = رفض)."""
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise InterviewParsingError(f"الحقل '{key}' يجب أن يكون قائمة نصوص.")
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise InterviewParsingError(f"عناصر '{key}' يجب أن تكون نصوصاً.")
        trimmed = item.strip()
        if trimmed:
            out.append(trimmed)
    return out


def _parse_symptoms(data: Dict[str, Any]) -> List[Symptom]:
    """قراءة الأعراض المستخرَجة. عنصر بلا ``text`` صالح يُرفض ولا يُخمَّن."""
    raw = data.get("symptoms")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InterviewParsingError("الحقل 'symptoms' يجب أن يكون قائمة.")

    symptoms: List[Symptom] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise InterviewParsingError("كل عنصر في 'symptoms' يجب أن يكون كائناً.")
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise InterviewParsingError("كل عرَض يحتاج 'text' نصياً غير فارغ.")

        negated = entry.get("negated", False)
        if not isinstance(negated, bool):
            raise InterviewParsingError("الحقل 'negated' يجب أن يكون boolean.")

        confidence = entry.get("confidence", 1.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise InterviewParsingError("الحقل 'confidence' يجب أن يكون رقماً.")
        # التقييد لا الرفض: قيمة خارج المدى خطأ تنسيق لا خطأ طبي.
        confidence = max(0.0, min(1.0, float(confidence)))

        symptoms.append(
            Symptom(text=text.strip(), negated=negated, confidence=confidence)
        )
    return symptoms


def parse_interview_turn(text: str) -> InterviewTurnOutput:
    """تحليل العقد الموحّد: السجل المنظَّم + الأعراض + قرار الدور."""
    data = _extract_json_object(text)
    decision = parse_interview_decision(text)

    record = ClinicalRecord(
        chief_complaint=_opt_str(data, "chief_complaint"),
        severity=_opt_str(data, "severity"),
        duration=_opt_str(data, "duration"),
        body_location=_opt_str(data, "body_location"),
        medications=_str_list(data, "medications"),
        allergies=_str_list(data, "allergies"),
        chronic_conditions=_str_list(data, "chronic_conditions"),
        family_history=_str_list(data, "family_history"),
        missing_fields=_str_list(data, "missing_fields"),
    )
    return InterviewTurnOutput(
        decision=decision,
        record=record,
        symptoms=_parse_symptoms(data),
    )


def validate_interview_shape(text: str) -> None:
    """بوّابة صارمة لإعادة المحاولة داخل مزوّد الـLLM (response_validator).

    تُمرَّر كـ``response_validator`` عند بناء المزوّد، فيُعيد المزوّد المحاولة
    تلقائياً عند مخرجات غير مطابقة بدل تمريرها للمجال.
    """
    parse_interview_turn(text)
