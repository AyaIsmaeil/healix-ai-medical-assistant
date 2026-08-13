"""
Healix — Slot Answer Validator
تحقّق حتمي من إجابات المريض مقابل الخانة المُطروحة.

مبنيّ على مبادئ أخذ التاريخ المُوجَّه (OLDCARTS) ومعايير triage السريرية —
لا يعتمد على الـ LLM لقبول/رفض الإجابة. يرفع موثوقية البيانات قبل التقييم.

مراجع منهجية:
- OLDCARTS / Calgary-Cambridge للمقابلة السريرية المُنظَّمة
- NICE CG95 / ESC Guidelines — أسئلة نعم/لا صريحة لألم الصدر المصاحب
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.domain.text_preprocessing import normalize_arabic

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_YES_RE = re.compile(
    r"^(?:نعم|أيو[ae]?|ايو[ae]?|صح|موافق|مع(?:ا|ه)?|طبع[ao]?|بالتأ?كيد|yes|y)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"^(?:لا|ل[ao]|مو|م[ao]|ابدا|أ?بدا|ابداً|ما\s*(?:في|فيه|عندي|صار)|no|n)\b",
    re.IGNORECASE,
)
_UNKNOWN_RE = re.compile(
    r"(?:لا\s*(?:أ|ا)?\s*عرف|لا\s*ادري|مش\s*عارف|ما\s*بعرف|unknown)",
    re.IGNORECASE,
)
_BARE_AGE_RE = re.compile(r"^\s*(\d{1,3})\s*$")
_EXPLICIT_AGE_RE = re.compile(r"عمري\s*(\d{1,3})")
_AGE_WITH_UNIT_RE = re.compile(r"(\d{1,3})\s*(?:سنة|سنه|عام|سنوات)")
_SEVERITY_NUMERIC_RE = re.compile(r"\b(10|[0-9])\b")
_SEVERITY_SCALE_RE = re.compile(r"(\d{1,2})\s*(?:من|/)\s*10")

_SEVERITY_WORDS = {
    "خفيف": 2, "بسيط": 2, "خفيفة": 2,
    "متوسط": 5, "متوسطة": 5,
    "شديد": 8, "شديدة": 8,
    "شديد جدا": 9, "شديد جداً": 9, "شديدة جدا": 9,
}

_GENDER_MALE = ("ذكر", "رجل", "male", "ولد", "صبي")
_GENDER_FEMALE = ("أنثى", "انثى", "انثي", "امرأة", "امراه", "مرأة", "مراه", "بنت", "female", "سيدة")

# خانات نعم/لا — مطابقة لـ SYMPTOM_SPECIFIC + CONTEXT (حمل)
_YES_NO_SLOT_SUFFIXES = frozenset({
    "chills", "cough", "sore_throat", "vomiting", "diarrhea", "constipation",
    "radiation", "dyspnea", "sweating", "exertion", "nausea", "photophobia",
    "neck_stiffness", "sputum", "hemoptysis", "onset_sudden", "side_affected",
    "pregnancy", "smoking",
})

_CHRONIC_CONDITION_HINTS = (
    "ضغط", "سكري", "سكر", "ربو", "قلب", "كلى", "كبد", "سرطان",
    "ارتفاع ضغط", "ارتفاع ضغط الدم", "hypertension", "diabetes",
)


class SlotAnswerStatus(str, Enum):
    VALID = "valid"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True)
class SlotAnswerResult:
    """نتيجة التحقّق من إجابة خانة واحدة."""

    status: SlotAnswerStatus
    clarification_question: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    pregnancy_possible: Optional[bool] = None
    yes_no: Optional[bool] = None
    severity_text: Optional[str] = None
    severity_numeric: Optional[int] = None
    chronic_condition: Optional[str] = None


def _normalize_message(message: str) -> str:
    return normalize_arabic(message.translate(_ARABIC_INDIC))


def _slot_kind(slot: str) -> str:
    """يُستخرج نوع الخانة من target (مثل severity@عرض أو context:age)."""
    if slot.startswith("context:"):
        return slot.split(":", 1)[1]
    if "@" in slot:
        return slot.split("@", 1)[0]
    return slot


def _parse_yes_no(text: str) -> Optional[bool]:
    normalized = _normalize_message(text)
    if _YES_RE.search(normalized):
        return True
    if _NO_RE.search(normalized):
        return False
    return None


def _parse_age(text: str) -> Optional[int]:
    raw = text.translate(_ARABIC_INDIC).strip()
    match = _EXPLICIT_AGE_RE.search(raw)
    if match:
        age = int(match.group(1))
        return age if 0 < age < 120 else None
    match = _AGE_WITH_UNIT_RE.search(raw)
    if match:
        age = int(match.group(1))
        return age if 0 < age < 120 else None
    match = _BARE_AGE_RE.match(raw)
    if match:
        age = int(match.group(1))
        return age if 0 < age < 120 else None
    return None


def _parse_gender(text: str) -> Optional[str]:
    normalized = _normalize_message(text)
    if any(h in normalized for h in _GENDER_FEMALE):
        return "female"
    if any(h in normalized for h in _GENDER_MALE):
        return "male"
    return None


def _parse_severity(text: str) -> tuple[Optional[str], Optional[int]]:
    raw = text.translate(_ARABIC_INDIC).strip()
    scale = _SEVERITY_SCALE_RE.search(raw)
    if scale:
        value = int(scale.group(1))
        if 0 <= value <= 10:
            return raw, value
    for word, value in _SEVERITY_WORDS.items():
        if word in _normalize_message(raw):
            return raw, value
    match = _SEVERITY_NUMERIC_RE.search(raw)
    if match and len(raw) <= 4:
        value = int(match.group(1))
        if 0 <= value <= 10:
            return raw, value
    return None, None


def _clarify_yes_no(slot: str) -> str:
    return "لم أفهم إجابتك — هل جوابك «نعم» أم «لا»؟"


def validate_slot_answer(slot: Optional[str], message: str) -> SlotAnswerResult:
    """يتحقّق من أن إجابة المريض تطابق الخانة المُطروحة.

    يُعاد ``VALID`` عند غياب ``slot`` (لا خانة معلّقة) — لا يُرفض شيء.
    """
    if not slot:
        return SlotAnswerResult(status=SlotAnswerStatus.VALID)

    kind = _slot_kind(slot)
    text = message.strip()
    if not text:
        return SlotAnswerResult(
            status=SlotAnswerStatus.NEEDS_CLARIFICATION,
            clarification_question="يرجى الإجابة على السؤال السابق.",
        )

    if _UNKNOWN_RE.search(_normalize_message(text)):
        return SlotAnswerResult(status=SlotAnswerStatus.VALID)

    if kind == "age":
        age = _parse_age(text)
        if age is None:
            return SlotAnswerResult(
                status=SlotAnswerStatus.NEEDS_CLARIFICATION,
                clarification_question="كم عمرك بالسنوات؟ (مثال: 23)",
                age=None,
            )
        return SlotAnswerResult(status=SlotAnswerStatus.VALID, age=age)

    if kind == "gender":
        gender = _parse_gender(text)
        if gender is None:
            return SlotAnswerResult(
                status=SlotAnswerStatus.NEEDS_CLARIFICATION,
                clarification_question="هل أنت ذكر أم أنثى؟",
            )
        return SlotAnswerResult(status=SlotAnswerStatus.VALID, gender=gender)

    if kind == "pregnancy":
        answer = _parse_yes_no(text)
        if answer is None:
            return SlotAnswerResult(
                status=SlotAnswerStatus.NEEDS_CLARIFICATION,
                clarification_question=_clarify_yes_no(slot),
            )
        return SlotAnswerResult(status=SlotAnswerStatus.VALID, pregnancy_possible=answer)

    if kind == "severity":
        severity_text, severity_numeric = _parse_severity(text)
        if severity_numeric is None:
            return SlotAnswerResult(
                status=SlotAnswerStatus.NEEDS_CLARIFICATION,
                clarification_question="ما شدّة الألم من 1 إلى 10؟ (أو: خفيف / متوسط / شديد)",
            )
        return SlotAnswerResult(
            status=SlotAnswerStatus.VALID,
            severity_text=severity_text,
            severity_numeric=severity_numeric,
        )

    if kind == "chronic_diseases":
        answer = _parse_yes_no(text)
        if answer is False:
            return SlotAnswerResult(status=SlotAnswerStatus.VALID, yes_no=False)
        if answer is True:
            return SlotAnswerResult(
                status=SlotAnswerStatus.NEEDS_CLARIFICATION,
                clarification_question="ما اسم المرض المزمن الذي تعاني منه؟",
            )
        if any(h in _normalize_message(text) for h in _CHRONIC_CONDITION_HINTS):
            return SlotAnswerResult(
                status=SlotAnswerStatus.VALID,
                chronic_condition=text.strip(),
            )
        return SlotAnswerResult(
            status=SlotAnswerStatus.NEEDS_CLARIFICATION,
            clarification_question="هل لديك أمراض مزمنة؟ (نعم/لا — أو اذكر اسم المرض)",
        )

    if kind in _YES_NO_SLOT_SUFFIXES:
        answer = _parse_yes_no(text)
        if answer is None:
            # إجابة وصفية على سؤال نعم/لا — نطلب توضيحاً
            return SlotAnswerResult(
                status=SlotAnswerStatus.NEEDS_CLARIFICATION,
                clarification_question=_clarify_yes_no(slot),
            )
        return SlotAnswerResult(status=SlotAnswerStatus.VALID, yes_no=answer)

    # خانات OLDCARTS الوصفية (موقع، جودة، ...) — أي نصّ غير فارغ مقبول
    return SlotAnswerResult(status=SlotAnswerStatus.VALID)


def apply_slot_answer_to_record(record, slot: str, result: SlotAnswerResult) -> None:
    """يُطبّق القيم المُتحقَّقة على ``ClinicalRecord`` (تعديل في المكان)."""
    from app.domain.clinical_record import ClinicalRecord, FactProvenance, FactSource

    if not isinstance(record, ClinicalRecord):
        return

    kind = _slot_kind(slot)
    turn = 0

    if result.age is not None:
        record.age = result.age
        record.provenance["age"] = FactProvenance(
            source=FactSource.PATIENT_EXPLICIT,
            evidence=str(result.age),
            turn_number=turn,
        )

    if result.gender is not None:
        record.gender = result.gender
        record.provenance["gender"] = FactProvenance(
            source=FactSource.PATIENT_EXPLICIT,
            evidence=result.gender,
            turn_number=turn,
        )

    if result.pregnancy_possible is not None:
        record.pregnancy_possible = result.pregnancy_possible

    if result.severity_text is not None:
        record.severity = result.severity_text
    if result.severity_numeric is not None:
        record.severity_numeric = result.severity_numeric
        record.provenance["severity"] = FactProvenance(
            source=FactSource.PATIENT_EXPLICIT,
            evidence=result.severity_text or str(result.severity_numeric),
            turn_number=turn,
        )

    if result.chronic_condition:
        if result.chronic_condition not in record.chronic_conditions:
            record.chronic_conditions.append(result.chronic_condition)
        record.provenance["chronic_conditions"] = FactProvenance(
            source=FactSource.PATIENT_EXPLICIT,
            turn_number=turn,
        )

    if kind == "chronic_diseases" and result.yes_no is False:
        record.provenance["chronic_conditions"] = FactProvenance(
            source=FactSource.PATIENT_EXPLICIT,
            evidence="لا",
            turn_number=turn,
        )
