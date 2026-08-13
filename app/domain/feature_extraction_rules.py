"""
Healix - Rule-Based Feature Extractor
مستخرج حتمي للقيم الصريحة والواضحة من رسائل المريض الخام.

بلا شبكة، بلا LLM، بلا FastAPI — منطق مجال خالص (regex + قواميس)، بالضبط
نفس فلسفة ``domain.clinical``. مسؤوليته الحقول الستة الصريحة فقط: العمر،
الجنس، التدخين، الحرارة، بداية الأعراض (duration بمصطلح الطلب — دلالياً
onset)، والشدّة الرقمية. أي حقل غامض أو يحتاج فهماً سياقياً يبقى None
ويُترك لـ LLMFeatureExtractor عبر ``unresolved_fields()``.

الحقول الستة لا تتقاطع مع مسؤولية استخراج الأعراض (وجود/نفي الأعراض) — يُعاد
استخدام تلك الأعراض كما هي، لا تُستنتج هنا إطلاقاً.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from app.domain.clinical import _MALE_HINTS  # إعادة استخدام مقصودة — لا تكرار

_FEMALE_HINTS = ("أنثى", "امرأة", "بنت", "female", "سيدة")

_SMOKING_NEGATIVE_PATTERNS = (
    r"ما\s*بدخن", r"لا\s*أدخن", r"غير\s*مدخن", r"مو\s*مدخن", r"مب\s*مدخن",
)
_SMOKING_POSITIVE_PATTERNS = (r"بدخن", r"مدخن", r"أدخن")

# كلمات شدّة نوعية → رقم تقريبي على مقياس 0-10 (قاموس، كما طُلب).
_SEVERITY_WORDS = {
    "خفيف": 2, "بسيط": 2,
    "متوسط": 5,
    "شديد جدا": 9, "شديد جداً": 9, "لا يطاق": 10,
    "شديد": 8,
}

# وحدات المدة الزمنية → عدد الأيام لكل وحدة واحدة.
_DURATION_UNIT_DAYS = {
    "يوم": 1, "أيام": 1,
    "اسبوع": 7, "أسبوع": 7, "اسابيع": 7, "أسابيع": 7,
    "شهر": 30, "أشهر": 30,
}
# صيغ مزدوجة/ثابتة شائعة (تصعب على regex العام).
_DURATION_FIXED_WORDS = {
    "يومين": 1 * 2,
    "أسبوعين": 7 * 2, "اسبوعين": 7 * 2,
    "شهرين": 30 * 2,
}

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalize_digits(text: str) -> str:
    """تحويل الأرقام العربية-الهندية إلى لاتينية لتبسيط الـregex."""
    return text.translate(_ARABIC_INDIC_DIGITS)


@dataclass
class RuleExtractionResult:
    """ناتج جزئي من الاستخراج القاعدي — حقل None يعني "غير محلول"."""

    age: Optional[int] = None
    gender: Optional[str] = None
    smoking: Optional[bool] = None
    temperature: Optional[float] = None
    duration_text: Optional[str] = None
    duration_days: Optional[int] = None
    severity: Optional[int] = None


# ترتيب الحقول ثابت — يُستخدم لبناء unresolved_fields() بشكل متّسق.
FIELDS = ("age", "gender", "smoking", "temperature", "duration", "severity")


class RuleBasedFeatureExtractor:
    """يستخرج القيم الصريحة الستة من رسائل المريض الخام عبر regex وقواميس."""

    def extract(self, raw_messages: List[str]) -> RuleExtractionResult:
        text = _normalize_digits(" ".join(raw_messages))
        return RuleExtractionResult(
            age=self._extract_age(text),
            gender=self._extract_gender(text),
            smoking=self._extract_smoking(text),
            temperature=self._extract_temperature(text),
            duration_text=self._extract_duration_text(text),
            duration_days=self._extract_duration_days(text),
            severity=self._extract_severity(text),
        )

    def unresolved_fields(self, result: RuleExtractionResult) -> List[str]:
        """أسماء الحقول التي لم يحلّها الاستخراج القاعدي — نفس مفهوم
        ``clinical.missing_targets`` لكن لحقول التقييم لا خانات المقابلة."""
        unresolved = []
        if result.age is None:
            unresolved.append("age")
        if result.gender is None:
            unresolved.append("gender")
        if result.smoking is None:
            unresolved.append("smoking")
        if result.temperature is None:
            unresolved.append("temperature")
        if result.duration_text is None:
            unresolved.append("duration")
        if result.severity is None:
            unresolved.append("severity")
        return unresolved

    # ------------------------------------------------------------------
    # مستخرِجات كل حقل
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_age(text: str) -> Optional[int]:
        explicit = list(re.finditer(r"عمري\s*(\d{1,3})", text))
        if explicit:
            age = int(explicit[-1].group(1))
            return age if 0 < age < 120 else None

        for match in re.finditer(r"(\d{1,3})\s*(?:سنة|سنه|عام|سنوات)", text):
            window = text[max(0, match.start() - 24):match.start()]
            if re.search(r"منذ|مدخن|دخن|تدخين|من\s+\d", window):
                continue
            age = int(match.group(1))
            if 0 < age < 120:
                return age

        stripped = text.strip()
        if re.fullmatch(r"\d{1,3}", stripped):
            age = int(stripped)
            return age if 0 < age < 120 else None

        return None

    @staticmethod
    def _extract_gender(text: str) -> Optional[str]:
        if any(hint in text for hint in _MALE_HINTS):
            return "male"
        if any(hint in text for hint in _FEMALE_HINTS):
            return "female"
        return None

    @staticmethod
    def _extract_smoking(text: str) -> Optional[bool]:
        if any(re.search(pattern, text) for pattern in _SMOKING_NEGATIVE_PATTERNS):
            return False
        if any(re.search(pattern, text) for pattern in _SMOKING_POSITIVE_PATTERNS):
            return True
        return None

    @staticmethod
    def _extract_temperature(text: str) -> Optional[float]:
        match = re.search(r"(\d{2}(?:\.\d)?)\s*درجة", text)
        return float(match.group(1)) if match else None

    @staticmethod
    def _extract_duration_text(text: str) -> Optional[str]:
        for word in _DURATION_FIXED_WORDS:
            if word in text:
                return word
        match = re.search(
            r"(?:منذ|من)\s*(\d+)\s*"
            r"(يوم|أيام|اسبوع|أسبوع|اسابيع|أسابيع|شهر|أشهر)",
            text,
        )
        if match:
            return match.group(0)
        return None

    @staticmethod
    def _extract_duration_days(text: str) -> Optional[int]:
        for word, days in _DURATION_FIXED_WORDS.items():
            if word in text:
                return days
        match = re.search(
            r"(?:منذ|من)\s*(\d+)\s*"
            r"(يوم|أيام|اسبوع|أسبوع|اسابيع|أسابيع|شهر|أشهر)",
            text,
        )
        if not match:
            return None
        count = int(match.group(1))
        unit_days = _DURATION_UNIT_DAYS.get(match.group(2))
        return count * unit_days if unit_days else None

    @staticmethod
    def _extract_severity(text: str) -> Optional[int]:
        match = re.search(r"(\d{1,2})\s*(?:من|/)\s*10", text)
        if match:
            value = int(match.group(1))
            return value if 0 <= value <= 10 else None
        for word, value in _SEVERITY_WORDS.items():
            if word in text:
                return value
        return None
