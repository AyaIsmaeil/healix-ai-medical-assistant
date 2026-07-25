"""
Healix - Feature Validator (Phase 3.2)
يتحقّق من صحّة ``ClinicalFeatureSet`` المُستخرَجة، يُطبّع القيم الخارجة عن
النطاق، ويُسجّل تقرير تحقّق (``ValidationReport``) قبل استهلاكها من أي
مكوّن لاحق (Phase 3.3+: FeatureEncoder، متنبّئ المرض...).

منطق مجال خالص — بلا FastAPI، بلا شبكة، بلا نظام ملفات، بلا أي اعتماد على
وكيل المقابلة أو Whisper أو مزوّدي الـLLM. قواعد النطاقات/التعداد تصله محقونة
بالمُنشئ (من ``infrastructure.dictionary_loader``) — لا قيم ثابتة بالكود؛
"النقاء" هنا يعني صفر عمليات I/O وقت التحقّق نفسه، لا غياب أي إعداد محقون
(نفس مبدأ حقن ``SessionStore`` بمحرّك المحادثة).

فشل ناعم (Fail-Soft): حقل اختياري خارج النطاق أو من نوع خاطئ يُصحَّح إن
أمكن (قصّ للحدود)، وإلا يُستبدَل بـnull، مع توثيق السبب — لا ينهار التحقّق
كاملاً. الاستثناء الوحيد: غياب معلومة سريرية جوهرية (لا أعراض إطلاقاً على
الإطلاق) يرفع ``FeatureValidationError`` صراحة، لأنّه لا معنى للمتابعة بلا
أي عرَض — يطابق نفس المبدأ الذي يعتمده ``domain.clinical`` (chief_complaint
عند غياب الأعراض المُثبَتة).

قاعدة تعارض الحمل/الجنس (pregnancy + gender=male) قاعدة بنيوية لا نطاق
رقمي أو تعداد قابل للتوصيف بملف JSON — لذا بقيت منطق مجال صريح هنا، لا
بيانات بالقاموس، بنفس فلسفة استبعاد سؤال الحمل للذكور بـ``clinical.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.domain.assessment import ClinicalFeatureSet, ValidationReport
from app.exceptions import FeatureValidationError


class FeatureValidator:
    """يتحقّق من ``ClinicalFeatureSet`` ويُطبّعه وفق قواعد محقونة من الخارج."""

    def __init__(self, rules: Dict[str, Any]):
        self._rules: Dict[str, Any] = rules.get("fields", {})

    def validate(self, features: ClinicalFeatureSet) -> ClinicalFeatureSet:
        """يتحقّق من ``features`` ويُعيده مع ``validation`` معبّأً.

        يُعدَّل الكائن الممرَّر في مكانه (نفس فلسفة ``ConversationState``
        بمحرّك المقابلة) ويُعاد أيضاً لتيسير التسلسل (chaining) بالراوت.
        """
        self._ensure_critical_information_present(features)

        corrected: List[str] = []
        rejected: List[str] = []
        warnings: List[str] = []
        checked_count = 0

        checked_count += self._validate_demographics(features, corrected, rejected, warnings)
        checked_count += self._validate_temperature(features, corrected, rejected, warnings)
        checked_count += self._validate_symptom_descriptors(features, corrected, rejected, warnings)
        checked_count += self._validate_pregnancy_gender_conflict(features, corrected, warnings)

        features.validation = ValidationReport(
            corrected_fields=corrected,
            rejected_fields=rejected,
            warnings=warnings,
            validity_score=self._compute_validity_score(
                checked_count, len(rejected), len(corrected)
            ),
        )
        return features

    # ------------------------------------------------------------------
    # الحرج فقط — بلا معلومة سريرية أساسية، لا معنى للمتابعة.
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_critical_information_present(features: ClinicalFeatureSet) -> None:
        if not features.symptoms:
            raise FeatureValidationError(
                "لا توجد أي أعراض بمجموعة الميزات — معلومة سريرية جوهرية غائبة، "
                "لا يمكن متابعة التقييم."
            )

    # ------------------------------------------------------------------
    # حقول بسيطة (نطاق/تعداد) — مدفوعة بالقاموس المحقون بالكامل.
    # ------------------------------------------------------------------
    def _validate_demographics(
        self,
        features: ClinicalFeatureSet,
        corrected: List[str],
        rejected: List[str],
        warnings: List[str],
    ) -> int:
        checked = 0
        demo = features.demographics

        if demo.age is not None:
            checked += 1
            new_value, outcome, message = self._check_numeric("age", demo.age)
            self._apply_outcome(
                "demographics.age", outcome, message, corrected, rejected, warnings
            )
            demo.age = new_value

        if demo.gender is not None:
            checked += 1
            new_value, outcome, message = self._check_enum("gender", demo.gender)
            self._apply_outcome(
                "demographics.gender", outcome, message, corrected, rejected, warnings
            )
            demo.gender = new_value

        return checked

    def _validate_temperature(
        self,
        features: ClinicalFeatureSet,
        corrected: List[str],
        rejected: List[str],
        warnings: List[str],
    ) -> int:
        if features.temperature_c is None:
            return 0

        new_value, outcome, message = self._check_numeric("temperature_c", features.temperature_c)
        self._apply_outcome("temperature_c", outcome, message, corrected, rejected, warnings)
        features.temperature_c = new_value
        return 1

    def _validate_symptom_descriptors(
        self,
        features: ClinicalFeatureSet,
        corrected: List[str],
        rejected: List[str],
        warnings: List[str],
    ) -> int:
        checked = 0
        for index, symptom in enumerate(features.symptoms):
            descriptors = symptom.descriptors

            if descriptors.severity_0_10 is not None:
                checked += 1
                label = f"symptoms[{index}].descriptors.severity_0_10"
                new_value, outcome, message = self._check_numeric(
                    "severity_0_10", descriptors.severity_0_10
                )
                self._apply_outcome(label, outcome, message, corrected, rejected, warnings)
                descriptors.severity_0_10 = new_value

            if descriptors.onset_days_ago is not None:
                checked += 1
                label = f"symptoms[{index}].descriptors.onset_days_ago"
                new_value, outcome, message = self._check_numeric(
                    "onset_days_ago", descriptors.onset_days_ago
                )
                self._apply_outcome(label, outcome, message, corrected, rejected, warnings)
                descriptors.onset_days_ago = new_value

        return checked

    # ------------------------------------------------------------------
    # قاعدة عبر-حقلية: الحمل لا يتّسق مع gender=male.
    #
    # يعمل هذا الفحص عمداً *بعد* ``_validate_demographics`` بترتيب الاستدعاء
    # بـ``validate()``، فيقرأ قيمة ``gender`` بعد تحقّقها هي نفسها (قد تكون
    # أُعيدت لـnull لو كانت غير صالحة أصلاً، كـ"alien"). لا يُشغَّل التعارض
    # إلا بدليل إيجابي مؤكَّد (gender == "male" فعلياً) — لو كان الجنس نفسه
    # غير صالح ولم يُحسَم، لا نخمّن أنّه كان "male" لنُصفّر الحمل أيضاً؛ ذلك
    # افتراض إضافي بلا دليل، يخالف مبدأ "لا تخمين" المُعتمَد بكل هذا التصميم.
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_pregnancy_gender_conflict(
        features: ClinicalFeatureSet, corrected: List[str], warnings: List[str]
    ) -> int:
        demo = features.demographics
        if demo.gender == "male" and demo.pregnancy_possible is True:
            demo.pregnancy_possible = None
            corrected.append("demographics.pregnancy_possible")
            warnings.append(
                "demographics.pregnancy_possible=true يتعارض مع demographics.gender=male "
                "— أُعيد إلى null (الافتراض الأأمن، لا تخمين)."
            )
            return 1
        return 0

    # ------------------------------------------------------------------
    # فحص عام مدفوع بالقاموس (بلا أي نطاق/تعداد ثابت بالكود)
    # ------------------------------------------------------------------
    def _check_numeric(self, field_key: str, value: Any) -> Tuple[Optional[Any], str, Optional[str]]:
        spec = self._rules.get(field_key)
        if spec is None:
            return value, "ok", None  # لا قاعدة معرَّفة لهذا الحقل — يمرّ بلا تغيير

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None, "rejected", f"{field_key}: القيمة ليست رقمية ({value!r})."

        minimum = spec.get("min")
        maximum = spec.get("max")
        below_min = minimum is not None and value < minimum
        above_max = maximum is not None and value > maximum

        if below_min or above_max:
            strategy = spec.get("on_out_of_range", "reject")
            if strategy == "clip":
                clipped = max(minimum, min(maximum, value))
                return clipped, "corrected", (
                    f"{field_key}={value} خارج النطاق المسموح [{minimum}, {maximum}] "
                    f"— قُصّ إلى {clipped}."
                )
            return None, "rejected", (
                f"{field_key}={value} خارج النطاق المسموح [{minimum}, {maximum}] "
                f"— استُبدل بـnull."
            )

        return value, "ok", None

    def _check_enum(self, field_key: str, value: Any) -> Tuple[Optional[str], str, Optional[str]]:
        spec = self._rules.get(field_key)
        if spec is None:
            return value, "ok", None  # لا قاعدة معرَّفة لهذا الحقل — يمرّ بلا تغيير

        allowed = spec.get("allowed", [])
        if not isinstance(value, str) or value not in allowed:
            return None, "rejected", (
                f"{field_key}={value!r} ليست ضمن القيم المسموحة {allowed} — استُبدل بـnull."
            )

        return value, "ok", None

    @staticmethod
    def _apply_outcome(
        field_label: str,
        outcome: str,
        message: Optional[str],
        corrected: List[str],
        rejected: List[str],
        warnings: List[str],
    ) -> None:
        if outcome == "corrected":
            corrected.append(field_label)
            warnings.append(f"{field_label}: {message}")
        elif outcome == "rejected":
            rejected.append(field_label)
            warnings.append(f"{field_label}: {message}")
        # outcome == "ok" → لا شيء يُسجَّل

    @staticmethod
    def _compute_validity_score(
        checked_count: int, rejected_count: int, corrected_count: int
    ) -> Optional[float]:
        """نسبة أوّلية بسيطة (رفض = عقوبة كاملة، تصحيح = نصف عقوبة) — قابلة
        للاستهلاك من ConfidenceEstimator لاحقاً (Phase 3.3+). الصيغة نفسها
        قابلة للتحسين مستقبلاً بلا تغيير عقد ValidationReport."""
        if checked_count == 0:
            return None
        penalty = rejected_count + 0.5 * corrected_count
        score = max(0.0, min(1.0, (checked_count - penalty) / checked_count))
        return round(score, 2)
