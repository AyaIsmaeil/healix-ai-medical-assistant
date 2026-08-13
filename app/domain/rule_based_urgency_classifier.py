"""
Healix - Rule-Based Urgency Classifier (Phase 3.5)
أول تنفيذ فعلي لـ``UrgencyClassifierPort`` — قواعد حتمية بسيطة (Placeholder)،
بلا أي مكتبة ML. يحقّق نفس العقد الذي ستستخدمه لاحقاً adapters حقيقية —
استبدال هذا الصنف فقط مستقبلاً، بلا تغيير على بقية النظام.

يعتمد حصراً على ``ClinicalFeatureSet`` (لا ``EncodedFeatures``، بخلاف
DiseasePredictor) — لا يعرف شيئاً عن AssessmentFeatureBuilder أو
FeatureValidator أو FeatureEncoder أو DiseasePredictor أو أي مزوّد LLM.

⚠️ ملاحظة تصميم (تكرار مقصود، لا اقتران): يستخرج شدّة العرَض الرئيسي من
``symptoms[primary].descriptors.severity_0_10`` مباشرة — لا من
``derived.primary_symptom_severity`` (نسخة قد لا يُزامنها FeatureValidator،
نفس الفجوة المُكتشَفة والمُصلَحة بـFeatureEncoder بمرحلة ٣.٣). منطق إيجاد
العرَض الرئيسي مُعاد تنفيذه محلياً هنا عمداً — لا استيراد من FeatureEncoder
(ممنوع صراحة بقيود هذه المرحلة، وحدود المنافذ تفرض استقلال كل adapter عن
غيره على أي حال).

لا يخترع بيانات غائبة: أي حقل None (حرارة/شدّة/حمل) يُتخطّى بلا افتراض
قيمة بديلة — يستقر التقييم على NON_URGENT إن لم تُطابَق أي قاعدة.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from app.domain.assessment import ClinicalFeatureSet, SymptomFeature
from app.domain.prediction import DiseasePredictionResult
from app.domain.urgency import UrgencyAssessment, UrgencyLevel

_LEVEL_SCORES = {
    UrgencyLevel.EMERGENCY: 1.0,
    UrgencyLevel.URGENT: 0.75,
    UrgencyLevel.SEMI_URGENT: 0.5,
    UrgencyLevel.NON_URGENT: 0.25,
}

_EMERGENCY_TEMPERATURE_C = 40.0
_URGENT_SEVERITY_MIN = 9
_SEMI_URGENT_SEVERITY_MIN = 6
_SEMI_URGENT_SEVERITY_MAX = 8


class RuleBasedUrgencyClassifier:
    """يطبّق قواعد حتمية بسيطة (Placeholder) على ``ClinicalFeatureSet`` مباشرة."""

    def classify(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: Optional[DiseasePredictionResult] = None,
    ) -> UrgencyAssessment:
        """يُعيد ``UrgencyAssessment`` دائماً — استقرار افتراضي NON_URGENT."""
        _ = prediction_result  # غير مستخدم — التوافق مع UrgencyClassifierPort
        temperature = clinical_features.temperature_c
        severity = self._primary_severity(clinical_features.symptoms)
        has_red_flag = clinical_features.derived.has_red_flag
        pregnancy_possible = clinical_features.demographics.pregnancy_possible

        level, explanation = self._determine_level(temperature, severity, has_red_flag)
        level, explanation = self._apply_pregnancy_floor(level, explanation, pregnancy_possible)

        return UrgencyAssessment(
            level=level,
            score=_LEVEL_SCORES[level],
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # تسلسل الأولوية: الأخطر أولاً — أول قاعدة تُطابَق تفوز (لا تراجع).
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_level(
        temperature: Optional[float], severity: Optional[int], has_red_flag: bool
    ) -> Tuple[UrgencyLevel, str]:
        if temperature is not None and temperature >= _EMERGENCY_TEMPERATURE_C:
            return (
                UrgencyLevel.EMERGENCY,
                f"درجة الحرارة {temperature} ≥ {_EMERGENCY_TEMPERATURE_C} — طارئ محتمل.",
            )

        if has_red_flag:
            return UrgencyLevel.EMERGENCY, "علامة خطر مُسجَّلة — طارئ محتمل."

        if severity is not None and severity >= _URGENT_SEVERITY_MIN:
            return (
                UrgencyLevel.URGENT,
                f"شدّة العرَض {severity} من 10 ≥ {_URGENT_SEVERITY_MIN} — حالة عاجلة.",
            )

        if severity is not None and _SEMI_URGENT_SEVERITY_MIN <= severity <= _SEMI_URGENT_SEVERITY_MAX:
            return (
                UrgencyLevel.SEMI_URGENT,
                f"شدّة العرَض {severity} من 10 بين {_SEMI_URGENT_SEVERITY_MIN} و"
                f"{_SEMI_URGENT_SEVERITY_MAX} — حالة شبه عاجلة.",
            )

        return UrgencyLevel.NON_URGENT, "لا مؤشّرات استعجال حرجة بالبيانات المتوفّرة حالياً."

    @staticmethod
    def _apply_pregnancy_floor(
        level: UrgencyLevel, explanation: str, pregnancy_possible: Optional[bool]
    ) -> Tuple[UrgencyLevel, str]:
        """احتمال الحمل يرفع الحدّ الأدنى للاستعجال إلى SEMI_URGENT على
        الأقل (حذر إضافي)، بلا خفض مستوى أعلى مُحسَّب أصلاً من القواعد
        الأخرى — لا يُطبَّق إلا عند غياب أي مؤشّر آخر (NON_URGENT فقط)."""
        if pregnancy_possible is not True or level != UrgencyLevel.NON_URGENT:
            return level, explanation
        return UrgencyLevel.SEMI_URGENT, explanation + " احتمال الحمل يستدعي حذراً إضافياً."

    # ------------------------------------------------------------------
    # استخراج شدّة العرَض الرئيسي (المُتحقَّق منها فعلياً) — منطق مستقل.
    # ------------------------------------------------------------------
    @staticmethod
    def _primary_severity(symptoms: List[SymptomFeature]) -> Optional[int]:
        primary = next((symptom for symptom in symptoms if symptom.is_primary), None)
        return primary.descriptors.severity_0_10 if primary else None
