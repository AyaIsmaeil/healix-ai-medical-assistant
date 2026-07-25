"""
Healix - Rule-Based Confidence Estimator (Phase 3.7)
أول تنفيذ فعلي لـ``ConfidenceEstimatorPort`` — قواعد حتمية بسيطة (Placeholder)،
بلا أي مكتبة ML. يحقّق نفس العقد الذي ستستخدمه لاحقاً adapters حقيقية
(مُعايِر ثقة مُدرَّب) — استبدال هذا الصنف فقط مستقبلاً، بلا تغيير على بقية
النظام.

الدور الوحيد: تقدير موثوقية التقييم الكامل الناتج قبله. لا يتنبّأ بمرض، لا
يغيّر الاستعجال، لا يغيّر التخصّص — يقرأ فقط ما أُنتج ويُصدر حكماً على الثقة.

يعتمد على مخرجات المراحل السابقة (ClinicalFeatureSet + ValidationReport +
DiseasePredictionResult + UrgencyAssessment + SpecialtyRecommendation) — لا
يعرف شيئاً عن وكيل المقابلة أو Whisper أو أي مزوّد LLM أو أي parser.

⚠️ قرار معماري موثَّق (Phase 3.7): ``ValidationReport.validity_score`` قد يكون
``None`` (انظر ``feature_validator._compute_validity_score`` — يُعيد None حين
لا حقل قابل للفحص أصلاً؛ حالة واردة فعلياً: أعراض بلا واصفات ولا ديموغرافيا).
قواعد Phase 3.7 كلها مقارنات رقمية (>=0.95 ... <0.70)، و``None`` بأي مقارنة
يرفع ``TypeError`` بـPython 3. الحل الأدنى: يُعامَل ``None`` كـ"لا إشارة صلاحية"
→ محايد (لا مكافأة ولا عقوبة، ولا يُشغّل مراجعة بشرية عبر بند الصلاحية) —
وتبقى عتبة ``overall_confidence < 0.60`` شبكةَ الأمان. لا يضيف هذا شيئاً على
النطاقات الرقمية المُحدَّدة؛ يحرس فقط حالة None التي لم تذكرها المواصفة.

⚠️ بند "لا تنبؤ مرض": المواصفة تنصّ صراحةً على أساس 0.40 عند غياب التنبؤات،
*و* عقوبة -0.20 عند غياب التنبؤ — كلاهما بند مستقل بالمواصفة، فيُطبَّقان معاً
(0.40 - 0.20 = 0.20 قبل بقية التعديلات)، تطبيقاً حرفياً لها.
"""

from __future__ import annotations

from typing import List, Optional

from app.domain.assessment import ClinicalFeatureSet, ValidationReport
from app.domain.confidence import ConfidenceAssessment
from app.domain.prediction import DiseasePredictionResult
from app.domain.specialty import SpecialtyRecommendation
from app.domain.urgency import UrgencyAssessment, UrgencyLevel

_BASE_WITHOUT_PREDICTIONS = 0.40

_VALIDITY_HIGH_MIN = 0.95
_VALIDITY_MID_MIN = 0.80
_VALIDITY_HIGH_BONUS = 0.10
_VALIDITY_LOW_PENALTY = -0.15

_UNRESOLVED_FIELDS_THRESHOLD = 3
_UNRESOLVED_PENALTY = -0.10

_NO_PREDICTION_PENALTY = -0.20

_EMERGENCY_BONUS = 0.05

_HUMAN_REVIEW_CONFIDENCE_FLOOR = 0.60
_HUMAN_REVIEW_VALIDITY_FLOOR = 0.70


class RuleBasedConfidenceEstimator:
    """يطبّق قواعد حتمية بسيطة (Placeholder) لتقدير موثوقية التقييم الكامل."""

    def estimate(
        self,
        clinical_features: ClinicalFeatureSet,
        validation: ValidationReport,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
    ) -> ConfidenceAssessment:
        """يُعيد ``ConfidenceAssessment`` دائماً — لا يرفع استثناءً، لا يغيّر
        أي مخرَج سابق. ``specialty`` جزء من العقد (قد يستهلكه معايِر مستقبلي)
        لكنه لا يؤثّر بقواعد Placeholder الحالية."""
        has_predictions = bool(prediction_result.predictions)
        validity_score = validation.validity_score  # قد يكون None (انظر توثيق الملف)

        confidence = self._base_confidence(prediction_result, has_predictions)
        confidence += self._validity_adjustment(validity_score)
        confidence += self._unresolved_adjustment(clinical_features.unresolved_fields)
        confidence += self._missing_prediction_adjustment(has_predictions)
        confidence += self._urgency_adjustment(urgency.level)

        overall_confidence = self._clamp(confidence)
        requires_review = self._requires_human_review(overall_confidence, validity_score)
        explanation = self._explain(
            has_predictions=has_predictions,
            validity_score=validity_score,
            validation=validation,
        )

        return ConfidenceAssessment(
            overall_confidence=overall_confidence,
            requires_human_review=requires_review,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # الأساس: أعلى تنبؤ (القائمة مرتّبة تنازلياً بالمُتنبِّئ) أو 0.40.
    # ------------------------------------------------------------------
    @staticmethod
    def _base_confidence(
        prediction_result: DiseasePredictionResult, has_predictions: bool
    ) -> float:
        if has_predictions:
            return prediction_result.predictions[0].score
        return _BASE_WITHOUT_PREDICTIONS

    # ------------------------------------------------------------------
    # تعديل الصلاحية — None يُعامَل كمحايد (لا مكافأة/عقوبة). انظر التوثيق.
    # ------------------------------------------------------------------
    @staticmethod
    def _validity_adjustment(validity_score: Optional[float]) -> float:
        if validity_score is None:
            return 0.0
        if validity_score >= _VALIDITY_HIGH_MIN:
            return _VALIDITY_HIGH_BONUS
        if validity_score >= _VALIDITY_MID_MIN:
            return 0.0
        return _VALIDITY_LOW_PENALTY

    @staticmethod
    def _unresolved_adjustment(unresolved_fields: List[str]) -> float:
        if len(unresolved_fields) > _UNRESOLVED_FIELDS_THRESHOLD:
            return _UNRESOLVED_PENALTY
        return 0.0

    @staticmethod
    def _missing_prediction_adjustment(has_predictions: bool) -> float:
        return 0.0 if has_predictions else _NO_PREDICTION_PENALTY

    @staticmethod
    def _urgency_adjustment(level: UrgencyLevel) -> float:
        return _EMERGENCY_BONUS if level == UrgencyLevel.EMERGENCY else 0.0

    @staticmethod
    def _clamp(value: float) -> float:
        return round(max(0.0, min(1.0, value)), 2)

    # ------------------------------------------------------------------
    # المراجعة البشرية — ثقة منخفضة أو صلاحية منخفضة مؤكَّدة.
    # None صلاحية لا يُشغّل البند (غير مؤكَّد، لا "< 0.70") — الأمان من عتبة الثقة.
    # ------------------------------------------------------------------
    @staticmethod
    def _requires_human_review(
        overall_confidence: float, validity_score: Optional[float]
    ) -> bool:
        if overall_confidence < _HUMAN_REVIEW_CONFIDENCE_FLOOR:
            return True
        if validity_score is not None and validity_score < _HUMAN_REVIEW_VALIDITY_FLOOR:
            return True
        return False

    # ------------------------------------------------------------------
    # تفسير حتمي — يُختار العامل الأبرز بترتيب أولوية ثابت (الأخطر أولاً).
    # ------------------------------------------------------------------
    @staticmethod
    def _explain(
        has_predictions: bool,
        validity_score: Optional[float],
        validation: ValidationReport,
    ) -> str:
        if not has_predictions:
            return "No disease prediction was available."
        if validity_score is not None and validity_score < _HUMAN_REVIEW_VALIDITY_FLOOR:
            return "Low validity score requires clinician review."
        if validation.corrected_fields or validation.rejected_fields:
            return "Several fields required correction, reducing confidence."
        if validity_score is not None and validity_score >= _VALIDITY_HIGH_MIN:
            return "High feature validity and strong disease prediction."
        return "Confidence estimated from the available clinical features."
