"""
Healix - Rule-Based Disease Predictor (Phase 3.4)
أول تنفيذ فعلي لـ``DiseasePredictorPort`` — قواعد حتمية بسيطة (Placeholder)،
بلا أي مكتبة ML (sklearn/xgboost/catboost/numpy محظورة صراحة). يحقّق نفس
العقد الذي ستستخدمه لاحقاً adapters حقيقية (XGBoost/RandomForest/CatBoost)
— استبدال هذا الصنف فقط مستقبلاً، بلا تغيير على بقية النظام.

يعتمد حصراً على ``EncodedFeatures`` — لا يعرف شيئاً عن وكيل المقابلة أو
FeatureValidator أو AssessmentFeatureBuilder أو أي مزوّد LLM.

⚠️ فجوة معماريّة مُكتشَفة وموثَّقة (Phase 3.4): قاعدة "symptom_count >= 3"
بالمهمّة الأصلية تتطلّب حقلاً (عدد الأعراض) غير موجود إطلاقاً بمخطّط ترميز
الميزات الحالي (v1.json يُصدِّر فقط: age, gender, severity, smoking,
temperature_c, progression). إضافته تتطلّب تعديل FeatureEncoder/v1.json —
ممنوع صراحة بقيود هذه المرحلة. القاعدة موجودة أدناه كدالة مستقلة موثَّقة،
تُعيد ``None`` دائماً (لا تُساهم بأي تنبؤ) حتى يُضاف الحقل بمخطّط ترميز
مستقبلي (v2.json) — بدل اختلاق قيمة بديلة أو تجاهل الفجوة بصمت.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.domain.feature_encoder import EncodedFeatures
from app.domain.prediction import DiseasePrediction, DiseasePredictionResult

PREDICTOR_VERSION = "rule-based-v1"

_SMOKING_SCORE_BOOST = 0.1


class RuleBasedDiseasePredictor:
    """يطبّق قواعد حتمية بسيطة (Placeholder) على ``EncodedFeatures.features``."""

    predictor_version: str = PREDICTOR_VERSION

    def predict(self, encoded_features: EncodedFeatures) -> DiseasePredictionResult:
        """يُعيد ``DiseasePredictionResult`` دائماً — حتى لو لم يُطابَق شيء."""
        values = encoded_features.features

        predictions: List[DiseasePrediction] = []
        for rule in (self._check_temperature, self._check_severity, self._check_symptom_count):
            prediction = rule(values)
            if prediction is not None:
                predictions.append(prediction)

        predictions = self._apply_smoking_boost(predictions, values)

        # ترتيب حتمي: الأعلى نقاطاً أولاً؛ عند التعادل، ترتيب الفحص أعلاه
        # (list.sort مستقرّ — لا يُبدِّل ترتيب العناصر متساوية النقاط).
        predictions.sort(key=lambda prediction: prediction.score, reverse=True)

        return DiseasePredictionResult(
            predictions=predictions,
            predictor_version=self.predictor_version,
        )

    # ------------------------------------------------------------------
    # قواعد فردية (كل قاعدة دالة صغيرة مستقلة — سهلة الاختبار والتوسيع)
    # ------------------------------------------------------------------
    @staticmethod
    def _check_temperature(values: Dict[str, Any]) -> Optional[DiseasePrediction]:
        temperature = values.get("temperature_c")
        if temperature is not None and temperature >= 38:
            return DiseasePrediction(
                disease="Febrile Illness",
                score=0.6,
                explanation=f"درجة الحرارة {temperature} ≥ 38 — مؤشّر حمّى.",
            )
        return None

    @staticmethod
    def _check_severity(values: Dict[str, Any]) -> Optional[DiseasePrediction]:
        severity = values.get("severity")
        if severity is not None and severity >= 8:
            return DiseasePrediction(
                disease="Severe Condition",
                score=0.7,
                explanation=f"شدّة العرَض {severity} من 10 ≥ 8 — حالة شديدة محتملة.",
            )
        return None

    @staticmethod
    def _check_symptom_count(values: Dict[str, Any]) -> Optional[DiseasePrediction]:
        """⚠️ Placeholder غير فعّال حالياً — انظر توثيق الفجوة أعلى الملف.
        ``EncodedFeatures`` لا يحمل عدد الأعراض بمخطّط v1، فلا قاعدة "عدد
        الأعراض ≥ 3 → General Viral Illness" ممكنة بلا اختلاق قيمة."""
        return None

    @staticmethod
    def _apply_smoking_boost(
        predictions: List[DiseasePrediction], values: Dict[str, Any]
    ) -> List[DiseasePrediction]:
        """التدخين لا يُنتج تشخيصاً مستقلاً بذاته — يرفع نقاط أي تنبؤ موجود
        أصلاً برفع بسيط وثابت (Placeholder)، مقصوص عند 1.0 كحدّ أقصى."""
        if values.get("smoking") != 1 or not predictions:
            return predictions

        return [
            DiseasePrediction(
                disease=prediction.disease,
                score=min(1.0, round(prediction.score + _SMOKING_SCORE_BOOST, 2)),
                explanation=prediction.explanation + " زيادة بسيطة بسبب التدخين.",
            )
            for prediction in predictions
        ]
