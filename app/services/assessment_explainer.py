"""
Healix - Assessment Explainer (Phase 3.8)
المرحلة الأخيرة بخطّ التقييم: تشرح ما حُسِب سلفاً بالعربية — لا تُشخّص، ولا
تُعدّل أي تنبؤ/استعجال/تخصّص/ثقة.

يعيد استخدام منفذ ``LLMProvider`` الحالي بلا أي تعديل عليه — نفس نمط
``ConversationService`` و``LLMFeatureExtractor``. يبني قاموساً منظَّماً من
مخرجات المراحل السابقة فقط (لا رسائل مريض خام إطلاقاً) ويمرّره للـLLM.

⚠️ تدهور لطيف (قرار معماري موثَّق — Phase 3.8): هذه أول مرحلة بالخطّ يعتمد
ناتجها الأساسي على استدعاء LLM حيّ. لضمان بقاء الخطّ كليّاً (كبقية المراحل)
لا يتسرّب أي فشل LLM/تحليل للراوت: عند تعذّر الاستدعاء أو رفض المُحلِّل
الصارم للمخرجات (بعد استنفاد إعادة المحاولة + التلقين بالمزوّد)، تُعاد نسخة
حتمية بديلة مبنيّة **حصراً** من البيانات المنظَّمة المحسوبة سلفاً — لا تخترع
مرضاً، ولا تغيّر استعجالاً/تخصّصاً/ثقة، وتتضمّن دائماً إخلاء المسؤولية. نفس
فلسفة التدهور اللطيف بـ``LLMFeatureExtractor``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.domain.assessment import ClinicalFeatureSet, SymptomFeature
from app.domain.confidence import ConfidenceAssessment
from app.domain.explanation import AssessmentExplanation
from app.domain.ports import LLMProvider
from app.domain.prediction import DiseasePredictionResult
from app.domain.specialty import SpecialtyRecommendation
from app.domain.urgency import UrgencyAssessment, UrgencyLevel
from app.parsing.assessment_explainer_parser import parse_assessment_explanation
from app.prompts.assessment_explainer_builder import AssessmentExplainerPromptBuilder

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "هذا التقييم ليس تشخيصاً طبياً نهائياً، بل نتيجة مبدئية مبنيّة على المعلومات "
    "المتوفّرة — يجب مراجعة طبيب مختصّ لتأكيد الحالة."
)

_URGENCY_LABELS = {
    UrgencyLevel.EMERGENCY: "طارئ",
    UrgencyLevel.URGENT: "عاجل",
    UrgencyLevel.SEMI_URGENT: "شبه عاجل",
    UrgencyLevel.NON_URGENT: "غير عاجل",
}


class AssessmentExplainer:
    """يشرح التقييم المُحسَب سلفاً عبر الـLLM، مع تدهور حتمي لطيف عند الفشل."""

    def __init__(self, provider: LLMProvider, prompt_builder: AssessmentExplainerPromptBuilder):
        self._provider = provider
        self._prompts = prompt_builder

    def explain(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
        confidence: ConfidenceAssessment,
    ) -> AssessmentExplanation:
        """يُعيد ``AssessmentExplanation`` دائماً — لا يرفع استثناءً، لا يغيّر
        أي مدخَل. عند فشل الـLLM/التحليل يُعاد التفسير الحتمي البديل."""
        structured = self._build_structured(
            clinical_features, prediction_result, urgency, specialty, confidence
        )
        system_prompt = self._prompts.system_prompt()
        user_prompt = self._prompts.explanation_prompt(structured)

        try:
            completion = self._provider.generate(system_prompt, user_prompt)
            return parse_assessment_explanation(completion.text)
        except Exception as exc:  # noqa: BLE001 - تدهور لطيف، لا يُسقط التقييم المحسوب
            logger.warning("فشل تفسير التقييم عبر الـLLM — بديل حتمي: %s", exc)
            return self._fallback(prediction_result, urgency, specialty, confidence)

    # ------------------------------------------------------------------
    # بناء القاموس المنظَّم — لا رسائل خام إطلاقاً (structured info only).
    # ------------------------------------------------------------------
    def _build_structured(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
        confidence: ConfidenceAssessment,
    ) -> Dict[str, Any]:
        primary = self._primary_symptom(clinical_features.symptoms)
        return {
            "CHIEF_COMPLAINT": self._chief_complaint(clinical_features),
            "IMPORTANT_VALIDATED_FEATURES": {
                "age": clinical_features.demographics.age,
                "gender": clinical_features.demographics.gender,
                "pregnancy_possible": clinical_features.demographics.pregnancy_possible,
                "temperature_c": clinical_features.temperature_c,
                "primary_symptom_severity_0_10": (
                    primary.descriptors.severity_0_10 if primary else None
                ),
                "smoking": clinical_features.lifestyle.smoking,
            },
            "DISEASE_PREDICTIONS": [
                {"disease": p.disease, "score": p.score}
                for p in prediction_result.predictions
            ],
            "URGENCY": {"level": urgency.level.value, "score": urgency.score},
            "SPECIALTY": {
                "specialty": specialty.specialty,
                "confidence": specialty.confidence,
            },
            "CONFIDENCE": {
                "overall_confidence": confidence.overall_confidence,
                "requires_human_review": confidence.requires_human_review,
            },
        }

    # ------------------------------------------------------------------
    # التفسير الحتمي البديل — من البيانات المنظَّمة فقط (لا اختلاق).
    # ------------------------------------------------------------------
    def _fallback(
        self,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
        confidence: ConfidenceAssessment,
    ) -> AssessmentExplanation:
        urgency_label = _URGENCY_LABELS.get(urgency.level, urgency.level.value)
        summary = (
            f"مستوى الاستعجال المقدَّر: {urgency_label}. "
            f"التخصّص المقترح: {specialty.specialty}."
        )

        if prediction_result.predictions:
            names = "، ".join(p.disease for p in prediction_result.predictions)
            top = prediction_result.predictions[0].disease
            medical_reasoning = (
                f"استناداً إلى الميزات المتوفّرة، أبرز الاحتمالات المطروحة: {names}. "
                f"الأعلى ترجيحاً حالياً: {top}."
            )
        else:
            medical_reasoning = (
                "لا تتوفّر تنبؤات مرضية كافية من البيانات الحالية لطرح احتمالات محدّدة."
            )

        recommendation = self._fallback_recommendation(
            urgency.level, specialty.specialty, confidence.requires_human_review
        )

        return AssessmentExplanation(
            summary=summary,
            medical_reasoning=medical_reasoning,
            recommendation=recommendation,
            disclaimer=_DISCLAIMER,
        )

    @staticmethod
    def _fallback_recommendation(
        level: UrgencyLevel, specialty: str, requires_human_review: bool
    ) -> str:
        if level == UrgencyLevel.EMERGENCY:
            # التوجّه للطوارئ هو بذاته تقييم مهني فوري (يفي بشرط المراجعة البشرية).
            return "يُنصح بالتوجّه إلى قسم الطوارئ فوراً."

        if level in (UrgencyLevel.URGENT, UrgencyLevel.SEMI_URGENT):
            base = f"يُنصح بحجز موعد مع {specialty} في أقرب وقت."
        else:  # NON_URGENT
            base = f"يمكن متابعة الأعراض في المنزل، مع مراجعة {specialty} إن استمرّت أو تفاقمت."

        if requires_human_review:
            base += " ونظراً لعدم اليقين في النتيجة، يُنصح بمراجعة طبيب مختصّ لتقييم مهني للحالة."
        return base

    # ------------------------------------------------------------------
    # مساعدات
    # ------------------------------------------------------------------
    @staticmethod
    def _primary_symptom(symptoms: List[SymptomFeature]) -> Optional[SymptomFeature]:
        primary = next((s for s in symptoms if s.is_primary), None)
        if primary is not None:
            return primary
        return symptoms[0] if symptoms else None

    @classmethod
    def _chief_complaint(cls, clinical_features: ClinicalFeatureSet) -> str:
        primary = cls._primary_symptom(clinical_features.symptoms)
        return primary.name if primary else "غير محدّد"
