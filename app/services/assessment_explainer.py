"""
Healix - Assessment Explainer (Phase 3.8 + Hybrid RAG)
المرحلة الأخيرة بخطّ التقييم: تشرح ما حُسِب سلفاً بالعربية — لا تُشخّص، ولا
تُعدّل أي تنبؤ/استعجال/تخصّص/ثقة.

مسار هجين: القواعد + ML + YAML = **القرار**؛ RAG (PubMed) = **سياق للشرح** فقط.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.domain.assessment import ClinicalFeatureSet, SymptomFeature
from app.domain.confidence import ConfidenceAssessment
from app.domain.explanation import AssessmentExplanation
from app.domain.ports import LLMProvider
from app.domain.prediction import DiseasePredictionResult
from app.domain.rag_context import HybridExplainResult, RagSource
from app.domain.specialty import SpecialtyRecommendation
from app.domain.urgency import UrgencyAssessment, UrgencyLevel
from app.parsing.assessment_explainer_parser import parse_assessment_explanation
from app.prompts.assessment_explainer_builder import AssessmentExplainerPromptBuilder
from app.rag.retriever import MedicalKnowledgeRetriever

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
    """يشرح التقييم — مسار هجين: قرار من قواعد/ML، سياق PubMed للشرح."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: AssessmentExplainerPromptBuilder,
        rag_retriever: Optional[MedicalKnowledgeRetriever] = None,
    ):
        self._provider = provider
        self._prompts = prompt_builder
        self._rag = rag_retriever

    def explain(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
        confidence: ConfidenceAssessment,
    ) -> AssessmentExplanation:
        return self.explain_hybrid(
            clinical_features, prediction_result, urgency, specialty, confidence
        ).explanation

    def explain_hybrid(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
        confidence: ConfidenceAssessment,
    ) -> HybridExplainResult:
        rag_sources: List[RagSource] = []
        if self._rag is not None and self._rag.is_ready():
            rag_sources = self._rag.retrieve_for_assessment(clinical_features, prediction_result)

        structured = self._build_structured(
            clinical_features, prediction_result, urgency, specialty, confidence, rag_sources
        )
        deterministic = self._build_deterministic_explanation(
            clinical_features, prediction_result, urgency, specialty, confidence
        )
        system_prompt = self._prompts.system_prompt(include_rag=bool(rag_sources))
        user_prompt = self._prompts.explanation_prompt(structured)

        try:
            completion = self._provider.generate(system_prompt, user_prompt)
            llm_parts = parse_assessment_explanation(completion.text)
            explanation = AssessmentExplanation(
                summary=deterministic.summary,
                medical_reasoning=llm_parts.medical_reasoning,
                recommendation=deterministic.recommendation,
                disclaimer=_DISCLAIMER,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("فشل تفسير التقييم عبر الـLLM — بديل حتمي: %s", exc)
            explanation = deterministic

        return HybridExplainResult(
            explanation=explanation,
            rag_enabled=bool(rag_sources),
            rag_sources=rag_sources,
        )

    def _build_structured(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
        confidence: ConfidenceAssessment,
        rag_sources: Optional[List[RagSource]] = None,
    ) -> Dict[str, Any]:
        primary = self._primary_symptom(clinical_features.symptoms)
        payload: Dict[str, Any] = {
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
        if rag_sources:
            payload["RAG_LITERATURE_CONTEXT"] = [
                {
                    "pmid": source.pmid,
                    "disease_name": source.disease_name,
                    "specialty": source.medical_specialty,
                    "triage_from_literature": source.triage_level,
                    "snippet": source.snippet,
                    "pubmed_url": source.pubmed_url,
                    "relevance_score": source.relevance_score,
                }
                for source in rag_sources
            ]
        return payload

    def _build_deterministic_explanation(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
        confidence: ConfidenceAssessment,
    ) -> AssessmentExplanation:
        """ملخّص وتوصية حتميان من البيانات المُتحقَّقة — لا هلوسة LLM."""
        demo = clinical_features.demographics
        primary = self._primary_symptom(clinical_features.symptoms)
        urgency_label = _URGENCY_LABELS.get(urgency.level, urgency.level.value)

        intro_parts: List[str] = []
        if demo.age is not None:
            gender_word = ""
            if demo.gender == "female":
                gender_word = "أنثى، "
            elif demo.gender == "male":
                gender_word = "ذكر، "
            intro_parts.append(
                f"المريض/المريضة ({gender_word}العمر {demo.age} سنة)"
            )

        complaint = self._chief_complaint(clinical_features)
        if primary and primary.descriptors.severity_0_10 is not None:
            intro_parts.append(
                f"يشكو/تشكو من {complaint} بشدّة "
                f"{primary.descriptors.severity_0_10} من 10"
            )
        elif complaint != "غير محدّد":
            intro_parts.append(f"الشكوى الرئيسية: {complaint}")

        summary_body = ". ".join(intro_parts) if intro_parts else "تقييم مبدئي للأعراض المُبلَّغ عنها."
        if prediction_result.predictions:
            top = prediction_result.predictions[0]
            pct = round(top.score * 100, 1)
            summary_body += (
                f". وفقاً للنموذج الإحصائي (DDXPlus)، أعلى احتمال: "
                f"{top.disease} ({pct}%)"
            )
        summary_body += f". درجة الاستعجال: {urgency_label}. التخصّص المقترح: {specialty.specialty}."

        if prediction_result.predictions:
            names = "، ".join(
                f"{p.disease} ({round(p.score * 100, 1)}%)"
                for p in prediction_result.predictions[:3]
            )
            medical_reasoning = (
                f"التنبؤات مبنيّة على رموز أدلة DDXPlus المستخرجة من المقابلة "
                f"(نموذج XGBoost المدرَّب). أبرز الاحتمالات: {names}."
            )
        else:
            medical_reasoning = (
                "لا تتوفّر تنبؤات مرضية كافية من البيانات الحالية."
            )

        recommendation = self._fallback_recommendation(
            urgency.level, specialty.specialty, confidence.requires_human_review
        )

        return AssessmentExplanation(
            summary=summary_body,
            medical_reasoning=medical_reasoning,
            recommendation=recommendation,
            disclaimer=_DISCLAIMER,
        )

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
            return "يُنصح بالتوجّه إلى قسم الطوارئ فوراً."

        if level in (UrgencyLevel.URGENT, UrgencyLevel.SEMI_URGENT):
            base = f"يُنصح بحجز موعد مع {specialty} في أقرب وقت."
        else:
            base = f"يمكن متابعة الأعراض في المنزل، مع مراجعة {specialty} إن استمرّت أو تفاقمت."

        if requires_human_review:
            base += " ونظراً لعدم اليقين في النتيجة، يُنصح بمراجعة طبيب مختصّ لتقييم مهني للحالة."
        return base

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
