"""
Healix - Hybrid Confidence Estimator
تقدير موثوقية هجين: قواعد الصلاحية والاكتمال + هامش ML + عدد الأدلة المطابقة.

لا يغيّر أي تنبؤ/استعجال/تخصّص — حكم كلّي على جودة التقييم الكامل.
"""

from __future__ import annotations

from typing import List, Optional

from app.domain.assessment import ClinicalFeatureSet, ValidationReport
from app.domain.confidence import ConfidenceAssessment
from app.domain.prediction import DiseasePredictionResult
from app.domain.rule_based_confidence_estimator import RuleBasedConfidenceEstimator
from app.domain.specialty import SpecialtyRecommendation
from app.domain.urgency import UrgencyAssessment, UrgencyLevel

_MARGIN_HIGH = 0.25
_MARGIN_LOW = 0.08
_MARGIN_HIGH_BONUS = 0.08
_MARGIN_LOW_PENALTY = -0.12

_EVIDENCE_STRONG = 3
_EVIDENCE_WEAK = 1
_EVIDENCE_STRONG_BONUS = 0.05
_EVIDENCE_WEAK_PENALTY = -0.08

_SPECIALTY_METADATA_CONFIDENCE = 0.90
_SPECIALTY_FALLBACK_CONFIDENCE = 0.50

_HUMAN_REVIEW_MARGIN_FLOOR = 0.08
_HUMAN_REVIEW_EVIDENCE_FLOOR = 1


class HybridConfidenceEstimator:
    """يوسّع المُقدِّر القاعدي بإشارات ML والتخصّص."""

    estimator_version = "hybrid-confidence-v1"

    def __init__(self, base_estimator: RuleBasedConfidenceEstimator) -> None:
        self._base = base_estimator

    def estimate(
        self,
        clinical_features: ClinicalFeatureSet,
        validation: ValidationReport,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
    ) -> ConfidenceAssessment:
        base = self._base.estimate(
            clinical_features, validation, prediction_result, urgency, specialty
        )

        confidence = base.overall_confidence
        confidence += self._margin_adjustment(prediction_result)
        confidence += self._evidence_adjustment(prediction_result.matched_evidence_count)
        confidence += self._specialty_adjustment(specialty)

        overall_confidence = self._clamp(confidence)
        requires_review = base.requires_human_review or self._extra_review_triggers(
            overall_confidence, prediction_result, urgency
        )
        explanation = self._explain(base.explanation, prediction_result, specialty)

        return ConfidenceAssessment(
            overall_confidence=overall_confidence,
            requires_human_review=requires_review,
            explanation=explanation,
        )

    @staticmethod
    def _margin_adjustment(prediction_result: DiseasePredictionResult) -> float:
        predictions = prediction_result.predictions
        if len(predictions) < 2:
            return 0.0
        margin = predictions[0].score - predictions[1].score
        if margin >= _MARGIN_HIGH:
            return _MARGIN_HIGH_BONUS
        if margin <= _MARGIN_LOW:
            return _MARGIN_LOW_PENALTY
        return 0.0

    @staticmethod
    def _evidence_adjustment(matched_evidence_count: Optional[int]) -> float:
        if matched_evidence_count is None:
            return 0.0
        if matched_evidence_count >= _EVIDENCE_STRONG:
            return _EVIDENCE_STRONG_BONUS
        if matched_evidence_count <= _EVIDENCE_WEAK:
            return _EVIDENCE_WEAK_PENALTY
        return 0.0

    @staticmethod
    def _specialty_adjustment(specialty: SpecialtyRecommendation) -> float:
        if specialty.confidence >= _SPECIALTY_METADATA_CONFIDENCE:
            return 0.03
        if specialty.confidence <= _SPECIALTY_FALLBACK_CONFIDENCE:
            return -0.05
        return 0.0

    @staticmethod
    def _extra_review_triggers(
        overall_confidence: float,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
    ) -> bool:
        if urgency.level == UrgencyLevel.EMERGENCY and overall_confidence < 0.70:
            return True

        predictions = prediction_result.predictions
        if len(predictions) >= 2:
            margin = predictions[0].score - predictions[1].score
            if margin <= _HUMAN_REVIEW_MARGIN_FLOOR:
                return True

        evidence_count = prediction_result.matched_evidence_count
        if evidence_count is not None and evidence_count <= _HUMAN_REVIEW_EVIDENCE_FLOOR:
            return True

        return False

    @staticmethod
    def _explain(
        base_explanation: str,
        prediction_result: DiseasePredictionResult,
        specialty: SpecialtyRecommendation,
    ) -> str:
        parts: List[str] = [base_explanation]
        predictions = prediction_result.predictions
        if len(predictions) >= 2:
            margin = predictions[0].score - predictions[1].score
            parts.append(f"هامش التنبؤ بين المركزين الأولين: {margin:.2f}.")
        if prediction_result.matched_evidence_count is not None:
            parts.append(
                f"عدد أدلة DDXPlus المطابقة: {prediction_result.matched_evidence_count}."
            )
        if specialty.confidence <= _SPECIALTY_FALLBACK_CONFIDENCE:
            parts.append("التخصّص بقي عاماً — مراجعة طبية موصى بها.")
        return " ".join(part for part in parts if part)

    @staticmethod
    def _clamp(value: float) -> float:
        return round(max(0.0, min(1.0, value)), 2)
