"""اختبارات HybridConfidenceEstimator — هامش ML وعدد الأدلة."""

from app.domain.assessment import ClinicalFeatureSet, Demographics, ValidationReport
from app.domain.confidence import ConfidenceAssessment
from app.domain.hybrid_confidence_estimator import HybridConfidenceEstimator
from app.domain.prediction import DiseasePrediction, DiseasePredictionResult
from app.domain.rule_based_confidence_estimator import RuleBasedConfidenceEstimator
from app.domain.specialty import SpecialtyRecommendation
from app.domain.urgency import UrgencyAssessment, UrgencyLevel


def _estimate(
    prediction_result: DiseasePredictionResult,
    specialty_confidence: float = 0.90,
) -> ConfidenceAssessment:
    estimator = HybridConfidenceEstimator(RuleBasedConfidenceEstimator())
    features = ClinicalFeatureSet(
        session_id="test",
        demographics=Demographics(),
        symptoms=[],
        unresolved_fields=[],
    )
    validation = ValidationReport(validity_score=0.96, corrected_fields=[], rejected_fields=[])
    urgency = UrgencyAssessment(level=UrgencyLevel.NON_URGENT, score=0.25, explanation="ok")
    specialty = SpecialtyRecommendation(
        specialty="General Medicine",
        confidence=specialty_confidence,
        explanation="test",
    )
    return estimator.estimate(features, validation, prediction_result, urgency, specialty)


def test_high_margin_increases_confidence():
    low_margin = _estimate(
        DiseasePredictionResult(
            predictions=[
                DiseasePrediction("A", 0.51, "x"),
                DiseasePrediction("B", 0.49, "x"),
            ],
            predictor_version="ml-xgboost-v1",
            matched_evidence_count=4,
        )
    )
    high_margin = _estimate(
        DiseasePredictionResult(
            predictions=[
                DiseasePrediction("A", 0.85, "x"),
                DiseasePrediction("B", 0.10, "x"),
            ],
            predictor_version="ml-xgboost-v1",
            matched_evidence_count=4,
        )
    )
    assert high_margin.overall_confidence > low_margin.overall_confidence


def test_low_evidence_triggers_review():
    result = _estimate(
        DiseasePredictionResult(
            predictions=[DiseasePrediction("A", 0.75, "x")],
            predictor_version="ml-xgboost-v1",
            matched_evidence_count=1,
        )
    )
    assert result.requires_human_review
