"""اختبارات HybridUrgencyClassifier — تصعيد فقط."""

from app.domain.assessment import (
    ClinicalFeatureSet,
    Demographics,
    DerivedFeatures,
    SymptomFeature,
    SymptomDescriptors,
)
from app.domain.hybrid_urgency_classifier import HybridUrgencyClassifier
from app.domain.prediction import DiseasePrediction, DiseasePredictionResult
from app.domain.rule_based_urgency_classifier import RuleBasedUrgencyClassifier
from app.domain.urgency import UrgencyLevel
from app.infrastructure.dictionary_loader import DictionaryLoader


def _classifier() -> HybridUrgencyClassifier:
    return HybridUrgencyClassifier(
        base_classifier=RuleBasedUrgencyClassifier(),
        disease_metadata=DictionaryLoader.load_disease_metadata(),
    )


def _features(severity: int | None = None) -> ClinicalFeatureSet:
    descriptors = SymptomDescriptors(severity_0_10=severity) if severity is not None else SymptomDescriptors()
    return ClinicalFeatureSet(
        session_id="test",
        demographics=Demographics(),
        symptoms=[
            SymptomFeature(
                name="صداع",
                negated=False,
                extraction_confidence=0.9,
                is_primary=True,
                descriptors=descriptors,
            ),
        ],
        derived=DerivedFeatures(has_red_flag=False),
        unresolved_fields=[],
    )


def test_disease_severity_floor_escalates_non_urgent():
    """Acute dystonic reactions severity=2 → URGENT floor."""
    prediction = DiseasePredictionResult(
        predictions=[
            DiseasePrediction(
                disease="Acute dystonic reactions",
                score=0.42,
                explanation="test",
            )
        ],
        predictor_version="ml-xgboost-v1",
    )
    result = _classifier().classify(_features(severity=3), prediction)
    assert result.level == UrgencyLevel.URGENT


def test_never_downgrades_emergency_from_red_flag():
    features = ClinicalFeatureSet(
        session_id="test",
        demographics=Demographics(),
        symptoms=[
            SymptomFeature(
                name="ألم صدر",
                negated=False,
                extraction_confidence=0.9,
                is_primary=True,
                descriptors=SymptomDescriptors(),
            )
        ],
        derived=DerivedFeatures(has_red_flag=True),
        unresolved_fields=[],
    )
    prediction = DiseasePredictionResult(
        predictions=[
            DiseasePrediction(
                disease="URTI",
                score=0.9,
                explanation="test",
            )
        ],
        predictor_version="ml-xgboost-v1",
    )
    result = _classifier().classify(features, prediction)
    assert result.level == UrgencyLevel.EMERGENCY
