"""
Healix - Hybrid Urgency Classifier
تقييم استعجال هجين: قواعد سريرية على بيانات المريض + حدّ أدنى من شدّة
المرض الأعلى احتمالاً (DDXPlus ``disease_metadata.severity``).

القاعدة الذهبية (CLAUDE.md): **القواعد ترفع الخطورة فقط، لا تخفّضها أبداً.**
"""

from __future__ import annotations

from typing import Dict, Optional

from app.domain.assessment import ClinicalFeatureSet
from app.domain.prediction import DiseasePredictionResult
from app.domain.rule_based_urgency_classifier import RuleBasedUrgencyClassifier
from app.domain.urgency import UrgencyAssessment, UrgencyLevel
from app.ml.ontology_mapper import OntologyMapper

_LEVEL_RANK = {
    UrgencyLevel.NON_URGENT: 0,
    UrgencyLevel.SEMI_URGENT: 1,
    UrgencyLevel.URGENT: 2,
    UrgencyLevel.EMERGENCY: 3,
}

_LEVEL_SCORES = {
    UrgencyLevel.EMERGENCY: 1.0,
    UrgencyLevel.URGENT: 0.75,
    UrgencyLevel.SEMI_URGENT: 0.5,
    UrgencyLevel.NON_URGENT: 0.25,
}


class HybridUrgencyClassifier:
    """يجمع قواعد المريض مع أرضية شدّة المرض المُتنبَّأ — تصعيد فقط."""

    classifier_version = "hybrid-urgency-v1"

    def __init__(
        self,
        base_classifier: RuleBasedUrgencyClassifier,
        disease_metadata: Dict[str, Dict[str, object]],
    ) -> None:
        self._base = base_classifier
        self._disease_metadata = disease_metadata.get("diseases", disease_metadata)

    def classify(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: Optional[DiseasePredictionResult] = None,
    ) -> UrgencyAssessment:
        base = self._base.classify(clinical_features)
        disease_level, disease_note = self._disease_severity_floor(prediction_result)

        if disease_level is None or _LEVEL_RANK[disease_level] <= _LEVEL_RANK[base.level]:
            return base

        explanation = base.explanation
        if disease_note:
            explanation = f"{explanation} {disease_note}".strip()

        return UrgencyAssessment(
            level=disease_level,
            score=_LEVEL_SCORES[disease_level],
            explanation=explanation,
        )

    def _disease_severity_floor(
        self, prediction_result: Optional[DiseasePredictionResult]
    ) -> tuple[Optional[UrgencyLevel], str]:
        if prediction_result is None or not prediction_result.predictions:
            return None, ""

        top_disease = prediction_result.predictions[0].disease
        metadata = self._disease_metadata.get(top_disease)
        if not metadata:
            return None, ""

        severity = metadata.get("severity")
        if not isinstance(severity, int):
            return None, ""

        urgency_label = OntologyMapper.severity_to_urgency(severity)
        if urgency_label is None:
            return None, ""

        try:
            level = UrgencyLevel(urgency_label)
        except ValueError:
            return None, ""

        return (
            level,
            f"شدّة المرض الأعلى احتمالاً ({top_disease}) تستدعي على الأقل {level.value}.",
        )
