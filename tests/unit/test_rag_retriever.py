"""اختبارات RAG retriever والمسار الهجين."""

from __future__ import annotations

from app.domain.assessment import (
    ClinicalFeatureSet,
    DerivedFeatures,
    SymptomFeature,
    SymptomDescriptors,
)
from app.domain.prediction import DiseasePrediction, DiseasePredictionResult
from app.rag.retriever import MedicalKnowledgeRetriever


def test_build_query_combines_disease_and_symptoms():
    retriever = MedicalKnowledgeRetriever(enabled=False)
    features = ClinicalFeatureSet(
        session_id="s1",
        symptoms=[
            SymptomFeature(
                name="صداع",
                negated=False,
                extraction_confidence=0.9,
                is_primary=True,
                descriptors=SymptomDescriptors(severity_0_10=7),
            ),
            SymptomFeature(name="حرارة", negated=False, extraction_confidence=0.8),
        ],
        derived=DerivedFeatures(has_red_flag=True),
    )
    predictions = DiseasePredictionResult(
        predictions=[DiseasePrediction(disease="Pneumonia", score=0.8, explanation="")],
        predictor_version="test",
    )

    query = retriever._build_query(features, predictions)
    assert "Pneumonia" in query
    assert "صداع" in query
    assert "حرارة" in query
    assert "emergency" in query.lower()


def test_retriever_disabled_returns_empty():
    retriever = MedicalKnowledgeRetriever(enabled=False)
    features = ClinicalFeatureSet(session_id="s1")
    predictions = DiseasePredictionResult(predictions=[], predictor_version="test")
    assert retriever.retrieve_for_assessment(features, predictions) == []
    assert retriever.is_ready() is False
