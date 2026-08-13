"""تكامل ML: من الأعراض العربية → E_* → MLDiseasePredictor (نموذج حقيقي)."""

import pytest

from app.domain.feature_encoder import EncodedFeatures
from app.domain.ml_disease_predictor import MLDiseasePredictor
from app.domain.symptom_evidence_encoder import SymptomEvidenceEncoder
from app.infrastructure.dictionary_loader import DictionaryLoader
from app.infrastructure.model_loader import ModelLoader
from app.services.rule_based_evidence_concept_extractor import RuleBasedEvidenceConceptExtractor


@pytest.fixture(scope="module")
def ml_predictor() -> MLDiseasePredictor:
    model, feature_names, label_encoder = ModelLoader.load_all()
    return MLDiseasePredictor(model=model, feature_names=feature_names, label_encoder=label_encoder)


def test_fever_case_produces_ddxplus_predictions(ml_predictor: MLDiseasePredictor):
    evidence_map = DictionaryLoader.load_symptom_evidence_map()
    rule_extractor = RuleBasedEvidenceConceptExtractor.from_dict(evidence_map)
    encoder = SymptomEvidenceEncoder.from_dict(evidence_map)

    concepts = rule_extractor.extract(["عندي حرارة وكحة من 3 أيام"], ["حرارة", "سعال"])
    evidence = encoder.encode(concepts)
    assert evidence, "expected at least one E_* code"

    encoded = EncodedFeatures(
        feature_schema_version="assessment-features-v1",
        features={
            "age": 35,
            "gender_male": 1,
            "gender_female": 0,
            **evidence,
        },
        categorical_index={},
    )
    result = ml_predictor.predict(encoded)
    assert result.predictions, "ML should return top diseases when E_* present"
    assert result.matched_evidence_count and result.matched_evidence_count >= 1
    assert all(0.0 <= p.score <= 1.0 for p in result.predictions)
