"""اختبارات وحدة لـRuleBasedSpecialtyRecommender (Phase 3.6) — قواعد حتمية
بسيطة (Placeholder)، بلا أي مكتبة ML. مدخل ClinicalFeatureSet/DiseasePredictionResult
حقيقي مباشر (بلا مكتبات mocking)، نفس روح باقي اختبارات الوحدة."""

from app.domain.assessment import (
    ClinicalFeatureSet,
    Demographics,
    DerivedFeatures,
    SymptomFeature,
)
from app.domain.prediction import DiseasePrediction, DiseasePredictionResult
from app.domain.rule_based_specialty_recommender import RuleBasedSpecialtyRecommender

_LOOKUP = {
    "influenza": {"specialty": "Family Medicine"},
    "pneumonia": {"specialty": "Pulmonology"},
    "migraine": {"specialty": "Neurology"},
    "stroke": {"specialty": "Neurology"},
    "appendicitis": {"specialty": "General Surgery"},
    "diabetes": {"specialty": "Endocrinology"},
    "hypertension": {"specialty": "Cardiology"},
    "dermatitis": {"specialty": "Dermatology"},
    "uti": {"specialty": "Urology"},
    "pregnancy": {"specialty": "Obstetrics and Gynecology"},
}


def _recommender():
    return RuleBasedSpecialtyRecommender(specialty_lookup=_LOOKUP)


def _symptom(name, negated=False):
    return SymptomFeature(name=name, negated=negated, extraction_confidence=0.9)


def _features(**overrides):
    defaults = dict(
        session_id="sid",
        demographics=Demographics(),
        symptoms=[],
        derived=DerivedFeatures(),
    )
    defaults.update(overrides)
    return ClinicalFeatureSet(**defaults)


def _predictions(*pairs):
    """pairs: (disease, score) tuples."""
    return DiseasePredictionResult(
        predictions=[
            DiseasePrediction(disease=disease, score=score, explanation="")
            for disease, score in pairs
        ],
        predictor_version="test-v1",
    )


# ----------------------------------------------------------------------
# الأولوية ١: أعلى مرض مُتنبَّأ به يطابق القاموس
# ----------------------------------------------------------------------
def test_top_disease_lookup_success():
    features = _features()
    prediction_result = _predictions(("influenza", 0.8))
    result = _recommender().recommend(features, prediction_result)
    assert result.specialty == "Family Medicine"
    assert result.confidence == 0.90
    assert "influenza" in result.explanation


def test_top_disease_picks_highest_score_not_first():
    features = _features()
    prediction_result = _predictions(("influenza", 0.3), ("pneumonia", 0.9))
    result = _recommender().recommend(features, prediction_result)
    assert result.specialty == "Pulmonology"


def test_top_disease_lookup_case_insensitive():
    features = _features()
    prediction_result = _predictions(("INFLUENZA", 0.8))
    result = _recommender().recommend(features, prediction_result)
    assert result.specialty == "Family Medicine"


def test_unknown_disease_falls_through_to_clinical_inference():
    """مرض غير موجود بالقاموس (مثل RuleBasedDiseasePredictor الحالي: Febrile
    Illness) لا يُطابَق تير 1 — ينتقل للاستدلال السريري."""
    features = _features(symptoms=[_symptom("صداع شديد")])
    prediction_result = _predictions(("Febrile Illness", 0.6))
    result = _recommender().recommend(features, prediction_result)
    assert result.specialty == "Neurology"
    assert result.confidence == 0.70


def test_no_prediction_result_falls_through_to_clinical_inference():
    features = _features(symptoms=[_symptom("صداع شديد")])
    result = _recommender().recommend(features, prediction_result=None)
    assert result.specialty == "Neurology"


def test_empty_predictions_list_falls_through():
    features = _features(symptoms=[_symptom("صداع شديد")])
    prediction_result = DiseasePredictionResult(predictions=[], predictor_version="v1")
    result = _recommender().recommend(features, prediction_result)
    assert result.specialty == "Neurology"


# ----------------------------------------------------------------------
# الأولوية ٢: استدلال سريري من ClinicalFeatureSet
# ----------------------------------------------------------------------
def test_pregnancy_recommends_obgyn():
    features = _features(demographics=Demographics(pregnancy_possible=True))
    result = _recommender().recommend(features)
    assert result.specialty == "Obstetrics and Gynecology"
    assert result.confidence == 0.70


def test_pregnancy_takes_priority_over_symptom_keywords():
    features = _features(
        demographics=Demographics(pregnancy_possible=True),
        symptoms=[_symptom("صداع شديد")],
    )
    result = _recommender().recommend(features)
    assert result.specialty == "Obstetrics and Gynecology"


def test_chest_pain_recommends_cardiology():
    features = _features(symptoms=[_symptom("ألم صدر")])
    result = _recommender().recommend(features)
    assert result.specialty == "Cardiology"


def test_headache_recommends_neurology():
    features = _features(symptoms=[_symptom("صداع شديد")])
    result = _recommender().recommend(features)
    assert result.specialty == "Neurology"


def test_cough_recommends_pulmonology():
    features = _features(symptoms=[_symptom("سعال جاف")])
    result = _recommender().recommend(features)
    assert result.specialty == "Pulmonology"


def test_skin_rash_recommends_dermatology():
    features = _features(symptoms=[_symptom("طفح جلدي")])
    result = _recommender().recommend(features)
    assert result.specialty == "Dermatology"


def test_abdominal_pain_recommends_general_surgery():
    features = _features(symptoms=[_symptom("مغص بالبطن")])
    result = _recommender().recommend(features)
    assert result.specialty == "General Surgery"


def test_joint_pain_recommends_orthopedics():
    features = _features(symptoms=[_symptom("ألم بالركبة")])
    result = _recommender().recommend(features)
    assert result.specialty == "Orthopedics"


def test_urinary_symptoms_recommend_urology():
    features = _features(symptoms=[_symptom("حرقان بالتبول")])
    result = _recommender().recommend(features)
    assert result.specialty == "Urology"


def test_fever_only_recommends_family_medicine():
    features = _features(symptoms=[_symptom("حرارة مرتفعة")])
    result = _recommender().recommend(features)
    assert result.specialty == "Family Medicine"


def test_negated_symptom_is_ignored():
    """عرَض منفي (negated=True) لا يُطابَق بالاستدلال السريري."""
    features = _features(symptoms=[_symptom("ألم صدر", negated=True)])
    result = _recommender().recommend(features)
    assert result.specialty != "Cardiology"


def test_symptom_keyword_priority_chest_over_headache():
    """ألم الصدر بالأولوية قبل الصداع لو ظهرا معاً (ترتيب _SYMPTOM_KEYWORD_RULES)."""
    features = _features(symptoms=[_symptom("صداع"), _symptom("ألم صدر")])
    result = _recommender().recommend(features)
    assert result.specialty == "Cardiology"


# ----------------------------------------------------------------------
# الاحتياطي: لا مطابقة إطلاقاً → General Medicine
# ----------------------------------------------------------------------
def test_no_match_at_all_falls_back_to_general_medicine():
    features = _features()
    result = _recommender().recommend(features)
    assert result.specialty == "General Medicine"
    assert result.confidence == 0.50
    assert result.explanation


def test_unrecognized_symptom_falls_back_to_general_medicine():
    features = _features(symptoms=[_symptom("عرَض غير معروف تماماً")])
    result = _recommender().recommend(features)
    assert result.specialty == "General Medicine"


# ----------------------------------------------------------------------
# خصائص عامة: قيم الثقة والتفسير دوماً موجودة
# ----------------------------------------------------------------------
def test_confidence_always_in_valid_range():
    cases = [
        _recommender().recommend(_features(), _predictions(("influenza", 0.9))),
        _recommender().recommend(_features(symptoms=[_symptom("صداع شديد")])),
        _recommender().recommend(_features()),
    ]
    for result in cases:
        assert 0.0 <= result.confidence <= 1.0


def test_explanation_never_empty():
    cases = [
        _recommender().recommend(_features(), _predictions(("influenza", 0.9))),
        _recommender().recommend(_features(symptoms=[_symptom("صداع شديد")])),
        _recommender().recommend(_features()),
    ]
    for result in cases:
        assert result.explanation
