"""اختبارات وحدة لـRuleBasedDiseasePredictor (Phase 3.4) — قواعد حتمية بسيطة
(Placeholder)، بلا أي مكتبة ML. مدخل EncodedFeatures حقيقي مباشر (بلا
مكتبات mocking)، نفس روح باقي اختبارات الوحدة."""

from app.domain.feature_encoder import EncodedFeatures
from app.domain.rule_based_predictor import PREDICTOR_VERSION, RuleBasedDiseasePredictor

_BASE_FEATURES = {
    "age": None,
    "gender_male": None,
    "gender_female": None,
    "severity": None,
    "smoking": None,
    "temperature_c": None,
    "progression_worsening": None,
    "progression_stable": None,
    "progression_improving": None,
}


def _predictor():
    return RuleBasedDiseasePredictor()


def _encoded(**overrides):
    features = dict(_BASE_FEATURES, **overrides)
    return EncodedFeatures(
        feature_schema_version="assessment-features-v1",
        features=features,
        categorical_index={"gender": ["male", "female"]},
    )


# ----------------------------------------------------------------------
# غياب الميزات — يُعيد نتيجة فارغة، لا خطأ إطلاقاً
# ----------------------------------------------------------------------
def test_no_features_returns_empty_predictions_not_error():
    result = _predictor().predict(_encoded())
    assert result.predictions == []
    assert result.predictor_version == PREDICTOR_VERSION


def test_always_returns_disease_prediction_result_instance():
    from app.domain.prediction import DiseasePredictionResult

    result = _predictor().predict(_encoded())
    assert isinstance(result, DiseasePredictionResult)


# ----------------------------------------------------------------------
# الحرارة
# ----------------------------------------------------------------------
def test_temperature_at_or_above_38_triggers_febrile_illness():
    result = _predictor().predict(_encoded(temperature_c=38.0))
    diseases = [p.disease for p in result.predictions]
    assert "Febrile Illness" in diseases


def test_temperature_below_38_does_not_trigger_febrile_illness():
    result = _predictor().predict(_encoded(temperature_c=37.5))
    diseases = [p.disease for p in result.predictions]
    assert "Febrile Illness" not in diseases


def test_temperature_none_does_not_trigger_febrile_illness():
    result = _predictor().predict(_encoded(temperature_c=None))
    assert result.predictions == []


# ----------------------------------------------------------------------
# الشدّة
# ----------------------------------------------------------------------
def test_severity_at_or_above_8_triggers_severe_condition():
    result = _predictor().predict(_encoded(severity=8))
    diseases = [p.disease for p in result.predictions]
    assert "Severe Condition" in diseases


def test_severity_below_8_does_not_trigger_severe_condition():
    result = _predictor().predict(_encoded(severity=7))
    diseases = [p.disease for p in result.predictions]
    assert "Severe Condition" not in diseases


# ----------------------------------------------------------------------
# التدخين — يرفع نقاط تنبؤات موجودة، لا يُنشئ تشخيصاً مستقلاً
# ----------------------------------------------------------------------
def test_smoking_boosts_existing_prediction_score():
    without_smoking = _predictor().predict(_encoded(temperature_c=39.0, smoking=0))
    with_smoking = _predictor().predict(_encoded(temperature_c=39.0, smoking=1))

    score_without = next(p.score for p in without_smoking.predictions if p.disease == "Febrile Illness")
    score_with = next(p.score for p in with_smoking.predictions if p.disease == "Febrile Illness")
    assert score_with > score_without


def test_smoking_alone_creates_no_prediction():
    """التدخين وحده (بلا تنبؤ آخر موجود) لا يُنتج أي تشخيص مستقلّ."""
    result = _predictor().predict(_encoded(smoking=1))
    assert result.predictions == []


def test_smoking_none_does_not_boost():
    result = _predictor().predict(_encoded(temperature_c=39.0, smoking=None))
    score = next(p.score for p in result.predictions if p.disease == "Febrile Illness")
    assert score == 0.6  # القيمة الأساسية بلا زيادة


# ----------------------------------------------------------------------
# عدد الأعراض (symptom_count) — فجوة معماريّة موثَّقة، Placeholder غير فعّال
# ----------------------------------------------------------------------
def test_symptom_count_rule_is_documented_placeholder_contributes_nothing():
    """EncodedFeatures لا يحمل عدد الأعراض بمخطّط v1 — القاعدة موجودة
    كدالة موثَّقة لكنها لا تُساهم بأي تنبؤ حالياً (لا اختلاق قيمة بديلة)."""
    result = _predictor().predict(_encoded())
    diseases = [p.disease for p in result.predictions]
    assert "General Viral Illness" not in diseases


def test_symptom_count_placeholder_even_with_other_signals_present():
    """حتى مع وجود إشارات أخرى (حرارة/شدّة)، Febrile/Severe فقط تظهران —
    لا "General Viral Illness" لغياب حقل عدد الأعراض بالمخطّط الحالي."""
    result = _predictor().predict(_encoded(temperature_c=39.0, severity=9))
    diseases = {p.disease for p in result.predictions}
    assert diseases == {"Febrile Illness", "Severe Condition"}


# ----------------------------------------------------------------------
# ترتيب النتائج — الأعلى نقاطاً أولاً، وحتمي عند التعادل
# ----------------------------------------------------------------------
def test_predictions_sorted_by_score_descending():
    result = _predictor().predict(_encoded(temperature_c=39.0, severity=9))
    scores = [p.score for p in result.predictions]
    assert scores == sorted(scores, reverse=True)
    assert result.predictions[0].disease == "Severe Condition"  # 0.7 > 0.6


def test_ordering_is_deterministic_across_repeated_calls():
    predictor = _predictor()
    encoded = _encoded(temperature_c=39.0, severity=9, smoking=1)
    first = predictor.predict(encoded)
    second = predictor.predict(encoded)
    assert [p.disease for p in first.predictions] == [p.disease for p in second.predictions]
    assert [p.score for p in first.predictions] == [p.score for p in second.predictions]


# ----------------------------------------------------------------------
# predictor_version
# ----------------------------------------------------------------------
def test_predictor_version_is_reported_on_every_result():
    result = _predictor().predict(_encoded(temperature_c=39.0))
    assert result.predictor_version == "rule-based-v1"
    assert result.predictor_version == PREDICTOR_VERSION


# ----------------------------------------------------------------------
# score ضمن [0, 1] دائماً — حتى بعد زيادة التدخين
# ----------------------------------------------------------------------
def test_all_scores_within_zero_one_bounds():
    result = _predictor().predict(_encoded(temperature_c=39.0, severity=9, smoking=1))
    for prediction in result.predictions:
        assert 0.0 <= prediction.score <= 1.0


def test_smoking_boost_never_exceeds_one():
    result = _predictor().predict(_encoded(severity=10, smoking=1))
    severe = next(p for p in result.predictions if p.disease == "Severe Condition")
    assert severe.score <= 1.0


# ----------------------------------------------------------------------
# كل تنبؤ يحمل تفسيراً غير فارغ
# ----------------------------------------------------------------------
def test_every_prediction_has_non_empty_explanation():
    result = _predictor().predict(_encoded(temperature_c=39.0, severity=9))
    assert all(isinstance(p.explanation, str) and p.explanation for p in result.predictions)


# ----------------------------------------------------------------------
# لا يعتمد إطلاقاً على أي شيء غير EncodedFeatures.features
# ----------------------------------------------------------------------
def test_predictor_reads_only_encoded_features_dict():
    """يضمن أنّ المُتنبِّئ لا يحتاج أي بيانات إضافية غير EncodedFeatures —
    مطابقة صريحة لعقد DiseasePredictorPort (لا ClinicalFeatureSet)."""
    encoded = EncodedFeatures(
        feature_schema_version="v-test",
        features={"temperature_c": 38.5},
        categorical_index={},
    )
    result = _predictor().predict(encoded)
    assert any(p.disease == "Febrile Illness" for p in result.predictions)
