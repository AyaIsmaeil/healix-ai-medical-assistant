"""اختبارات وحدة لـMLDiseasePredictor (Phase 3.4، مسار موازٍ) — نفس روح
test_rule_based_predictor.py: مدخل EncodedFeatures حقيقي مباشر، بلا مكتبات
mocking، ونموذج LogisticRegression صغير حقيقي مُدرَّب داخل الاختبار نفسه
(لا الملف الحقيقي بـmodels/ — لا اعتماد على القرص، سرعة وعزل تام)."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import pytest

from app.domain.feature_encoder import EncodedFeatures
from app.domain.ml_disease_predictor import PREDICTOR_VERSION, MLDiseasePredictor
from app.exceptions import InferenceError

_FEATURE_NAMES = ["E_1", "E_2", "E_3", "AGE", "SEX_M"]

_BASE_FEATURES = {
    "age": None,
    "gender_male": None,
    "gender_female": None,
    "severity": None,
    "smoking": None,
    "temperature_c": None,
}


def _encoded(**overrides):
    features = dict(_BASE_FEATURES, **overrides)
    return EncodedFeatures(
        feature_schema_version="assessment-features-v-test",
        features=features,
        categorical_index={},
    )


def _train_small_model():
    """نموذج LogisticRegression صغير حقيقي (3 فئات، عمودان أول لكل عرض دليل
    مميّز + AGE + SEX_M) — بيانات اصطناعية بسيطة، `random_state=42`."""
    X = np.array(
        [
            [1, 0, 0, 30, 1],
            [1, 0, 0, 25, 1],
            [1, 0, 0, 28, 0],
            [0, 1, 0, 40, 0],
            [0, 1, 0, 45, 0],
            [0, 1, 0, 42, 1],
            [0, 0, 1, 60, 1],
            [0, 0, 1, 65, 0],
            [0, 0, 1, 62, 1],
        ]
    )
    y = ["Flu", "Flu", "Flu", "Cold", "Cold", "Cold", "Allergy", "Allergy", "Allergy"]

    label_encoder = LabelEncoder().fit(y)
    model = LogisticRegression(random_state=42).fit(X, label_encoder.transform(y))
    return model, label_encoder


def _predictor(feature_names=None):
    model, label_encoder = _train_small_model()
    return MLDiseasePredictor(
        model=model,
        feature_names=feature_names if feature_names is not None else _FEATURE_NAMES,
        label_encoder=label_encoder,
    )


# ----------------------------------------------------------------------
# غياب رموز E_* — نتيجة فارغة، لا خطأ (فجوة v1.json الموثَّقة)
# ----------------------------------------------------------------------
def test_no_evidence_codes_returns_empty_predictions_not_error():
    result = _predictor().predict(_encoded())
    assert result.predictions == []
    assert result.predictor_version == PREDICTOR_VERSION


def test_only_age_and_sex_without_evidence_returns_empty():
    """AGE/SEX_M وحدهما بلا أي E_* لا يكفيان لبدء الاستنتاج."""
    result = _predictor().predict(_encoded(age=30, gender_male=1))
    assert result.predictions == []


# ----------------------------------------------------------------------
# رمز غير معروف يُتجاهَل بصمت، لا يُسقط الاستنتاج ولا يُغيّر النتيجة
# ----------------------------------------------------------------------
def test_unknown_evidence_code_is_ignored():
    encoded_with_unknown = _encoded(E_1=1, E_999=1, age=30, gender_male=1)
    encoded_without_unknown = _encoded(E_1=1, age=30, gender_male=1)

    predictor = _predictor()
    result_with = predictor.predict(encoded_with_unknown)
    result_without = predictor.predict(encoded_without_unknown)

    assert result_with.predictions != []
    scores_with = [(p.disease, p.score) for p in result_with.predictions]
    scores_without = [(p.disease, p.score) for p in result_without.predictions]
    assert scores_with == scores_without  # E_999 لم يغيّر شيئاً


# ----------------------------------------------------------------------
# مطابقة رمز E_* حقيقي واحد على الأقل يكفي لبدء الاستنتاج
# ----------------------------------------------------------------------
def test_single_matched_evidence_code_triggers_inference():
    result = _predictor().predict(_encoded(E_1=1, age=28, gender_male=0))
    assert result.predictions != []
    assert result.predictions[0].disease == "Flu"  # الإشارة الأقوى في بيانات التدريب


def test_top_predictions_capped_at_five():
    result = _predictor().predict(_encoded(E_3=1, age=60, gender_male=1))
    assert len(result.predictions) <= 5
    assert len(result.predictions) == 3  # 3 فئات فقط بنموذج الاختبار


# ----------------------------------------------------------------------
# الترتيب حتمي عبر نداءات متكررة
# ----------------------------------------------------------------------
def test_ordering_is_deterministic_across_repeated_calls():
    predictor = _predictor()
    encoded = _encoded(E_2=1, age=42, gender_male=1)
    first = predictor.predict(encoded)
    second = predictor.predict(encoded)
    assert [p.disease for p in first.predictions] == [p.disease for p in second.predictions]
    assert [p.score for p in first.predictions] == [p.score for p in second.predictions]


def test_predictions_sorted_by_score_descending():
    result = _predictor().predict(_encoded(E_2=1, age=42, gender_male=1))
    scores = [p.score for p in result.predictions]
    assert scores == sorted(scores, reverse=True)


# ----------------------------------------------------------------------
# عدم تطابق الأبعاد يرفع InferenceError، لا انهياراً غامضاً
# ----------------------------------------------------------------------
def test_feature_names_length_mismatch_raises_inference_error():
    """feature_names أقصر مما دُرِّب عليه النموذج (4 بدل 5) → predict_proba
    يفشل داخلياً؛ يجب أن يُغلَّف بـInferenceError لا استثناء sklearn خام."""
    mismatched_predictor = _predictor(feature_names=_FEATURE_NAMES[:-1])
    with pytest.raises(InferenceError):
        mismatched_predictor.predict(_encoded(E_1=1, age=28, gender_male=0))


# ----------------------------------------------------------------------
# predictor_version
# ----------------------------------------------------------------------
def test_predictor_version_is_reported_on_every_result():
    result = _predictor().predict(_encoded(E_1=1, age=28, gender_male=0))
    assert result.predictor_version == "ml-logreg-v1"
    assert result.predictor_version == PREDICTOR_VERSION


# ----------------------------------------------------------------------
# كل تنبؤ يحمل تفسيراً غير فارغ، وscore ضمن [0, 1]
# ----------------------------------------------------------------------
def test_every_prediction_has_non_empty_explanation_and_bounded_score():
    result = _predictor().predict(_encoded(E_1=1, age=28, gender_male=0))
    assert result.predictions
    for prediction in result.predictions:
        assert isinstance(prediction.explanation, str) and prediction.explanation
        assert 0.0 <= prediction.score <= 1.0


# ----------------------------------------------------------------------
# لا يعتمد إطلاقاً على أي شيء غير EncodedFeatures.features
# ----------------------------------------------------------------------
def test_predictor_reads_only_encoded_features_dict():
    """يضمن أنّ المُتنبِّئ لا يحتاج أي بيانات إضافية غير EncodedFeatures —
    مطابقة صريحة لعقد DiseasePredictorPort (لا ClinicalFeatureSet)."""
    encoded = EncodedFeatures(
        feature_schema_version="v-test",
        features={"E_1": 1, "age": 28, "gender_male": 0},
        categorical_index={},
    )
    result = _predictor().predict(encoded)
    assert result.predictions

