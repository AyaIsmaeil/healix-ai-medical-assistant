"""اختبارات وحدة لـFeatureEncoder (Phase 3.3) — ترميز حتمي، بلا اختلاق قيم،
بلا numpy/pandas. مخطّط وهمي محقون (بلا قراءة ملف)، نفس روح باقي اختبارات
الوحدة (كائنات حقيقية، لا مكتبات mocking)."""

from app.domain.assessment import (
    ClinicalFeatureSet,
    Demographics,
    DerivedFeatures,
    Lifestyle,
    SymptomDescriptors,
    SymptomFeature,
)
from app.domain.feature_encoder import EncodedFeatures, FeatureEncoder

SCHEMA = {
    "schema_version": "assessment-features-v1",
    "feature_order": [
        "age",
        "gender_male",
        "gender_female",
        "severity",
        "smoking",
        "temperature_c",
        "progression_worsening",
        "progression_stable",
        "progression_improving",
    ],
    "numeric_fields": {
        "age": {"source": "demographics.age", "nullable": True},
        "severity": {"source": "derived.primary_symptom_severity", "nullable": True},
        "temperature_c": {"source": "temperature_c", "nullable": True},
    },
    "boolean_fields": {
        "smoking": {"source": "lifestyle.smoking", "nullable": True},
    },
    "categorical_fields": {
        "gender": {
            "source": "demographics.gender",
            "values": ["male", "female"],
            "one_hot_prefix": "gender",
            "nullable": True,
        },
        "progression": {
            "source": "primary_symptom.descriptors.progression",
            "values": ["worsening", "stable", "improving"],
            "one_hot_prefix": "progression",
            "nullable": True,
        },
    },
}


def _encoder():
    return FeatureEncoder(schema=SCHEMA)


def _primary_symptom(**descriptor_overrides):
    return SymptomFeature(
        name="صداع شديد",
        negated=False,
        extraction_confidence=0.97,
        is_primary=True,
        descriptors=SymptomDescriptors(**descriptor_overrides),
    )


def _features(**overrides):
    defaults = dict(
        session_id="sid",
        demographics=Demographics(),
        symptoms=[_primary_symptom()],
        lifestyle=Lifestyle(),
        derived=DerivedFeatures(),
    )
    defaults.update(overrides)
    return ClinicalFeatureSet(**defaults)


# ----------------------------------------------------------------------
# مطابقة المثال الحرفي بالمهمّة (اختبار امتثال شامل)
# ----------------------------------------------------------------------
def test_matches_literal_task_example():
    features = _features(
        demographics=Demographics(age=34, gender="male"),
        symptoms=[_primary_symptom(severity_0_10=8, progression="worsening")],
        lifestyle=Lifestyle(smoking=True),
        temperature_c=38.5,
    )
    result = _encoder().encode(features)

    assert result.feature_schema_version == "assessment-features-v1"
    assert result.features == {
        "age": 34,
        "gender_male": 1,
        "gender_female": 0,
        "severity": 8,
        "smoking": 1,
        "temperature_c": 38.5,
        "progression_worsening": 1,
        "progression_stable": 0,
        "progression_improving": 0,
    }
    assert result.categorical_index == {
        "gender": ["male", "female"],
        "progression": ["worsening", "stable", "improving"],
    }


# ----------------------------------------------------------------------
# ترميز رقمي — يبقى كما هو
# ----------------------------------------------------------------------
def test_numeric_value_passes_through_unchanged():
    features = _features(demographics=Demographics(age=45))
    result = _encoder().encode(features)
    assert result.features["age"] == 45


def test_numeric_missing_becomes_null_not_zero():
    features = _features(demographics=Demographics(age=None))
    result = _encoder().encode(features)
    assert result.features["age"] is None


# ----------------------------------------------------------------------
# ترميز منطقي — 0/1، لا اختلاق
# ----------------------------------------------------------------------
def test_boolean_true_encodes_to_one():
    features = _features(lifestyle=Lifestyle(smoking=True))
    result = _encoder().encode(features)
    assert result.features["smoking"] == 1


def test_boolean_false_encodes_to_zero():
    features = _features(lifestyle=Lifestyle(smoking=False))
    result = _encoder().encode(features)
    assert result.features["smoking"] == 0


def test_boolean_missing_becomes_null_not_zero():
    """صفر كان سيَختلق تأكيداً كاذباً ('مؤكَّد غير مدخّن') من غياب معرفة."""
    features = _features(lifestyle=Lifestyle(smoking=None))
    result = _encoder().encode(features)
    assert result.features["smoking"] is None


# ----------------------------------------------------------------------
# ترميز فئوي (one-hot)
# ----------------------------------------------------------------------
def test_categorical_one_hot_marks_correct_category():
    features = _features(demographics=Demographics(gender="female"))
    result = _encoder().encode(features)
    assert result.features["gender_male"] == 0
    assert result.features["gender_female"] == 1


def test_categorical_missing_makes_all_columns_null_not_zero():
    """القيمة الفئوية المفقودة → كل أعمدة one-hot المقابلة null، لا 0 —
    تصفيرها كان سيَختلق تأكيداً كاذباً ('مؤكَّد ليس ذكراً وليس أنثى')."""
    features = _features(demographics=Demographics(gender=None))
    result = _encoder().encode(features)
    assert result.features["gender_male"] is None
    assert result.features["gender_female"] is None


def test_categorical_unknown_value_treated_as_missing():
    """قيمة خارج قائمة المخطّط (لم تُحسم null بعد التحقّق نظرياً) — لا تُختلَق
    فئة جديدة، تُعامَل كمفقودة (كل الأعمدة null)."""
    features = _features(demographics=Demographics(gender="unknown"))
    result = _encoder().encode(features)
    assert result.features["gender_male"] is None
    assert result.features["gender_female"] is None


def test_progression_one_hot_for_stable():
    features = _features(symptoms=[_primary_symptom(progression="stable")])
    result = _encoder().encode(features)
    assert result.features["progression_worsening"] == 0
    assert result.features["progression_stable"] == 1
    assert result.features["progression_improving"] == 0


# ----------------------------------------------------------------------
# categorical_index
# ----------------------------------------------------------------------
def test_categorical_index_matches_schema_values():
    result = _encoder().encode(_features())
    assert result.categorical_index["gender"] == ["male", "female"]
    assert result.categorical_index["progression"] == ["worsening", "stable", "improving"]


# ----------------------------------------------------------------------
# نسخة المخطّط
# ----------------------------------------------------------------------
def test_feature_schema_version_matches_injected_schema():
    result = _encoder().encode(_features())
    assert result.feature_schema_version == SCHEMA["schema_version"]


def test_different_schema_version_is_reflected():
    other_schema = dict(SCHEMA, schema_version="assessment-features-v2-test")
    result = FeatureEncoder(schema=other_schema).encode(_features())
    assert result.feature_schema_version == "assessment-features-v2-test"


# ----------------------------------------------------------------------
# الترتيب الحتمي
# ----------------------------------------------------------------------
def test_feature_keys_follow_schema_feature_order_exactly():
    result = _encoder().encode(_features())
    assert list(result.features.keys()) == SCHEMA["feature_order"]


def test_identical_input_produces_identical_output_value_and_order():
    features_a = _features(
        demographics=Demographics(age=30, gender="male"),
        symptoms=[_primary_symptom(severity_0_10=7, progression="stable")],
        lifestyle=Lifestyle(smoking=False),
        temperature_c=37.2,
    )
    features_b = _features(
        demographics=Demographics(age=30, gender="male"),
        symptoms=[_primary_symptom(severity_0_10=7, progression="stable")],
        lifestyle=Lifestyle(smoking=False),
        temperature_c=37.2,
    )
    encoder = _encoder()
    result_a = encoder.encode(features_a)
    result_b = encoder.encode(features_b)

    assert result_a.features == result_b.features
    assert list(result_a.features.keys()) == list(result_b.features.keys())
    assert result_a.categorical_index == result_b.categorical_index


def test_encode_has_no_side_effects_on_input():
    """الترميز لا يُعدَّل ClinicalFeatureSet المُمرَّر — عملية قراءة خالصة."""
    features = _features(demographics=Demographics(age=30, gender="male"))
    _encoder().encode(features)
    assert features.demographics.age == 30
    assert features.demographics.gender == "male"


# ----------------------------------------------------------------------
# EncodedFeatures نفسها ليست numpy ولا pandas — قاموس صرف
# ----------------------------------------------------------------------
def test_encoded_features_uses_plain_dict_no_numpy_no_pandas():
    result = _encoder().encode(_features())
    assert isinstance(result, EncodedFeatures)
    assert isinstance(result.features, dict)
    assert isinstance(result.categorical_index, dict)
    assert type(result.features) is dict  # لا OrderedDict مخصّصة ولا أي نوع آخر


# ----------------------------------------------------------------------
# قفل إصلاح: الشدّة تُقرَأ من وصف العرَض الرئيسي المُتحقَّق منه، لا من
# derived.primary_symptom_severity (نسخة قد لا تُزامنها FeatureValidator).
# ----------------------------------------------------------------------
def test_severity_reads_from_validated_primary_descriptor_not_stale_derived_copy():
    """محاكاة تباعد ما بعد التحقّق: FeatureValidator يُصحِّح
    descriptors.severity_0_10 لكن لا يُزامن derived.primary_symptom_severity —
    المشفِّر يجب أن يعتمد النسخة المُتحقَّق منها (الوصف)، لا النسخة القديمة."""
    features = _features(
        symptoms=[_primary_symptom(severity_0_10=10)],  # كأنّها بعد قصّ 99 -> 10
        derived=DerivedFeatures(primary_symptom_severity=99),  # نسخة قديمة لم تُزامَن
    )
    result = _encoder().encode(features)
    assert result.features["severity"] == 10  # القيمة المُتحقَّق منها، لا 99


def test_no_primary_symptom_severity_is_null():
    features = _features(symptoms=[
        SymptomFeature(name="عرَض", negated=False, extraction_confidence=0.5, is_primary=False)
    ])
    result = _encoder().encode(features)
    assert result.features["severity"] is None


def test_no_symptoms_at_all_severity_and_progression_are_null():
    features = _features(symptoms=[])
    result = _encoder().encode(features)
    assert result.features["severity"] is None
    assert result.features["progression_worsening"] is None
    assert result.features["progression_stable"] is None
    assert result.features["progression_improving"] is None
