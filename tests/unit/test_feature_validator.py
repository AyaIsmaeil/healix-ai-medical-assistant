"""اختبارات وحدة لـFeatureValidator (Phase 3.2) — فشل ناعم للحقول الاختيارية،
واستثناء صريح فقط عند غياب معلومة سريرية جوهرية. قواعد وهمية محقونة (بلا
قراءة ملف)، نفس روح باقي اختبارات الوحدة (Fakes لا مكتبات mocking)."""

import pytest

from app.domain.assessment import (
    ClinicalFeatureSet,
    Demographics,
    SymptomDescriptors,
    SymptomFeature,
)
from app.domain.feature_validator import FeatureValidator
from app.exceptions import FeatureValidationError

RULES = {
    "fields": {
        "age": {"type": "int", "min": 0, "max": 120, "on_out_of_range": "reject"},
        "temperature_c": {"type": "float", "min": 30.0, "max": 45.0, "on_out_of_range": "reject"},
        "severity_0_10": {"type": "int", "min": 0, "max": 10, "on_out_of_range": "clip"},
        "onset_days_ago": {"type": "int", "min": 0, "max": 3650, "on_out_of_range": "reject"},
        "gender": {"type": "enum", "allowed": ["male", "female"]},
    }
}

_DEFAULT_SYMPTOM = SymptomFeature(name="صداع", negated=False, extraction_confidence=0.9)


def _validator():
    return FeatureValidator(rules=RULES)


def _features(**overrides):
    defaults = dict(
        session_id="sid",
        demographics=Demographics(),
        symptoms=[_DEFAULT_SYMPTOM],
    )
    defaults.update(overrides)
    return ClinicalFeatureSet(**defaults)


# ----------------------------------------------------------------------
# العمر
# ----------------------------------------------------------------------
def test_valid_age_passes_unchanged():
    features = _features(demographics=Demographics(age=34))
    result = _validator().validate(features)

    assert result.demographics.age == 34
    assert "demographics.age" not in result.validation.rejected_fields
    assert "demographics.age" not in result.validation.corrected_fields


def test_invalid_age_is_rejected_to_null():
    features = _features(demographics=Demographics(age=200))
    result = _validator().validate(features)

    assert result.demographics.age is None
    assert "demographics.age" in result.validation.rejected_fields
    assert any("age" in w for w in result.validation.warnings)


# ----------------------------------------------------------------------
# الحرارة
# ----------------------------------------------------------------------
def test_invalid_temperature_is_rejected_to_null():
    features = _features()
    features.temperature_c = 100.0
    result = _validator().validate(features)

    assert result.temperature_c is None
    assert "temperature_c" in result.validation.rejected_fields


def test_valid_temperature_passes_unchanged():
    features = _features()
    features.temperature_c = 38.5
    result = _validator().validate(features)

    assert result.temperature_c == 38.5
    assert result.validation.rejected_fields == []


# ----------------------------------------------------------------------
# قصّ الشدّة (clip) — لا رفض
# ----------------------------------------------------------------------
def test_severity_out_of_range_is_clipped_not_rejected():
    symptom = SymptomFeature(
        name="صداع", negated=False, extraction_confidence=0.9,
        descriptors=SymptomDescriptors(severity_0_10=15),
    )
    features = _features(symptoms=[symptom])
    result = _validator().validate(features)

    assert result.symptoms[0].descriptors.severity_0_10 == 10
    assert any("severity_0_10" in c for c in result.validation.corrected_fields)
    assert not any("severity_0_10" in r for r in result.validation.rejected_fields)


def test_severity_negative_clipped_to_minimum():
    symptom = SymptomFeature(
        name="صداع", negated=False, extraction_confidence=0.9,
        descriptors=SymptomDescriptors(severity_0_10=-3),
    )
    features = _features(symptoms=[symptom])
    result = _validator().validate(features)

    assert result.symptoms[0].descriptors.severity_0_10 == 0


# ----------------------------------------------------------------------
# تعارض الحمل × الجنس
# ----------------------------------------------------------------------
def test_pregnancy_true_with_male_gender_is_corrected_to_null():
    features = _features(demographics=Demographics(gender="male", pregnancy_possible=True))
    result = _validator().validate(features)

    assert result.demographics.pregnancy_possible is None
    assert "demographics.pregnancy_possible" in result.validation.corrected_fields
    assert any("gender=male" in w for w in result.validation.warnings)


def test_pregnancy_true_with_female_gender_is_untouched():
    features = _features(demographics=Demographics(gender="female", pregnancy_possible=True))
    result = _validator().validate(features)

    assert result.demographics.pregnancy_possible is True
    assert "demographics.pregnancy_possible" not in result.validation.corrected_fields


def test_pregnancy_conflict_not_inferred_when_gender_itself_invalid():
    """gender غير صالح (كـ'alien') يُرفض بذاته أولاً ويُعاد لـnull؛ تعارض
    الحمل يتطلّب دليلاً إيجابياً (gender == 'male' فعلياً)، فلا يُشغَّل هنا —
    لا نخمّن أنّ الجنس المشوَّه كان 'male' لنُصفّر الحمل أيضاً بلا دليل."""
    features = _features(demographics=Demographics(gender="alien", pregnancy_possible=True))
    result = _validator().validate(features)

    assert result.demographics.gender is None
    assert "demographics.gender" in result.validation.rejected_fields
    assert result.demographics.pregnancy_possible is True  # بلا تغيير — بلا دليل تعارض مؤكَّد
    assert "demographics.pregnancy_possible" not in result.validation.corrected_fields


# ----------------------------------------------------------------------
# تعداد غير صالح
# ----------------------------------------------------------------------
def test_invalid_gender_enum_is_rejected_to_null():
    features = _features(demographics=Demographics(gender="alien"))
    result = _validator().validate(features)

    assert result.demographics.gender is None
    assert "demographics.gender" in result.validation.rejected_fields


def test_valid_gender_enum_passes_unchanged():
    features = _features(demographics=Demographics(gender="female"))
    result = _validator().validate(features)

    assert result.demographics.gender == "female"
    assert result.validation.rejected_fields == []


# ----------------------------------------------------------------------
# الحقول الفارغة (None) ليست "غير صالحة"
# ----------------------------------------------------------------------
def test_none_fields_are_not_flagged_as_invalid():
    """حقل None يعني 'غير محلول' لا 'غير صالح' — unresolved_fields هو
    المسؤول عن تتبّعه، لا ValidationReport."""
    features = _features(demographics=Demographics(age=None, gender=None))
    result = _validator().validate(features)

    assert result.validation.rejected_fields == []
    assert result.validation.corrected_fields == []
    assert result.validation.validity_score is None  # لا شيء فُحص فعلياً


# ----------------------------------------------------------------------
# validity_score
# ----------------------------------------------------------------------
def test_validity_score_full_when_everything_valid():
    features = _features(demographics=Demographics(age=30, gender="male"))
    result = _validator().validate(features)
    assert result.validation.validity_score == 1.0


def test_validity_score_zero_when_everything_rejected():
    features = _features(demographics=Demographics(age=999, gender="unknown"))
    result = _validator().validate(features)
    assert result.validation.validity_score == 0.0


def test_validity_score_partial_penalty_for_correction():
    symptom = SymptomFeature(
        name="صداع", negated=False, extraction_confidence=0.9,
        descriptors=SymptomDescriptors(severity_0_10=15),
    )
    features = _features(demographics=Demographics(age=30), symptoms=[symptom])
    result = _validator().validate(features)
    # فُحص حقلان: age (صالح) + severity (مُصحَّح، عقوبة 0.5) => (2-0.5)/2 = 0.75
    assert result.validation.validity_score == 0.75


# ----------------------------------------------------------------------
# FeatureValidationError — فقط عند غياب معلومة جوهرية
# ----------------------------------------------------------------------
def test_raises_feature_validation_error_when_no_symptoms_at_all():
    features = _features(symptoms=[])
    with pytest.raises(FeatureValidationError):
        _validator().validate(features)


def test_does_not_raise_when_only_negated_symptoms_present():
    """عرَض منفي واحد على الأقل لا يزال 'معلومة سريرية' — لا يرفع استثناءً."""
    features = _features(symptoms=[SymptomFeature(name="حرارة", negated=True, extraction_confidence=0.8)])
    result = _validator().validate(features)  # يجب ألا يرفع
    assert result.symptoms[0].negated is True


# ----------------------------------------------------------------------
# التسلسل (chaining) وعدم وجود قاعدة لحقل غير معروف
# ----------------------------------------------------------------------
def test_returns_same_object_for_chaining():
    features = _features()
    result = _validator().validate(features)
    assert result is features


def test_field_without_defined_rule_passes_through_unchanged():
    validator = FeatureValidator(rules={"fields": {"age": RULES["fields"]["age"]}})
    features = _features()
    features.temperature_c = 999.0  # لا قاعدة لـtemperature_c بهذا القاموس الجزئي
    result = validator.validate(features)

    assert result.temperature_c == 999.0
    assert result.validation.rejected_fields == []
