"""اختبارات وحدة لـRuleBasedUrgencyClassifier (Phase 3.5) — قواعد حتمية
بسيطة (Placeholder)، بلا أي مكتبة ML. مدخل ClinicalFeatureSet حقيقي مباشر
(بلا مكتبات mocking)، نفس روح باقي اختبارات الوحدة."""

from app.domain.assessment import (
    ClinicalFeatureSet,
    Demographics,
    DerivedFeatures,
    SymptomDescriptors,
    SymptomFeature,
)
from app.domain.rule_based_urgency_classifier import RuleBasedUrgencyClassifier
from app.domain.urgency import UrgencyAssessment, UrgencyLevel


def _classifier():
    return RuleBasedUrgencyClassifier()


def _primary_symptom(**descriptor_overrides):
    return SymptomFeature(
        name="صداع شديد",
        negated=False,
        extraction_confidence=0.9,
        is_primary=True,
        descriptors=SymptomDescriptors(**descriptor_overrides),
    )


def _features(**overrides):
    defaults = dict(
        session_id="sid",
        demographics=Demographics(),
        symptoms=[_primary_symptom()],
        derived=DerivedFeatures(),
    )
    defaults.update(overrides)
    return ClinicalFeatureSet(**defaults)


# ----------------------------------------------------------------------
# الحرارة ≥ 40 → EMERGENCY
# ----------------------------------------------------------------------
def test_temperature_40_triggers_emergency():
    features = _features(temperature_c=40.0)
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.EMERGENCY


def test_temperature_above_40_triggers_emergency():
    features = _features(temperature_c=41.2)
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.EMERGENCY


def test_temperature_just_below_40_does_not_trigger_emergency():
    features = _features(temperature_c=39.9)
    result = _classifier().classify(features)
    assert result.level != UrgencyLevel.EMERGENCY


# ----------------------------------------------------------------------
# الشدّة ≥ 9 → URGENT
# ----------------------------------------------------------------------
def test_severity_9_triggers_urgent():
    features = _features(symptoms=[_primary_symptom(severity_0_10=9)])
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.URGENT


def test_severity_10_triggers_urgent():
    features = _features(symptoms=[_primary_symptom(severity_0_10=10)])
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.URGENT


# ----------------------------------------------------------------------
# الشدّة بين 6 و8 → SEMI_URGENT
# ----------------------------------------------------------------------
def test_severity_6_triggers_semi_urgent():
    features = _features(symptoms=[_primary_symptom(severity_0_10=6)])
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.SEMI_URGENT


def test_severity_8_triggers_semi_urgent():
    features = _features(symptoms=[_primary_symptom(severity_0_10=8)])
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.SEMI_URGENT


def test_severity_9_is_urgent_not_semi_urgent():
    """حدّ فاصل: 9 يجب أن يكون URGENT لا SEMI_URGENT رغم قربه من 8."""
    features = _features(symptoms=[_primary_symptom(severity_0_10=9)])
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.URGENT


# ----------------------------------------------------------------------
# الشدّة المنخفضة → NON_URGENT
# ----------------------------------------------------------------------
def test_low_severity_is_non_urgent():
    features = _features(symptoms=[_primary_symptom(severity_0_10=3)])
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.NON_URGENT


def test_severity_5_is_non_urgent():
    """حدّ فاصل: 5 أقل من نطاق SEMI_URGENT (6-8) فيبقى NON_URGENT."""
    features = _features(symptoms=[_primary_symptom(severity_0_10=5)])
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.NON_URGENT


# ----------------------------------------------------------------------
# عدم وجود بيانات — لا اختلاق، NON_URGENT افتراضياً
# ----------------------------------------------------------------------
def test_no_data_at_all_defaults_to_non_urgent():
    features = _features()  # لا حرارة، لا شدّة، لا حمل
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.NON_URGENT


def test_no_primary_symptom_does_not_crash():
    features = _features(
        symptoms=[SymptomFeature(name="عرَض", negated=False, extraction_confidence=0.5, is_primary=False)]
    )
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.NON_URGENT


def test_empty_symptoms_does_not_crash():
    features = _features(symptoms=[])
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.NON_URGENT


# ----------------------------------------------------------------------
# احتمال الحمل — حدّ أدنى SEMI_URGENT، بلا خفض مستوى أعلى
# ----------------------------------------------------------------------
def test_pregnancy_possible_raises_non_urgent_floor_to_semi_urgent():
    features = _features(demographics=Demographics(pregnancy_possible=True))
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.SEMI_URGENT
    assert "حمل" in result.explanation


def test_pregnancy_possible_does_not_downgrade_emergency():
    features = _features(temperature_c=41.0, demographics=Demographics(pregnancy_possible=True))
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.EMERGENCY


def test_pregnancy_false_has_no_effect():
    features = _features(demographics=Demographics(pregnancy_possible=False))
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.NON_URGENT


def test_pregnancy_none_has_no_effect():
    features = _features(demographics=Demographics(pregnancy_possible=None))
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.NON_URGENT


# ----------------------------------------------------------------------
# علامة الخطر (red flag) — EMERGENCY لو مُسجَّلة
# ----------------------------------------------------------------------
def test_red_flag_triggers_emergency():
    features = _features(derived=DerivedFeatures(has_red_flag=True))
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.EMERGENCY


# ----------------------------------------------------------------------
# ترتيب الأولويات — الأخطر يفوز دائماً عند تعدّد المؤشّرات معاً
# ----------------------------------------------------------------------
def test_emergency_wins_over_urgent_when_both_conditions_met():
    features = _features(
        temperature_c=41.0,
        symptoms=[_primary_symptom(severity_0_10=10)],
    )
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.EMERGENCY


def test_urgent_wins_over_semi_urgent_when_severity_is_9():
    features = _features(symptoms=[_primary_symptom(severity_0_10=9)])
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.URGENT


# ----------------------------------------------------------------------
# score دائماً ضمن [0, 1]
# ----------------------------------------------------------------------
def test_all_levels_produce_score_within_bounds():
    scenarios = [
        _features(temperature_c=41.0),
        _features(symptoms=[_primary_symptom(severity_0_10=9)]),
        _features(symptoms=[_primary_symptom(severity_0_10=7)]),
        _features(),
    ]
    for features in scenarios:
        result = _classifier().classify(features)
        assert 0.0 <= result.score <= 1.0


def test_emergency_score_is_maximum():
    result = _classifier().classify(_features(temperature_c=41.0))
    assert result.score == 1.0


def test_non_urgent_score_is_minimum_among_levels():
    result = _classifier().classify(_features())
    assert result.score == min(
        _classifier().classify(_features(temperature_c=41.0)).score,
        _classifier().classify(_features(symptoms=[_primary_symptom(severity_0_10=9)])).score,
        _classifier().classify(_features(symptoms=[_primary_symptom(severity_0_10=7)])).score,
        result.score,
    )


# ----------------------------------------------------------------------
# explanation غير فارغ دائماً
# ----------------------------------------------------------------------
def test_explanation_never_empty():
    scenarios = [
        _features(temperature_c=41.0),
        _features(symptoms=[_primary_symptom(severity_0_10=9)]),
        _features(symptoms=[_primary_symptom(severity_0_10=7)]),
        _features(symptoms=[_primary_symptom(severity_0_10=2)]),
        _features(),
    ]
    for features in scenarios:
        result = _classifier().classify(features)
        assert isinstance(result.explanation, str) and result.explanation.strip()


# ----------------------------------------------------------------------
# Enum الصحيح ونوع النتيجة
# ----------------------------------------------------------------------
def test_result_is_urgency_assessment_instance():
    result = _classifier().classify(_features())
    assert isinstance(result, UrgencyAssessment)


def test_level_is_urgency_level_enum_member():
    result = _classifier().classify(_features(temperature_c=41.0))
    assert isinstance(result.level, UrgencyLevel)
    assert result.level.value == "EMERGENCY"


def test_all_four_levels_are_reachable():
    levels = {
        _classifier().classify(_features(temperature_c=41.0)).level,
        _classifier().classify(_features(symptoms=[_primary_symptom(severity_0_10=9)])).level,
        _classifier().classify(_features(symptoms=[_primary_symptom(severity_0_10=7)])).level,
        _classifier().classify(_features()).level,
    }
    assert levels == {
        UrgencyLevel.EMERGENCY,
        UrgencyLevel.URGENT,
        UrgencyLevel.SEMI_URGENT,
        UrgencyLevel.NON_URGENT,
    }


# ----------------------------------------------------------------------
# قفل إصلاح: الشدّة تُقرَأ من وصف العرَض الرئيسي المُتحقَّق منه، لا من
# derived.primary_symptom_severity (نفس فجوة FeatureEncoder مرحلة ٣.٣).
# ----------------------------------------------------------------------
def test_severity_reads_from_validated_primary_descriptor_not_stale_derived_copy():
    features = _features(
        symptoms=[_primary_symptom(severity_0_10=7)],  # كأنّها بعد قصّ 99 -> 7
        derived=DerivedFeatures(primary_symptom_severity=99),  # نسخة قديمة لم تُزامَن
    )
    result = _classifier().classify(features)
    assert result.level == UrgencyLevel.SEMI_URGENT  # وفق 7 المُتحقَّق منها، لا 99


# ----------------------------------------------------------------------
# لا جوانب تأثير على المدخل
# ----------------------------------------------------------------------
def test_classify_has_no_side_effects_on_input():
    features = _features(temperature_c=39.0, symptoms=[_primary_symptom(severity_0_10=7)])
    _classifier().classify(features)
    assert features.temperature_c == 39.0
    assert features.symptoms[0].descriptors.severity_0_10 == 7
