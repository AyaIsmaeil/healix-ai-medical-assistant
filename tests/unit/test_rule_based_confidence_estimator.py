"""اختبارات وحدة لـRuleBasedConfidenceEstimator (Phase 3.7) — قواعد حتمية
بسيطة (Placeholder)، بلا أي مكتبة ML. مدخلات حقيقية مباشرة (بلا مكتبات
mocking)، نفس روح باقي اختبارات الوحدة. المُقدِّر لا يتنبّأ بمرض ولا يغيّر
استعجالاً ولا تخصّصاً — يقرأ فقط ويُصدر حكماً على الثقة."""

from app.domain.assessment import ClinicalFeatureSet, Demographics, ValidationReport
from app.domain.confidence import ConfidenceAssessment
from app.domain.prediction import DiseasePrediction, DiseasePredictionResult
from app.domain.rule_based_confidence_estimator import RuleBasedConfidenceEstimator
from app.domain.specialty import SpecialtyRecommendation
from app.domain.urgency import UrgencyAssessment, UrgencyLevel


def _estimator():
    return RuleBasedConfidenceEstimator()


def _features(unresolved_fields=None, validity_score=0.90, corrected=None, rejected=None):
    """ClinicalFeatureSet مبسَّط + ValidationReport متّسق داخله (نفس القيمة
    تُمرَّر لاحقاً كمعامل validation المنفصل بالمنفذ)."""
    validation = ValidationReport(
        corrected_fields=corrected or [],
        rejected_fields=rejected or [],
        warnings=[],
        validity_score=validity_score,
    )
    features = ClinicalFeatureSet(
        session_id="sid",
        demographics=Demographics(),
        symptoms=[],
        unresolved_fields=unresolved_fields or [],
        validation=validation,
    )
    return features, validation


def _predictions(*scores):
    """DiseasePredictionResult مرتّب تنازلياً (مثل RuleBasedDiseasePredictor)."""
    ordered = sorted(scores, reverse=True)
    return DiseasePredictionResult(
        predictions=[
            DiseasePrediction(disease=f"D{i}", score=s, explanation="x")
            for i, s in enumerate(ordered)
        ],
        predictor_version="test-v1",
    )


def _urgency(level=UrgencyLevel.NON_URGENT):
    return UrgencyAssessment(level=level, score=0.25, explanation="x")


def _specialty():
    return SpecialtyRecommendation(specialty="General Medicine", confidence=0.5, explanation="x")


def _estimate(features_validation, prediction_result, urgency=None, specialty=None):
    features, validation = features_validation
    return _estimator().estimate(
        clinical_features=features,
        validation=validation,
        prediction_result=prediction_result,
        urgency=urgency or _urgency(),
        specialty=specialty or _specialty(),
    )


# ----------------------------------------------------------------------
# نوع النتيجة — دائماً ConfidenceAssessment
# ----------------------------------------------------------------------
def test_always_returns_confidence_assessment_instance():
    result = _estimate(_features(), _predictions(0.7))
    assert isinstance(result, ConfidenceAssessment)


# ----------------------------------------------------------------------
# ثقة عالية — تنبؤ قوي + صلاحية عالية
# ----------------------------------------------------------------------
def test_high_confidence_strong_prediction_high_validity():
    # 0.85 base + 0.10 (validity>=0.95) = 0.95
    result = _estimate(_features(validity_score=0.97), _predictions(0.85))
    assert result.overall_confidence == 0.95
    assert result.requires_human_review is False
    assert result.explanation == "High feature validity and strong disease prediction."


# ----------------------------------------------------------------------
# ثقة متوسطة — تنبؤ متوسط + نطاق صلاحية محايد
# ----------------------------------------------------------------------
def test_medium_confidence_mid_validity_band_no_adjustment():
    # 0.70 base + 0 (0.80<=validity<0.95) = 0.70
    result = _estimate(_features(validity_score=0.85), _predictions(0.70))
    assert result.overall_confidence == 0.70
    assert result.requires_human_review is False


# ----------------------------------------------------------------------
# ثقة منخفضة — صلاحية منخفضة تُعاقِب
# ----------------------------------------------------------------------
def test_low_confidence_low_validity_penalty_applied():
    # 0.65 base - 0.15 (validity<0.80) = 0.50
    result = _estimate(_features(validity_score=0.60, corrected=["a"]), _predictions(0.65))
    assert result.overall_confidence == 0.50
    assert result.requires_human_review is True  # confidence<0.60 و validity<0.70


# ----------------------------------------------------------------------
# غياب التنبؤات — أساس 0.40 وعقوبة -0.20 (كلاهما بالمواصفة)
# ----------------------------------------------------------------------
def test_missing_predictions_uses_base_and_penalty():
    # لا تنبؤات: 0.40 base - 0.20 (no prediction) = 0.20
    empty = DiseasePredictionResult(predictions=[], predictor_version="v")
    result = _estimate(_features(validity_score=0.90), empty)
    assert result.overall_confidence == 0.20
    assert result.explanation == "No disease prediction was available."
    assert result.requires_human_review is True  # confidence<0.60


def test_missing_predictions_explanation_takes_top_priority():
    """رسالة 'لا تنبؤ' لها الأولوية حتى مع وجود حقول مُصحَّحة."""
    empty = DiseasePredictionResult(predictions=[], predictor_version="v")
    result = _estimate(_features(validity_score=0.50, corrected=["a"]), empty)
    assert result.explanation == "No disease prediction was available."


# ----------------------------------------------------------------------
# معلومات ناقصة — unresolved_fields > 3
# ----------------------------------------------------------------------
def test_many_unresolved_fields_reduce_confidence():
    many = ["a", "b", "c", "d"]  # > 3
    with_many = _estimate(_features(unresolved_fields=many, validity_score=0.90), _predictions(0.80))
    without = _estimate(_features(unresolved_fields=[], validity_score=0.90), _predictions(0.80))
    assert with_many.overall_confidence < without.overall_confidence
    # 0.80 - 0.10 = 0.70 مقابل 0.80
    assert with_many.overall_confidence == 0.70
    assert without.overall_confidence == 0.80


def test_exactly_three_unresolved_fields_no_penalty():
    """الحدّ الفاصل: 3 بالضبط ليست > 3 فلا عقوبة."""
    three = ["a", "b", "c"]
    result = _estimate(_features(unresolved_fields=three, validity_score=0.90), _predictions(0.80))
    assert result.overall_confidence == 0.80


# ----------------------------------------------------------------------
# تعديل الاستعجال — EMERGENCY يرفع +0.05، الباقي 0
# ----------------------------------------------------------------------
def test_emergency_urgency_adds_bonus():
    result = _estimate(
        _features(validity_score=0.85),
        _predictions(0.70),
        urgency=_urgency(UrgencyLevel.EMERGENCY),
    )
    # 0.70 + 0 (mid validity) + 0.05 (emergency) = 0.75
    assert result.overall_confidence == 0.75


def test_non_emergency_urgency_adds_nothing():
    for level in (UrgencyLevel.URGENT, UrgencyLevel.SEMI_URGENT, UrgencyLevel.NON_URGENT):
        result = _estimate(
            _features(validity_score=0.85), _predictions(0.70), urgency=_urgency(level)
        )
        assert result.overall_confidence == 0.70


# ----------------------------------------------------------------------
# المراجعة البشرية
# ----------------------------------------------------------------------
def test_human_review_triggered_by_low_confidence():
    empty = DiseasePredictionResult(predictions=[], predictor_version="v")
    result = _estimate(_features(validity_score=0.90), empty)  # 0.20 < 0.60
    assert result.requires_human_review is True


def test_human_review_triggered_by_low_validity_even_if_confidence_ok():
    # base عالٍ يُبقي الثقة >= 0.60، لكن validity<0.70 يفرض المراجعة
    # 0.95 - 0.15 (validity<0.80) = 0.80 >= 0.60، لكن validity=0.65<0.70
    result = _estimate(_features(validity_score=0.65), _predictions(0.95))
    assert result.overall_confidence >= 0.60
    assert result.requires_human_review is True


def test_no_human_review_when_confidence_and_validity_good():
    result = _estimate(_features(validity_score=0.90), _predictions(0.80))
    assert result.requires_human_review is False


def test_validity_exactly_070_does_not_trigger_review():
    """الحدّ الفاصل: 0.70 ليست < 0.70 فلا مراجعة من بند الصلاحية."""
    # 0.90 base - 0.15 (validity<0.80) = 0.75 >= 0.60
    result = _estimate(_features(validity_score=0.70), _predictions(0.90))
    assert result.requires_human_review is False


# ----------------------------------------------------------------------
# القصّ (clamp) بين 0.0 و1.0
# ----------------------------------------------------------------------
def test_confidence_clamped_to_upper_bound():
    # 1.0 base + 0.10 (validity>=0.95) + 0.05 (emergency) = 1.15 → 1.0
    result = _estimate(
        _features(validity_score=0.97),
        _predictions(1.0),
        urgency=_urgency(UrgencyLevel.EMERGENCY),
    )
    assert result.overall_confidence == 1.0


def test_confidence_clamped_to_lower_bound():
    # لا تنبؤات + صلاحية منخفضة + حقول ناقصة كثيرة:
    # 0.40 - 0.20 - 0.15 - 0.10 = -0.05 → 0.0
    empty = DiseasePredictionResult(predictions=[], predictor_version="v")
    result = _estimate(
        _features(unresolved_fields=["a", "b", "c", "d"], validity_score=0.10), empty
    )
    assert result.overall_confidence == 0.0
    assert result.requires_human_review is True


# ----------------------------------------------------------------------
# validity_score = None — يُعامَل كمحايد (قرار معماري موثَّق)
# ----------------------------------------------------------------------
def test_none_validity_is_neutral_no_crash():
    """أعراض قابلة للفحص غائبة → validity_score=None. لا انهيار، لا تعديل صلاحية."""
    result = _estimate(_features(validity_score=None), _predictions(0.70))
    assert result.overall_confidence == 0.70  # لا مكافأة ولا عقوبة
    assert result.requires_human_review is False


def test_none_validity_does_not_trigger_review_by_itself():
    """None صلاحية لا يُشغّل بند المراجعة (غير مؤكَّد، ليس '< 0.70')."""
    result = _estimate(_features(validity_score=None), _predictions(0.80))
    assert result.requires_human_review is False


def test_none_validity_still_reviewed_via_confidence_floor():
    """لكن لو انخفضت الثقة الكلّية (لا تنبؤ)، تبقى شبكة الأمان تعمل."""
    empty = DiseasePredictionResult(predictions=[], predictor_version="v")
    result = _estimate(_features(validity_score=None), empty)  # 0.20 < 0.60
    assert result.requires_human_review is True


# ----------------------------------------------------------------------
# التفسير — حتمي وغير فارغ دائماً
# ----------------------------------------------------------------------
def test_explanation_correction_message():
    result = _estimate(_features(validity_score=0.85, corrected=["a", "b"]), _predictions(0.75))
    assert result.explanation == "Several fields required correction, reducing confidence."


def test_explanation_low_validity_clinician_review():
    result = _estimate(_features(validity_score=0.55), _predictions(0.90))
    assert result.explanation == "Low validity score requires clinician review."


def test_explanation_never_empty_across_cases():
    empty = DiseasePredictionResult(predictions=[], predictor_version="v")
    cases = [
        _estimate(_features(validity_score=0.97), _predictions(0.85)),
        _estimate(_features(validity_score=0.85), _predictions(0.70)),
        _estimate(_features(validity_score=0.55), _predictions(0.90)),
        _estimate(_features(validity_score=None), _predictions(0.70)),
        _estimate(_features(validity_score=0.90), empty),
    ]
    for result in cases:
        assert isinstance(result.explanation, str) and result.explanation


def test_deterministic_across_repeated_calls():
    fv = _features(validity_score=0.90, corrected=["a"])
    preds = _predictions(0.7, 0.6)
    first = _estimate(fv, preds)
    second = _estimate(fv, preds)
    assert first == second


# ----------------------------------------------------------------------
# لا يغيّر مخرجات المراحل السابقة (يقرأ فقط)
# ----------------------------------------------------------------------
def test_does_not_mutate_prior_outputs():
    features, validation = _features(validity_score=0.90)
    preds = _predictions(0.80)
    urgency = _urgency(UrgencyLevel.EMERGENCY)
    specialty = _specialty()

    _estimator().estimate(features, validation, preds, urgency, specialty)

    # لا شيء تغيّر بالمخرجات السابقة
    assert urgency.level == UrgencyLevel.EMERGENCY
    assert specialty.specialty == "General Medicine"
    assert preds.predictions[0].score == 0.80
    assert validation.validity_score == 0.90
