"""اختبارات وحدة لـAssessmentExplainer + مُحلِّله (Phase 3.8) — يشرح التقييم
المُحسَب سلفاً بالعربية فقط، بتدهور حتمي لطيف عند فشل الـLLM. مزوّدون وهميون
حقيقيون (Fakes لا مكتبات mocking)، نفس روح باقي اختبارات الوحدة."""

import json
import re

import pytest

from app.domain.assessment import (
    ClinicalFeatureSet,
    Demographics,
    Lifestyle,
    SymptomDescriptors,
    SymptomFeature,
)
from app.domain.confidence import ConfidenceAssessment
from app.domain.explanation import AssessmentExplanation
from app.domain.ports import Completion
from app.domain.prediction import DiseasePrediction, DiseasePredictionResult
from app.domain.specialty import SpecialtyRecommendation
from app.domain.urgency import UrgencyAssessment, UrgencyLevel
from app.exceptions import AssessmentExplanationError
from app.parsing.assessment_explainer_parser import (
    parse_assessment_explanation,
    validate_explanation_shape,
)
from app.prompts.assessment_explainer_builder import AssessmentExplainerPromptBuilder
from app.services.assessment_explainer import AssessmentExplainer

_ARABIC_RE = re.compile(r"[؀-ۿ]")

_VALID_PAYLOAD = {
    "summary": "الشكوى صداع مع حرارة.",
    "medical_reasoning": "المؤشرات تتوافق مع حالة حموية.",
    "recommendation": "يُنصح بحجز موعد مع طب الأعصاب.",
    "disclaimer": "هذا التقييم ليس تشخيصاً طبياً نهائياً.",
}


# ----------------------------------------------------------------------
# مزوّدون وهميون
# ----------------------------------------------------------------------
class StaticProvider:
    """يُرجع نصّاً ثابتاً (JSON أو غيره) ويعدّ الاستدعاءات ويحفظ آخر تعليمات."""

    name = "static"

    def __init__(self, text: str):
        self._text = text
        self.calls = 0
        self.last_user_prompt = None

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        self.calls += 1
        self.last_user_prompt = user_prompt
        return Completion(self._text, model="static")


class RaisingProvider:
    name = "raising"

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        raise RuntimeError("تعذّر الاتصال بالمزوّد")


def _json_provider(payload: dict) -> StaticProvider:
    return StaticProvider(json.dumps(payload, ensure_ascii=False))


def _explainer(provider):
    return AssessmentExplainer(provider=provider, prompt_builder=AssessmentExplainerPromptBuilder())


# ----------------------------------------------------------------------
# صانعات المدخلات
# ----------------------------------------------------------------------
def _features(symptom_name="صداع شديد", **demo):
    return ClinicalFeatureSet(
        session_id="sid",
        demographics=Demographics(**demo),
        symptoms=[
            SymptomFeature(
                name=symptom_name,
                negated=False,
                extraction_confidence=0.9,
                is_primary=True,
                descriptors=SymptomDescriptors(severity_0_10=7),
            )
        ],
        temperature_c=38.5,
        lifestyle=Lifestyle(smoking=False),
    )


def _predictions(*pairs):
    return DiseasePredictionResult(
        predictions=[
            DiseasePrediction(disease=d, score=s, explanation="x") for d, s in pairs
        ],
        predictor_version="test-v1",
    )


def _urgency(level=UrgencyLevel.SEMI_URGENT):
    return UrgencyAssessment(level=level, score=0.5, explanation="x")


def _specialty(name="Neurology"):
    return SpecialtyRecommendation(specialty=name, confidence=0.7, explanation="x")


def _confidence(requires_human_review=False, overall=0.7):
    return ConfidenceAssessment(
        overall_confidence=overall,
        requires_human_review=requires_human_review,
        explanation="x",
    )


def _explain(provider, features=None, predictions=None, urgency=None, specialty=None, confidence=None):
    return _explainer(provider).explain(
        clinical_features=features or _features(),
        prediction_result=predictions or _predictions(("Febrile Illness", 0.6)),
        urgency=urgency or _urgency(),
        specialty=specialty or _specialty(),
        confidence=confidence or _confidence(),
    )


# ======================================================================
# المُحلِّل (parser) — يجب أن يرفض المشوّه والناقص
# ======================================================================
def test_parser_accepts_valid_json():
    result = parse_assessment_explanation(json.dumps(_VALID_PAYLOAD, ensure_ascii=False))
    assert isinstance(result, AssessmentExplanation)
    assert result.summary == _VALID_PAYLOAD["summary"]
    assert result.disclaimer == _VALID_PAYLOAD["disclaimer"]


def test_parser_accepts_fenced_json():
    fenced = "```json\n" + json.dumps(_VALID_PAYLOAD, ensure_ascii=False) + "\n```"
    result = parse_assessment_explanation(fenced)
    assert result.recommendation == _VALID_PAYLOAD["recommendation"]


def test_parser_rejects_malformed_json():
    with pytest.raises(AssessmentExplanationError):
        parse_assessment_explanation("ليس JSON إطلاقاً")


def test_parser_rejects_empty_output():
    with pytest.raises(AssessmentExplanationError):
        parse_assessment_explanation("")


def test_parser_rejects_missing_field():
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "recommendation"}
    with pytest.raises(AssessmentExplanationError):
        parse_assessment_explanation(json.dumps(payload, ensure_ascii=False))


def test_parser_rejects_empty_string_field():
    payload = {**_VALID_PAYLOAD, "summary": "   "}
    with pytest.raises(AssessmentExplanationError):
        parse_assessment_explanation(json.dumps(payload, ensure_ascii=False))


def test_validate_shape_passes_for_valid_and_raises_for_missing():
    validate_explanation_shape(json.dumps(_VALID_PAYLOAD, ensure_ascii=False))  # لا استثناء
    bad = {k: v for k, v in _VALID_PAYLOAD.items() if k != "disclaimer"}
    with pytest.raises(AssessmentExplanationError):
        validate_explanation_shape(json.dumps(bad, ensure_ascii=False))


# ======================================================================
# الخدمة — المسار الناجح (JSON صالح من الـLLM)
# ======================================================================
def test_valid_json_is_parsed_and_returned():
    provider = _json_provider(_VALID_PAYLOAD)
    result = _explain(provider)
    assert provider.calls == 1
    # الملخّص والتوصية حتميان — LLM يُنتج medical_reasoning فقط
    assert "23" in result.summary or "صداع" in result.summary
    assert result.medical_reasoning == _VALID_PAYLOAD["medical_reasoning"]
    assert "Neurology" in result.summary or "Neurology" in result.recommendation


def test_prompt_never_contains_raw_messages():
    """المبدأ الحاسم: معلومات منظَّمة فقط — لا رسائل مريض خام تصل للـLLM."""
    provider = _json_provider(_VALID_PAYLOAD)
    features = _features()
    # نضع نصّاً مميّزاً بالرسائل الخام ثم نتأكد أنه لا يظهر بالتعليمات إطلاقاً
    features.raw_messages = ["نصّ_خام_سرّي_يجب_ألا_يظهر"]
    _explainer(provider).explain(
        features, _predictions(("Febrile Illness", 0.6)), _urgency(), _specialty(), _confidence()
    )
    assert "نصّ_خام_سرّي_يجب_ألا_يظهر" not in provider.last_user_prompt


# ======================================================================
# الخدمة — التدهور اللطيف (فشل الـLLM أو JSON مشوّه) → بديل حتمي
# ======================================================================
def test_invalid_json_degrades_to_fallback_not_raises():
    result = _explain(StaticProvider("رد ليس JSON"))
    assert isinstance(result, AssessmentExplanation)
    assert result.disclaimer  # بديل حتمي كامل


def test_missing_field_degrades_to_fallback():
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "medical_reasoning"}
    result = _explain(_json_provider(payload))
    # البديل الحتمي يملأ كل الحقول
    assert result.medical_reasoning


def test_provider_exception_degrades_to_fallback():
    result = _explain(RaisingProvider())
    assert isinstance(result, AssessmentExplanation)
    assert result.summary


# ======================================================================
# قواعد البديل الحتمي — مراجعة بشرية
# ======================================================================
def test_human_review_recommendation_encourages_professional_evaluation():
    result = _explain(
        RaisingProvider(),
        urgency=_urgency(UrgencyLevel.SEMI_URGENT),
        confidence=_confidence(requires_human_review=True),
    )
    assert "طبيب مختص" in result.recommendation or "تقييم مهني" in result.recommendation


def test_no_human_review_recommendation_has_no_forced_professional_clause():
    result = _explain(
        RaisingProvider(),
        urgency=_urgency(UrgencyLevel.SEMI_URGENT),
        confidence=_confidence(requires_human_review=False),
    )
    assert "تقييم مهني" not in result.recommendation


def test_emergency_recommendation_points_to_emergency_department():
    result = _explain(RaisingProvider(), urgency=_urgency(UrgencyLevel.EMERGENCY))
    assert "الطوارئ" in result.recommendation


def test_non_urgent_recommendation_allows_home_monitoring():
    result = _explain(
        RaisingProvider(),
        urgency=_urgency(UrgencyLevel.NON_URGENT),
        confidence=_confidence(requires_human_review=False),
    )
    assert "المنزل" in result.recommendation


# ======================================================================
# قواعد البديل الحتمي — التنبؤات
# ======================================================================
def test_fallback_mentions_only_provided_diseases():
    result = _explain(
        RaisingProvider(), predictions=_predictions(("Febrile Illness", 0.6), ("Severe Condition", 0.7))
    )
    assert "Febrile Illness" in result.medical_reasoning
    assert "Severe Condition" in result.medical_reasoning


def test_empty_predictions_invents_no_disease():
    empty = DiseasePredictionResult(predictions=[], predictor_version="v")
    result = _explain(RaisingProvider(), predictions=empty)
    assert isinstance(result, AssessmentExplanation)
    # لا يخترع مرضاً: لا أحرف لاتينية لاسم مرض، وينصّ على غياب التنبؤات
    assert "لا تتوفّر تنبؤات" in result.medical_reasoning


# ======================================================================
# ضمانات عامّة
# ======================================================================
def test_disclaimer_always_mentions_not_final_diagnosis_in_fallback():
    result = _explain(RaisingProvider())
    assert "ليس تشخيصاً طبياً نهائياً" in result.disclaimer


def test_fallback_output_is_arabic():
    result = _explain(RaisingProvider())
    for text in (result.summary, result.medical_reasoning, result.recommendation, result.disclaimer):
        assert _ARABIC_RE.search(text)


def test_all_fields_non_empty_in_fallback():
    result = _explain(RaisingProvider())
    assert all(
        isinstance(t, str) and t.strip()
        for t in (result.summary, result.medical_reasoning, result.recommendation, result.disclaimer)
    )


def test_does_not_mutate_inputs():
    features = _features()
    preds = _predictions(("Febrile Illness", 0.6))
    urgency = _urgency(UrgencyLevel.EMERGENCY)
    specialty = _specialty("Neurology")
    confidence = _confidence(requires_human_review=True, overall=0.3)

    _explainer(RaisingProvider()).explain(features, preds, urgency, specialty, confidence)

    assert urgency.level == UrgencyLevel.EMERGENCY
    assert specialty.specialty == "Neurology"
    assert preds.predictions[0].score == 0.6
    assert confidence.overall_confidence == 0.3
    assert confidence.requires_human_review is True
