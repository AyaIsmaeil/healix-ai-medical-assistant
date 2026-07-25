"""اختبارات وحدة لمُحلِّل استخراج ميزات التقييم — يجب أن يكون متساهلاً بالمفاتيح."""

import json

import pytest

from app.exceptions import FeatureExtractionError
from app.parsing.assessment_extraction_parser import (
    parse_extraction_result,
    validate_extraction_shape,
)


def test_parses_full_valid_json():
    text = json.dumps({
        "age": 34, "gender": "male", "smoking": False,
        "temperature": 39.0, "duration": "منذ يومين", "severity": 8,
    }, ensure_ascii=False)
    result = parse_extraction_result(text)
    assert result.age == 34
    assert result.gender == "male"
    assert result.smoking is False
    assert result.temperature == 39.0
    assert result.duration_text == "منذ يومين"
    assert result.severity == 8


def test_strips_code_fences():
    text = "```json\n" + json.dumps({"age": 25}) + "\n```"
    result = parse_extraction_result(text)
    assert result.age == 25


def test_missing_keys_become_none_not_error():
    text = json.dumps({"age": 25})  # بلا باقي المفاتيح
    result = parse_extraction_result(text)
    assert result.age == 25
    assert result.gender is None
    assert result.severity is None


def test_wrong_shape_json_resolves_to_all_none():
    """محاكاة تعارض json_schema مع عقد المقابلة — لا ينهار، يُحسم كل شيء None."""
    text = json.dumps({"finished": True})  # شكل قرار المقابلة، لا شكل الاستخراج
    result = parse_extraction_result(text)
    assert result.age is None
    assert result.gender is None
    assert result.smoking is None
    assert result.temperature is None
    assert result.duration_text is None
    assert result.severity is None


def test_wrong_type_values_become_none():
    text = json.dumps({"age": "غير معروف", "smoking": "ربما"})
    result = parse_extraction_result(text)
    assert result.age is None
    assert result.smoking is None


def test_bool_not_coerced_to_int_for_age():
    text = json.dumps({"age": True})
    result = parse_extraction_result(text)
    assert result.age is None


def test_unparseable_json_raises():
    with pytest.raises(FeatureExtractionError):
        parse_extraction_result("ليس JSON إطلاقاً")


def test_empty_text_raises():
    with pytest.raises(FeatureExtractionError):
        parse_extraction_result("")


# ----------------------------------------------------------------------
# validate_extraction_shape — بوّابة إعادة المحاولة الصارمة (تُستخدم
# كـresponse_validator داخل QwenOpenRouterProvider، لا لبناء النتيجة النهائية)
# ----------------------------------------------------------------------
def test_validate_extraction_shape_passes_with_all_six_keys():
    text = json.dumps({
        "age": None, "gender": None, "smoking": None,
        "temperature": None, "duration": None, "severity": None,
    })
    validate_extraction_shape(text)  # لا يرفع استثناءً


def test_validate_extraction_shape_rejects_interview_shape():
    """نفس السيناريو الذي كان يسبّب البقاء "غير محلول" بصمت قبل الإصلاح —
    الآن يُرفض صراحة فتُشغَّل إعادة المحاولة داخل المزوّد."""
    text = json.dumps({"finished": True})
    with pytest.raises(FeatureExtractionError):
        validate_extraction_shape(text)


def test_validate_extraction_shape_rejects_partial_keys():
    text = json.dumps({"age": 30, "gender": "male"})  # ناقص 4 مفاتيح
    with pytest.raises(FeatureExtractionError):
        validate_extraction_shape(text)
