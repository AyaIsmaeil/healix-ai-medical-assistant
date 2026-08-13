"""اختبارات SlotAnswerValidator — تحقّق حتمي من إجابات المقابلة."""

from app.domain.slot_answer_validator import (
    SlotAnswerStatus,
    apply_slot_answer_to_record,
    validate_slot_answer,
)
from app.domain.clinical_record import ClinicalRecord


def test_age_bare_number_valid():
    result = validate_slot_answer("context:age", "23")
    assert result.status == SlotAnswerStatus.VALID
    assert result.age == 23


def test_age_invalid_returns_clarification():
    result = validate_slot_answer("context:age", "كبير")
    assert result.status == SlotAnswerStatus.NEEDS_CLARIFICATION


def test_gender_female_valid():
    result = validate_slot_answer("context:gender", "انثى")
    assert result.status == SlotAnswerStatus.VALID
    assert result.gender == "female"


def test_yes_no_dyspnea_slot():
    assert validate_slot_answer("dyspnea@ضيق نفس", "نعم").yes_no is True
    assert validate_slot_answer("dyspnea@ضيق نفس", "لا").yes_no is False


def test_off_topic_on_yes_no_needs_clarification():
    result = validate_slot_answer("sweating@ضغط", "ارتفاع ضغط الدم")
    assert result.status == SlotAnswerStatus.NEEDS_CLARIFICATION


def test_severity_shadeed_maps_to_8():
    result = validate_slot_answer("severity@ضيق نفس", "شديد")
    assert result.status == SlotAnswerStatus.VALID
    assert result.severity_numeric == 8


def test_chronic_disease_named_directly():
    result = validate_slot_answer("context:chronic_diseases", "ارتفاع ضغط الدم")
    assert result.status == SlotAnswerStatus.VALID
    assert result.chronic_condition == "ارتفاع ضغط الدم"


def test_apply_slot_answer_sets_record_age():
    record = ClinicalRecord()
    result = validate_slot_answer("context:age", "23")
    apply_slot_answer_to_record(record, "context:age", result)
    assert record.age == 23
